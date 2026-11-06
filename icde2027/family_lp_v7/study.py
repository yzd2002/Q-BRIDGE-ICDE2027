"""Sealed input-only scheduling and execution for the registered V7 comparison.

This driver does not compute capacities, decompose results, or read historical
answers. The separately reviewed service produces proofs; a later independent
audit interprets them. Preparation performs no mathematical solver calls.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import types


SEED = 2026110601
HEX = re.compile(r"[0-9a-f]{64}\Z")
FAMILY_KEYS = {"program", "context_sizes", "exact_weights", "exact_reference"}
LIMITS = {"startup_limit_s": 30, "service_limit_s": 10,
          "diagnostic_limit_s": 30, "memory_limit_bytes": 8 * 1024 ** 3,
          "max_line_bytes": 32 * 1024 ** 2}
GLOBAL_LIMIT_SECONDS = 8 * 60 * 60


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": h.hexdigest()}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def loads(raw):
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=_unique_pairs,
                      parse_constant=invalid_constant)


def read(path):
    return loads(Path(path).read_bytes())


def records(path):
    with Path(path).open("rb") as stream:
        for line in stream:
            need(line.endswith(b"\n"), "Incomplete JSONL record")
            yield loads(line)


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_state(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def bound_file(path, expected):
    actual = identity(path)
    need(actual == expected, "File identity differs: " + str(path))


def prepared_inventory(directory):
    directory = Path(directory).resolve()
    preparation_path = directory / "PREPARATION.json"
    prep = read(preparation_path)
    need(prep["status"] == "PREPARED_NOT_RUN", "Wrong preparation status")
    result = {str(preparation_path): identity(preparation_path)}
    for name, expected in prep["files"].items():
        need(Path(name).name == name, "Nonlocal preparation filename")
        path = directory / name
        bound_file(path, expected)
        result[str(path)] = expected
    return prep, result


def index_unique(items, field):
    result = {}
    for item in items:
        key = item[field]
        need(type(key) is str and HEX.fullmatch(key), "Malformed task identity")
        need(key not in result, "Duplicate task identity")
        result[key] = item
    return result


def family_payload(item):
    source = item["request"]
    need(set(source) == FAMILY_KEYS | {"schema_version"}, "Unexpected family fields")
    return {key: source[key] for key in sorted(FAMILY_KEYS)}


def load_inputs(synthetic, coat):
    """Load only requests and input memberships; never consume saved answers."""
    synthetic, coat = Path(synthetic), Path(coat)
    families = index_unique(records(synthetic / "family_requests.jsonl"), "task_id")
    programs = index_unique(records(synthetic / "programs.jsonl"), "program_identity")
    fixed = index_unique(records(synthetic / "fixed_objectives.jsonl"), "task_id")
    real = index_unique(records(coat / "family_requests.jsonl"), "task_id")
    for key, item in families.items():
        need(digest(item["request"]) == key, "Synthetic request identity differs")
        family_payload(item)
    for key, item in programs.items():
        need(digest(item["program"]) == key, "Program matrix identity differs")
    for key, item in fixed.items():
        task = item["task"]
        need(set(task) == {"schema_version", "program_identity", "objective"}, "Unexpected fixed fields")
        need(task["schema_version"] == "verified-fixed-task/1" and digest(task) == key,
             "Fixed task identity differs")
        need(task["program_identity"] in programs, "Missing fixed program")
    for key, item in real.items():
        need(type(item["partition_supported"]) is bool, "Bad domain flag")
        need(digest(item["request"]) == key, "Coat request identity differs")
        family_payload(item)
    # Values of old classifications, lower bounds and eligibility are not used.
    family_strata, fixed_strata = defaultdict(set), defaultdict(set)
    occurrence_count = 0
    for row in records(synthetic / "occurrences.jsonl"):
        occurrence_count += 1
        stratum = canonical([row["group"], row["size"]]).decode("ascii")
        rid = row["family_task_id"]
        need(rid in families, "Occurrence has missing family request")
        family_strata[rid].add(stratum)
        for fid in row["fixed_tasks"].values():
            need(fid in fixed, "Occurrence has missing fixed request")
            fixed_strata[fid].add(stratum)
    need(set(family_strata) == set(families), "Missing family memberships")
    need(set(fixed_strata) == set(fixed), "Missing fixed memberships")
    timing = read(synthetic / "timing_selection.json")
    timing_ids = timing["unique_request_ids"]
    need(len(timing_ids) == len(set(timing_ids)) and set(timing_ids) <= set(families),
         "Invalid timing selection")
    return {"families": families, "programs": programs, "fixed": fixed,
            "coat": real, "family_strata": family_strata, "fixed_strata": fixed_strata,
            "timing_ids": timing_ids, "occurrence_count": occurrence_count}


def balanced_ids(ids, memberships, seed):
    """Input-only owner-stratum round robin; shared memberships stay in inputs."""
    groups = defaultdict(list)
    for key in sorted(ids):
        need(key in memberships and memberships[key], "Missing scheduling stratum")
        groups[min(memberships[key])].append(key)
    rng = random.Random(seed)
    order = sorted(groups)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(groups[key])
        groups[key] = deque(groups[key])
    result = []
    while order:
        remaining = []
        for key in order:
            result.append(groups[key].popleft())
            if groups[key]:
                remaining.append(key)
        order = remaining
    need(len(result) == len(set(result)) == len(ids), "Scheduling census differs")
    return result


def request_for(data, namespace, task_id):
    if namespace in ("synthetic", "coat"):
        source = data["families"] if namespace == "synthetic" else data["coat"]
        return family_payload(source[task_id])
    need(namespace == "fixed", "Unknown request namespace")
    task = data["fixed"][task_id]["task"]
    return {"program": data["programs"][task["program_identity"]]["program"],
            "exact_objective": task["objective"]}


def receiver_digest(payload, method):
    """Map canonical prepared scalars to the documented receiver schema.

    A task identity, transport identity and receiver identity are distinct.
    The worker and final independent audit recompute the receiver identity.
    """
    if method in ("S", "G_partition"):
        return digest({"schema_version": "partition-capacity-input/1", **payload})
    if method == "G_general":
        return digest({"schema_version": "family-capacity-input/1", "program": payload["program"],
                       "context_sizes": payload["context_sizes"], "weights": payload["exact_weights"],
                       "reference": payload["exact_reference"]})
    need(method == "G_fixed", "Unknown receiver method")
    return digest({"schema_version": "fixed-policy-optimum-input/1", "program": payload["program"],
                   "objective": payload["exact_objective"]})


def make_queue(data):
    queue = []
    def add(phase, namespace, rid, method, repetition=0):
        payload = request_for(data, namespace, rid)
        row = {"sequence": len(queue), "phase": phase, "namespace": namespace,
               "task_id": rid, "method": method, "round": repetition,
               "request_sha256": digest(payload), "request_bytes": len(canonical(payload)),
               "receiver_input_sha256": receiver_digest(payload, method)}
        row["call_id"] = digest(row)
        queue.append(row)
    synthetic_order = balanced_ids(data["families"], data["family_strata"], SEED)
    for i, rid in enumerate(synthetic_order):
        for method in (("S", "G_partition") if i % 2 == 0 else ("G_partition", "S")):
            add("synthetic_quality", "synthetic", rid, method)
    for rid in balanced_ids(data["fixed"], data["fixed_strata"], SEED + 1):
        add("fixed_quality", "fixed", rid, "G_fixed")
    real_order = sorted(data["coat"])
    random.Random(SEED + 2).shuffle(real_order)
    supported_ordinal = 0
    for rid in real_order:
        if data["coat"][rid]["partition_supported"]:
            for method in (("S", "G_partition") if supported_ordinal % 2 == 0 else ("G_partition", "S")):
                add("coat_quality", "coat", rid, method)
            supported_ordinal += 1
        else:
            add("coat_quality", "coat", rid, "G_general")
    for repetition in range(3):
        order = balanced_ids(data["timing_ids"], data["family_strata"], SEED + 3 + repetition)
        for i, rid in enumerate(order):
            for method in (("S", "G_partition") if (i + repetition) % 2 == 0 else ("G_partition", "S")):
                add("controlled_timing", "synthetic", rid, method, repetition + 1)
    return queue


def prepare(synthetic, coat, runtime_path, dependency_path, plan_path, service_path, output, cpu):
    output = Path(output).resolve()
    need(not output.exists(), "Output already exists")
    need(type(cpu) is int and cpu in os.sched_getaffinity(0) and cpu != 4,
         "CPU unavailable or reserved")
    synth_prep, synth_files = prepared_inventory(synthetic)
    coat_prep, coat_files = prepared_inventory(coat)
    data = load_inputs(synthetic, coat)
    expected = (2160, 7762, 125, 78, 5760)
    actual = tuple(len(data[k]) for k in ("families", "fixed", "coat", "timing_ids")) + (data["occurrence_count"],)
    need(actual == expected, "Registered input census differs")
    need(sum(row["partition_supported"] for row in data["coat"].values()) == 95,
         "Coat structural census differs")
    queue = make_queue(data)
    counts = dict(Counter(row["phase"] for row in queue))
    need(counts == {"synthetic_quality": 4320, "fixed_quality": 7762,
                    "coat_quality": 220, "controlled_timing": 468}, "Queue census differs")
    runtime_path, dependency_path, plan_path, service_path = [Path(p).resolve() for p in (runtime_path, dependency_path, plan_path, service_path)]
    runtime = read(runtime_path)
    runtime_files = {}
    for name, entry in runtime.items():
        need(type(entry) is dict and {"path", "sha256"} <= set(entry), "Invalid runtime binding")
        path = Path(entry["path"]).resolve()
        actual_id = identity(path)
        need(actual_id["sha256"] == entry["sha256"], "Runtime pin differs: " + name)
        runtime_files[str(path)] = actual_id
    dependency_record = read(dependency_path)
    need(dependency_record.get("status") == "PASS_SYNTHETIC_RUNTIME_PROBE",
         "Dependency setup probe not complete")
    dependency_files = dependency_record["files"]
    need(type(dependency_files) is dict and dependency_files, "No runtime dependency files")
    for path, expected in dependency_files.items():
        need(Path(path).is_absolute(), "Non-absolute runtime dependency")
        bound_file(path, expected)
    source_files = {str(p): identity(p) for p in (runtime_path, dependency_path, plan_path, service_path, Path(__file__).resolve(), Path(sys.executable).resolve())}
    control = {"schema_version": "verified-family-study-preparation/1",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "PREPARED_NOT_RUN", "cpu": cpu, "seed": SEED,
               "worker_integer_string_digits": 100000,
               "limits": LIMITS, "global_limit_seconds": GLOBAL_LIMIT_SECONDS,
               "synthetic_directory": str(Path(synthetic).resolve()),
               "coat_directory": str(Path(coat).resolve()),
               "runtime_config": runtime, "service_path": str(service_path),
               "input_files": {**synth_files, **coat_files},
               "source_files": source_files, "runtime_files": runtime_files,
               "dependency_files": dependency_files,
               "counts": counts, "calls_planned": len(queue),
               "domain_excluded_S_coat": sorted(rid for rid, row in data["coat"].items() if not row["partition_supported"])}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "queue.jsonl").open("xb") as stream:
        for row in queue:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    control["queue"] = identity(output / "queue.jsonl")
    write_new(output / "PREPARATION.json", control)
    return {"status": "PREPARED_NOT_RUN", "calls_planned": len(queue), "counts": counts,
            "preparation": identity(output / "PREPARATION.json")}


def execute(prepared, gate_path, output):
    started_ns = time.monotonic_ns()
    started_utc = datetime.now(timezone.utc).isoformat()
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    need(not output.exists(), "Run output already exists")
    prep_path = prepared / "PREPARATION.json"
    prep, gate = read(prep_path), read(gate_path)
    need(gate.get("approved") is True, "Missing independent admission gate")
    need(gate.get("preparation") == identity(prep_path), "Gate preparation differs")
    need(type(gate.get("reviews")) is dict and gate["reviews"], "Missing bound independent reviews")
    for path, expected in gate["reviews"].items():
        bound_file(path, expected)
    for group in ("input_files", "source_files", "runtime_files", "dependency_files"):
        for path, expected in prep[group].items():
            bound_file(path, expected)
    need(prep["limits"] == LIMITS and prep["global_limit_seconds"] == GLOBAL_LIMIT_SECONDS,
         "Resource contract differs")
    need(prep["worker_integer_string_digits"] == 100000, "Worker integer conversion guard differs")
    bound_file(prepared / "queue.jsonl", prep["queue"])
    queue = list(records(prepared / "queue.jsonl"))
    data = load_inputs(prep["synthetic_directory"], prep["coat_directory"])
    need(queue == make_queue(data), "Queue differs from input-only registration")
    need(prep["cpu"] in os.sched_getaffinity(0) and prep["cpu"] != 4, "CPU unavailable")
    os.sched_setaffinity(0, {prep["cpu"]})
    service_path = Path(prep["service_path"])
    service = types.ModuleType("verified_family_service_v7")
    service.__file__ = str(service_path)
    sys.modules[service.__name__] = service
    service_bytes = service_path.read_bytes()
    need(hashlib.sha256(service_bytes).hexdigest() == prep["source_files"][str(service_path)]["sha256"],
         "Service source changed during load")
    exec(compile(service_bytes, str(service_path), "exec"), service.__dict__)
    output.mkdir(parents=True, exist_ok=False)
    (output / "calls").mkdir()
    write_new(output / "ADMISSION.json", {"gate_path": str(Path(gate_path).resolve()),
               "gate": identity(gate_path), "preparation_path": str(prep_path),
               "preparation": identity(prep_path), "started_at_utc": started_utc,
               "hostname": os.uname().nodename, "pid": os.getpid(),
               "affinity": sorted(os.sched_getaffinity(0)),
               "python": sys.version, "python_executable": str(Path(sys.executable).resolve())})
    completed, incomplete, counts = 0, 0, Counter()
    status = "running"
    ledger_path = output / "outputs.jsonl"
    with ledger_path.open("xb") as ledger:
        for row in queue:
            if (time.monotonic_ns() - started_ns) / 1e9 >= GLOBAL_LIMIT_SECONDS:
                status = "global_cap_incomplete"
                break
            payload = request_for(data, row["namespace"], row["task_id"])
            need(digest(payload) == row["request_sha256"], "Runtime payload differs")
            call_started = time.monotonic_ns()
            contract_error = None
            returned = None
            try:
                result = service.run_call(payload, method=row["method"], runtime_config=prep["runtime_config"],
                                          cpu=prep["cpu"], **LIMITS)
                # Preserve JSON-safe raw returns even if their contract fails.
                canonical(result)
                returned = result
                # A terminal refusal/timeout is retained, not converted into an answer.
                need(type(result) is dict and type(result.get("record_complete")) is bool,
                     "Service did not return its terminal record contract")
                need(result.get("request_sha256") == row["request_sha256"],
                     "Service transport request binding differs")
                need(result.get("method") == row["method"] and result.get("request_bytes") == row["request_bytes"],
                     "Service method or request length differs")
                if result.get("ready") is not None:
                    need(result["ready"]["integer_string_digits"] == prep["worker_integer_string_digits"],
                         "Worker integer conversion guard receipt differs")
                response = result.get("response")
                if type(response) is dict and response.get("status") == "verified_optimal":
                    need(response["certificate"]["input_sha256"] == row["receiver_input_sha256"]
                         == response["receiver_receipt"]["input_sha256"],
                         "Service receiver request binding differs")
                if result["record_complete"]:
                    inventory = result["diagnostics"]["runtime_inventory"]
                    for key in ("missing_module_files", "missing_mapped_files", "deleted_mappings"):
                        need(inventory[key] == [], "Runtime provenance contains missing/deleted files")
                    actual_runtime = (set(inventory["module_files"]) | set(inventory["mapped_files"])
                                      | set(inventory["module_cache_files"]))
                    need(actual_runtime <= set(prep["dependency_files"]),
                         "Worker loaded a file outside the registered runtime closure")
            except Exception as error:
                contract_error = type(error).__name__ + ": " + str(error)
                result = {"schema_version": "verified-family-driver-error/1",
                          "status": "service_contract_error", "record_complete": False,
                          "public_response": None, "error": contract_error,
                          "method": row["method"], "request_sha256": row["request_sha256"],
                          "unaccepted_returned_record": returned}
            item = {"registration": row, "result": result,
                    "parent_call_wall_ns": time.monotonic_ns() - call_started}
            path = output / "calls" / (f"{row['sequence']:05d}_" + row["call_id"] + ".json")
            write_new(path, item)
            ledger_row = {"sequence": row["sequence"], "call_id": row["call_id"],
                          "path": str(path.relative_to(output)), **identity(path)}
            ledger.write(canonical(ledger_row) + b"\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            completed += 1
            incomplete += not result["record_complete"]
            counts[row["phase"]] += 1
            write_state(output / "progress.json", {"status": "running", "calls_planned": len(queue),
                        "calls_recorded": completed, "incomplete_calls": incomplete,
                        "phase_counts": dict(counts), "last_sequence": row["sequence"]})
            if contract_error is not None:
                status = "driver_error_incomplete"
                break
        else:
            status = "completed" if incomplete == 0 else "completed_with_incomplete_calls"
    postflight, changed = {}, []
    for group in ("input_files", "source_files", "runtime_files", "dependency_files"):
        for path, expected in prep[group].items():
            try:
                actual = identity(path)
                postflight[path] = actual
                if actual != expected:
                    changed.append({"path": path, "expected": expected, "actual": actual})
            except Exception as error:
                changed.append({"path": path, "expected": expected,
                                "error": type(error).__name__ + ": " + str(error)})
    if changed:
        status = "invalid_source_postflight"
    completion = {"schema_version": "verified-family-study-completion/1", "status": status,
                  "started_at_utc": started_utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "wall_ns": time.monotonic_ns() - started_ns, "calls_planned": len(queue),
                  "calls_recorded": completed, "not_started": len(queue) - completed,
                  "incomplete_calls": incomplete, "phase_counts": dict(counts),
                  "outputs_ledger": identity(ledger_path), "source_postflight": postflight,
                  "changed_inputs_or_sources": changed,
                  "interpretation": "Execution record only; no scientific result audit or acceptance claim"}
    write_new(output / "COMPLETION.json", completion)
    write_state(output / "progress.json", {k: v for k, v in completion.items() if k != "source_postflight"})
    return {k: v for k, v in completion.items() if k != "source_postflight"}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for arg in ("synthetic", "coat", "runtime", "dependencies", "plan", "service", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--cpu", type=int, required=True)
    r = sub.add_parser("run")
    for arg in ("prepared", "gate", "output"):
        r.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.synthetic, args.coat, args.runtime, args.dependencies, args.plan, args.service, args.output, args.cpu)
    else:
        result = execute(args.prepared, args.gate, args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if args.command == "run" and result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
