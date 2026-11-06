"""Single request worker. Only its selected, pinned algorithm is loaded.

No study files, policies, old results, or other method's answers are accessed.
There are no child processes. RLIMIT_AS applies to this process's virtual space.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import types


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def forbidden(_):
        raise ValueError("floating/nonfinite JSON number forbidden")
    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
                      parse_float=forbidden, parse_constant=forbidden)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def checked_bytes(entry):
    raw = Path(entry["path"]).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == entry["sha256"], "runtime source SHA mismatch: " + entry["path"])
    return raw


def load_source(entry, name):
    raw = checked_bytes(entry)
    module = types.ModuleType(name)
    module.__file__ = entry["path"]
    sys.modules[name] = module
    exec(compile(raw, entry["path"], "exec"), module.__dict__)
    return module


def send(value, limit):
    raw = encode(value)
    if len(raw) > limit:
        raise ValueError("output line exceeds registered byte limit")
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()
    return len(raw)


def runtime_inventory():
    """Actual file identities by path only, collected after the public reply.

    The outer study must bind these files' bytes before/after the full batch and
    check each observed set against its admitted closure. This inventory is not
    a claim that every historical import is still present in sys.modules.
    """
    modules, caches, missing_modules, mappings, missing_mappings, deleted = set(), set(), set(), set(), set(), set()
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if type(value) is not str:
            continue
        path = Path(value).absolute()
        if path.is_file():
            modules.add(str(path.resolve(strict=True)))
        else:
            missing_modules.add(str(path))
        cached = getattr(module, "__cached__", None)
        if type(cached) is str and Path(cached).is_file():
            caches.add(str(Path(cached).resolve(strict=True)))
    for line in Path("/proc/self/maps").read_text().splitlines():
        pieces = line.split(maxsplit=5)
        if len(pieces) != 6 or not pieces[5].startswith("/"):
            continue
        value = pieces[5].replace("\\012", "\n")
        if value.endswith(" (deleted)"):
            deleted.add(value)
            value = value[:-10]
        path = Path(value)
        if path.is_file():
            mappings.add(str(path.resolve(strict=True)))
        else:
            missing_mappings.add(str(path))
    return {"scope": "sys.modules file attributes and /proc/self/maps after public response; existing __cached__ paths are possible bytecode inputs, not proof of cache use; no content hashing here",
            "module_files": sorted(modules), "mapped_files": sorted(mappings),
            "module_cache_files": sorted(caches),
            "missing_module_files": sorted(missing_modules),
            "missing_mapped_files": sorted(missing_mappings), "deleted_mappings": sorted(deleted)}


def main():
    started = time.perf_counter_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-json", required=True)
    parser.add_argument("--method", choices=("S", "G_partition", "G_general", "G_fixed"), required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--max-line-bytes", type=int, required=True)
    args = parser.parse_args()
    try:
        need(args.memory_limit_bytes >= 1024 ** 2 and args.max_line_bytes > 0, "invalid resource limits")
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_limit_bytes, args.memory_limit_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.sched_setaffinity(0, {args.cpu})
        # Restore the original v6 study's registered representation domain for
        # every method equally. This process-local setting does not alter the
        # caller or any producer/receiver algorithm and never disables a limit.
        sys.set_int_max_str_digits(100000)
        config = loads(args.runtime_json.encode("utf-8"))
        checked_bytes(config["worker"])
        names = (("core", "old_receiver") if args.method == "S" else
                 ("adapter", "library", "old_receiver" if args.method == "G_partition" else "general_receiver"))
        for name in names:
            checked_bytes(config[name])
        if args.method == "S":
            core = load_source(config["core"], "family_service_partition_core")
            receiver = load_source(config["old_receiver"], "family_service_partition_receiver")
            engine = None
        else:
            adapter = load_source(config["adapter"], "family_service_generic_adapter")
            kwargs = {"expected_library_sha256": config["library"]["sha256"]}
            if args.method == "G_partition":
                kwargs["receiver_path"] = config["old_receiver"]["path"]
            else:
                kwargs["general_receiver_path"] = config["general_receiver"]["path"]
                kwargs["expected_general_receiver_sha256"] = config["general_receiver"]["sha256"]
            engine = adapter.ExactLP(config["library"]["path"], **kwargs)
        send({"schema_version": "family-lp-worker-ready/1", "ready": True, "method": args.method,
              "pid": os.getpid(), "setup_ns": time.perf_counter_ns() - started,
              "affinity": sorted(os.sched_getaffinity(0)), "memory_limit_bytes": args.memory_limit_bytes,
              "integer_string_digits": sys.get_int_max_str_digits(),
              "loaded_sources": {name: config[name]["sha256"] for name in names}}, args.max_line_bytes)
    except Exception as error:
        send({"schema_version": "family-lp-worker-init-refusal/1", "method": args.method,
              "error": type(error).__name__ + ": " + str(error)}, args.max_line_bytes)
        return 2

    request_start = time.perf_counter_ns()
    raw = sys.stdin.buffer.readline(args.max_line_bytes + 2)
    # EOF is supplied by the supervisor after this sole request; do not wait
    # for a second request. All input blocking is bounded by the outer timer.
    request_sha = hashlib.sha256(raw[:-1] if raw.endswith(b"\n") else raw).hexdigest()
    response = {"status": "refused", "certificate": None, "receiver_receipt": None}
    operation = None
    timings = {}
    error_text = None
    outcome_kind = "refused"
    active_phase = "request_read_and_decode"
    active_started = request_start
    try:
        need(raw.endswith(b"\n") and len(raw) - 1 <= args.max_line_bytes, "incomplete or oversized request line")
        request = loads(raw[:-1])
        expected_keys = {"program", "exact_objective"} if args.method == "G_fixed" else {
            "program", "context_sizes", "exact_weights", "exact_reference"}
        need(type(request) is dict and set(request) == expected_keys, "request fields differ")
        timings["request_read_and_decode_ns"] = time.perf_counter_ns() - request_start
        if args.method == "S":
            family = tuple(request[k] for k in ("program", "context_sizes", "exact_weights", "exact_reference"))
            phase = time.perf_counter_ns()
            active_phase, active_started = "request_binding", phase
            expected = receiver.input_digest(*family)
            timings["request_binding_ns"] = time.perf_counter_ns() - phase
            phase = time.perf_counter_ns()
            active_phase, active_started = "construction", phase
            certificate = core.solve(*family)
            timings["construction_ns"] = time.perf_counter_ns() - phase
            phase = time.perf_counter_ns()
            active_phase, active_started = "public_certificate_adaptation", phase
            auxiliary = {"contexts": certificate["contexts"], "internal_check": certificate["internal_check"]}
            certificate["contexts"] = [{} for _ in request["context_sizes"]]
            certificate["internal_check"] = {}
            timings["public_certificate_adaptation_ns"] = time.perf_counter_ns() - phase
            phase = time.perf_counter_ns()
            active_phase, active_started = "receiver", phase
            receipt = receiver.verify_certificate(*family, certificate, expected_input_sha256=expected)
            timings["receiver_ns"] = time.perf_counter_ns() - phase
            operation = {"construction_metadata": auxiliary,
                         "mathematical_receiver_invocations": 1}
            response = {"status": "verified_optimal", "certificate": certificate, "receiver_receipt": receipt}
        else:
            active_phase, active_started = "generic_call", time.perf_counter_ns()
            if args.method == "G_fixed":
                result = engine.solve_fixed_objective(request["program"], request["exact_objective"], native_time_limit_ns=None)
            else:
                family = tuple(request[k] for k in ("program", "context_sizes", "exact_weights", "exact_reference"))
                function = engine.solve_family if args.method == "G_partition" else engine.solve_general_family
                result = function(*family, native_time_limit_ns=None)
            timings["generic_call_ns"] = time.perf_counter_ns() - active_started
            need(result["status"] == "verified_optimal", "adapter returned an unverified result")
            response = {k: result[k] for k in ("status", "certificate", "receiver_receipt")}
            operation = {k: v for k, v in result.items() if k not in response}
            operation["mathematical_receiver_invocations"] = 1
        outcome_kind = "verified_optimal"
    except Exception as error:
        timings.setdefault(active_phase + "_ns", time.perf_counter_ns() - active_started)
        error_text = type(error).__name__ + ": " + str(error)
        if isinstance(error, MemoryError):
            outcome_kind = "resource_refusal"
        diagnostics = getattr(error, "diagnostics", None)
        if type(diagnostics) is dict:
            operation = diagnostics
    timings["before_response_ns"] = time.perf_counter_ns() - request_start
    phase = time.perf_counter_ns()
    try:
        send(response, args.max_line_bytes)
    except Exception as error:
        # An oversized certificate is a resource refusal, never truncated proof.
        active_phase = "public_serialization"
        response = {"status": "refused", "certificate": None, "receiver_receipt": None}
        error_text = "Resource refusal while serializing public response: " + type(error).__name__ + ": " + str(error)
        outcome_kind = "resource_refusal"
        send(response, args.max_line_bytes)
    timings["public_serialization_and_write_ns"] = time.perf_counter_ns() - phase
    phase = time.perf_counter_ns()
    inventory = runtime_inventory()
    timings["runtime_inventory_ns"] = time.perf_counter_ns() - phase
    diagnostics = {"schema_version": "family-lp-worker-diagnostics/1", "method": args.method,
                   "request_sha256": request_sha, "complete": True, "error": error_text,
                   "outcome_kind": outcome_kind,
                   "failure_stage": active_phase if error_text is not None else None,
                   "worker_timing_ns": timings, "operation": operation,
                   "runtime_inventory": inventory,
                   "peak_rss_kib_before_diagnostics": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                   "peak_rss_scope": "worker before diagnostic serialization; full-life wait4 value is in supervisor record"}
    try:
        send(diagnostics, args.max_line_bytes)
    except (ValueError, MemoryError) as error:
        send({"schema_version": "family-lp-worker-diagnostics/1", "method": args.method,
              "request_sha256": request_sha, "complete": False, "resource_refusal": True,
              "error": "Diagnostic output resource refusal: " + type(error).__name__ + ": " + str(error)},
             args.max_line_bytes)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
