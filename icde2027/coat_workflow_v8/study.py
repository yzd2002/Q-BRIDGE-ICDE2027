"""Freeze and execute the registered 32-stream workflow, preserving failures.

The service owns scientific clocks and child cleanup. This outer driver owns
input/source snapshots, independent admission, sequence and post-service disk
logging. A separately supervised process supplies the global emergency guard.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_pinned(path, pin):
    require(identity(path) == pin, "File differs from admission: " + str(path))
    return json.loads(path.read_text())


def prepare(inputs, output, library, design, method_review):
    """Freeze only; no native initialization, policy query or optimizer call."""
    from icde2027.coat_workflow_v8.service import runtime_configuration
    require(not output.exists(), "Output exists")
    manifest = json.loads((inputs / "INPUT_MANIFEST.json").read_text())
    require(manifest["status"] == "METADATA_PREPARED_NOT_PRODUCTION_RUN", "Unrecognized input manifest")
    expected = {row["file"]: {k: row[k] for k in ("bytes", "sha256")} for row in manifest["streams"]}
    expected.update({"schedule.json": manifest["schedule"], "original_workflow_historical.json": manifest["original_workflow_historical"]})
    for name, pin in expected.items():
        require(identity(inputs / name) == pin, "Prepared input changed: " + name)
    schedule = json.loads((inputs / "schedule.json").read_text())["streams"]
    require(len(schedule) == 32 and [r["stream_index"] for r in schedule] == list(range(32)), "Schedule census")
    output.mkdir(parents=True)
    frozen = output / "source_snapshot"
    # The separately reviewed post-run auditor is not an execution dependency
    # and may still be under development while this snapshot is frozen.
    post_run_only = {"icde2027/coat_workflow_v8/audit.py", "icde2027/coat_workflow_v8/test_audit.py"}
    source_files = [path for path in sorted((ROOT / "icde2027").rglob("*.py"))
                    if path.relative_to(ROOT).as_posix() not in post_run_only]
    original_sources = {}
    for path in source_files:
        relative = path.relative_to(ROOT)
        dest = frozen / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        original_sources[str(relative)] = identity(path)
        require(identity(dest) == identity(path), "Source changed during snapshot")
    frozen_inputs = output / "inputs"
    frozen_inputs.mkdir()
    for name in [*expected, "INPUT_MANIFEST.json"]:
        shutil.copyfile(inputs / name, frozen_inputs / name)
    shutil.copyfile(design, output / "EXPERIMENTS.md")
    shutil.copyfile(method_review, output / "DESIGN_REVIEW.json")
    config = runtime_configuration(frozen, library)
    write_new(output / "runtime.json", config)
    files = {str(path.relative_to(output)): identity(path) for path in sorted(output.rglob("*")) if path.is_file()}
    for name, pin in original_sources.items():
        require(identity(ROOT / name) == pin, "Live source changed during snapshot")
    write_new(output / "PREPARATION.json", {"schema_version": "coat-workflow-preparation/1", "status": "PREPARED_NOT_RUN",
        "created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "input_manifest_sha256": identity(inputs / "INPUT_MANIFEST.json")["sha256"],
        "design_sha256": identity(design)["sha256"], "method_review_sha256": identity(method_review)["sha256"],
        "original_project_sources": original_sources, "library": {"path": str(library), **identity(library)},
        "stream_count": 32, "positions": 7680, "cpu": 12, "global_cap_seconds": 7200,
        "stream_cap_seconds": 180, "new_optimizer_calls": 0})
    return identity(output / "PREPARATION.json")


def verify_prepared(prepared, expected_sha):
    path = prepared / "PREPARATION.json"
    require(identity(path)["sha256"] == expected_sha, "Preparation digest mismatch")
    p = json.loads(path.read_text())
    require(p["status"] == "PREPARED_NOT_RUN" and p["schema_version"] == "coat-workflow-preparation/1", "Preparation schema")
    for name, pin in p["files"].items():
        candidate = prepared / name
        require(candidate.resolve().is_relative_to(prepared.resolve()), "Preparation path escapes")
        require(not candidate.is_symlink() and identity(candidate) == pin, "Changed frozen file: " + name)
    require(identity(Path(p["library"]["path"])) == {k: p["library"][k] for k in ("bytes", "sha256")}, "Native library changed")
    return p


def runtime_admission(result, allowed_paths):
    """Check observed module/cache/mapping sets, not just preexisting hashes."""
    fields = ("module_files", "module_cache_files", "mapped_files")
    bad_fields = ("missing_module_files", "missing_mapped_files", "deleted_mappings")
    inventories, gaps, issues, observed = [], [], [], set()
    allowed = {str(Path(p).resolve()) for p in allowed_paths}

    def check_inventory(value, location):
        if not isinstance(value, dict):
            gaps.append(location)
            return
        inventories.append(location)
        for field in fields + bad_fields:
            values = value.get(field)
            if not isinstance(values, list) or any(type(v) is not str for v in values):
                issues.append({"location": location, "field": field, "reason": "invalid_inventory_field"})
                continue
            if field in bad_fields:
                issues.extend({"location": location, "field": field, "path": p, "reason": "missing_or_deleted"} for p in values)
            else:
                for name in values:
                    path = str(Path(name).resolve())
                    observed.add(path)
                    if path not in allowed:
                        issues.append({"location": location, "field": field, "path": path, "reason": "unadmitted_file"})

    def visit(value, location):
        if isinstance(value, dict):
            if value.get("schema_version") == "family-lp-service-record/1":
                diagnostics = value.get("diagnostics")
                if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get("runtime_inventory"), dict):
                    gaps.append(location + "/exact_runtime_inventory")
            for key, child in value.items():
                child_location = location + "/" + key
                if key in ("runtime_inventory", "supervisor_runtime_inventory"):
                    check_inventory(child, child_location)
                elif isinstance(child, (dict, list)):
                    visit(child, child_location)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    visit(child, location + "/" + str(i))

    visit(result, "stream")
    if not isinstance(result.get("supervisor_runtime_inventory"), dict):
        gaps.append("stream/supervisor_runtime_inventory")
    native = result.get("native_cleanup", {}).get("diagnostics")
    if not isinstance(native, dict) or not isinstance(native.get("runtime_inventory"), dict):
        gaps.append("stream/native_runtime_inventory")
    gaps = sorted(set(gaps))
    return {"admitted": not issues and not gaps, "inventories_checked": len(inventories),
            "observed_file_count": len(observed), "observed_files": sorted(observed),
            "violations": issues, "observations_missing": gaps,
            "scope": "Reported sys.modules, possible caches and mapped files; not OS-level isolation or historical-import tracing."}


def execute(prepared, output, gate_path, expected_gate_sha, *, run_stream=None, monotonic=time.monotonic):
    """Run once; test injection supplies synthetic service results only."""
    require(not output.exists(), "Run output exists")
    require(identity(gate_path)["sha256"] == expected_gate_sha, "Admission gate hash mismatch")
    gate = json.loads(gate_path.read_text())
    require(gate.get("status") == "GO" and gate.get("production_authorized") is True, "Production not independently admitted")
    preparation = verify_prepared(prepared, gate["preparation_sha256"])
    for name, pin in gate["review_files"].items():
        require(identity(Path(name)) == pin, "Independent review changed")
    for name, pin in gate.get("runtime_dependency_files", {}).items():
        require(identity(Path(name)) == pin, "Runtime dependency changed")
    require(identity(Path(__file__))["sha256"] == gate["study_sha256"], "Outer driver source changed")
    if run_stream is None:
        require(Path(__file__).resolve().is_relative_to((prepared / "source_snapshot").resolve()), "Run the frozen driver")
        from icde2027.coat_workflow_v8.service import run_stream
    require(os.sched_getaffinity(0) == {preparation["cpu"]}, "Driver CPU affinity differs")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(key) == "1", "Single-thread declaration missing")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU visibility must be empty")
    sys.set_int_max_str_digits(100000)
    output.mkdir(parents=True)
    shutil.copyfile(gate_path, output / "ADMISSION.json")
    runtime = json.loads((prepared / "runtime.json").read_text())
    schedule = json.loads((prepared / "inputs/schedule.json").read_text())["streams"]
    manifest = json.loads((prepared / "inputs/INPUT_MANIFEST.json").read_text())
    input_pins = {row["file"]: {k: row[k] for k in ("bytes", "sha256")} for row in manifest["streams"]}
    admitted_paths = {str(prepared / name) for name in preparation["files"]}
    admitted_paths.update(gate.get("runtime_dependency_files", {}))
    started = monotonic()
    rows, stopped = [], None
    for entry in schedule:
        remaining = preparation["global_cap_seconds"] - (monotonic() - started)
        if stopped is not None or remaining <= 5:
            rows.append({**entry, "status": "unstarted", "reason": stopped or "global_budget_exhausted"})
            continue
        stream_input = read_pinned(prepared / "inputs" / entry["input_file"], input_pins[entry["input_file"]])
        require(stream_input["view_id"] == entry["view_id"] and len(stream_input["queries"]) == 240, "Schedule/input mismatch")
        canonical = hashlib.sha256(encoded(stream_input)).hexdigest()
        stream_start = monotonic()
        outer_error = None
        try:
            result = run_stream(stream_input, arm=entry["arm"], runtime_config=runtime,
                                expected_input_sha256=canonical, cpu=preparation["cpu"],
                                stream_limit_s=min(preparation["stream_cap_seconds"], remaining))
        except BaseException as error:
            outer_error = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            stopped = "outer_service_exception"
            # No returned record means the current stream's execution extent is
            # unknown. Do not turn its positions into unstarted or zero calls.
            result = {"status": "outer_service_exception", "complete": False,
                      "counts": {"represented": {}, "ideal": {}},
                      "workload_census": {"requested_positions": 240, "completed_positions": None,
                          "incomplete_positions": None, "unstarted_positions": None, "unrecovered_positions": 240},
                      "children": [], "cleanup_observation": "unavailable after service exception",
                      "resources": {"full_wall_exceeded_registered_limit": None}}
        returned = monotonic()
        closure = runtime_admission(result, admitted_paths)
        record = {"schedule": entry, "input_raw_identity": input_pins[entry["input_file"]],
                  "input_canonical_sha256": canonical, "result": result,
                  "runtime_admission": closure,
                  "outer_service_error": outer_error,
                  "outer_function_wall_seconds": returned - stream_start}
        name = f"stream_{entry['stream_index']:02d}.json"
        log_start = monotonic()
        write_new(output / name, record)
        log_elapsed = monotonic() - log_start
        rows.append({**entry, "status": result["status"], "complete": result["complete"],
                     "file": name, **identity(output / name), "logging_seconds": log_elapsed,
                     "runtime_admission": {k: v for k, v in closure.items() if k != "observed_files"},
                     "workload_census": result["workload_census"], "counts": result["counts"]})
        write_new(output / f"progress_{entry['stream_index']:02d}.json", {"streams_so_far": rows, "elapsed_seconds": monotonic() - started})
        if result["status"] == "correctness_mismatch":
            stopped = "correctness_mismatch"
        if result.get("native_cleanup", {}).get("error") or any(not p["reaped"] for p in result.get("children", [])):
            stopped = "process_cleanup_failed"
        if result["resources"]["full_wall_exceeded_registered_limit"]:
            stopped = "registered_stream_limit_exceeded"
        if closure["violations"] or (result["complete"] and closure["observations_missing"]):
            stopped = "runtime_admission_failed"
    pin_error = None
    try:
        verify_prepared(prepared, gate["preparation_sha256"])
        for name, pin in gate.get("runtime_dependency_files", {}).items():
            require(identity(Path(name)) == pin, "Runtime dependency changed during study")
    except Exception as error:
        pin_error = type(error).__name__ + ": " + str(error)
        stopped = stopped or "source_or_runtime_changed"
    completed = sum(row.get("complete", False) for row in rows)
    write_new(output / "COMPLETION.json", {"schema_version": "coat-workflow-completion/1",
          "status": "complete" if completed == 32 and stopped is None else "incomplete", "finished_utc": datetime.now(timezone.utc).isoformat(),
          "preparation_sha256": gate["preparation_sha256"], "gate_sha256": expected_gate_sha,
          "streams": rows, "completed_streams": completed, "registered_streams": 32,
          "elapsed_seconds_including_logging": monotonic() - started, "stop_reason": stopped,
          "source_and_input_pins_unchanged": pin_error is None, "pin_recheck_error": pin_error,
          "mathematical_post_audit_pending": True})
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("inputs", "output", "library", "design", "method-review"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("run")
    for name in ("prepared", "output", "gate"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gate-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.inputs.resolve(), args.output.resolve(), args.library.resolve(), args.design.resolve(), args.method_review.resolve())))
    else:
        n = execute(args.prepared.resolve(), args.output.resolve(), args.gate.resolve(), args.gate_sha256)
        print(json.dumps({"completed_streams": n, "post_audit_pending": True}))
        status = json.loads((args.output.resolve() / "COMPLETION.json").read_text())["status"]
        return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
