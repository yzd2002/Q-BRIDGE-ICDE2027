"""Synthetic outer-driver admission, failure and sequence regressions."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from . import study


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    inputs = prepared / "inputs"
    inputs.mkdir(parents=True)
    library = tmp_path / "toy-library"
    library.write_bytes(b"not a loaded library")
    dependency = tmp_path / "toy-runtime"
    dependency.write_bytes(b"pinned dependency")
    streams = []
    for label in ("positive", "zero_control"):
        value = {"view_id": label, "queries": [{"request_index": i} for i in range(240)]}
        study.write_new(inputs / (label + ".json"), value)
    arms = ["H", "O", "O+S", "O+G"]
    for round_ in range(4):
        views = ["positive", "zero_control"] if round_ % 2 == 0 else ["zero_control", "positive"]
        for view in views:
            for arm in arms[round_:] + arms[:round_]:
                streams.append({"stream_index": len(streams), "stage": "quality" if round_ == 0 else "timing",
                                "round": round_, "view_id": view, "label": view, "arm": arm,
                                "input_file": view + ".json", "requests": 240})
    study.write_new(inputs / "schedule.json", {"streams": streams})
    study.write_new(inputs / "INPUT_MANIFEST.json", {"streams": [
        {"file": label + ".json", **study.identity(inputs / (label + ".json"))}
        for label in ("positive", "zero_control")]})
    study.write_new(prepared / "runtime.json", {"synthetic_only": True})
    p = {"schema_version": "coat-workflow-preparation/1", "status": "PREPARED_NOT_RUN",
         "files": {str(f.relative_to(prepared)): study.identity(f) for f in prepared.rglob("*") if f.is_file()},
         "library": {"path": str(library), **study.identity(library)}, "cpu": 12,
         "global_cap_seconds": 7200, "stream_cap_seconds": 180}
    study.write_new(prepared / "PREPARATION.json", p)
    gate = tmp_path / "gate.json"
    study.write_new(gate, {"status": "GO", "production_authorized": True,
                          "preparation_sha256": study.identity(prepared / "PREPARATION.json")["sha256"],
                          "review_files": {}, "runtime_dependency_files": {str(dependency): study.identity(dependency)},
                          "study_sha256": study.identity(Path(study.__file__))["sha256"]})
    monkeypatch.setattr(study.os, "sched_getaffinity", lambda _: {12})
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return {"prepared": prepared, "gate": gate, "gate_sha": study.identity(gate)["sha256"],
            "output": tmp_path / "out", "dependency": dependency, "expected_streams": streams}


def response(status="complete"):
    inventory = {key: [] for key in ("module_files", "module_cache_files", "mapped_files", "missing_module_files", "missing_mapped_files", "deleted_mappings")}
    return {"status": status, "complete": status == "complete", "counts": {"represented": {"unknown": 240}, "ideal": {"unknown": 240}},
            "workload_census": {"requested_positions": 240, "completed_positions": 240 if status == "complete" else 0,
                                "incomplete_positions": 0, "unstarted_positions": 0 if status == "complete" else 240},
            "native_cleanup": {"error": None, "reaped": True, "diagnostics": {"runtime_inventory": inventory}}, "children": [],
            "supervisor_runtime_inventory": copy.deepcopy(inventory),
            "resources": {"full_wall_exceeded_registered_limit": False}}


def execute(f, fake):
    return study.execute(f["prepared"], f["output"], f["gate"], f["gate_sha"], run_stream=fake)


def test_full_schedule_and_raw_vs_canonical_binding(fixture):
    seen = []
    def fake(value, **kw):
        seen.append((value["view_id"], kw["arm"]))
        assert kw["expected_input_sha256"] == hashlib.sha256(study.encoded(value)).hexdigest()
        assert kw["stream_limit_s"] == 180 and kw["cpu"] == 12
        return response()
    assert execute(fixture, fake) == 32
    assert seen == [(r["view_id"], r["arm"]) for r in fixture["expected_streams"]]
    first = json.loads((fixture["output"] / "stream_00.json").read_text())
    assert first["input_canonical_sha256"] != first["input_raw_identity"]["sha256"]
    assert len(list(fixture["output"].glob("stream_*.json"))) == 32


@pytest.mark.parametrize("changed", ["gate", "source", "dependency"])
def test_changed_admission_never_calls_service(fixture, changed):
    path = {"gate": fixture["gate"], "source": fixture["prepared"] / "runtime.json", "dependency": fixture["dependency"]}[changed]
    path.write_text(path.read_text() + " ")
    called = []
    with pytest.raises(ValueError):
        execute(fixture, lambda *a, **k: called.append(1))
    assert not called and not fixture["output"].exists()


def test_timeout_kept_then_next_scheduled_stream(fixture):
    called = []
    def fake(*a, **kw):
        called.append(kw["arm"])
        return response("timeout" if len(called) == 1 else "complete")
    assert execute(fixture, fake) == 31
    summary = json.loads((fixture["output"] / "COMPLETION.json").read_text())
    assert summary["status"] == "incomplete" and summary["streams"][0]["status"] == "timeout"
    assert len(called) == 32 and summary["stop_reason"] is None


@pytest.mark.parametrize("reason", ["correctness_mismatch", "cleanup", "overrun"])
def test_safety_failure_keeps_first_record_and_stops_remaining(fixture, reason):
    calls = []
    def fake(*a, **kw):
        calls.append(1)
        result = response("correctness_mismatch" if reason == "correctness_mismatch" else "refused")
        if reason == "cleanup":
            result["children"] = [{"reaped": False}]
        if reason == "overrun":
            result["resources"]["full_wall_exceeded_registered_limit"] = True
        return result
    assert execute(fixture, fake) == 0 and len(calls) == 1
    summary = json.loads((fixture["output"] / "COMPLETION.json").read_text())
    assert len(summary["streams"]) == 32
    assert all(row["status"] == "unstarted" for row in summary["streams"][1:])
    assert (fixture["output"] / "stream_00.json").is_file()


def test_refuse_overwrite(fixture):
    fixture["output"].mkdir()
    sentinel = fixture["output"] / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="exists"):
        execute(fixture, lambda *a, **kw: pytest.fail("must not run"))
    assert sentinel.read_text() == "keep"


def test_runtime_change_after_execution_preserves_records_and_marks_failed_completion(fixture):
    calls = []
    def fake(*a, **kw):
        calls.append(1)
        if len(calls) == 32:
            fixture["dependency"].write_bytes(b"changed")
        return response()
    assert execute(fixture, fake) == 32
    assert len(list(fixture["output"].glob("stream_*.json"))) == 32
    summary = json.loads((fixture["output"] / "COMPLETION.json").read_text())
    assert summary["status"] == "incomplete" and not summary["source_and_input_pins_unchanged"]
    assert "changed during" in summary["pin_recheck_error"]


@pytest.mark.parametrize("field", ["module_files", "module_cache_files", "mapped_files", "missing_module_files", "missing_mapped_files", "deleted_mappings"])
def test_observed_runtime_violation_keeps_actual_reply_counts_and_stops(fixture, field):
    def fake(*a, **kw):
        result = response()
        result["supervisor_runtime_inventory"][field] = [str(fixture["output"] / "unadmitted")]
        return result
    assert execute(fixture, fake) == 1
    first = json.loads((fixture["output"] / "stream_00.json").read_text())
    assert first["result"]["workload_census"]["completed_positions"] == 240
    assert not first["runtime_admission"]["admitted"]
    summary = json.loads((fixture["output"] / "COMPLETION.json").read_text())
    assert summary["stop_reason"] == "runtime_admission_failed"
    assert all(r["status"] == "unstarted" for r in summary["streams"][1:])


def test_missing_exact_inventory_is_explicit_without_rewriting_calls(fixture):
    def fake(*a, **kw):
        result = response()
        result["family"] = {"schema_version": "family-lp-service-record/1", "status": "service_timeout", "diagnostics": None}
        return result
    assert execute(fixture, fake) == 1
    first = json.loads((fixture["output"] / "stream_00.json").read_text())
    assert first["result"]["family"]["status"] == "service_timeout"
    assert first["runtime_admission"]["observations_missing"] == ["stream/family/exact_runtime_inventory"]


def test_admitted_external_observation_is_allowed(fixture):
    def fake(*a, **kw):
        result = response()
        result["supervisor_runtime_inventory"]["module_files"] = [str(fixture["dependency"])]
        return result
    assert execute(fixture, fake) == 32


def test_outer_service_exception_retains_prior_and_unknown_current_extent(fixture):
    calls = []
    def fake(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic post-service assembly failure")
        return response()
    assert execute(fixture, fake) == 1 and len(calls) == 2
    summary = json.loads((fixture["output"] / "COMPLETION.json").read_text())
    failed = json.loads((fixture["output"] / "stream_01.json").read_text())
    assert summary["stop_reason"] == "outer_service_exception"
    assert failed["result"]["workload_census"]["completed_positions"] is None
    assert failed["result"]["workload_census"]["unrecovered_positions"] == 240
    assert failed["outer_service_error"]["type"] == "RuntimeError"
    assert "synthetic post-service" in failed["outer_service_error"]["traceback"]
    assert all(r["status"] == "unstarted" for r in summary["streams"][2:])
