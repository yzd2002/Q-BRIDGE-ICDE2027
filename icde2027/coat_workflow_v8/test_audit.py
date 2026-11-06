"""Saved two-coordinate evidence only; no producer imports or optimizer calls."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from . import audit

ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "runs/icde2027/analysis/coat_workflow_v8_service_candidate/synthetic_receipts_01"


@pytest.fixture
def toy():
    return wrapped_toy()


def wrapped_toy():
    data = json.loads((SAVED / "TOY_INPUT.json").read_bytes())
    data["family_request"]["schema_version"] = "verified-coat-family-task/1"
    return data


def checked_stream(data, saved, **kwargs):
    runtime = json.loads((SAVED / "RUNTIME_CONFIG.json").read_bytes())
    return audit.audit_stream(data, saved, runtime_config=runtime, **kwargs)


def result(arm="O_S"):
    # Preserve the recorded mathematical evidence; only bind this synthetic
    # test variant to the production-shaped, five-key task envelope.
    saved = json.loads((SAVED / (arm + ".json")).read_bytes())
    saved["input_sha256"] = audit.digest(wrapped_toy())
    return saved


@pytest.mark.parametrize("arm", ["H", "O", "O_S", "O_G"])
def test_saved_all_arms_online_replay(arm, toy):
    r = checked_stream(toy, result(arm), synthetic=True)
    assert r["audit_status"] == "PASS_COMPLETE"
    assert r["denominator"] == 3 and r["exact_fixed_calls"] == 1
    assert [row["represented"]["decision"] for row in r["ledger"]] == ["true", "true", "false"]


def test_production_default_rejects_small_saved_toy(toy):
    with pytest.raises(audit.AuditError, match="production stream census"):
        checked_stream(toy, result())


@pytest.mark.parametrize("mutation", ["query", "threshold", "ideal", "decision", "bounds", "family", "fixed", "wire", "clock", "calls", "prefix", "future"])
def test_tampering_is_rejected(toy, mutation):
    r = result()
    if mutation == "query":
        r["responses"][0]["query"]["request_index"] = 1
    elif mutation == "threshold":
        r["responses"][1]["query"]["threshold"] = .05
    elif mutation == "ideal":
        r["responses"][0]["query"]["ideal_objective_exact"][0] = "0/1"
    elif mutation == "decision":
        r["responses"][0]["receipt"]["represented"]["decision"] = "false"
    elif mutation == "bounds":
        r["responses"][0]["receipt"]["represented"]["lower_bound"] = "1/1"
    elif mutation == "family":
        r["family"]["response"]["certificate"]["point"][0] = "1/2"
    elif mutation == "fixed":
        r["responses"][1]["exact_fixed"]["response"]["certificate"]["optimum"] = "0/1"
    elif mutation == "wire":
        r["responses"][0]["public_response_wire_utf8"] += " "
    elif mutation == "clock":
        r["timing_scope"]["disjoint_phase_keys"].append("receiver_ns")
    elif mutation == "calls":
        r["responses"][0]["native"]["actual_native_calls"] += 1
    elif mutation == "prefix":
        r["prefixes"]["240"] = {"full_cumulative_ns": 1, "ready_cumulative_ns": 1}
    elif mutation == "future":
        r["responses"][0]["before_native_receipt"] = copy.deepcopy(r["responses"][1]["receipt"])
    with pytest.raises((audit.AuditError, ValueError)):
        checked_stream(toy, r, synthetic=True)


def test_current_runtime_closure_recomputed_without_flags(toy):
    r = result()
    identities = json.loads((SAVED / "OBSERVED_RUNTIME_IDENTITIES.json").read_bytes())
    allowed = [row["path"] for row in identities["files"]]
    closure = audit.runtime_closure(r, allowed)
    assert closure["admitted"], closure
    r["supervisor_runtime_inventory"]["module_files"].append("/unadmitted/future.py")
    with pytest.raises(audit.AuditError, match="unadmitted runtime"):
        checked_stream(toy, r, synthetic=True, allowed_runtime_paths=allowed)


def test_import_is_optimizer_and_producer_free():
    code = """import sys
sys.path.insert(0,sys.argv[1])
import icde2027.coat_workflow_v8.audit
bad=[n for n in sys.modules if n.split('.')[0] in ('numpy','scipy','highspy') or
 n in ('icde2027.coat_workflow_v8.adapter','icde2027.coat_workflow_v8.service','icde2027.coat_workflow_v8.study','icde2027.family_lp_v7.adapter','icde2027.partition_capacity_v6.core')]
assert not bad,bad
print('NO_OPTIMIZER_OR_PRODUCER_IMPORTS')
"""
    r = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "NO_OPTIMIZER" in r.stdout


def test_rehashed_forged_family_point_still_fails_math(toy):
    r = result()
    call = r["family"]
    call["response"]["certificate"]["point"][0] = "1/2"
    raw = audit.encode(call["response"])
    call["response_bytes"] = len(raw)
    call["response_sha256"] = audit.hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        checked_stream(toy, r, synthetic=True)


def test_success_flags_cannot_replace_fixed_proof(toy):
    r = result()
    call = r["responses"][1]["exact_fixed"]
    call["response"]["certificate"]["original_row_lower_dual"] = ["0/1"]
    raw = audit.encode(call["response"])
    call["response_bytes"] = len(raw)
    call["response_sha256"] = audit.hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        checked_stream(toy, r, synthetic=True)


def test_saved_partial_retains_online_denominator_and_has_no_acceleration_ratio(toy):
    r = result("H")
    partial = copy.deepcopy(r["responses"][1])
    partial.pop("receipt")
    partial["exact_fixed"] = None
    r["responses"] = r["responses"][:1]
    r.update(complete=False, status="timeout", incomplete_request=partial, unstarted_request_indices=[2])
    r["retained_points"] = r["retained_points"][:1]
    r["counts"] = {domain: {"true": 1} for domain in ("represented", "ideal")}
    r["workload_census"].update(completed_positions=1, incomplete_positions=1, unstarted_positions=1)
    r["time_ns"]["ready_to_final_response_ns"] = r["responses"][0]["cumulative_ready_ns"]
    r["bytes"]["public_responses"] = r["responses"][0]["public_response_bytes"]
    r["bytes"]["public_certificates_nested_in_responses"] = r["responses"][0]["public_certificate_bytes"]
    r["children"] = r["children"][:1]
    r["statistics"] = {**partial["statistics"], "actual_exact_worker_launches": 0,
                       "completed_response_native_calls": partial["statistics"]["actual_native_calls"]}
    r["exact_call_outcomes"] = {}
    r["refusal_timeout_counts"].update(stream_timeout=1, incomplete_positions=1, unstarted_positions=1)
    audited = checked_stream(toy, r, synthetic=True)
    assert [row["status"] for row in audited["ledger"]] == ["verified_replay", "partial", "unstarted"]
    assert audited["denominator"] == 3 and audited["completed_positions"] == 1
    full = checked_stream(toy, result("O"), synthetic=True)
    comparison = audit.compare_streams(audited, full, label="H/O")
    assert comparison["ratio"] is None and comparison["ratio_status"] == "NA"


def test_quality_difference_retains_costs_but_suppresses_ratio(toy):
    a = checked_stream(toy, result("O_S"), synthetic=True)
    b = checked_stream(toy, result("O"), synthetic=True)
    assert audit.compare_streams(a, b, label="O+S/O")["ratio"] is not None
    a["ledger"][0]["represented"]["decision"] = "unknown"
    comparison = audit.compare_streams(a, b, label="O+S/O")
    assert comparison["ratio"] is None
    assert comparison["numerator_full_wall_ns"] > 0
    assert comparison["differences"]["represented"][0]["kind"] == "loss"


def registered_schedule():
    result = []
    for round_ in range(4):
        for view in ((0, 1) if round_ % 2 == 0 else (1, 0)):
            label = ("positive", "zero_control")[view]
            for arm in audit.ARMS[round_:] + audit.ARMS[:round_]:
                result.append({"stream_index": len(result), "stage": "quality" if round_ == 0 else "timing", "round": round_,
                    "view_id": audit.VIEW_IDS[view], "label": label, "arm": arm, "input_file": label + ".json",
                    "requests": 240, "stream_wall_limit_seconds": 180})
    return result


@pytest.mark.parametrize("change", ["short", "order", "requests", "view", "round"])
def test_production_schedule_is_exact_and_cannot_admit_toy(change):
    schedule = registered_schedule()
    assert audit.check_schedule(schedule) == schedule
    if change == "short":
        schedule.pop()
    elif change == "order":
        schedule[0], schedule[1] = schedule[1], schedule[0]
    elif change == "requests":
        schedule[0]["requests"] = 3
    elif change == "view":
        schedule[0]["view_id"] = "1" * 64
    else:
        schedule[8]["round"] = 0
    with pytest.raises(audit.AuditError):
        audit.check_schedule(schedule)


def test_frozen_source_pin_and_path_escape_rejected(tmp_path):
    p = tmp_path / "source.py"
    p.write_text("value = 1\n")
    pin = audit.identity(p)
    assert audit.checked_tree(tmp_path, {"source.py": pin})
    p.write_text("value = 2\n")
    with pytest.raises(audit.AuditError, match="changed"):
        audit.checked_tree(tmp_path, {"source.py": pin})
    with pytest.raises(audit.AuditError, match="escapes"):
        audit.checked_tree(tmp_path, {"../escaped.py": pin})


def test_scaled_dual_is_exact_product_of_two_binary64_values(toy):
    from icde2027.threshold_extension.proofs import ThresholdChecker, bundle_proof
    checker = ThresholdChecker(toy["legacy_view"], cache_size=0)
    query = copy.deepcopy(toy["queries"][0])
    query["objective"], query["ideal_objective_exact"] = [1., 0.], ["1/1", "0/1"]
    lower = checker.lower_proof(query["objective"], [.1], [0.], threshold=0., scale=.3)
    proof = bundle_proof(checker.digest, query["objective"], 0., lower)
    answer = {"certificate": proof, "status": "certified", "robust_threshold_decision": True,
              "lower_bound": lower["claimed_lower"], "upper_bound": None}
    receiver = audit.RationalReceiver(toy["family_request"]["program"], view_sha256=toy["view_id"],
                                     expected_program_sha256=audit.program_input_digest(toy["family_request"]["program"]))
    receiver.accept_point(audit.point_wire(receiver, toy["seed_point_exact"]))
    request = receiver.bind_request(["1/1", "0/1"], ["1/1", "0/1"], "0/1", request_index=0, policy_index=0)
    wire, checked = audit.legacy_packet(checker, receiver, request, query, {"answer": answer})
    actual = audit.Fraction(json.loads(wire)["original_row_lower_dual"][0])
    assert actual == audit.Fraction.from_float(.1) * audit.Fraction.from_float(.3)
    assert actual != audit.Fraction.from_float(.1 * .3)


def test_explicit_synthetic_bundle_entrypoint(tmp_path):
    report = audit.audit_synthetic_bundle(SAVED, tmp_path / "audited", expected_receipt_sha256=audit.identity(SAVED / "RECEIPT.json")["sha256"])
    assert report["mode"] == "synthetic" and report["production_admission"] is False
    assert len(report["streams"]) == 4
    assert len(list((tmp_path / "audited").glob("*_ledger.jsonl"))) == 4
    with pytest.raises(audit.AuditError, match="output exists"):
        audit.audit_synthetic_bundle(SAVED, tmp_path / "audited", expected_receipt_sha256=audit.identity(SAVED / "RECEIPT.json")["sha256"])


def test_original_query_order_thresholds_and_pair_mapping():
    policies = [{"policy_index": i, "objective": [.5, -.5], "ideal_objective_exact": [["1", "2"], ["-1", "2"]]} for i in range(80)]
    requests = [{"query_index": t * 80 + i, "policy_index": i, "objective": [.5, -.5],
                 "threshold": delta, "features": ["synthetic"], "multipliers": [2.]}
                for t, delta in enumerate([0., .001, .01]) for i in range(80)]
    workflow = {"format": "qbridge-coat-metadata-workflow-v5", "policies": policies, "requests": requests}
    queries = audit.original_queries(workflow)
    assert len(queries) == 240 and queries[80]["threshold"] == .001
    assert audit.Fraction(audit.represented(queries[80]["threshold"])) != audit.Fraction(1, 1000)
    workflow["requests"][80]["threshold"] = .002
    with pytest.raises(audit.AuditError, match="threshold differs"):
        audit.original_queries(workflow)


@pytest.mark.parametrize("change", ["gate", "preparation", "source", "dependency"])
def test_package_rejects_changed_admission_source_and_dependency_before_input_reads(tmp_path, change):
    prepared, run = tmp_path / "prepared", tmp_path / "run"
    prepared.mkdir(); run.mkdir()
    source, dependency, library = prepared / "frozen.py", tmp_path / "runtime.bin", tmp_path / "library.bin"
    source.write_text("frozen = True\n")
    dependency.write_bytes(b"runtime pin")
    library.write_bytes(b"library pin, never loaded")
    preparation = {"schema_version": "coat-workflow-preparation/1", "status": "PREPARED_NOT_RUN", "stream_count": 32,
        "positions": 7680, "cpu": 12, "stream_cap_seconds": 180, "global_cap_seconds": 7200,
        "files": {"frozen.py": audit.identity(source)}, "library": {"path": str(library), **audit.identity(library)}}
    audit.write_new(prepared / "PREPARATION.json", preparation)
    gate = {"status": "GO", "production_authorized": True,
            "preparation_sha256": audit.identity(prepared / "PREPARATION.json")["sha256"],
            "review_files": {}, "runtime_dependency_files": {str(dependency): audit.identity(dependency)}}
    audit.write_new(run / "ADMISSION.json", gate)
    gate_sha = audit.identity(run / "ADMISSION.json")["sha256"]
    path = {"gate": run / "ADMISSION.json", "preparation": prepared / "PREPARATION.json",
            "source": source, "dependency": dependency}[change]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(audit.AuditError, match="pin|changed"):
        audit.verify_package(prepared, run, expected_gate_sha256=gate_sha, expected_completion_sha256="0" * 64)


def test_individually_valid_future_point_is_not_available_online(toy):
    r = result("H")
    receiver = audit.RationalReceiver(toy["family_request"]["program"], view_sha256=toy["view_id"],
        expected_program_sha256=audit.program_input_digest(toy["family_request"]["program"]))
    receiver.accept_point(audit.point_wire(receiver, toy["seed_point_exact"]))
    future = r["responses"][1]["exact_fixed"]["response"]["certificate"]["point"]
    assert receiver.accept_point(audit.point_wire(receiver, future))["verified"]
    q = toy["queries"][0]
    request = receiver.bind_request(list(map(audit.represented, q["objective"])), audit.ideal(q["ideal_objective_exact"]),
                audit.represented(q["threshold"]), request_index=0, policy_index=0)
    valid_but_future = receiver.evaluate(request)
    assert valid_but_future["verified"] and valid_but_future["request_sha256"] == r["responses"][0]["request_sha256"]
    r["responses"][0]["before_native_receipt"] = valid_but_future
    with pytest.raises(audit.AuditError, match="possible future evidence"):
        checked_stream(toy, r, synthetic=True)


def test_runtime_gap_is_not_silently_ignored(toy):
    r = result()
    r["family"]["diagnostics"].pop("runtime_inventory")
    rehash_diagnostics(r["family"])
    allowed = [row["path"] for row in json.loads((SAVED / "OBSERVED_RUNTIME_IDENTITIES.json").read_bytes())["files"]]
    with pytest.raises(audit.AuditError, match="missing runtime inventory"):
        checked_stream(toy, r, synthetic=True, allowed_runtime_paths=allowed)


def test_family_envelope_keeps_raw_identity_and_exact_service_four_field_binding(toy):
    raw = copy.deepcopy(toy["family_request"])
    mathematical = audit.family_math_request(raw)
    assert len(raw) == 5 and len(mathematical) == 4
    assert raw == toy["family_request"]
    assert audit.digest(raw) != audit.digest(mathematical)
    saved = result()
    assert saved["family"]["request_sha256"] == audit.digest(mathematical)
    audited = checked_stream(toy, saved, synthetic=True)
    assert audited["family_task_envelope_sha256"] == audit.digest(raw)
    assert audited["family_mathematical_request_sha256"] == audit.digest(mathematical)


@pytest.mark.parametrize("mutation", ["schema", "extra", "missing", "missing_version", "schema_bool"])
def test_family_envelope_unknown_or_extra_fields_are_rejected(toy, mutation):
    family = toy["family_request"]
    if mutation == "schema":
        family["schema_version"] = "verified-coat-family-task/2"
    elif mutation == "schema_bool":
        family["schema_version"] = True
    elif mutation == "extra":
        family["unused_metadata"] = "not accepted"
    elif mutation == "missing":
        del family["exact_reference"]
    else:
        del family["schema_version"]
    with pytest.raises(audit.AuditError, match="family task envelope"):
        audit.family_math_request(family)


def test_legacy_four_field_family_is_explicit_synthetic_compatibility_only():
    historical = json.loads((SAVED / "TOY_INPUT.json").read_bytes())["family_request"]
    assert audit.family_math_request(historical, allow_saved_legacy=True) == historical
    with pytest.raises(audit.AuditError, match="family task envelope fields differ"):
        audit.family_math_request(historical)


def rehash_diagnostics(call):
    raw = audit.encode(call["diagnostics"])
    call["diagnostic_bytes"] = len(raw)
    call["diagnostic_sha256"] = audit.hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("name", ["negative_unlisted_receiver_clock", "family_cost_hidden_in_residual",
    "false_refusal_timeout_counts", "wrong_exact_worker_resources", "wrong_exact_runtime_source_binding",
    "contradictory_exact_public_response", "omitted_rejected_native_primal"])
def test_independent_saved_counterexamples_are_rejected(name):
    path = ROOT / "runs/icde2027/analysis/coat_workflow_v8_auditor_independent" / (name + ".json")
    case = json.loads(path.read_bytes())
    assert case["synthetic_only"] is True
    with pytest.raises(audit.AuditError):
        checked_stream(case["input"], case["mutated_result"], synthetic=True)


def interrupted_fixture(*, native=False):
    """A consistent two-started/one-unstarted variant of saved H evidence."""
    value = result("H")
    partial = copy.deepcopy(value["responses"][1])
    for key in list(partial):
        if key == "receipt" or key.startswith(("public_", "cumulative_")) or key == "elapsed_query_ns":
            partial.pop(key)
    partial["exact_fixed"] = None
    value["responses"] = value["responses"][:1]
    value.update(complete=False, status="timeout", incomplete_request=partial, unstarted_request_indices=[2])
    value["retained_points"] = value["retained_points"][:1]
    value["counts"] = {domain: {"true": 1} for domain in ("represented", "ideal")}
    value["workload_census"].update(completed_positions=1, incomplete_positions=1, unstarted_positions=1)
    value["time_ns"]["ready_to_final_response_ns"] = value["responses"][0]["cumulative_ready_ns"]
    value["bytes"]["public_responses"] = value["responses"][0]["public_response_bytes"]
    value["bytes"]["public_certificates_nested_in_responses"] = value["responses"][0]["public_certificate_bytes"]
    value["exact_call_outcomes"] = {}
    value["refusal_timeout_counts"].update(stream_timeout=1, incomplete_positions=1, unstarted_positions=1)
    value["statistics"] = {**partial["statistics"], "actual_exact_worker_launches": 1, "completed_response_native_calls": 2}
    if native:
        partial.update(native=None, native_point_checks=[], legacy_adaptation=None, route="pending")
        value["children"] = value["children"][:1]
        value["statistics"] = {**value["responses"][0]["statistics"], "requests": 2, "producer_attempts": 2,
            "actual_exact_worker_launches": 0, "exact_fixed_calls": 0, "actual_native_calls": None,
            "completed_response_native_calls": 1}
        value["native_call_accounting"].update(complete=False, actual_native_calls=None,
            completed_response_call_count=1, inflight_call_count_unknown=True)
        value["time_ns"]["supervisor_residual_ns"] += value["time_ns"].pop("exact_fixed_service_ns")
        value["timing_scope"]["disjoint_phase_keys"].remove("exact_fixed_service_ns")
    return value


def test_inflight_fixed_attempt_and_launch_are_preserved(toy):
    saved = interrupted_fixture()
    checked = checked_stream(toy, saved, synthetic=True)
    assert checked["exact_fixed_calls"] == 1
    assert checked["call_activity"]["fixed"] == {"attempted": 1, "launched": 1, "returned_records": 0,
        "complete_records": 0, "verified_successes": 0, "inflight_attempts": 1}
    assert checked["ledger"][1]["inflight_exact_attempt"] is True
    assert checked["refusal_timeout_counts"]["stream_timeout"] == 1
    assert checked["refusal_timeout_counts"]["unstarted_positions"] == 1


def test_inflight_native_retains_null_and_rejects_imputed_complete_count(toy):
    saved = interrupted_fixture(native=True)
    checked = checked_stream(toy, saved, synthetic=True)
    assert checked["actual_native_calls"] is None and checked["observed_native_calls"] == 1
    assert checked["call_activity"]["native"]["attempted"] == 2
    assert checked["ledger"][1]["inflight_native_calls_unknown"] is True
    saved["native_call_accounting"].update(complete=True, actual_native_calls=1, inflight_call_count_unknown=False)
    saved["statistics"]["actual_native_calls"] = 1
    with pytest.raises(audit.AuditError, match="unreturned attempt chronology"):
        checked_stream(toy, saved, synthetic=True)


@pytest.mark.parametrize("field", ["requests", "producer_attempts", "exact_fixed_calls", "actual_exact_worker_launches", "rejected_native_points"])
def test_partial_counters_are_not_exempt_from_checks(toy, field):
    saved = interrupted_fixture()
    saved["statistics"][field] += 99
    with pytest.raises(audit.AuditError):
        checked_stream(toy, saved, synthetic=True)


@pytest.mark.parametrize("field", ["receiver_ns", "post_service_report_assembly_ns", "native_service_ns", "exact_fixed_service_ns"])
def test_all_timing_scopes_remain_validated(toy, field):
    saved = result()
    if field == "post_service_report_assembly_ns":
        saved[field] = -1
        saved["total_run_stream_call_wall_ns"] = saved["time_ns"]["full_wall_ns"] - 1
    else:
        saved["time_ns"]["supervisor_residual_ns"] += saved["time_ns"].pop(field)
        saved["timing_scope"]["disjoint_phase_keys"].remove(field)
    with pytest.raises(audit.AuditError):
        checked_stream(toy, saved, synthetic=True)


def test_bad_gate_produces_untrusted_failure_denominator(tmp_path):
    prepared, run, output = tmp_path / "prepared", tmp_path / "run", tmp_path / "audited"
    prepared.mkdir()
    run.mkdir()
    audit.write_new(run / "ADMISSION.json", {"synthetic_only": True})
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(100000)
    try:
        with pytest.raises(audit.AuditError, match="pin mismatch"):
            audit.audit_run(prepared, run, output, expected_gate_sha256="0" * 64, expected_completion_sha256="0" * 64)
    finally:
        sys.set_int_max_str_digits(previous)
    failure = json.loads((output / "FAILURE.json").read_bytes())
    assert failure["metadata_verified"] is False and failure["registered_positions"] is None
    assert failure["expected_registered_positions"] == failure["unverified_positions"] == 7680
    assert failure["caller_inputs_unverified_until_metadata_gate"] == {"prepared": str(prepared), "run": str(run),
        "expected_gate_sha256": "0" * 64, "expected_completion_sha256": "0" * 64}


def exact_check(call, toy):
    return audit.check_exact_call(call, audit.family_math_request(toy["family_request"]), "S",
        runtime_config=json.loads((SAVED / "RUNTIME_CONFIG.json").read_bytes()), cpu=14)


def rehash_public(call):
    raw = audit.encode(call["response"])
    call["response_bytes"] = len(raw)
    call["response_sha256"] = audit.hashlib.sha256(raw).hexdigest()


def test_clean_exact_refusal_preserves_no_optimum(toy):
    call = result()["family"]
    call.update(status="refused", verified_success=False)
    call["response"] = {"status": "refused", "certificate": None, "receiver_receipt": None}
    call["diagnostics"].update(outcome_kind="refused", error="synthetic mathematical domain refusal", failure_stage="construction", operation=None)
    rehash_public(call)
    rehash_diagnostics(call)
    assert exact_check(call, toy) is None


def test_valid_proof_before_diagnostic_failure_is_checked_but_not_success(toy):
    call = result()["family"]
    call.update(status="diagnostic_timeout", record_complete=False, verified_success=False, error="synthetic diagnostic timeout")
    call["diagnostics"] = None
    call["diagnostic_bytes"] = call["diagnostic_sha256"] = None
    call["time_ns"]["diagnostics_ns"] = call["limits_ns"]["diagnostics"]
    call["time_ns"]["total_wall_ns"] = sum(value for key, value in call["time_ns"].items() if key != "total_wall_ns")
    assert exact_check(call, toy) is None
    call["response"]["certificate"]["point"][0] = "1/2"
    rehash_public(call)
    with pytest.raises(ValueError):
        exact_check(call, toy)


def test_preflight_refusal_has_no_launched_worker_or_fabricated_response(toy):
    call = {"schema_version": "family-lp-service-record/1", "method": "S", "status": "input_refusal",
        "record_complete": False, "verified_success": False, "reaped": False, "ready": None,
        "request_sha256": None, "request_bytes": None, "response": None, "response_sha256": None,
        "response_bytes": None, "diagnostics": None, "diagnostic_bytes": None, "diagnostic_sha256": None,
        "time_ns": {"total_wall_ns": 123}, "error": "synthetic preflight refusal"}
    assert exact_check(call, toy) is None
    call["worker_pid"] = 123
    with pytest.raises(audit.AuditError, match="preflight refusal"):
        exact_check(call, toy)


@pytest.mark.parametrize("field", ["cpu", "affinity", "memory", "guard", "service_cap", "startup_cap", "config", "loaded_source", "diagnostic_request", "receiver_receipt"])
def test_exact_metadata_fields_are_bound_individually_even_when_wire_rehashed(toy, field):
    call = result()["family"]
    if field == "cpu": call["cpu"] = 99
    elif field == "affinity": call["ready"]["affinity"] = [99]
    elif field == "memory": call["ready"]["memory_limit_bytes"] = 2 ** 40
    elif field == "guard": call["integer_string_digits"] = 0
    elif field == "service_cap": call["limits_ns"]["service"] = 60_000_000_000
    elif field == "startup_cap": call["limits_ns"]["startup"] = 60_000_000_000
    elif field == "config": call["runtime_config_sha256"] = "0" * 64
    elif field == "loaded_source": call["ready"]["loaded_sources"]["core"] = "0" * 64
    elif field == "diagnostic_request": call["diagnostics"]["request_sha256"] = "0" * 64
    else: call["response"]["receiver_receipt"]["verified"] = False
    rehash_public(call)
    rehash_diagnostics(call)
    with pytest.raises(audit.AuditError):
        exact_check(call, toy)


def test_partial_native_ready_resources_are_still_checked(toy):
    saved = interrupted_fixture(native=True)
    saved["ready"]["affinity"] = [99]
    with pytest.raises(audit.AuditError, match="native startup identity/resources"):
        checked_stream(toy, saved, synthetic=True)


def test_missing_admitted_runtime_cannot_silently_pass(toy):
    with pytest.raises(audit.AuditError, match="admitted runtime_config is required"):
        audit.audit_stream(toy, result(), synthetic=True)


def test_prequery_family_timeout_preserves_launched_attempt_without_partial_request(toy):
    saved = result()
    saved.update(complete=False, status="timeout", responses=[], family=None, incomplete_request=None,
                 unstarted_request_indices=[0, 1, 2], prefixes={}, exact_call_outcomes={})
    saved["retained_points"] = saved["retained_points"][:1]
    saved["statistics"] = {"constant_shortcuts": 0, "family_calls": 1, "actual_exact_worker_launches": 1,
                           "actual_native_calls": 0, "completed_response_native_calls": 0}
    saved["native_call_accounting"].update(complete=True, actual_native_calls=0, completed_response_call_count=0,
                                         inflight_call_count_unknown=False)
    saved["children"] = [child for child in saved["children"] if child["method"] in ("native_producer", "S")]
    saved["counts"] = {domain: {} for domain in ("represented", "ideal")}
    saved["workload_census"].update(completed_positions=0, incomplete_positions=0, unstarted_positions=3)
    saved["refusal_timeout_counts"] = {"unknown_responses": 0, "native_refusals": 0, "exact_non_successful_calls": 0,
        "stream_timeout": 1, "incomplete_positions": 0, "unstarted_positions": 3}
    saved["bytes"]["public_responses"] = saved["bytes"]["public_certificates_nested_in_responses"] = 0
    saved["time_ns"]["ready_to_final_response_ns"] = None
    for phase in ("native_service_ns", "exact_fixed_service_ns", "receiver_ns", "legacy_receiver_ns", "response_encoding_ns"):
        saved["time_ns"]["supervisor_residual_ns"] += saved["time_ns"].pop(phase)
        saved["timing_scope"]["disjoint_phase_keys"].remove(phase)
    saved["native_cleanup"]["diagnostics"] = None
    checked = checked_stream(toy, saved, synthetic=True)
    assert checked["completed_positions"] == 0 and checked["unstarted_positions"] == 3
    assert checked["family_calls"] == 1
    assert checked["call_activity"]["family"] == {"attempted": 1, "launched": 1, "returned_records": 0,
        "complete_records": 0, "verified_successes": 0, "inflight_attempts": 1}


def test_exact_timeout_cannot_claim_a_shorter_than_deadline_clock(toy):
    call = result()["family"]
    call.update(status="diagnostic_timeout", record_complete=False, verified_success=False, error="synthetic timeout")
    with pytest.raises(audit.AuditError, match="before its recorded deadline"):
        exact_check(call, toy)
