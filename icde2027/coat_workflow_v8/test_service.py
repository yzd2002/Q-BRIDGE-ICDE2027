"""Small synthetic workflow, native IPC and unchanged exact service tests."""
import copy
from fractions import Fraction
import os
from pathlib import Path
import json
import signal
import subprocess
import sys
import time

import pytest

from .adapter import (LegacyEvidenceAdapter, NativeProducer, binary64_exact,
                      constant_intersection, effective_legacy_multipliers, encode,
                      ideal_exact_vector, legacy_view_from_program, point_wire, sha256)
from .service import (ChildRegistry, Workflow, exact_service_module, identity,
                      run_stream, runtime_configuration)

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "runs/icde2027/analysis/family_lp_v7_implementation/build_attempt_04/libfamily_lp_v7.so"
CPU = 14


def toy(*, constant=False):
    program = {"lower": ["0/1", "0/1"], "upper": ["1/1", "1/1"],
               "rows": [{"coefficients": ["1/1", "0/1"], "lower": "3/4", "upper": "1/1"},
                        {"coefficients": ["0/1", "1/1"], "lower": "0/1", "upper": "1/4"}]}
    if constant:
        program["rows"] = []
    family = {"schema_version": "verified-coat-family-task/1", "program": program, "context_sizes": [2], "exact_weights": ["1/1"],
              "exact_reference": [["1/2", "1/2"]]}
    seed = ["3/4", "1/4"]
    queries = [{"request_index": i, "policy_index": 0,
                "objective": [.5, -.5], "ideal_objective_exact": ["1/2", "-1/2"],
                "threshold": threshold} for i, threshold in enumerate([0., .25, .5])]
    return {"schema_version": "coat-workflow-input/1", "view_id": "1" * 64,
            "family_request": family, "legacy_view": legacy_view_from_program(program, seed),
            "seed_point_exact": seed, "queries": queries}


def refused(request, method):
    return {"verified_success": False, "status": "refused", "response": None,
            "method": method, "request": request}


def native_refused(c, d):
    return {"kind": "producer_refused", "error": "synthetic refusal"}


def native_real(arm, view):
    producer = NativeProducer(arm, view)
    return lambda c, d: {"kind": "response", **producer.query(c, d)}


def test_saved_family_envelope_is_retained_but_exact_service_gets_only_math_fields():
    data = toy()
    original = encode(data)
    calls = []
    def capture(request, method):
        calls.append((request, method))
        return refused(request, method)
    workflow = Workflow(data, "O+S", native=native_refused, exact_call=capture)
    workflow.preprocess()
    assert len(calls) == 1 and calls[0][1] == "S"
    assert set(calls[0][0]) == {"program", "context_sizes", "exact_weights", "exact_reference"}
    assert calls[0][0] == {k: v for k, v in data["family_request"].items() if k != "schema_version"}
    assert encode(data) == original


@pytest.mark.parametrize("change", [{"schema_version": "unsupported/1"}, {"untrusted_hint": True}])
def test_unknown_family_envelope_refused(change):
    data = toy()
    data["family_request"].update(change)
    with pytest.raises(ValueError, match="family task envelope"):
        Workflow(data, "O", native=native_refused, exact_call=refused)


def test_exact_scaled_legacy_multipliers_do_not_round_product():
    got = Fraction(effective_legacy_multipliers([.1], .3)[0])
    assert got == Fraction.from_float(.1) * Fraction.from_float(.3)
    assert got != Fraction.from_float(.1 * .3)


def test_owned_child_explicitly_hides_gpus_without_mutating_minimal_environment():
    registry = ChildRegistry()
    environment = {"LANG": "C.UTF-8", "OMP_NUM_THREADS": "1"}
    process = registry.launch([sys.executable, "-I", "-c", "import os; print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')))"],
                              env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              start_new_session=True)
    output, error = process.communicate(timeout=3)
    assert process.returncode == 0 and output.strip() == b"''" and not error
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert all(item["reaped"] for item in registry.cleanup(time.perf_counter_ns() + 1_000_000_000))


def test_saved_ideal_integer_text_pairs_are_exact_and_original_pairs_retained():
    data = toy()
    data["queries"][0]["ideal_objective_exact"] = [["1", "3"], ["-1", "3"]]
    w = Workflow(data, "O", native=native_real("O", data["legacy_view"]), exact_call=refused)
    r = w.query(data["queries"][0])
    assert r["query"]["ideal_objective_exact"] == [["1", "3"], ["-1", "3"]]
    assert r["receipt"]["ideal"]["objective"] == ["1/3", "-1/3"]
    assert r["ideal_source_mapping"]["canonical_sha256"] == sha256(["1/3", "-1/3"])


@pytest.mark.parametrize("pair", [[True, 1], ["01", "3"], ["-0", "1"], ["+1", "3"],
                                   ["1", "0"], ["1", "-3"], ["2", "4"], ["0", "2"], [.5, 1]])
def test_noncanonical_ideal_pairs_refused(pair):
    with pytest.raises(ValueError):
        ideal_exact_vector([pair])


def test_constant_point_deterministic_and_family_skipped_in_all_arms():
    data = toy(constant=True)
    assert constant_intersection(data["family_request"]) == ["0/1", "0/1"]
    for arm in ("H", "O", "O+S", "O+G"):
        w = Workflow(data, arm, native=native_refused,
                     exact_call=lambda *args: pytest.fail("constant shortcut must avoid family solve"))
        w.preprocess()
        assert w.query(data["queries"][0])["receipt"]["represented"]["decision"] == "false"
        assert w.stats["family_calls"] == 0


def test_failed_pairs_and_objectives_not_retried_but_later_points_help():
    data = toy(constant=True)
    data["queries"] = [{"request_index": i, "policy_index": 0, "objective": [1., -1.],
                        "ideal_objective_exact": ["1/1", "-1/1"], "threshold": -.75} for i in range(3)]
    w = Workflow(data, "O", native=native_refused, exact_call=refused)
    assert w.query(data["queries"][0])["status"] == "unknown"
    assert w.query(data["queries"][1])["status"] == "unknown"
    assert w.stats["producer_attempts"] == w.stats["exact_fixed_calls"] == 1
    w.receiver.accept_point(point_wire(w.receiver, ["0/1", "1/1"]))
    third = w.query(data["queries"][2])
    assert third["receipt"]["represented"]["decision"] == "false"
    assert w.stats["producer_attempts"] == w.stats["exact_fixed_calls"] == 1


def test_threshold_changes_recompare_complete_bound_cache():
    data = toy()
    w = Workflow(data, "O", native=native_real("O", data["legacy_view"]), exact_call=refused)
    result = [w.query(q) for q in data["queries"]]
    assert [r["receipt"]["represented"]["decision"] for r in result] == ["true", "false", "false"]
    assert w.stats["producer_attempts"] == 1
    assert result[0]["legacy_adaptation"]["accepted_legacy_proof"] is True


def test_native_outer_tampering_rejected_before_new_receiver():
    data = toy()
    w = Workflow(data, "H", native=native_refused, exact_call=refused)
    q = data["queries"][0]
    bound = w.receiver.bind_request(["1/2", "-1/2"], ["1/2", "-1/2"], "0/1", request_index=0, policy_index=0)
    answer = NativeProducer("H", data["legacy_view"]).query(q["objective"], q["threshold"])["answer"]
    answer["robust_threshold_decision"] = False
    with pytest.raises(ValueError, match="outer packet"):
        w.legacy.verify_and_adapt(answer, q["objective"], q["threshold"], w.receiver, bound)


def test_legacy_program_and_seed_require_exact_binding():
    data = toy()
    data["legacy_view"]["rhs_lower"][0] = .5
    with pytest.raises(ValueError, match="trusted program differ"):
        Workflow(data, "O", native=native_refused, exact_call=refused)
    data = toy()
    data["seed_point_exact"] = ["1/1", "0/1"]
    with pytest.raises(ValueError, match="seed differs"):
        Workflow(data, "O", native=native_refused, exact_call=refused)
    with pytest.raises(ValueError, match="exactly representable"):
        legacy_view_from_program({"lower": ["1/3"], "upper": ["1/1"], "rows": []}, ["1/2"])


@pytest.mark.parametrize("arm", ["H", "O", "O+S", "O+G"])
def test_tiny_persistent_native_stream_with_full_wall_accounting(arm):
    assert CPU in os.sched_getaffinity(0), "synthetic tests must use CPU14"
    data = toy()
    config = runtime_configuration(ROOT, LIBRARY)
    result = run_stream(data, arm=arm, runtime_config=config,
                        expected_input_sha256=sha256(data), cpu=CPU)
    assert result["complete"], result
    assert result["status"] == "complete"
    assert [r["receipt"]["represented"]["decision"] for r in result["responses"]] == ["true", "false", "false"]
    assert result["native_cleanup"]["exit_code"] == 0
    assert result["time_ns"]["supervisor_residual_ns"] >= 0
    assert sum(result["time_ns"][key] for key in result["timing_scope"]["disjoint_phase_keys"]) + result["time_ns"]["supervisor_residual_ns"] == result["time_ns"]["full_wall_ns"]
    if arm in ("O+S", "O+G"):
        assert result["family"]["verified_success"]
        assert result["statistics"]["family_calls"] == 1
    assert result["unstarted_request_indices"] == []


def test_stream_input_pin_failure_launches_nothing():
    data = toy()
    result = run_stream(data, arm="H", runtime_config={}, expected_input_sha256="0" * 64, cpu=CPU)
    assert result["status"] == "input_refusal"
    assert "native_cleanup" not in result
    assert result["unstarted_request_indices"] == [0, 1, 2]


def nondyadic_toy():
    data = toy()
    p = data["family_request"]["program"]
    p["upper"][1] = "1/4"
    p["rows"] = [{"coefficients": ["3/1", "0/1"], "lower": "1/1", "upper": "3/1"}]
    data["seed_point_exact"] = ["1/2", "1/4"]
    data["legacy_view"] = legacy_view_from_program(p, data["seed_point_exact"])
    return data


@pytest.mark.parametrize("arm", ["O+S", "O+G"])
def test_family_rational_one_third_is_preserved_without_binary64_rounding(arm):
    from icde2027.family_lp_v7.service import run_call
    data = nondyadic_toy()
    config = runtime_configuration(ROOT, LIBRARY)
    w = Workflow(data, arm, native=native_refused, exact_call=lambda request, method:
                 run_call(request, method=method, runtime_config=config["exact_runtime"], cpu=CPU))
    family = w.preprocess()
    assert family["verified_success"], family
    assert family["response"]["certificate"]["point"] == ["1/3", "1/4"]
    assert ["1/3", "1/4"] in [p["point"] for p in w.receiver.export_points()]
    assert [binary64_exact(1/3), "1/4"] not in [p["point"] for p in w.receiver.export_points()]


def test_exact_fixed_fallback_is_once_and_ideal_unknown_does_not_trigger_it():
    from icde2027.family_lp_v7.service import run_call
    data = nondyadic_toy()
    data["queries"] = [{"request_index": i, "policy_index": 0, "objective": [.5, -.5],
                        "ideal_objective_exact": ["0/1", "1/1"], "threshold": d}
                       for i, d in enumerate([0., float(Fraction(1, 24)), .1])]
    config = runtime_configuration(ROOT, LIBRARY)
    calls = []
    def exact(request, method):
        calls.append(method)
        return run_call(request, method=method, runtime_config=config["exact_runtime"], cpu=CPU)
    w = Workflow(data, "O", native=native_refused, exact_call=exact)
    results = [w.query(q) for q in data["queries"]]
    assert calls == ["G_fixed"]
    assert [r["receipt"]["represented"]["optimum"] for r in results] == ["1/24"] * 3
    assert [r["receipt"]["represented"]["decision"] for r in results] == ["true", "true", "false"]
    assert [r["receipt"]["ideal"]["decision"] for r in results] == ["unknown"] * 3
    assert w.stats["producer_attempts"] == 1


@pytest.mark.parametrize("arm", ["H", "O"])
def test_positive_native_branch_exports_and_checks_all_primal_proposals(arm):
    data = toy()
    data["seed_point_exact"] = ["1/1", "0/1"]
    data["legacy_view"] = legacy_view_from_program(data["family_request"]["program"], data["seed_point_exact"])
    w = Workflow(data, arm, native=native_real(arm, data["legacy_view"]), exact_call=refused)
    result = w.query(data["queries"][0])
    assert result["native"]["answer"]["certificate"]["upper_proof"] is None
    assert result["native"]["proposed_points"]
    assert result["native_point_checks"][0]["accepted"]
    assert w.receiver.point_count >= 2
    assert result["receipt"]["represented"]["optimum"] == "1/4"


def test_old_unknown_may_strengthen_to_true_under_exact_receiver():
    from icde2027.threshold_extension.proofs import ThresholdChecker, bundle_proof
    c = float.fromhex("0x1.0000000000001p+0")
    data = toy(constant=True)
    p = {"lower": ["1/1"], "upper": ["1/1"], "rows": []}
    data["family_request"] = {"program": p, "context_sizes": [1], "exact_weights": ["1/1"], "exact_reference": [["1/1"]]}
    data["seed_point_exact"] = ["1/1"]
    data["legacy_view"] = legacy_view_from_program(p, ["1/1"])
    w = Workflow(data, "H", native=native_refused, exact_call=refused)
    checker = ThresholdChecker(data["legacy_view"])
    lower = checker.lower_proof([c], [], [], threshold=1.)
    proof = bundle_proof(checker.digest, [c], 1., lower)
    assert proof["robust_threshold_decision"] is None
    answer = {"certificate": proof, "status": "unresolved", "lower_bound": lower["claimed_lower"],
              "upper_bound": None, "robust_threshold_decision": None}
    bound = w.receiver.bind_request([binary64_exact(c)], [binary64_exact(c)], "1/1", request_index=0, policy_index=0)
    wire, receipt = w.legacy.verify_and_adapt(answer, [c], 1., w.receiver, bound)
    assert receipt["old_decision"] is None
    assert w.receiver.evaluate(bound, wire)["represented"]["decision"] == "true"


MOCK_NATIVE = '''import argparse,json,os,sys,time
p=argparse.ArgumentParser()
for n in ('runtime-json','cpu','memory-limit-bytes','max-line-bytes'):p.add_argument('--'+n)
a=p.parse_args();c=json.loads(a.runtime_json);initial=json.loads(sys.stdin.buffer.readline())
def send(v):print(json.dumps(v,separators=(',',':')),flush=True)
send({'kind':'ready','schema_version':'coat-workflow-native-ready/1','pid':os.getpid(),'arm':initial['arm'],'affinity':[int(a.cpu)],
      'memory_limit_bytes':int(a.memory_limit_bytes),'integer_string_digits':100000,
      'sources':{k:v['sha256'] for k,v in c['sources'].items()}})
request=json.loads(sys.stdin.buffer.readline())
'''


@pytest.mark.parametrize("body, expected", [("time.sleep(60)\n", "timeout"), ("sys.exit(7)\n", "refused")])
def test_mock_native_timeout_or_death_stops_stream_without_rebuilding(tmp_path, body, expected):
    path = tmp_path / "mock_native.py"
    path.write_text(MOCK_NATIVE + body)
    config = runtime_configuration(ROOT, LIBRARY)
    config["worker"] = identity(path)
    data = toy()
    result = run_stream(data, arm="H", runtime_config=config, expected_input_sha256=sha256(data),
                        cpu=CPU, stream_limit_s=.8)
    assert result["status"] == expected, result
    assert not result["complete"]
    assert result["incomplete_request"]["query"]["request_index"] == 0
    assert result["unstarted_request_indices"] == [1, 2]
    assert len(result["children"]) == 1
    assert all(item["reaped"] for item in result["children"])
    assert result["time_ns"]["full_wall_ns"] < 800_000_000


def test_owned_children_in_distinct_sessions_are_all_killed_and_reaped():
    registry = ChildRegistry()
    command = [sys.executable, "-I", "-S", "-c", "import time;time.sleep(60)"]
    one = registry.launch(command, start_new_session=True)
    two = registry.launch(command, start_new_session=True)
    assert os.getsid(one.pid) == one.pid and os.getsid(two.pid) == two.pid
    records = registry.cleanup(time.perf_counter_ns() + 2_000_000_000)
    assert len(records) == 2
    assert all(r["reaped"] and r["exit_code"] == -signal.SIGKILL for r in records)


def test_child_parent_death_signal_is_installed_before_exec():
    registry = ChildRegistry()
    code = "import ctypes;v=ctypes.c_int();assert ctypes.CDLL(None).prctl(2,ctypes.byref(v),0,0,0)==0;print(v.value,flush=True)"
    process = registry.launch([sys.executable, "-I", "-S", "-c", code], start_new_session=True, stdout=subprocess.PIPE)
    output, _ = process.communicate(timeout=2)
    assert output.strip() == str(signal.SIGKILL).encode()
    assert registry.cleanup(time.perf_counter_ns() + 1_000_000_000)[0]["reaped"]


MOCK_EXACT = '''import argparse,json,os,sys,time
p=argparse.ArgumentParser()
for n in ('runtime-json','method','cpu','memory-limit-bytes','max-line-bytes'):p.add_argument('--'+n)
a=p.parse_args();c=json.loads(a.runtime_json)
print(json.dumps({'schema_version':'family-lp-worker-ready/1','ready':True,'method':a.method,'pid':os.getpid(),
 'setup_ns':0,'affinity':[int(a.cpu)],'memory_limit_bytes':int(a.memory_limit_bytes),'integer_string_digits':100000,
 'loaded_sources':{k:c[k]['sha256'] for k in ('core','old_receiver')}}),flush=True)
sys.stdin.buffer.readline()
time.sleep(60)
'''


def test_exact_service_timeout_keeps_native_basis_and_reaps_both_sessions(tmp_path):
    path = tmp_path / "mock_exact.py"
    path.write_text(MOCK_EXACT)
    config = runtime_configuration(ROOT, LIBRARY)
    config["exact_runtime"]["worker"] = identity(path)
    data = toy()
    result = run_stream(data, arm="O+S", runtime_config=config, expected_input_sha256=sha256(data),
                        cpu=CPU, exact_service_limit_s=.03)
    assert result["complete"], result
    assert result["family"]["status"] == "service_timeout", result["family"]
    assert result["statistics"]["family_failures"] == 1
    assert [c["method"] for c in result["children"]] == ["native_producer", "S"]
    assert all(c["reaped"] for c in result["children"])
    assert result["children"][1]["exit_code"] == -signal.SIGKILL
    assert len({c["pid"] for c in result["children"] if c["method"] == "native_producer"}) == 1
    assert result["native_call_accounting"]["complete"]


def test_correctness_failure_preserves_packet_and_marks_remaining_unstarted(monkeypatch):
    from . import service
    class BadWorker:
        def __init__(self, config, arm, view, **kwargs):
            self.producer = NativeProducer(arm, view)
            self.ready = {"synthetic": True}
        def query(self, c, d, deadline):
            result = {"kind": "response", **self.producer.query(c, d)}
            result["answer"]["robust_threshold_decision"] = False
            return result
        def close(self, **kwargs):
            return {"error": None, "reaped": True, "exit_code": 0}
        _remaining = staticmethod(service.NativeWorker._remaining)
    monkeypatch.setattr(service, "NativeWorker", BadWorker)
    data = toy()
    result = run_stream(data, arm="H", runtime_config=runtime_configuration(ROOT, LIBRARY),
                        expected_input_sha256=sha256(data), cpu=CPU)
    assert result["status"] == "correctness_mismatch"
    assert result["incomplete_request"]["native"]["answer"]["certificate"]
    assert result["unstarted_request_indices"] == [1, 2]
    assert result["responses"] == []


def test_response_wire_bytes_and_post_service_clock_are_literal():
    data = toy(constant=True)
    result = run_stream(data, arm="O+S", runtime_config=runtime_configuration(ROOT, LIBRARY),
                        expected_input_sha256=sha256(data), cpu=CPU)
    assert result["complete"], result
    assert result["family"]["status"] == "constant_shortcut"
    for response in result["responses"]:
        raw = response["public_response_wire_utf8"].encode("ascii")
        assert len(raw) == response["public_response_bytes"]
        assert json.loads(raw)["receipt"] == response["receipt"]
    assert result["total_run_stream_call_wall_ns"] == result["time_ns"]["full_wall_ns"] + result["post_service_report_assembly_ns"]
    assert result["statistics"]["actual_exact_worker_launches"] == 0
