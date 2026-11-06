"""Independent, optimizer-free replay of saved Coat workflow evidence.

The auditor never imports the producer adapters, services, study driver or
optimizers. Inputs and saved output are evidence only, and each emitted receipt
is reconstructed at its original online position before later points exist.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

from .receiver import RationalReceiver, program_input_digest, POINT_SCHEMA, PARTIAL_SCHEMA
from icde2027.threshold_extension.proofs import ThresholdChecker

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("H", "O", "O+S", "O+G")
PREFIXES = (1, 10, 50, 80, 160, 240)
WORKFLOW_SHA = "68d408c420da27322f3044c8de3f8af32dd55518c125d7b3ad85cb4852bd64e5"
VIEW_IDS = ("8cc608157bcd58e0c9458060ecdf8fd3009070b1cc293084cf1906ead6f6a7dc",
            "9740606138a13f4e01077444f054a23fcc02ac93eaa9137281c5e36e5436f287")


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def loads(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def forbidden(_):
        raise AuditError("nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=forbidden)


def identity(path):
    raw = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def read_pinned(path, pin):
    found = identity(path)
    need(found["sha256"] == pin["sha256"] and ("bytes" not in pin or found["bytes"] == pin["bytes"]),
         "file pin mismatch: " + str(path))
    return loads(Path(path).read_bytes())


def rational(value):
    q = Fraction(value)
    return str(q.numerator) + "/" + str(q.denominator)


def represented(value):
    need(type(value) in (float, int) and math.isfinite(value), "invalid represented binary64 number")
    return rational(Fraction.from_float(float(value)))


def ideal(values):
    result = []
    for value in values:
        if type(value) is str:
            q = Fraction(value)
            need(rational(q) == value, "noncanonical ideal rational")
        else:
            need(type(value) in (list, tuple) and len(value) == 2, "invalid ideal numerator/denominator pair")
            parts = []
            for part in value:
                need(type(part) is int or (type(part) is str and re.fullmatch(r"(?:0|-?[1-9][0-9]*)", part)),
                     "noncanonical ideal pair integer")
                parts.append(int(part))
            need(parts[1] > 0, "nonpositive ideal denominator")
            q = Fraction(*parts)
            need((q.numerator, q.denominator) == tuple(parts), "nonreduced ideal pair")
        result.append(rational(q))
    return result


def verifier(relative, name):
    """Load the independent verifier directly, without producer package init."""
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PARTITION = verifier("icde2027/partition_capacity_v6/verify.py", "_coat_audit_partition_verifier")
FIXED = verifier("icde2027/family_lp_v7/verify_general.py", "_coat_audit_fixed_verifier")


def family_math_request(family, *, allow_saved_legacy=False):
    """Validate the immutable task envelope, then select old-verifier fields.

    The task ID hashes the entire five-key input. The unchanged exact service
    and verifiers bind only the four mathematical fields. Saved historical toy
    bundles alone may omit the wrapper version in explicit synthetic mode.
    """
    fields = ("program", "context_sizes", "exact_weights", "exact_reference")
    need(type(family) is dict, "family task envelope must be an object")
    expected = set(fields) | {"schema_version"}
    if not (allow_saved_legacy and set(family) == set(fields)):
        need(set(family) == expected, "family task envelope fields differ")
        need(family["schema_version"] == "verified-coat-family-task/1", "unsupported family task envelope schema")
    mathematical = {name: family[name] for name in fields}
    FIXED.family_input_digest(**mathematical)
    return mathematical


def point_wire(receiver, point):
    return encode({"schema_version": POINT_SCHEMA, "view_sha256": receiver.view_sha256,
                   "program_sha256": receiver.program_sha256, "point": point})


def partial_wire(receiver, request, *, point, lower, upper):
    return encode({"schema_version": PARTIAL_SCHEMA, "view_sha256": receiver.view_sha256,
                   "program_sha256": receiver.program_sha256, "request_sha256": request.request_sha256,
                   "point": point, "original_row_lower_dual": lower, "original_row_upper_dual": upper})


def validate_program_pair(data):
    view, program = data["legacy_view"], data["family_request"]["program"]
    old = ThresholdChecker(view, cache_size=0)
    need([represented(v) for v in view["lower"]] == program["lower"] and
         [represented(v) for v in view["upper"]] == program["upper"], "native box differs from exact program")
    need(len(program["rows"]) == old.j, "native/exact row count differs")
    matrix = view["matrix"]
    for j, row in enumerate(program["rows"]):
        coefficients = [Fraction(0) for _ in view["lower"]]
        for k in range(matrix["indptr"][j], matrix["indptr"][j + 1]):
            coefficients[matrix["indices"][k]] += Fraction(represented(matrix["data"][k]))
        need(list(map(rational, coefficients)) == row["coefficients"] and
             represented(view["rhs_lower"][j]) == row["lower"] and
             represented(view["rhs_upper"][j]) == row["upper"], "native/exact original row differs")
    need(list(map(represented, view["feasible_point"])) == data["seed_point_exact"], "seed binding differs")
    bindings = data.get("input_bindings", {})
    for key, value in (("family_task_id", data["view_id"]), ("legacy_view_sha256", old.digest),
                       ("program_sha256", program_input_digest(program))):
        if key in bindings:
            need(bindings[key] == value, "prepared input identity differs: " + key)
    return old


def constant_witness(family):
    """Independently intersect the one-variable constant constraints per context."""
    p, sizes = family["program"], family["context_sizes"]
    labels = [x for x, size in enumerate(sizes) for _ in range(size)]
    limits = []
    for x in range(len(sizes)):
        coordinates = [i for i, label in enumerate(labels) if label == x]
        limits.append([max(Fraction(p["lower"][i]) for i in coordinates),
                       min(Fraction(p["upper"][i]) for i in coordinates)])
    for row in p["rows"]:
        values = list(map(Fraction, row["coefficients"]))
        active = {labels[i] for i, value in enumerate(values) if value}
        if not active:
            if not Fraction(row["lower"]) <= 0 <= Fraction(row["upper"]):
                return None
            continue
        if len(active) != 1:
            return None
        total = sum(values, Fraction())
        if not total:
            if not Fraction(row["lower"]) <= 0 <= Fraction(row["upper"]):
                return None
            continue
        lo, hi = sorted((Fraction(row["lower"]) / total, Fraction(row["upper"]) / total))
        interval = limits[next(iter(active))]
        interval[:] = max(interval[0], lo), min(interval[1], hi)
    if any(lo > hi for lo, hi in limits):
        return None
    return [rational(limits[label][0]) for label in labels]


METHOD_SOURCES = {"S": ("core", "old_receiver"),
                  "G_partition": ("adapter", "library", "old_receiver"),
                  "G_fixed": ("adapter", "library", "general_receiver")}
WORKFLOW_COUNTERS = {"constant_shortcuts", "requests", "common_evidence_hits", "objective_cache_hits",
    "point_pool_decisive_hits", "lower_evidence_decisive_hits", "producer_attempts", "actual_native_calls",
    "accepted_native_points", "rejected_native_points", "producer_refusals", "failed_pair_not_retried",
    "exact_fixed_calls", "exact_fixed_failures", "failed_exact_objective_not_retried", "family_calls", "family_failures"}


def nonnegative_ns(value, label):
    need(type(value) is int and value >= 0, "invalid nonnegative nanoseconds: " + label)
    return value


def exact_metadata(call, method, runtime_config, cpu):
    """Check saved transport/resource claims against the admitted configuration."""
    need(type(call.get("record_complete")) is bool and type(call.get("verified_success")) is bool
         and type(call.get("reaped")) is bool, "invalid exact completeness/success flags")
    status = call.get("status")
    need(status in {"verified_optimal", "refused", "resource_refusal", "input_refusal", "startup_timeout", "service_timeout",
                    "diagnostic_timeout", "initialization_refused", "protocol_error", "premature_eof", "io_error",
                    "process_exit", "incomplete", "supervisor_error"}, "unknown exact service outcome")
    config = runtime_config["exact_runtime"]
    # A preflight refusal may precede recording a request/configuration. A
    # launched worker never qualifies for this exception.
    preflight_only = call.get("request_sha256") is None
    if preflight_only:
        need(status in ("input_refusal", "resource_refusal") and not call["record_complete"] and not call["verified_success"]
             and call.get("worker_pid") is None and call.get("ready") is None and call.get("response") is None
             and call.get("diagnostics") is None, "preflight refusal contains later-stage evidence")
    else:
        need(call.get("runtime_config_sha256") == digest(config), "exact runtime configuration digest differs")
        need(type(call.get("cpu")) is int and call["cpu"] == cpu and call.get("memory_limit_bytes") == 8 * 1024 ** 3
             and type(call.get("integer_string_digits")) is int and call["integer_string_digits"] == 100000,
             "exact worker registered resources differ")
        need(call.get("max_line_bytes") == 32 * 1024 ** 2 and call.get("max_stderr_bytes") == 1024 ** 2,
             "exact transport limits differ")
        caps = call.get("limits_ns")
        need(type(caps) is dict and set(caps) == {"startup", "service", "diagnostics"}, "exact phase caps missing")
        for phase, limit in (("startup", 30_000_000_000), ("service", 10_000_000_000), ("diagnostics", 30_000_000_000)):
            need(type(caps[phase]) is int and 0 < caps[phase] <= limit, "exact phase cap exceeds registration")
        need(caps["startup"] == caps["diagnostics"] and caps["service"] <= caps["startup"], "exact clipped phase allocation differs")
    ready = call.get("ready")
    if ready is not None:
        need(type(ready) is dict and set(ready) == {"schema_version", "ready", "method", "pid", "setup_ns", "affinity",
             "memory_limit_bytes", "integer_string_digits", "loaded_sources"}, "exact ready fields differ")
        need(ready["schema_version"] == "family-lp-worker-ready/1" and ready["ready"] is True and ready["method"] == method
             and type(ready["pid"]) is int and ready["pid"] > 0 and ready["pid"] == call.get("worker_pid"), "exact worker identity differs")
        need(ready["affinity"] == [cpu] and ready["memory_limit_bytes"] == 8 * 1024 ** 3
             and type(ready["integer_string_digits"]) is int and ready["integer_string_digits"] == 100000,
             "exact ready resource declaration differs")
        need(ready["loaded_sources"] == {key: config[key]["sha256"] for key in METHOD_SOURCES[method]}, "exact loaded source binding differs")
        nonnegative_ns(ready["setup_ns"], "exact setup")
    timing = call.get("time_ns")
    need(type(timing) is dict and "total_wall_ns" in timing, "exact clock missing")
    phases = {"preflight_ns", "startup_ns", "service_ns", "diagnostics_ns", "supervision_unassigned_ns"}
    need(set(timing) <= phases | {"total_wall_ns"}, "unknown exact clock scope")
    for key, value in timing.items():
        nonnegative_ns(value, "exact " + key)
    if preflight_only:
        need(set(timing) == {"total_wall_ns"}, "preflight-only exact refusal has later phase clock")
    else:
        need(sum(timing.get(key, 0) for key in phases) == timing["total_wall_ns"], "exact clock scopes do not add")
        if ready is not None:
            need(timing.get("startup_ns", -1) >= ready["setup_ns"], "exact setup exceeds startup wall")
        if call["record_complete"]:
            need(set(timing) == phases | {"total_wall_ns"}, "complete exact call lacks recorded phases")
        if status in ("startup_timeout", "service_timeout", "diagnostic_timeout"):
            phase = {"startup_timeout": "startup", "service_timeout": "service", "diagnostic_timeout": "diagnostics"}[status]
            need(timing.get(phase + "_ns", -1) >= call["limits_ns"][phase], "exact timeout occurs before its recorded deadline")
    if call.get("stderr_utf8") is not None:
        raw = call["stderr_utf8"].encode("utf-8")
        # Replacement decoding is explicitly lossy for non-UTF8 stderr. The
        # recorded bytes/hash remain opaque in that case; never invent bytes.
        if "\ufffd" not in call["stderr_utf8"]:
            need(call.get("stderr_bytes") == len(raw) and call.get("stderr_sha256") == hashlib.sha256(raw).hexdigest(),
                 "literal exact stderr identity differs")
    return preflight_only


def check_exact_call(call, request, method, *, runtime_config, cpu):
    """Validate transport/completeness and independently check every present proof."""
    need(call.get("schema_version") == "family-lp-service-record/1" and call.get("method") == method,
         "exact service method/schema differs")
    preflight_only = exact_metadata(call, method, runtime_config, cpu)
    request_wire = encode(request)
    need(preflight_only or (call.get("request_sha256") == hashlib.sha256(request_wire).hexdigest() and
         call.get("request_bytes") == len(request_wire)), "exact call request binding differs")
    response = call.get("response")
    checked = None
    if response is not None:
        need(call.get("ready") is not None and type(response) is dict and
             set(response) == {"status", "certificate", "receiver_receipt"}, "exact public response fields/stage differ")
        raw = encode(response)
        need(call.get("response_bytes") == len(raw) and call.get("response_sha256") == hashlib.sha256(raw).hexdigest(),
             "exact response wire byte/digest mismatch")
        cert = response.get("certificate")
        if cert is not None:
            need(response.get("status") == "verified_optimal", "exact proof contradicts public response status")
            if method == "G_fixed":
                checked = FIXED.verify_fixed_certificate(request["program"], request["exact_objective"], cert,
                    expected_input_sha256=FIXED.fixed_input_digest(request["program"], request["exact_objective"]))
            else:
                checked = PARTITION.verify_certificate(**request, certificate=cert,
                    expected_input_sha256=PARTITION.input_digest(**request))
            need(response["receiver_receipt"] == checked, "exact public receiver receipt differs from independent verification")
        else:
            need(response.get("status") == "refused" and response.get("receiver_receipt") is None,
                 "proofless exact response is not a refusal")
    else:
        need(call.get("response_bytes") is None and call.get("response_sha256") is None, "missing exact response has a wire identity")
    diagnostics = call.get("diagnostics")
    if diagnostics is not None:
        need(response is not None and type(diagnostics) is dict and diagnostics.get("schema_version") == "family-lp-worker-diagnostics/1"
             and diagnostics.get("method") == method and diagnostics.get("request_sha256") == call["request_sha256"],
             "exact diagnostics binding/stage differs")
        raw = encode(diagnostics)
        need(call.get("diagnostic_bytes") == len(raw) and call.get("diagnostic_sha256") == hashlib.sha256(raw).hexdigest(),
             "exact diagnostic wire identity differs")
        if diagnostics.get("complete") is True:
            need(diagnostics.get("outcome_kind") in ("verified_optimal", "refused", "resource_refusal"), "unknown exact diagnostic outcome")
            need((diagnostics["outcome_kind"] == "verified_optimal") is (checked is not None), "exact diagnostics contradict public proof")
            if checked is not None:
                need(diagnostics.get("error") is None and diagnostics.get("failure_stage") is None
                     and diagnostics.get("operation", {}).get("mathematical_receiver_invocations") == 1,
                     "successful exact diagnostics contain failure or extra receiver calls")
            for key, value in diagnostics.get("worker_timing_ns", {}).items():
                nonnegative_ns(value, "exact worker " + key)
            worker_times = diagnostics.get("worker_timing_ns", {})
            operation_phases = {"request_read_and_decode_ns", "request_binding_ns", "construction_ns",
                                "public_certificate_adaptation_ns", "receiver_ns", "generic_call_ns"}
            need(set(worker_times) <= operation_phases | {"before_response_ns", "public_serialization_and_write_ns", "runtime_inventory_ns"},
                 "unrecognized exact worker clock scope")
            before_response = worker_times.get("before_response_ns")
            if before_response is not None:
                # Worker timing begins before its blocking request read; the
                # supervisor service clock starts only after receiving ready.
                need(before_response <= call["time_ns"]["total_wall_ns"], "exact worker clock exceeds full call wall")
                need(sum(worker_times.get(key, 0) for key in operation_phases) <= before_response,
                     "exact worker operation phases exceed enclosing before-response clock")
                need(before_response + worker_times.get("public_serialization_and_write_ns", 0) + worker_times.get("runtime_inventory_ns", 0)
                     <= call["time_ns"]["total_wall_ns"], "exact worker disjoint scopes exceed full call wall")
        else:
            need(diagnostics.get("complete") is False and diagnostics.get("resource_refusal") is True
                 and not call["record_complete"], "invalid incomplete exact diagnostics")
    else:
        need(call.get("diagnostic_bytes") is None and call.get("diagnostic_sha256") is None,
             "missing exact diagnostics have a wire identity")
    usable = bool(call.get("record_complete") is True and call.get("status") == "verified_optimal"
                  and call.get("reaped") is True and call.get("exit_code") == 0 and checked is not None)
    need(call.get("verified_success") is usable, "exact success flag disagrees with independently checked evidence and exit")
    if call.get("record_complete"):
        need(isinstance(diagnostics, dict) and diagnostics.get("complete") is True and call["reaped"] is True
             and type(call["exit_code"]) is int and call["exit_code"] == 0 and call.get("error") is None,
             "complete exact service lacks complete diagnostics")
        expected_status = "resource_refusal" if diagnostics["outcome_kind"] == "resource_refusal" else response["status"]
        need(call["status"] == expected_status, "exact transport/public status differs")
    elif call.get("status") == "verified_optimal":
        raise AuditError("incomplete exact call claims verified optimal status")
    if not call["record_complete"]:
        need(type(call.get("error")) is str and call["error"].strip(), "incomplete exact call lacks failure reason")
    if diagnostics is not None and diagnostics.get("complete") and checked is None:
        need(type(diagnostics.get("error")) is str and diagnostics["error"].strip(), "exact refusal lacks worker reason")
    return response["certificate"] if usable else None


def legacy_packet(old, receiver, request, query, native):
    answer = native["answer"]
    proof = answer.get("certificate")
    if proof is None:
        need(answer.get("status") == "uncertain" and answer.get("robust_threshold_decision") is None
             and answer.get("lower_bound") is None and answer.get("upper_bound") is None
             and isinstance(answer.get("reason"), str) and answer["reason"].strip(), "malformed legacy refusal")
        return None, None
    receipt = old.verify(proof, expected_objective=query["objective"], expected_threshold=query["threshold"])
    need(answer.get("robust_threshold_decision") is receipt["robust_threshold_decision"]
         and answer.get("lower_bound") == receipt["lower_bound"]
         and answer.get("upper_bound") == receipt["upper_bound"]
         and answer.get("status") == ("certified" if receipt["robust_threshold_decision"] is True else "unresolved"),
         "legacy outer packet differs from original checked contract")
    lower = proof["lower_proof"]
    scale = Fraction(represented(lower.get("dual_scale", 1.0))) if lower["format"] == "qbridge-scaled-dual-proof-v1" else Fraction(1)
    eta = [rational(Fraction(represented(v)) * scale) for v in lower["lower_multipliers"]]
    lam = [rational(Fraction(represented(v)) * scale) for v in lower["upper_multipliers"]]
    point = None if proof["upper_proof"] is None else list(map(represented, proof["upper_proof"]["point"]))
    return partial_wire(receiver, request, point=point, lower=eta, upper=lam), receipt


class OnlineReplay:
    """Chronological evidence reconstruction, with no future-point preload."""
    def __init__(self, data, arm, *, synthetic=False, runtime_config, cpu):
        self.data, self.arm = data, arm
        self.runtime_config, self.cpu = runtime_config, cpu
        self.family = family_math_request(data["family_request"], allow_saved_legacy=synthetic)
        self.old = validate_program_pair(data)
        self.receiver = RationalReceiver(self.family["program"], view_sha256=data["view_id"],
                         expected_program_sha256=program_input_digest(self.family["program"]))
        self.receiver.accept_point(point_wire(self.receiver, data["seed_point_exact"]))
        self.constant = constant_witness(self.family)
        if self.constant is not None:
            self.receiver.accept_point(point_wire(self.receiver, self.constant))
        self.attempted_pairs, self.attempted_exact, self.seen_objectives = set(), set(), set()
        self.stats = Counter(constant_shortcuts=int(self.constant is not None))
        self.call_outcomes = Counter()
        self.family_certificate = None

    def family_step(self, saved, complete):
        if saved is None:
            need(not complete, "complete stream lacks family disposition")
            return
        if self.arm in ("H", "O"):
            need(saved == {"status": "not_requested"}, "unregistered family acquisition in native-only arm")
        elif self.constant is not None:
            need(saved == {"status": "constant_shortcut", "point": self.constant, "family_solve_performed": False},
                 "constant shortcut changed or unnecessary family purchase")
        else:
            self.stats["family_calls"] += 1
            self.call_outcomes[saved["status"]] += 1
            cert = check_exact_call(saved, self.family, "S" if self.arm == "O+S" else "G_partition",
                                    runtime_config=self.runtime_config, cpu=self.cpu)
            if cert is None:
                self.stats["family_failures"] += 1
            else:
                self.family_certificate = cert
                receipt = self.receiver.accept_point(point_wire(self.receiver, cert["point"]))
                need(saved.get("accepted_point") == receipt, "family point acceptance receipt differs")
                checked = PARTITION.verify_certificate(**self.family, certificate=cert,
                            expected_input_sha256=PARTITION.input_digest(**self.family))
                need(saved.get("external_family_receiver") == checked, "family external receipt differs")

    def response(self, row, index, *, partial=False):
        query = self.data["queries"][index]
        need(encode(row.get("query")) == encode(query), "saved request/order/original binary64 input differs")
        rep, companion = list(map(represented, query["objective"])), ideal(query["ideal_objective_exact"])
        delta = represented(query["threshold"])
        bound = self.receiver.bind_request(rep, companion, delta, request_index=index, policy_index=query["policy_index"])
        if partial and (row.get("request_sha256") is None or "before_native_receipt" not in row):
            need(row.get("native") is None and row.get("exact_fixed") is None and not row.get("native_point_checks"),
                 "early partial contains evidence acquired before recorded request binding")
            return {"request_index": index, "status": "partial", "evidence_extent": "request binding/common checks incomplete",
                    "expected_request_sha256": bound.request_sha256, "not_counted_as_success": True}
        need(row.get("request_sha256") == bound.request_sha256, "request digest differs")
        mapping = row.get("ideal_source_mapping")
        if mapping is not None:
            need(mapping.get("source_sha256") == digest(query["ideal_objective_exact"]) and
                 mapping.get("canonical_sha256") == digest(companion), "ideal source mapping digest differs")
        elif not partial:
            raise AuditError("complete response lacks ideal source mapping")
        objective, pair = tuple(rep), (tuple(rep), delta)
        cached_before = objective in self.seen_objectives
        self.seen_objectives.add(objective)
        self.stats["requests"] += 1
        calculated = self.receiver.evaluate(bound)
        need(row.get("before_native_receipt") == calculated, "pre-query cache/point state differs (possible future evidence)")
        before = calculated["represented"]["decision"]
        native = row.get("native")
        expected_route = "common_cache_or_point" if before != "unknown" else "previous_pair_unresolved"
        if before != "unknown":
            need(native is None and row.get("exact_fixed") is None, "solving despite resolved represented objective")
            self.stats["common_evidence_hits"] += 1
            self.stats["objective_cache_hits"] += int(cached_before)
            self.stats["point_pool_decisive_hits"] += int(before == "false")
            self.stats["lower_evidence_decisive_hits"] += int(before == "true")
        elif native is not None:
            need(pair not in self.attempted_pairs, "producer pair retried")
            self.attempted_pairs.add(pair)
            self.stats["producer_attempts"] += 1
            expected_route = "native_producer"
            need(native.get("kind") in ("response", "producer_refused"), "unknown native outcome")
            need(native.get("sequence") == self.stats["producer_attempts"] - 1, "persistent producer sequence differs")
            diagnostics = native if native["kind"] == "response" else native.get("partial_diagnostics") or {}
            actual_calls = diagnostics.get("actual_native_calls", 0)
            need(type(actual_calls) is int and 0 <= actual_calls <= (1 if self.arm == "H" else 2), "native call count invalid")
            observed = diagnostics.get("observed_native_calls", [])
            need(len(observed) == actual_calls and all(type(c.get("elapsed_ns")) is int and c["elapsed_ns"] >= 0 for c in observed),
                 "native call observations differ")
            need(nonnegative_ns(diagnostics.get("producer_ns"), "native producer") >= sum(c["elapsed_ns"] for c in observed),
                 "native call clocks exceed producer wall")
            self.stats["actual_native_calls"] += actual_calls
            proposals, checks = diagnostics.get("proposed_points", []), row.get("native_point_checks", [])
            if native["kind"] == "response":
                engine_calls = native["answer"].get("native_calls")
                need(type(engine_calls) is list and len(engine_calls) == actual_calls, "native diagnostics count differs")
                for engine_call, observer_call in zip(engine_calls, observed, strict=True):
                    need(type(engine_call.get("primal_values_valid")) is bool and
                         observer_call.get("status") == engine_call.get("run_status") and observer_call.get("error") is None,
                         "native engine/observer call source differs")
                    seconds = engine_call.get("seconds")
                    need(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0 and
                         observer_call["elapsed_ns"] <= math.ceil(seconds * 1e9), "observed native time exceeds enclosing engine call")
                # Each unchanged engine _native call invokes getSolution once.
                # The transparent observer exports every value_valid solution,
                # including points later rejected by either exact checker.
                need(len(proposals) == sum(c["primal_values_valid"] for c in engine_calls),
                     "native primal export omits or adds value-valid solutions")
            need(len(proposals) == len(checks), "missing/extra native proposal checks")
            for proposed, check in zip(proposals, checks, strict=True):
                need(check.get("binary64_point") == proposed, "native proposal changed before verification")
                try:
                    receipt = self.receiver.accept_point(point_wire(self.receiver, list(map(represented, proposed))))
                except ValueError:
                    need(check.get("accepted") is False and isinstance(check.get("error"), str), "invalid proposal was accepted")
                    self.stats["rejected_native_points"] += 1
                else:
                    need(check.get("accepted") is True and check.get("receipt") == receipt, "native point receipt differs")
                    self.stats["accepted_native_points"] += 1
            if native["kind"] == "response":
                need(len(native["answer"].get("native_calls", [])) == actual_calls, "native diagnostics count differs")
                wire, legacy = legacy_packet(self.old, self.receiver, bound, query, native)
                calculated = self.receiver.evaluate(bound, wire)
                adaptation = row.get("legacy_adaptation")
                need(isinstance(adaptation, dict), "missing legacy adaptation")
                old_decision = None if legacy is None else legacy["robust_threshold_decision"]
                need(adaptation.get("old_decision") is old_decision and
                     adaptation.get("old_lower_bound") == (None if legacy is None else legacy["lower_bound"]) and
                     adaptation.get("old_upper_bound") == (None if legacy is None else legacy["upper_bound"]), "legacy adaptation old bounds differ")
                if legacy is not None:
                    need(adaptation.get("legacy_receiver_receipt") == legacy, "legacy verifier receipt differs")
                for name in ("decision", "lower_bound", "upper_bound"):
                    need(adaptation.get("new_represented_" + name) == calculated["represented"][name], "adapted exact bound differs")
                old_text = "unknown" if old_decision is None else "true" if old_decision else "false"
                need(old_text == "unknown" or old_text == calculated["represented"]["decision"], "exact proof contradicts checked old verdict")
                need(adaptation.get("verdict_changed") is (old_text != calculated["represented"]["decision"]), "verdict-change annotation differs")
            else:
                self.stats["producer_refusals"] += 1
                calculated = self.receiver.evaluate(bound)
        elif before == "unknown":
            if partial and pair not in self.attempted_pairs:
                return {"request_index": index, "status": "partial", "evidence_extent": "before native response",
                        "request_sha256": bound.request_sha256, "last_verified_receipt": calculated,
                        "native_response_missing": True}
            need(pair in self.attempted_pairs, "unresolved new pair omitted required producer attempt")
            self.stats["failed_pair_not_retried"] += 1
        exact = row.get("exact_fixed")
        if exact is not None:
            need(calculated["represented"]["decision"] == "unknown" and objective not in self.attempted_exact,
                 "extra/repeated fixed solve or ideal-only trigger")
            self.attempted_exact.add(objective)
            self.stats["exact_fixed_calls"] += 1
            self.call_outcomes[exact["status"]] += 1
            request = {"program": self.family["program"], "exact_objective": rep}
            cert = check_exact_call(exact, request, "G_fixed", runtime_config=self.runtime_config, cpu=self.cpu)
            if cert is None:
                self.stats["exact_fixed_failures"] += 1
            else:
                checked = FIXED.verify_fixed_certificate(request["program"], rep, cert,
                                expected_input_sha256=FIXED.fixed_input_digest(request["program"], rep))
                need(exact.get("external_fixed_receiver") == checked, "saved fixed receipt differs")
                calculated = self.receiver.evaluate(bound, partial_wire(self.receiver, bound, point=cert["point"],
                    lower=cert["original_row_lower_dual"], upper=cert["original_row_upper_dual"]))
                need(calculated["represented"]["optimum"] == cert["optimum"], "fixed optimum differs")
                expected_route = "exact_fixed_fallback"
        elif calculated["represented"]["decision"] == "unknown":
            if objective in self.attempted_exact:
                self.stats["failed_exact_objective_not_retried"] += 1
            elif not partial:
                raise AuditError("unresolved objective omitted mandatory first exact fallback")
        if partial and "receipt" not in row:
            return {"request_index": index, "status": "partial", "request_sha256": bound.request_sha256,
                    "last_verified_receipt": calculated, "evidence_extent": "partial query"}
        need(row.get("receipt") == calculated, "online exact receipt differs (future/missing/changed evidence)")
        need(row.get("route") == expected_route, "saved route differs from reconstructed state machine")
        status = "unknown" if calculated["represented"]["decision"] == "unknown" else "verified"
        need(row.get("status") == status and row.get("point_count") == self.receiver.point_count, "response status/point census differs")
        for key in WORKFLOW_COUNTERS:
            need(row.get("statistics", {}).get(key, 0) == self.stats[key], "per-response statistic differs: " + key)
        return {"request_index": index, "status": "partial" if partial else "verified_replay",
                "request_sha256": bound.request_sha256, "route": expected_route,
                "represented": calculated["represented"], "ideal": calculated["ideal"],
                "point_count": self.receiver.point_count, "producer_attempts": self.stats["producer_attempts"],
                "actual_native_calls": self.stats["actual_native_calls"], "exact_fixed_calls": self.stats["exact_fixed_calls"]}


def clocks_and_bytes(result, queries):
    clock, scope = result["time_ns"], result["timing_scope"]
    keys = scope["disjoint_phase_keys"]
    allowed = {"preflight_ns", "common_preparation_ns", "native_startup_ns", "family_service_ns", "native_service_ns",
               "exact_fixed_service_ns", "receiver_ns", "legacy_receiver_ns", "response_encoding_ns", "cleanup_ns"}
    nested = {"ready_to_final_response_ns", "supervisor_residual_ns", "full_wall_ns"}
    need(type(keys) is list and len(keys) == len(set(keys)) and set(keys) <= allowed and set(clock) == set(keys) | nested,
         "overlapping/unrecognized/missing disjoint clock phase")
    for key, value in clock.items():
        if key != "ready_to_final_response_ns" or value is not None:
            nonnegative_ns(value, key)
    for key in ("post_service_report_assembly_ns", "total_run_stream_call_wall_ns"):
        nonnegative_ns(result[key], key)
    mandatory = {"cleanup_ns"}
    if result.get("ready") is not None:
        mandatory |= {"preflight_ns", "common_preparation_ns", "native_startup_ns"}
        need(clock.get("native_startup_ns", -1) >= nonnegative_ns(result["ready"].get("setup_ns"), "native setup"),
             "native setup exceeds startup phase")
    all_rows = [*result["responses"], *([result["incomplete_request"]] if result.get("incomplete_request") else [])]
    calls = [r["exact_fixed"] for r in all_rows if r.get("exact_fixed") is not None]
    family = result.get("family")
    if isinstance(family, dict) and family.get("schema_version") == "family-lp-service-record/1":
        mandatory.add("family_service_ns")
        need(clock.get("family_service_ns", -1) >= family["time_ns"]["total_wall_ns"], "family cost hidden outside its phase")
    if calls:
        mandatory.add("exact_fixed_service_ns")
        need(clock.get("exact_fixed_service_ns", -1) >= sum(c["time_ns"]["total_wall_ns"] for c in calls), "fixed-call cost hidden outside its phase")
    native_rows = [r for r in all_rows if r.get("native") is not None]
    if native_rows:
        mandatory.add("native_service_ns")
        producer_ns = sum((r["native"] if r["native"]["kind"] == "response" else r["native"].get("partial_diagnostics") or {}).get("producer_ns", 0)
                          for r in native_rows)
        need(clock.get("native_service_ns", -1) >= producer_ns, "native producer costs exceed outer native phase")
    if result["responses"]:
        mandatory |= {"response_encoding_ns", "receiver_ns"}
    if any(r.get("legacy_adaptation") is not None for r in all_rows):
        mandatory.add("legacy_receiver_ns")
    if isinstance(family, dict) and family.get("verified_success"):
        mandatory.add("receiver_ns")
    statistics = result.get("statistics", {})
    for phase, counter in (("family_service_ns", "family_calls"), ("exact_fixed_service_ns", "exact_fixed_calls"),
                           ("native_service_ns", "producer_attempts")):
        need(phase not in keys or statistics.get(counter, 0) > 0, "clock phase has no corresponding attempted operation")
    need(mandatory <= set(keys), "required disjoint clock phase absent")
    need(all(clock[key] > 0 for key in mandatory), "executed phase has no measured duration")
    need(sum(clock[k] for k in keys) + clock["supervisor_residual_ns"] == clock["full_wall_ns"], "full-wall clock decomposition differs")
    need(result["total_run_stream_call_wall_ns"] == clock["full_wall_ns"] + result["post_service_report_assembly_ns"], "report/service clock overlap")
    rows, prefix, previous = result["responses"], {}, (0, 0)
    response_bytes = certificate_bytes = 0
    offsets = set()
    for i, row in enumerate(rows):
        full, ready = row["cumulative_full_ns"], row["cumulative_ready_ns"]
        need(type(full) is int and type(ready) is int and previous[0] <= full <= clock["full_wall_ns"]
             and previous[1] <= ready <= full, "nonmonotone/impossible prefix clock")
        need(type(row["elapsed_query_ns"]) is int and 0 <= row["elapsed_query_ns"] <= ready - previous[1], "query clock exceeds ready interval")
        call = row.get("exact_fixed")
        producer = row.get("native")
        producer = producer if producer and producer.get("kind") == "response" else (producer or {}).get("partial_diagnostics") or {}
        need(row["elapsed_query_ns"] >= (call or {}).get("time_ns", {}).get("total_wall_ns", 0) + producer.get("producer_ns", 0),
             "query clock excludes nested producer/exact call")
        previous = full, ready
        offsets.add(full - ready)
        if i + 1 in PREFIXES:
            prefix[str(i + 1)] = {"full_cumulative_ns": full, "ready_cumulative_ns": ready}
        raw = row["public_response_wire_utf8"].encode("ascii")
        public = {"schema_version": "coat-workflow-public-response/1", "status": row["status"],
                  "request_sha256": row["request_sha256"], "receipt": row["receipt"]}
        need(raw == encode(public) and loads(raw) == public, "literal public wire differs from verified receipt")
        need(row["public_response_bytes"] == len(raw) and row["public_response_sha256"] == hashlib.sha256(raw).hexdigest(), "public wire byte/hash mismatch")
        need(row["public_certificate_bytes"] == len(encode(row["receipt"])), "nested certificate byte count differs")
        response_bytes += len(raw)
        certificate_bytes += row["public_certificate_bytes"]
    need(len(offsets) <= 1 and result["prefixes"] == prefix, "prefix startup/preprocessing offset or prefix census differs")
    if rows:
        need(clock["ready_to_final_response_ns"] == rows[-1]["cumulative_ready_ns"], "ready-to-final-response clock differs")
        need(rows[-1]["cumulative_full_ns"] + clock["cleanup_ns"] <= clock["full_wall_ns"], "cleanup overlaps completed response clock")
        need(rows[0]["cumulative_full_ns"] - rows[0]["cumulative_ready_ns"] >=
             sum(clock.get(k, 0) for k in ("preflight_ns", "common_preparation_ns", "native_startup_ns")),
             "prefix startup offset excludes recorded initialization")
    byte_record = result["bytes"]
    need(byte_record["public_responses"] == response_bytes and byte_record["public_certificates_nested_in_responses"] == certificate_bytes,
         "stream response/certificate bytes differ")
    cleanup = result.get("native_cleanup", {})
    need(byte_record["native_ipc_request"] == cleanup.get("request_bytes", 0) and
         byte_record["native_ipc_response"] == cleanup.get("response_bytes", 0), "native IPC totals differ")
    if result["complete"]:
        requests = [encode({"operation": "initialize", "arm": result["arm"], "legacy_view": queries["legacy_view"]})]
        replies = [encode(result["ready"])]
        sequence = 0
        for row in rows:
            if row.get("native") is not None:
                q = row["query"]
                requests.append(encode({"operation": "query", "sequence": sequence, "objective": q["objective"], "threshold": q["threshold"]}))
                replies.append(encode(row["native"]))
                sequence += 1
        requests.append(encode({"operation": "stop"}))
        replies.append(encode(cleanup["diagnostics"]))
        need(sum(map(len, requests)) == cleanup["request_bytes"] and sum(map(len, replies)) == cleanup["response_bytes"], "literal native IPC byte reconstruction differs")
    return {"clock_ns": clock, "prefixes": prefix, "bytes": byte_record,
            "receiver_external_share_full_wall": (clock.get("receiver_ns", 0) + clock.get("legacy_receiver_ns", 0)) / clock["full_wall_ns"] if clock["full_wall_ns"] else None,
            "receiver_share_scope": "external request/family/fixed/legacy checks only; common initialization and nested exact-worker checks separately recorded"}


def runtime_closure(result, allowed_paths):
    """Independently traverse all observed inventories, including failed calls."""
    allowed = {str(Path(p).resolve()) for p in allowed_paths}
    observed, locations, missing, issues = set(), [], [], []
    fields = ("module_files", "module_cache_files", "mapped_files")
    absent = ("missing_module_files", "missing_mapped_files", "deleted_mappings")
    def inventory(value, location):
        if not isinstance(value, dict):
            missing.append(location)
            return
        locations.append(location)
        for key in fields + absent:
            values = value.get(key)
            if type(values) is not list or any(type(v) is not str for v in values):
                issues.append({"location": location, "field": key, "reason": "invalid_inventory_field"})
                continue
            if key in absent:
                issues.extend({"location": location, "field": key, "path": v, "reason": "missing_or_deleted"} for v in values)
            else:
                for item in values:
                    path = str(Path(item).resolve())
                    observed.add(path)
                    if path not in allowed:
                        issues.append({"location": location, "field": key, "path": path, "reason": "unadmitted_file"})
    def visit(value, location):
        if isinstance(value, dict):
            if value.get("schema_version") == "family-lp-service-record/1":
                diag = value.get("diagnostics")
                if not isinstance(diag, dict) or not isinstance(diag.get("runtime_inventory"), dict):
                    missing.append(location + "/exact_runtime_inventory")
            for key, child in value.items():
                child_location = location + "/" + key
                if key in ("runtime_inventory", "supervisor_runtime_inventory"):
                    inventory(child, child_location)
                elif isinstance(child, (dict, list)):
                    visit(child, child_location)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    visit(child, location + "/" + str(i))
    visit(result, "stream")
    if not isinstance(result.get("supervisor_runtime_inventory"), dict):
        missing.append("stream/supervisor_runtime_inventory")
    native = result.get("native_cleanup", {}).get("diagnostics")
    if not isinstance(native, dict) or not isinstance(native.get("runtime_inventory"), dict):
        missing.append("stream/native_runtime_inventory")
    return {"admitted": not issues and not missing, "inventories_checked": len(locations),
            "observed_file_count": len(observed), "observed_files": sorted(observed),
            "violations": issues, "observations_missing": sorted(set(missing))}


def activity_accounting(result, replay, ledger):
    """Keep attempted/launched/returned work distinct, including interrupted calls."""
    stats = result.get("statistics", {})
    need(type(stats) is dict and set(stats) <= WORKFLOW_COUNTERS | {"actual_exact_worker_launches", "completed_response_native_calls"},
         "unknown final workflow counter")
    for key, value in stats.items():
        need((key == "actual_native_calls" and value is None) or (type(value) is int and value >= 0), "invalid final workflow counter: " + key)
    partial = result.get("incomplete_request")
    pending_ledger = ledger[len(result["responses"])] if partial is not None else {}
    missing_native = pending_ledger.get("native_response_missing") is True
    last = pending_ledger.get("last_verified_receipt", {})
    missing_exact = bool(partial is not None and partial.get("exact_fixed") is None and not missing_native and
                         last.get("represented", {}).get("decision") == "unknown")
    missing_family = bool(result.get("family") is None and not result["responses"] and partial is None and
                          replay.arm in ("O+S", "O+G") and replay.constant is None)
    inflight = {}
    for counter, possible, phase in (("producer_attempts", missing_native, "native_service_ns"),
                                     ("exact_fixed_calls", missing_exact, "exact_fixed_service_ns"),
                                     ("family_calls", missing_family, "family_service_ns")):
        delta = stats.get(counter, 0) - replay.stats[counter]
        need(delta == 0 or (possible and delta == 1 and result["time_ns"].get(phase, 0) > 0),
             "attempt counter differs from online/partial evidence: " + counter)
        inflight[counter] = delta
    early_request = int(partial is not None and pending_ledger.get("evidence_extent") == "request binding/common checks incomplete")
    for key in WORKFLOW_COUNTERS - {"producer_attempts", "exact_fixed_calls", "family_calls", "actual_native_calls"}:
        actual, expected = stats.get(key, 0), replay.stats[key]
        need(actual == expected or (key == "requests" and early_request and actual == expected + 1),
             "final statistic differs from complete and partial chronology: " + key)
    known = replay.stats["actual_native_calls"]
    unknown_native = inflight["producer_attempts"] == 1
    coverage = result["native_call_accounting"]
    need(type(coverage.get("completed_response_call_count")) is int and
         (coverage.get("actual_native_calls") is None or type(coverage["actual_native_calls"]) is int), "invalid native accounting number")
    need(type(coverage.get("complete")) is bool and type(coverage.get("inflight_call_count_unknown")) is bool and
         coverage["complete"] is (not unknown_native) and coverage["inflight_call_count_unknown"] is unknown_native,
         "native coverage contradicts unreturned attempt chronology")
    expected_calls = None if unknown_native else known
    need(coverage.get("actual_native_calls") == expected_calls and stats.get("actual_native_calls", 0) == expected_calls
         and coverage.get("completed_response_call_count") == known and stats.get("completed_response_native_calls", 0) == known,
         "native observed/unknown count differs")
    if missing_native:
        pending_ledger["inflight_native_calls_unknown"] = unknown_native
        pending_ledger["native_attempt_started"] = unknown_native
    if missing_exact:
        pending_ledger["inflight_exact_attempt"] = bool(inflight["exact_fixed_calls"])
    children = result.get("children", [])
    by_pid = {child["pid"]: child for child in children}
    need(len(by_pid) == len(children), "duplicate child identities")
    all_rows = [*result["responses"], *([partial] if partial is not None else [])]
    calls = [row["exact_fixed"] for row in all_rows if row.get("exact_fixed") is not None]
    family = result.get("family")
    if isinstance(family, dict) and family.get("schema_version") == "family-lp-service-record/1":
        calls.append(family)
    matched = set()
    for call in calls:
        pid = call.get("worker_pid")
        if pid is not None:
            need(pid in by_pid and by_pid[pid]["method"] == call["method"] and pid not in matched,
                 "returned exact call is absent/duplicated in child registry")
            need(by_pid[pid]["reaped"] == call["reaped"] and by_pid[pid]["exit_code"] == call["exit_code"],
                 "exact worker exit/cleanup records differ")
            matched.add(pid)
    expected_unreturned = Counter({"G_fixed": inflight["exact_fixed_calls"]})
    if inflight["family_calls"]:
        expected_unreturned["S" if replay.arm == "O+S" else "G_partition"] += 1
    unmatched = Counter(child["method"] for child in children if child["method"] != "native_producer" and child["pid"] not in matched)
    need(all(count <= expected_unreturned[method] for method, count in unmatched.items()), "exact launch without returned or pending attempt")
    need(stats.get("actual_exact_worker_launches", 0) == sum(child["method"] != "native_producer" for child in children),
         "actual exact worker launch count differs")
    outcomes = dict(Counter(call["status"] for call in calls))
    need(result.get("exact_call_outcomes") == outcomes, "exact outcomes omit or add returned calls")
    failures = {"unknown_responses": result["counts"]["represented"].get("unknown", 0),
        "native_refusals": replay.stats["producer_refusals"], "exact_non_successful_calls": sum(not call["verified_success"] for call in calls),
        "stream_timeout": int(result["status"] == "timeout"), "incomplete_positions": int(partial is not None),
        "unstarted_positions": len(result["unstarted_request_indices"])}
    need(result.get("refusal_timeout_counts") == failures and all(type(value) is int for value in result["refusal_timeout_counts"].values()),
         "refusal/timeout counts differ from evidence")
    for counter, cap in (("producer_attempts", 150), ("exact_fixed_calls", 50), ("family_calls", 1)):
        need(stats.get(counter, 0) <= cap, "registered attempt limit exceeded")
    activity = {"native": {"attempted": stats.get("producer_attempts", 0), "returned_records": replay.stats["producer_attempts"],
                    "inflight_attempts": inflight["producer_attempts"], "actual_native_calls": expected_calls, "observed_native_calls": known}}
    for label, method, key in (("fixed", "G_fixed", "exact_fixed_calls"),
                               ("family", "S" if replay.arm == "O+S" else "G_partition", "family_calls")):
        selected = [call for call in calls if call["method"] == method]
        activity[label] = {"attempted": stats.get(key, 0), "launched": sum(c["method"] == method for c in children),
            "returned_records": len(selected), "complete_records": sum(call["record_complete"] for call in selected),
            "verified_successes": sum(call["verified_success"] for call in selected), "inflight_attempts": inflight[key]}
    return activity, failures


def audit_stream(data, result, *, runtime_config=None, synthetic=False, allowed_runtime_paths=None):
    """Reconstruct a single saved stream; production defaults cannot be relaxed implicitly."""
    queries, arm = data["queries"], result.get("arm")
    need(arm in ARMS, "unregistered arm")
    need(all(type(q.get("request_index")) is int and type(q.get("policy_index")) is int for q in queries) and
         [q.get("request_index") for q in queries] == list(range(len(queries))), "input request order differs")
    if not synthetic:
        need(len(queries) == 240 and data["view_id"] in VIEW_IDS and
             len(data["family_request"]["program"]["lower"]) == 300 and
             len(data["family_request"]["program"]["rows"]) == 16 and
             data["family_request"]["context_sizes"] == [300], "production stream census/view differs")
    need(type(runtime_config) is dict and runtime_config.get("schema_version") == "coat-workflow-runtime/1",
         "admitted runtime_config is required for source/resource bindings")
    cpu = 14 if synthetic else 12
    if result.get("ready") is not None:
        need(result["ready"].get("sources") == {key: entry["sha256"] for key, entry in runtime_config["sources"].items()},
             "native source declaration differs from admitted runtime")
    need(result.get("schema_version") == "coat-workflow-stream/1" and result.get("input_sha256") == digest(data),
         "stream schema/input binding differs")
    total, rows = len(queries), result["responses"]
    need(type(rows) is list and len(rows) <= total, "response denominator exceeds original input")
    partial = result.get("incomplete_request")
    need(partial is None or isinstance(partial, dict), "invalid partial record")
    n_partial = int(partial is not None)
    need(len(rows) + n_partial <= total, "partial request exceeds denominator")
    need(result.get("unstarted_request_indices") == list(range(len(rows) + n_partial, total)), "unstarted positions differ")
    need(type(result.get("complete")) is bool, "invalid completeness flag")
    if result["complete"]:
        need(result["status"] == "complete" and len(rows) == total and partial is None, "false complete stream declaration")
    replay = OnlineReplay(data, arm, synthetic=synthetic, runtime_config=runtime_config, cpu=cpu)
    if "input_bindings" in result:
        expected_bindings = {**data.get("input_bindings", {}), "family_task_id": data["view_id"],
                            "program_sha256": replay.receiver.program_sha256,
                            "legacy_view_sha256": replay.old.digest}
        need(result["input_bindings"] == expected_bindings, "saved result provenance bindings differ")
    replay.family_step(result.get("family"), result["complete"])
    ledger = []
    for index, row in enumerate(rows):
        try:
            ledger.append(replay.response(row, index))
        except Exception as error:
            raise AuditError("request " + str(index) + ": " + str(error)) from error
    if partial is not None:
        try:
            ledger.append(replay.response(partial, len(rows), partial=True))
        except Exception as error:
            need(result["status"] == "correctness_mismatch", "unexplained invalid partial evidence: " + str(error))
            ledger.append({"request_index": len(rows), "status": "partial_rejected_evidence",
                           "error": type(error).__name__ + ": " + str(error),
                           "not_counted_as_success": True})
    for index in result["unstarted_request_indices"]:
        ledger.append({"request_index": index, "status": "unstarted", "not_counted_as_success": True})
    counts = {name: dict(Counter(row["receipt"][name]["decision"] for row in rows)) for name in ("represented", "ideal")}
    need(result["counts"] == counts, "response decision counts differ")
    census = result["workload_census"]
    for key, expected in (("requested_positions", total), ("completed_positions", len(rows)),
                          ("incomplete_positions", n_partial), ("unstarted_positions", total - len(rows) - n_partial)):
        need(type(census.get(key)) is int and census[key] == expected, "workload denominator differs: " + key)
    unique = {tuple(map(represented, q["objective"])) for q in queries}
    pairs = {(tuple(map(represented, q["objective"])), represented(q["threshold"])) for q in queries}
    need(census["distinct_represented_objectives"] == len(unique) and census["distinct_represented_pairs"] == len(pairs), "objective/pair census differs")
    if not synthetic:
        need(len(unique) == 50 and len(pairs) == 150, "production objective/pair counts differ")
    need(replay.stats["producer_attempts"] <= 150 and replay.stats["actual_native_calls"] <= (150 if arm == "H" else 300)
         and replay.stats["exact_fixed_calls"] <= 50 and replay.stats["family_calls"] <= 1, "registered attempt bound exceeded")
    stats = result.get("statistics", {})
    activity, failure_counts = activity_accounting(result, replay, ledger)
    coverage = result["native_call_accounting"]
    known = replay.stats["actual_native_calls"]
    need(coverage["completed_response_call_count"] == known, "observed native count differs")
    # activity_accounting already reconciles complete and interrupted work.
    # A pre-query family interruption has no partial request, but still has a
    # started family attempt; do not compare it to returned-call counts again.
    if "retained_points" in result:
        need(result["retained_points"] == replay.receiver.export_points(), "retained points differ from online acquisitions")
    children = result.get("children", [])
    need(len({c["pid"] for c in children}) == len(children), "duplicate child identities")
    native_children = [c for c in children if c["method"] == "native_producer"]
    need(len(native_children) <= 1, "native basis was rebuilt within stream")
    ready = result.get("ready")
    if ready is not None:
        need(len(native_children) == 1 and ready.get("kind") == "ready" and
             ready.get("schema_version") == "coat-workflow-native-ready/1" and ready["arm"] == arm and
             ready["memory_limit_bytes"] == 8 * 1024 ** 3 and type(ready["integer_string_digits"]) is int and
             ready["integer_string_digits"] == 100000 and ready["affinity"] == [cpu] and ready["pid"] == native_children[0]["pid"],
             "native startup identity/resources differ")
    if result["complete"]:
        need(len(native_children) == 1 and all(c["reaped"] for c in children), "complete stream contains unreaped children")
        need(result.get("native_cleanup", {}).get("exit_code") == 0 and not result["native_cleanup"].get("error"), "complete stream has native cleanup failure")
        ready = result["ready"]
        need(ready["arm"] == arm and ready["memory_limit_bytes"] == 8 * 1024 ** 3 and ready["integer_string_digits"] == 100000,
             "native startup resource declarations differ")
        if not synthetic:
            need(ready["affinity"] == [12], "production native affinity differs")
        need(ready["affinity"] == [cpu] and ready["pid"] == native_children[0]["pid"], "native worker affinity/identity differs")
        need(result["native_cleanup"].get("diagnostics", {}).get("queries") == replay.stats["producer_attempts"],
             "native final diagnostics query count differs")
    need(stats.get("actual_exact_worker_launches", 0) == sum(c["method"] != "native_producer" for c in children), "exact child launch count differs")
    measurements = clocks_and_bytes(result, data)
    resources = result["resources"]
    for name in ("stream_limit_s", "exact_service_limit_s"):
        need(type(resources[name]) in (int, float) and math.isfinite(resources[name]), "invalid resource duration")
    for name in ("absolute_deadline_ns", "service_deadline_ns", "cleanup_reserved_ns"):
        nonnegative_ns(resources[name], name)
    need(resources["memory_limit_bytes"] == 8 * 1024 ** 3 and 0 < resources["stream_limit_s"] <= 180
         and 0 < resources["exact_service_limit_s"] <= 10, "registered service resources differ")
    need(resources["absolute_deadline_ns"] - resources["service_deadline_ns"] == resources["cleanup_reserved_ns"], "cleanup reservation differs")
    exceeded = result["time_ns"]["full_wall_ns"] > int(resources["stream_limit_s"] * 1e9)
    need(resources["full_wall_exceeded_registered_limit"] is exceeded, "stream deadline overrun flag differs")
    if result["complete"]:
        need(not exceeded, "complete stream exceeded registered limit")
    closure = None if allowed_runtime_paths is None else runtime_closure(result, allowed_runtime_paths)
    if closure is not None:
        need(not closure["violations"], "unadmitted runtime observations")
        need(not result["complete"] or not closure["observations_missing"], "complete stream missing runtime inventory")
    prefix_quality = {position: {domain: dict(Counter(row["receipt"][domain]["decision"] for row in rows[:int(position)]))
                                for domain in ("represented", "ideal")} for position in measurements["prefixes"]}
    return {"audit_status": "PASS_COMPLETE" if result["complete"] else "PASS_RECORDED_INCOMPLETE",
            "mode": "synthetic" if synthetic else "production", "view_id": data["view_id"], "arm": arm,
            "family_task_envelope_sha256": digest(data["family_request"]),
            "family_mathematical_request_sha256": digest(replay.family),
            "source_status": result["status"], "complete": result["complete"], "denominator": total,
            "completed_positions": len(rows), "partial_positions": n_partial,
            "unstarted_positions": total - len(rows) - n_partial, "counts": counts,
            "actual_native_calls": coverage["actual_native_calls"], "observed_native_calls": known,
            "producer_attempts_observed": activity["native"]["attempted"], "exact_fixed_calls": activity["fixed"]["attempted"],
            "family_calls": activity["family"]["attempted"], "retained_point_count": replay.receiver.point_count,
            "call_activity": activity, "refusal_timeout_counts": failure_counts,
            "exact_call_outcomes": dict(replay.call_outcomes), "runtime_closure": closure,
            **measurements, "prefix_quality": prefix_quality, "ledger": ledger}


def compare_streams(numerator, denominator, *, label, primary=False):
    reasons = []
    if not numerator["complete"] or not denominator["complete"]:
        reasons.append("incomplete service")
    if numerator["audit_status"] != "PASS_COMPLETE" or denominator["audit_status"] != "PASS_COMPLETE":
        reasons.append("missing or rejected audit evidence")
    if numerator["denominator"] != denominator["denominator"]:
        reasons.append("different request denominators")
    differences = {"represented": [], "ideal": []}
    for domain in differences:
        for left, right in zip(numerator["ledger"], denominator["ledger"]):
            a, b = left.get(domain, {}).get("decision"), right.get(domain, {}).get("decision")
            if a != b:
                differences[domain].append({"request_index": left["request_index"], "numerator": a, "denominator": b,
                    "kind": "gain" if b == "unknown" and a in ("true", "false") else
                            "loss" if a == "unknown" and b in ("true", "false") else "difference"})
    if differences["represented"]:
        reasons.append("represented per-request quality differs")
    a, b = numerator["clock_ns"]["full_wall_ns"], denominator["clock_ns"]["full_wall_ns"]
    if type(a) is not int or type(b) is not int or b <= 0:
        reasons.append("nonpositive reference cost")
    return {"comparison": label, "primary": primary, "ratio": None if reasons else a / b,
            "ratio_status": "NA" if reasons else "descriptive_complete_represented_quality_match",
            "na_reasons": reasons, "numerator_full_wall_ns": a, "denominator_full_wall_ns": b,
            "differences": differences, "two_semantics_quality_match": not reasons and not differences["ideal"],
            "interpretation": "whole preprocessing/execution choices; no isolated algorithmic causal effect"}


def check_schedule(schedule):
    need(type(schedule) is list and len(schedule) == 32, "production requires all 32 registered schedule rows")
    expected = []
    labels = ("positive", "zero_control")
    for round_ in range(4):
        order = (0, 1) if round_ % 2 == 0 else (1, 0)
        for view in order:
            for arm in ARMS[round_:] + ARMS[:round_]:
                expected.append({"stream_index": len(expected), "stage": "quality" if round_ == 0 else "timing",
                    "round": round_, "view_id": VIEW_IDS[view], "label": labels[view], "arm": arm,
                    "input_file": labels[view] + ".json", "requests": 240, "stream_wall_limit_seconds": 180})
    need(schedule == expected and all(all(type(row.get(k)) is int for k in
         ("stream_index", "round", "requests", "stream_wall_limit_seconds")) for row in schedule),
         "production schedule order/arms/views/budgets differ")
    return expected


def original_queries(workflow):
    need(workflow.get("format") == "qbridge-coat-metadata-workflow-v5", "original workflow schema differs")
    policies = {row["policy_index"]: row for row in workflow["policies"]}
    need(len(policies) == 80 and len(workflow["requests"]) == 240, "original query census differs")
    result = []
    for i, request in enumerate(workflow["requests"]):
        need(request["query_index"] == i and request["policy_index"] == i % 80,
             "original policy/request sequence differs")
        p = policies[request["policy_index"]]
        need(encode(request["objective"]) == encode(p["objective"]) and
             represented(request["threshold"]) == represented((0.0, .001, .01)[i // 80]), "original represented query/threshold differs")
        result.append({"request_index": i, "policy_index": request["policy_index"],
             "objective": request["objective"], "ideal_objective_exact": p["ideal_objective_exact"],
             "threshold": request["threshold"], "features": request["features"], "multipliers": request["multipliers"]})
    return result


def checked_tree(root, files):
    root = Path(root).resolve()
    verified = {}
    for name, pin in files.items():
        path = root / name
        need(path.resolve().is_relative_to(root) and not path.is_symlink(), "pinned path escapes or is a symlink")
        found = identity(path)
        need(found == pin, "pinned preparation member changed: " + name)
        verified[str(path.resolve())] = found
    return verified


def source_view(data):
    family_math_request(data["family_request"])
    binding = data["input_bindings"]
    source = binding["source"]
    path = ROOT / source["path"]
    document = read_pinned(path, source)
    original = document["document"]["view"]
    need(digest(original) == binding["original_full_view_sha256"], "original full-view digest differs")
    rows = binding["membership"]["original_row_indices"]
    need(rows == list(range(2, 18)), "original moment subset differs")
    matrix = {"indptr": [0], "indices": [], "data": []}
    for index in rows:
        begin, end = original["matrix"]["indptr"][index:index + 2]
        matrix["indices"].extend(original["matrix"]["indices"][begin:end])
        matrix["data"].extend(original["matrix"]["data"][begin:end])
        matrix["indptr"].append(len(matrix["indices"]))
    expected = {"format": original["format"], "lower": original["lower"], "upper": original["upper"],
                "rhs_lower": original["rhs_lower"][2:18], "rhs_upper": original["rhs_upper"][2:18],
                "matrix": matrix, "feasible_point": original["feasible_point"]}
    need(encode(data["legacy_view"]) == encode(expected), "native view differs from original row subset")
    need(digest(data["family_request"]) == data["view_id"], "family request/task ID differs")
    need(binding["intervals_recomputed"] is False and binding["source_budget_reallocated"] is False and
         binding["historical_rank_span_fields_are_not_active_inputs"] is True, "statistical scope declaration differs")
    return {"path": str(path.resolve()), **identity(path)}


def verify_package(prepared, run, *, expected_gate_sha256, expected_completion_sha256):
    """Pin-only plus original-input checks; never import the saved executor."""
    prepared, run = Path(prepared).resolve(), Path(run).resolve()
    gate = read_pinned(run / "ADMISSION.json", {"sha256": expected_gate_sha256})
    need(gate.get("status") == "GO" and gate.get("production_authorized") is True, "production gate is not GO")
    preparation = read_pinned(prepared / "PREPARATION.json", {"sha256": gate["preparation_sha256"]})
    need(preparation.get("schema_version") == "coat-workflow-preparation/1" and preparation.get("status") == "PREPARED_NOT_RUN",
         "preparation declaration differs")
    need(preparation["stream_count"] == 32 and preparation["positions"] == 7680 and preparation["cpu"] == 12 and
         preparation["stream_cap_seconds"] == 180 and preparation["global_cap_seconds"] == 7200, "registered production resources/census differ")
    files = checked_tree(prepared, preparation["files"])
    for kind in ("review_files", "runtime_dependency_files"):
        need(isinstance(gate.get(kind), dict), "missing admission pins: " + kind)
        for name, pin in gate[kind].items():
            need(identity(name) == pin, "changed admitted " + kind + ": " + name)
            files[str(Path(name).resolve())] = pin
    library = preparation["library"]
    need(identity(library["path"]) == {k: library[k] for k in ("bytes", "sha256")}, "native library pin changed")
    files[str(Path(library["path"]).resolve())] = {k: library[k] for k in ("bytes", "sha256")}
    need(identity(prepared / "source_snapshot/icde2027/coat_workflow_v8/study.py")["sha256"] == gate["study_sha256"], "driver source pin differs")
    runtime = loads((prepared / "runtime.json").read_bytes())
    for entry in [runtime["worker"], *runtime["sources"].values(), *runtime["exact_runtime"].values()]:
        need(str(Path(entry["path"]).resolve()) in files or entry == {"path": library["path"], "sha256": library["sha256"]},
             "runtime config source absent from admission")
        need(identity(entry["path"])["sha256"] == entry["sha256"], "runtime-config source SHA differs")
    manifest = loads((prepared / "inputs/INPUT_MANIFEST.json").read_bytes())
    need(identity(prepared / "inputs/INPUT_MANIFEST.json")["sha256"] == preparation["input_manifest_sha256"], "input manifest pin differs")
    schedule = check_schedule(loads((prepared / "inputs/schedule.json").read_bytes())["streams"])
    historical = prepared / "inputs/original_workflow_historical.json"
    need(identity(historical)["sha256"] == WORKFLOW_SHA, "original catalogue raw bytes differ")
    queries = original_queries(loads(historical.read_bytes()))
    inputs, source_pins = {}, []
    need(len(manifest["streams"]) == 2, "production needs exactly two prepared inputs")
    for item in manifest["streams"]:
        data = read_pinned(prepared / "inputs" / item["file"], item)
        need(data["view_id"] in VIEW_IDS and encode(data["queries"]) == encode(queries), "prepared queries differ from all original 240 positions")
        need(data["input_bindings"]["workflow_sha256"] == WORKFLOW_SHA, "catalogue binding differs")
        source_pin = source_view(data)
        source_pins.append(source_pin)
        files[source_pin["path"]] = {key: source_pin[key] for key in ("bytes", "sha256")}
        inputs[item["file"]] = data
    completion = read_pinned(run / "COMPLETION.json", {"sha256": expected_completion_sha256})
    need(completion.get("schema_version") == "coat-workflow-completion/1" and completion["gate_sha256"] == expected_gate_sha256
         and completion["preparation_sha256"] == gate["preparation_sha256"] and completion["registered_streams"] == 32,
         "completion/admission binding differs")
    need(len(completion["streams"]) == 32 and completion["source_and_input_pins_unchanged"] is True,
         "completion omits registered streams or records changed pins")
    return {"gate": gate, "preparation": preparation, "completion": completion, "schedule": schedule,
            "inputs": inputs, "allowed_runtime_paths": sorted(files), "verified_files": files,
            "source_pins": source_pins, "runtime": runtime}


def compact_audit(record):
    result = {k: v for k, v in record.items() if k != "ledger"}
    result["ledger"] = [{"request_index": row["request_index"], "status": row["status"],
                         **{domain: {"decision": row[domain]["decision"]} for domain in ("represented", "ideal") if domain in row}}
                        for row in record["ledger"]]
    return result


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_ledger(path, rows):
    with Path(path).open("xb") as stream:
        for row in rows:
            stream.write(encode(row) + b"\n")


def auditor_sources():
    names = ("coat_workflow_v8/audit.py", "coat_workflow_v8/receiver.py", "threshold_extension/proofs.py",
             "proof_check.py", "family_lp_v7/verify_general.py", "partition_capacity_v6/verify.py")
    return {str(ROOT / "icde2027" / name): identity(ROOT / "icde2027" / name) for name in names}


def audit_run(prepared, run, output, *, expected_gate_sha256, expected_completion_sha256):
    started = time.perf_counter_ns()
    need(sys.get_int_max_str_digits() == 100000, "production auditor requires the registered 100000-digit guard; CLI sets it")
    output = Path(output)
    need(not output.exists(), "audit output exists; never overwrite prior evidence")
    output.mkdir(parents=True)
    summaries, indices, errors = [], [], []
    stopped = False
    metadata_verified = False
    try:
        package = verify_package(prepared, run, expected_gate_sha256=expected_gate_sha256,
                                 expected_completion_sha256=expected_completion_sha256)
        metadata_verified = True
        checker_pins = auditor_sources()
        for expected, item in zip(package["schedule"], package["completion"]["streams"], strict=True):
            need(all(item.get(k) == v for k, v in expected.items()), "completion stream schedule differs")
            if item["status"] == "unstarted":
                need("file" not in item and isinstance(item.get("reason"), str), "unstarted stream carries a result or lacks reason")
                stopped = True
                summary = {"audit_status": "PRESERVED_UNSTARTED", "complete": False, "source_status": "unstarted",
                    "denominator": 240, "completed_positions": 0, "partial_positions": 0, "unstarted_positions": 240,
                    "counts": {"represented": {}, "ideal": {}}, "clock_ns": {"full_wall_ns": None},
                    "ledger": [{"request_index": i, "status": "unstarted"} for i in range(240)]}
            else:
                need(not stopped, "stream started after schedule had stopped")
                path = Path(run) / item["file"]
                need(path.resolve().is_relative_to(Path(run).resolve()) and not path.is_symlink(), "stream path escapes run")
                saved = read_pinned(path, item)
                need(saved["schedule"] == expected, "saved stream schedule differs")
                data = package["inputs"][expected["input_file"]]
                raw_pin = identity(Path(prepared) / "inputs" / expected["input_file"])
                need(saved["input_raw_identity"] == raw_pin and saved["input_canonical_sha256"] == digest(data), "stream raw/canonical input pins differ")
                result = saved["result"]
                need(item.get("status") == result["status"] and item.get("complete") is result["complete"],
                     "completion stream status differs from saved service")
                need(item.get("runtime_admission") == {k: v for k, v in saved["runtime_admission"].items() if k != "observed_files"},
                     "completion runtime-admission summary differs")
                if result.get("ready"):
                    need(result["ready"]["sources"] == {k: v["sha256"] for k, v in package["runtime"]["sources"].items()},
                         "actual native loaded source declarations differ from frozen runtime")
                if result["status"] == "outer_service_exception":
                    need(not result["complete"] and saved.get("outer_service_error") and
                         result["workload_census"].get("unrecovered_positions") == 240, "outer exception falsely describes unstarted/successful positions")
                    summary = {"audit_status": "UNRECOVERED_EVIDENCE", "complete": False, "source_status": result["status"],
                        "denominator": 240, "completed_positions": None, "partial_positions": None, "unstarted_positions": None,
                        "unrecovered_positions": 240, "counts": {"represented": {}, "ideal": {}}, "clock_ns": {"full_wall_ns": None},
                        "ledger": [{"request_index": i, "status": "unrecovered"} for i in range(240)]}
                    stopped = True
                else:
                    try:
                        summary = audit_stream(data, result, runtime_config=package["runtime"],
                                               allowed_runtime_paths=package["allowed_runtime_paths"])
                        closure = summary["runtime_closure"]
                        need(all(saved["runtime_admission"].get(k) == v for k, v in closure.items()), "saved runtime admission differs from independent traversal")
                        need(item.get("counts") == summary["counts"] and item.get("workload_census") == result["workload_census"], "completion counts/denominators differ")
                    except Exception as error:
                        error_text = type(error).__name__ + ": " + str(error)
                        errors.append({"stream_index": expected["stream_index"], "error": error_text})
                        summary = {"audit_status": "AUDIT_FAILED", "complete": False,
                            "source_status": result["status"], "source_complete_claim": result["complete"],
                            "error": error_text, "denominator": 240, "completed_positions": None,
                            "partial_positions": None, "unstarted_positions": None,
                            "counts": {"represented": {}, "ideal": {}}, "reported_unverified_counts": result.get("counts"),
                            "clock_ns": {"full_wall_ns": None}, "reported_unverified_clock_ns": result.get("time_ns"),
                            "ledger": [{"request_index": i, "status": "unverified_due_to_stream_audit_failure"} for i in range(240)]}
                    if result["status"] == "correctness_mismatch":
                        stopped = True
            summary["schedule"] = expected
            ledger_path = output / ("stream_" + str(expected["stream_index"]).zfill(2) + "_ledger.jsonl")
            write_ledger(ledger_path, summary["ledger"])
            indices.append({"file": ledger_path.name, **identity(ledger_path)})
            summaries.append(compact_audit(summary))
        comparisons = []
        for round_ in (1, 2, 3):
            for view in VIEW_IDS:
                table = {s["schedule"]["arm"]: s for s in summaries if s["schedule"]["round"] == round_ and s["schedule"]["view_id"] == view}
                for arm in ("O+S", "O+G", "H"):
                    comparison = compare_streams(table[arm], table["O"], label=arm + "/O", primary=arm == "O+S" and view == VIEW_IDS[0])
                    comparisons.append({"round": round_, "view_id": view, **comparison})
        need(package["completion"]["completed_streams"] == sum(bool(item.get("complete")) for item in package["completion"]["streams"]), "completed-stream count differs")
        for path, pin in package["verified_files"].items():
            need(identity(path) == pin, "admitted file changed during audit: " + path)
        need(auditor_sources() == checker_pins, "independent verifier source changed during audit")
        status = "AUDIT_FAILED" if errors else "AUDIT_HAS_UNRECOVERED_GAPS" if any(s["audit_status"] == "UNRECOVERED_EVIDENCE" for s in summaries) else "PASS_SAVED_EVIDENCE_AUDIT"
        report = {"schema_version": "coat-workflow-independent-audit/1", "mode": "production", "status": status,
            "all_streams_complete": all(s["complete"] for s in summaries), "registered_streams": 32,
            "registered_positions": 7680, "quality_in_timing_comparisons": False,
            "gate_sha256": expected_gate_sha256, "completion_sha256": expected_completion_sha256,
            "streams": summaries, "comparisons": comparisons, "ledger_files": indices, "errors": errors,
            "auditor_sources": checker_pins, "optimizer_calls": 0,
            "audit_elapsed_ns": time.perf_counter_ns() - started}
        write_new(output / "AUDIT.json", report)
        return report
    except BaseException as error:
        write_new(output / "FAILURE.json", {"status": "AUDIT_FAILED", "error": type(error).__name__ + ": " + str(error),
                 "caller_inputs_unverified_until_metadata_gate": {"prepared": str(prepared), "run": str(run),
                     "expected_gate_sha256": expected_gate_sha256, "expected_completion_sha256": expected_completion_sha256},
                 "completed_audit_streams": len(summaries), "ledger_files": indices, "optimizer_calls": 0,
                 "metadata_verified": metadata_verified, "expected_registered_streams": 32, "expected_registered_positions": 7680,
                 "registered_streams": 32 if metadata_verified else None,
                 "registered_positions": 7680 if metadata_verified else None,
                 "unverified_stream_indices": list(range(len(summaries), 32)),
                 "unverified_positions": (32 - len(summaries)) * 240})
        raise


def audit_synthetic_bundle(saved, output, *, expected_receipt_sha256):
    """Explicit separate mode; cannot satisfy a production audit invocation."""
    saved, output = Path(saved), Path(output)
    need(not output.exists(), "audit output exists")
    receipt = read_pinned(saved / "RECEIPT.json", {"sha256": expected_receipt_sha256})
    checker_pins = auditor_sources()
    need(receipt.get("status") == "PASS_SYNTHETIC_ONLY" and receipt.get("real_streams_executed") == 0,
         "synthetic source lacks explicit synthetic-only declaration")
    data = read_pinned(saved / "TOY_INPUT.json", receipt["input"])
    identities = read_pinned(saved / "OBSERVED_RUNTIME_IDENTITIES.json", receipt["runtime_identities"])
    runtime = read_pinned(saved / "RUNTIME_CONFIG.json", receipt["config"])
    allowed = [r["path"] for r in identities["files"]]
    need(receipt["arms"] == list(ARMS) and len(receipt["stream_files"]) == 4, "synthetic four-arm census differs")
    output.mkdir(parents=True)
    summaries = []
    for arm, member in zip(ARMS, receipt["stream_files"], strict=True):
        path = saved / Path(member["path"]).name
        result = read_pinned(path, member)
        need(result["arm"] == arm, "synthetic arm order differs")
        audited = audit_stream(data, result, runtime_config=runtime, synthetic=True, allowed_runtime_paths=allowed)
        write_ledger(output / (arm.replace("+", "_") + "_ledger.jsonl"), audited["ledger"])
        summaries.append(compact_audit(audited))
    report = {"schema_version": "coat-workflow-independent-audit/1", "mode": "synthetic",
              "status": "PASS_SAVED_SYNTHETIC_EVIDENCE", "production_admission": False, "optimizer_calls": 0,
              "source_receipt_sha256": expected_receipt_sha256, "streams": summaries, "auditor_sources": checker_pins,
              "scope": "saved toy evidence and recorded runtime closure; historical live-source paths are not re-admitted as a production snapshot"}
    need(auditor_sources() == checker_pins, "independent verifier source changed during audit")
    write_new(output / "AUDIT.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prod = sub.add_parser("production")
    for name in ("prepared", "run", "output"):
        prod.add_argument("--" + name, type=Path, required=True)
    prod.add_argument("--gate-sha256", required=True)
    prod.add_argument("--completion-sha256", required=True)
    toy = sub.add_parser("synthetic")
    toy.add_argument("--saved", type=Path, required=True)
    toy.add_argument("--output", type=Path, required=True)
    toy.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    sys.set_int_max_str_digits(100000)
    if args.mode == "production":
        report = audit_run(args.prepared, args.run, args.output, expected_gate_sha256=args.gate_sha256,
                           expected_completion_sha256=args.completion_sha256)
    else:
        report = audit_synthetic_bundle(args.saved, args.output, expected_receipt_sha256=args.receipt_sha256)
    print(json.dumps({"status": report["status"], "mode": report["mode"], "optimizer_calls": 0}))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
