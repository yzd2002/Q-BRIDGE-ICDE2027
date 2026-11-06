"""Coat workflow adapters for unchanged native producers and exact services.

Only the new workflow owns caches and rational point transport.  Existing
producer implementations and their legacy evidence checks stay unchanged.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import re
import time

ARMS = ("H", "O", "O+S", "O+G")


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256(value):
    return hashlib.sha256(encode(value)).hexdigest()


def render(value):
    value = Fraction(value)
    return str(value.numerator) + "/" + str(value.denominator)


def binary64_exact(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite represented binary64 value required")
    return render(Fraction.from_float(float(value)))


def ideal_exact_vector(values):
    """Preserve saved integer numerator/denominator pairs without float math."""
    def integer(value):
        if type(value) is int:
            return value
        if type(value) is str and re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value):
            return int(value)
        raise ValueError("ideal rational pair contains a noncanonical integer")
    result = []
    for value in values:
        if type(value) is str:
            result.append(value)  # The receiver enforces canonical text.
        elif type(value) in (list, tuple) and len(value) == 2:
            numerator, denominator = map(integer, value)
            if denominator <= 0:
                raise ValueError("ideal rational pair requires a positive denominator")
            canonical = render(Fraction(numerator, denominator))
            if canonical != str(numerator) + "/" + str(denominator):
                raise ValueError("ideal rational pair is not reduced canonical form")
            result.append(canonical)
        else:
            raise ValueError("saved ideal coefficient must be canonical text or integer numerator/denominator pair")
    return result


def effective_legacy_multipliers(base, scale):
    """Preserve the old contract: exact binary64 base times exact scale.

    Converting float(base * scale) would silently change the valid proof.
    """
    factor = Fraction.from_float(float(scale))
    if factor < 0:
        raise ValueError("negative dual scale")
    return [render(Fraction.from_float(float(value)) * factor) for value in base]


def strict_loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def forbidden(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=forbidden)


def legacy_program(view):
    """Translate the represented CSR program, preserving duplicate-row sums."""
    lower = list(map(binary64_exact, view["lower"]))
    upper = list(map(binary64_exact, view["upper"]))
    n, matrix = len(lower), view["matrix"]
    rows = []
    for j, (lo, hi) in enumerate(zip(view["rhs_lower"], view["rhs_upper"], strict=True)):
        values = [Fraction(0)] * n
        for k in range(matrix["indptr"][j], matrix["indptr"][j + 1]):
            values[matrix["indices"][k]] += Fraction(binary64_exact(matrix["data"][k]))
        rows.append({"coefficients": list(map(render, values)),
                     "lower": binary64_exact(lo), "upper": binary64_exact(hi)})
    return {"lower": lower, "upper": upper, "rows": rows}


def legacy_view_from_program(program, seed_point_exact):
    """Only exact binary64-representable inputs can enter the old producer."""
    def f(value):
        exact = Fraction(value)
        represented = float(exact)
        if not math.isfinite(represented) or Fraction.from_float(represented) != exact:
            raise ValueError("legacy producer input is not exactly representable in binary64")
        return represented
    data, indices, indptr = [], [], [0]
    for row in program["rows"]:
        for i, value in enumerate(row["coefficients"]):
            if Fraction(value):
                indices.append(i)
                data.append(f(value))
        indptr.append(len(indices))
    return {"format": "qbridge-confidence-view-v1", "lower": list(map(f, program["lower"])),
            "upper": list(map(f, program["upper"])),
            "rhs_lower": [f(row["lower"]) for row in program["rows"]],
            "rhs_upper": [f(row["upper"]) for row in program["rows"]],
            "matrix": {"indptr": indptr, "indices": indices, "data": data},
            "feasible_point": list(map(f, seed_point_exact))}


def constant_intersection(family):
    """Deterministic constant point for disjoint rows contained in contexts.

    Return None for an empty intersection or an unsupported cross-context row.
    Zero-weight contexts are still required to admit this stronger common-point
    shortcut.  No family capacity or fixed-query decision is inferred here.
    """
    program, sizes = family["program"], family["context_sizes"]
    lower, upper = list(map(Fraction, program["lower"])), list(map(Fraction, program["upper"]))
    starts, offset = [], 0
    for size in sizes:
        starts.append((offset, offset + size))
        offset += size
    if offset != len(lower):
        raise ValueError("context sizes do not partition coordinates")
    intervals = [[max(lower[a:b]), min(upper[a:b])] for a, b in starts]
    for row in program["rows"]:
        coefficients = list(map(Fraction, row["coefficients"]))
        support = [i for i, value in enumerate(coefficients) if value]
        lo, hi = Fraction(row["lower"]), Fraction(row["upper"])
        if not support:
            if not lo <= 0 <= hi:
                return None
            continue
        contexts = [x for x, (a, b) in enumerate(starts) if all(a <= i < b for i in support)]
        if len(contexts) != 1:
            return None
        total = sum(coefficients, Fraction(0))
        if not total:
            if not lo <= 0 <= hi:
                return None
            continue
        bounds = sorted((lo / total, hi / total))
        interval = intervals[contexts[0]]
        interval[0], interval[1] = max(interval[0], bounds[0]), min(interval[1], bounds[1])
    if any(lo > hi for lo, hi in intervals):
        return None
    return [render(intervals[x][0]) for x, size in enumerate(sizes) for _ in range(size)]


def point_wire(receiver, point):
    from .receiver import POINT_SCHEMA
    return encode({"schema_version": POINT_SCHEMA, "view_sha256": receiver.view_sha256,
                   "program_sha256": receiver.program_sha256, "point": list(point)})


def partial_wire(receiver, request, *, point=None, lower=None, upper=None):
    from .receiver import PARTIAL_SCHEMA
    return encode({"schema_version": PARTIAL_SCHEMA, "view_sha256": receiver.view_sha256,
                   "program_sha256": receiver.program_sha256,
                   "request_sha256": request.request_sha256, "point": point,
                   "original_row_lower_dual": lower, "original_row_upper_dual": upper})


class LegacyEvidenceAdapter:
    """Verify old outer packet, proof and verdict before rational adaptation."""
    def __init__(self, legacy_view, program):
        from icde2027.threshold_extension.proofs import ThresholdChecker
        self.checker = ThresholdChecker(legacy_view, cache_size=0)
        if legacy_program(legacy_view) != program:
            raise ValueError("legacy native view and rational trusted program differ")

    def verify_and_adapt(self, answer, objective, threshold, receiver, request):
        proof = answer.get("certificate")
        if proof is None:
            if (answer.get("status") != "uncertain" or answer.get("robust_threshold_decision") is not None
                    or answer.get("lower_bound") is not None or answer.get("upper_bound") is not None
                    or not isinstance(answer.get("reason"), str) or not answer["reason"].strip()):
                raise ValueError("malformed native refusal")
            return None, {"accepted_legacy_proof": False, "reason": answer["reason"],
                          "old_decision": None, "old_lower_bound": None, "old_upper_bound": None}
        checked = self.checker.verify(proof, expected_objective=objective, expected_threshold=threshold)
        decision = checked["robust_threshold_decision"]
        if (checked.get("valid") is not True or answer.get("robust_threshold_decision") is not decision
                or answer.get("lower_bound") != checked["lower_bound"]
                or answer.get("upper_bound") != checked["upper_bound"]
                or answer.get("status") != ("certified" if decision is True else "unresolved")):
            raise ValueError("native outer packet disagrees with checked old proof")
        lower = proof["lower_proof"]
        factor = lower["dual_scale"] if lower["format"] == "qbridge-scaled-dual-proof-v1" else 1.0
        upper = proof.get("upper_proof")
        point = None if upper is None else list(map(binary64_exact, upper["point"]))
        wire = partial_wire(receiver, request, point=point,
                            lower=effective_legacy_multipliers(lower["lower_multipliers"], factor),
                            upper=effective_legacy_multipliers(lower["upper_multipliers"], factor))
        return wire, {"accepted_legacy_proof": True, "old_decision": decision,
                      "old_lower_bound": checked["lower_bound"], "old_upper_bound": checked["upper_bound"],
                      "legacy_receiver_receipt": checked,
                      "adaptation": "exact binary64 base times exact binary64 scale; no float multiplication"}


class NativeProducer:
    """One unchanged persistent H/O engine; no wrapper-level answer cache."""
    def __init__(self, arm, legacy_view):
        if arm not in ARMS:
            raise ValueError("unknown workflow arm")
        from icde2027.optimized_exact_v5 import OptimizedNativeFullHotEngine, OptimizedCheckingScheduleEngine
        constructor = OptimizedNativeFullHotEngine if arm == "H" else OptimizedCheckingScheduleEngine
        self.engine = constructor(legacy_view, simplex_strategy=1)
        self.observer = _ObservedHiGHS(self.engine.h)
        self.engine.h = self.observer
        self.last_diagnostics = None

    def query(self, objective, threshold):
        started = time.perf_counter_ns()
        before_calls, before_points = len(self.observer.calls), len(self.observer.points)
        try:
            answer = self.engine.query(objective, threshold)
        finally:
            calls = self.observer.calls[before_calls:]
            self.last_diagnostics = {"producer_ns": time.perf_counter_ns() - started,
                    "actual_native_calls": len(calls), "observed_native_calls": calls,
                    "proposed_points": self.observer.points[before_points:]}
        if len(answer.get("native_calls", [])) != self.last_diagnostics["actual_native_calls"]:
            raise RuntimeError("unchanged engine and actual HiGHS call accounting differ")
        return {"answer": answer, **self.last_diagnostics}


class _ObservedHiGHS:
    """Transparent call/point observation; no option, basis or result changes."""
    def __init__(self, native):
        self.native, self.calls, self.points = native, [], []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def run(self):
        started = time.perf_counter_ns()
        record = {"status": None, "error": None}
        self.calls.append(record)
        try:
            result = self.native.run()
            record["status"] = str(result)
            return result
        except BaseException as error:
            record["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            record["elapsed_ns"] = time.perf_counter_ns() - started

    def getSolution(self):
        result = self.native.getSolution()
        if result.value_valid:
            self.points.append(list(result.col_value))
        return result
