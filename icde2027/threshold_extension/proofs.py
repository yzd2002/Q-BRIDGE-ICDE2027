"""Standard-library-only proofs of a represented LP threshold decision.

A checked feasible point bounds the *minimum over the represented view* from
above. It does not upper-bound the true policy effect. Unknown is distinct from
a checked negative decision. No optimizer status is accepted as a proof.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import sys
import weakref

from ..proof_check import CheckedView, _numbers


FORMAT = "qbridge-threshold-proof-v1"
LOWER_FORMATS = {"qbridge-dual-proof-v1", "qbridge-scaled-dual-proof-v1"}


def _floats(values):
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("finite numeric coefficient required")
        try:
            value = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError("finite represented coefficient required") from error
        if not math.isfinite(value):
            raise ValueError("finite represented coefficient required")
        result.append(value)
    return tuple(result)


def upper_float(exact):
    """Smallest available float not below exact; keep exact equalities intact."""
    value = float(exact)
    if not math.isfinite(value):
        raise ValueError("finite upper bound required")
    if Fraction.from_float(value) < exact:
        value = math.nextafter(value, math.inf)
    if not math.isfinite(value):
        raise ValueError("finite upper bound required")
    return value


@dataclass(frozen=True)
class CheckedPrimal:
    view_digest: str
    raw: tuple
    point: tuple


def bundle_proof(view_digest, objective, threshold, lower_proof, upper_proof=None):
    """Serialize components produced by checked factories; not a verifier.

External consumers must call ThresholdChecker.verify. Keeping serialization
separate avoids repeating exact arithmetic already performed by those factories.
    """
    c = _floats(objective)
    delta = _floats([threshold])[0]
    if (lower_proof.get("format") not in LOWER_FORMATS
            or lower_proof.get("view_digest") != view_digest
            or _floats(lower_proof["objective"]) != c
            or _floats([lower_proof["decision_threshold"]])[0] != delta):
        raise ValueError("lower proof does not describe this threshold query")
    lower = _floats([lower_proof["claimed_lower"]])[0]
    upper = None if upper_proof is None else _floats([upper_proof["claimed_upper"]])[0]
    if upper is not None and lower > upper:
        raise ValueError("contradictory checked bounds")
    positive = lower > delta
    negative = upper is not None and upper <= delta
    if positive and negative:
        raise ValueError("contradictory threshold evidence")
    decision = True if positive else False if negative else None
    return {"format": FORMAT, "view_digest": view_digest,
            "objective": list(c), "decision_threshold": delta,
            "lower_proof": lower_proof, "upper_proof": upper_proof,
            "robust_threshold_decision": decision}


class ThresholdChecker:
    """Trusted-view checker with bounded FIFO point and prepared-dual caches.

The independent receiver uses the same cache rules for every execution plan.
cache_size=0 is a stateless-per-proof arithmetic path after view initialization.
Tokens are issued locally; arbitrary producer tokens are never accepted.
    """
    def __init__(self, view, cache_size=16):
        if type(cache_size) is not int or not 0 <= cache_size <= 16:
            raise ValueError("cache_size must be an integer between 0 and 16")
        if not isinstance(view, dict):
            raise ValueError("a caller-trusted view object is required")
        trusted = json.loads(json.dumps(view, allow_nan=False))
        self.view_checker = CheckedView(trusted)
        self.digest = self.view_checker.digest
        self.d, self.j = self.view_checker.d, self.view_checker.j
        self.cache_size = cache_size
        self._point_cache = OrderedDict()
        self._dual_cache = OrderedDict()
        self._issued_points = {}
        self._issued_duals = {}
        self.stats = {"primal_cache_hits": 0, "primal_exact_checks": 0,
                      "dual_cache_hits": 0, "dual_preparations": 0,
                      "prepared_state_reuses": 0,
                      "invalid_cache_entries": 0}

    def _register(self, table_name, state, record):
        table = getattr(self, table_name)
        identity = id(state)
        owner = weakref.ref(self)
        def cleanup(reference):
            checker = owner()
            if checker is not None:
                records = getattr(checker, table_name)
                saved = records.get(identity)
                if saved is not None and saved[0] is reference:
                    del records[identity]
        table[identity] = (weakref.ref(state, cleanup), record)

    def _remember(self, cache, key, state):
        if self.cache_size:
            cache[key] = state
            while len(cache) > self.cache_size:
                cache.popitem(last=False)

    def owns_primal(self, state, point=None):
        if not isinstance(state, CheckedPrimal):
            return False
        saved = self._issued_points.get(id(state))
        if saved is None or saved[0]() is not state:
            return False
        raw, exact = saved[1]
        return (state.view_digest == self.digest and state.raw is raw and state.point is exact
                and (point is None or _floats(point) == raw))

    def primal_state(self, point):
        raw = _floats(point)
        if len(raw) != self.d:
            raise ValueError("primal dimension differs from trusted view")
        state = self._point_cache.get(raw)
        if state is not None:
            if self.owns_primal(state, raw):
                self.stats["primal_cache_hits"] += 1
                return state
            self.stats["invalid_cache_entries"] += 1
            del self._point_cache[raw]
        exact = _numbers(raw)
        self.stats["primal_exact_checks"] += 1
        if not self.view_checker.is_feasible(exact, already_rational=True):
            raise ValueError("primal point does not exactly satisfy the whole trusted view")
        state = CheckedPrimal(self.digest, raw, exact)
        self._register("_issued_points", state, (state.raw, state.point))
        self._remember(self._point_cache, raw, state)
        return state

    def owns_dual(self, state, lower_multipliers, upper_multipliers):
        saved = self._issued_duals.get(id(state))
        if saved is None or saved[0]() is not state:
            return False
        constant, offset, base_lo, base_hi = saved[1]
        return (self.view_checker.owns_prepared(state, lower_multipliers, upper_multipliers)
                and state.constant is constant and state.offset is offset
                and state.base_lower is base_lo and state.base_upper is base_hi)

    def dual_state(self, lower_multipliers, upper_multipliers):
        lo, hi = _floats(lower_multipliers), _floats(upper_multipliers)
        if len(lo) != self.j or len(hi) != self.j or any(v < 0 for v in lo + hi):
            raise ValueError("matching nonnegative dual multipliers required")
        key = (lo, hi)
        state = self._dual_cache.get(key)
        if state is not None:
            if self.owns_dual(state, lo, hi):
                self.stats["dual_cache_hits"] += 1
                return state
            self.stats["invalid_cache_entries"] += 1
            del self._dual_cache[key]
        state = self.view_checker.prepare_dual(lo, hi)
        self.stats["dual_preparations"] += 1
        self._register("_issued_duals", state,
                       (state.constant, state.offset, state.base_lower, state.base_upper))
        self._remember(self._dual_cache, key, state)
        return state

    def lower_proof(self, objective, lower_multipliers, upper_multipliers, *, threshold, scale=None,
                    prepared=None):
        """Check a lower bound, optionally reusing a locally issued exact state.

        This optional producer path does not enlarge either FIFO cache. A
        foreign or modified state is never trusted: the complete multipliers
        are checked and prepared again. Receivers still reconstruct from the
        complete serialized proof, without any producer-owned token.
        """
        c = _floats(objective)
        delta = _floats([threshold])[0]
        if len(c) != self.d:
            raise ValueError("objective dimension differs from trusted view")
        lo, hi = _floats(lower_multipliers), _floats(upper_multipliers)
        if len(lo) != self.j or len(hi) != self.j or any(v < 0 for v in lo + hi):
            raise ValueError("matching nonnegative dual multipliers required")
        if prepared is not None and self.owns_dual(prepared, lo, hi):
            state = prepared
            self.stats["prepared_state_reuses"] += 1
        else:
            if prepared is not None:
                self.stats["invalid_cache_entries"] += 1
            state = self.dual_state(lo, hi)
        factor = 1. if scale is None else _floats([scale])[0]
        if factor < 0:
            raise ValueError("nonnegative dual scale required")
        exact = state.exact_bound(c, factor)
        value = math.nextafter(float(exact), -math.inf)
        if not math.isfinite(value):
            raise ValueError("finite lower bound required")
        result = {"format": "qbridge-dual-proof-v1" if scale is None else "qbridge-scaled-dual-proof-v1",
                  "view_digest": self.digest, "objective": list(c),
                  "lower_multipliers": list(state.base_lower), "upper_multipliers": list(state.base_upper),
                  "claimed_lower": value, "decision_threshold": delta}
        if scale is not None:
            result["dual_scale"] = factor
        return result

    def upper_proof(self, objective, state):
        c = _numbers(objective)
        if len(c) != self.d or not self.owns_primal(state):
            raise ValueError("matching objective and locally issued primal state required")
        exact = sum((ci * mi for ci, mi in zip(c, state.point)), Fraction(0))
        return {"point": list(state.raw), "claimed_upper": upper_float(exact)}

    def _verify_lower(self, proof, c, delta):
        if (not isinstance(proof, dict) or proof.get("format") not in LOWER_FORMATS or proof.get("view_digest") != self.digest
                or _numbers(proof["objective"]) != c
                or _numbers([proof["decision_threshold"]])[0] != delta):
            raise ValueError("lower proof is not bound to the requested view/query/threshold")
        claim = _numbers([proof["claimed_lower"]])[0]
        state = self.dual_state(proof["lower_multipliers"], proof["upper_multipliers"])
        scale = proof["dual_scale"] if proof["format"] == "qbridge-scaled-dual-proof-v1" else 1.
        exact = state.exact_bound(proof["objective"], scale)
        if claim > exact:
            raise ValueError("claimed lower bound exceeds the exact weak dual")
        return claim

    def verify(self, proof, *, expected_objective, expected_threshold):
        if not isinstance(proof, dict) or proof.get("format") != FORMAT or proof.get("view_digest") != self.digest:
            raise ValueError("threshold proof is not bound to the trusted view")
        c, requested_c = _numbers(proof["objective"]), _numbers(expected_objective)
        delta, requested_delta = _numbers([proof["decision_threshold"]])[0], _numbers([expected_threshold])[0]
        if len(c) != self.d or c != requested_c or delta != requested_delta:
            raise ValueError("threshold proof differs from the caller's requested query")
        lower = self._verify_lower(proof["lower_proof"], c, delta)
        upper = None
        upper_record = proof.get("upper_proof")
        if upper_record is not None:
            if not isinstance(upper_record, dict):
                raise ValueError("upper proof must contain a point and its bound")
            state = self.primal_state(upper_record["point"])
            exact = sum((ci * mi for ci, mi in zip(c, state.point)), Fraction(0))
            upper = _numbers([upper_record["claimed_upper"]])[0]
            if upper != Fraction.from_float(upper_float(exact)):
                raise ValueError("upper bound is not the outward-rounded exact feasible objective")
            if lower > exact:
                raise ValueError("lower and feasible upper evidence contradict each other")
        decision = True if lower > delta else False if upper is not None and upper <= delta else None
        declared = proof.get("robust_threshold_decision", "missing")
        if declared is not None and type(declared) is not bool:
            raise ValueError("threshold decision must be true, false, or null")
        if declared is not decision:
            raise ValueError("declared threshold decision does not follow from its evidence")
        return {"valid": True, "valid_lp_lower_bound": True,
                "valid_lp_upper_bound": upper is not None,
                "view_digest": self.digest, "lower_bound": float(lower),
                "upper_bound": None if upper is None else float(upper),
                "robust_threshold_decision": decision,
                "scientific_assumptions_verified": False,
                "negative_scope": "minimum over this represented confidence view only",
                "requested_query_checked": True}

    def cache_info(self):
        return {**self.stats, "primal_cache_entries": len(self._point_cache),
                "dual_cache_entries": len(self._dual_cache), "cache_size": self.cache_size}

    def cache_payload_bytes(self, retained_states=()):
        """Approximate Python object payload, excluding shared view and solver."""
        seen = set()
        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            total = sys.getsizeof(value)
            if isinstance(value, Fraction):
                total += size(value.numerator) + size(value.denominator)
            elif isinstance(value, (tuple, list)):
                total += sum(size(v) for v in value)
            elif isinstance(value, dict):
                total += sum(size(k) + size(v) for k, v in value.items())
            elif isinstance(value, CheckedPrimal):
                total += size(value.raw) + size(value.point)
            elif hasattr(value, "offset") and hasattr(value, "constant"):
                total += sum(size(getattr(value, k)) for k in ("offset", "constant", "base_lower", "base_upper"))
            return total
        return size(self._point_cache) + size(self._dual_cache) + size(retained_states)


def verify_certificate(view, proof, *, expected_objective, expected_threshold, cache_size=0):
    return ThresholdChecker(view, cache_size).verify(
        proof, expected_objective=expected_objective, expected_threshold=expected_threshold)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    args = parser.parse_args()
    try:
        query = json.loads(args.query.read_text())
        result = verify_certificate(json.loads(args.view.read_text()), json.loads(args.proof.read_text()),
                                    expected_objective=query["objective"], expected_threshold=query["threshold"])
    except (KeyError, TypeError, ValueError, ArithmeticError, OSError) as error:
        parser.exit(2, f"Proof did not verify: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
