"""Standard-library exact partial-bound receiver for fixed-policy requests.

Only the externally supplied original program and separately held digests are
authoritative.  A feasible original-program point proves an upper bound; a
nonnegative original-row dual, with support over the original box, proves a
lower bound.  Every bound is recomputed as a Fraction.  The represented and
ideal objectives remain separate, including at a zero threshold.

The wire accepts canonical reduced n/d text only.  It carries witnesses, never
a producer's bound, Boolean, family capacity, or optimum claim.  A returned
``false`` means the robust minimum is not strictly above delta; it says nothing
about the unknown real effect.  ``optimum`` is populated only if a checked
point and dual attain the same value for that particular objective.

Private parsing helpers from the already reviewed family_lp_v7.verify_general
are deliberately reused: _Budget, _program, _vector, _number, _object, _need,
_digest, _render and _HASH_TEXT.  Mathematical partial-bound computation and
cache management are implemented here.  No producer or optimizer is imported.

Requests are opaque, receiver-issued immutable handles.  Caller inputs are
copied into exact tuples; exported receipts/points are detached JSON values.
No public API accepts a caller-created cache record or a claimed verified flag.
This is an evidence boundary, not isolation from code that deliberately changes
the receiver's private Python state.  Resource/time limits belong to the host
service; this module does not change process-global integer-string settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
from typing import NamedTuple


# Import only this reviewed standard-library file.  Importing the v7 package
# would execute its __init__, which imports its ExactLP producer adapter.
_parser_path = Path(__file__).resolve().parent.parent / "family_lp_v7" / "verify_general.py"
_parser_spec = importlib.util.spec_from_file_location("_workflow_v8_strict_parsing", _parser_path)
if _parser_spec is None or _parser_spec.loader is None:
    raise ImportError("cannot load the pinned exact parsing dependency")
_v7 = importlib.util.module_from_spec(_parser_spec)
_parser_spec.loader.exec_module(_v7)


PROGRAM_SCHEMA = "workflow-original-program/1"
REQUEST_SCHEMA = "workflow-fixed-request/1"
OBJECTIVE_SCHEMA = "workflow-fixed-objective/1"
POINT_SCHEMA = "workflow-rational-point/1"
PARTIAL_SCHEMA = "workflow-rational-partial/1"
RECEIPT_SCHEMA = "workflow-rational-verification/1"

VerificationError = _v7.VerificationError
RequestValidationError = _v7.RequestValidationError
CertificateValidationError = _v7.CertificateValidationError
WireFormatError = _v7.WireFormatError
strict_json_loads = _v7.strict_json_loads

_ZERO = Fraction(0)
_POINT_KEYS = {"schema_version", "view_sha256", "program_sha256", "point"}
_PARTIAL_KEYS = {
    "schema_version", "view_sha256", "program_sha256", "request_sha256",
    "point", "original_row_lower_dual", "original_row_upper_dual",
}


def encode_wire(value):
    """Encode a JSON value for the strict receiver boundary (not verification)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _validated_program(program):
    lo, hi, rows, canonical = _v7._program(
        program, _v7._Budget(RequestValidationError))
    digest = _v7._digest({"schema_version": PROGRAM_SCHEMA, "program": canonical})
    return lo, hi, rows, digest


def program_input_digest(program):
    """Validate and digest the exact original program, independent of any proof."""
    return _validated_program(program)[-1]


def _hash(value, where, error):
    _v7._need(type(value) is str and _v7._HASH_TEXT.fullmatch(value) is not None,
              where + ": lowercase SHA256 required", error)
    return value


def _index(value, where):
    _v7._need(type(value) is int and 0 <= value <= 2**63 - 1,
              where + ": nonnegative signed-64-bit integer required",
              RequestValidationError)
    return value


def _texts(values):
    return [_v7._render(value) for value in values]


class BoundRequest(NamedTuple):
    """A receiver-issued handle; reconstructing an equal tuple is not authority."""
    view_sha256: str
    program_sha256: str
    request_sha256: str
    request_index: int
    policy_index: int


@dataclass(frozen=True)
class _RequestData:
    c_rep: tuple
    c_ideal: tuple
    delta: Fraction


@dataclass(frozen=True)
class _Point:
    values: tuple
    digest: str


@dataclass(frozen=True)
class _Dual:
    lower: tuple
    upper: tuple
    origin_request_sha256: str | None


@dataclass
class _ObjectiveState:
    objective: tuple
    digest: str
    lower: Fraction
    dual: _Dual
    upper: Fraction | None = None
    point: _Point | None = None
    scanned_points: int = 0


class RationalReceiver:
    """One original view and one stream's verified points/objective-bound cache.

    ``view_sha256`` must come from the trusted view registry; the program digest
    must be independently held, not adopted from a submitted certificate.
    Instantiate a fresh receiver per stream.  Points are reusable across that
    stream's objectives; nonzero duals are cached only for objectives explicitly
    named in the bound request whose certificate supplied the dual.
    """

    def __init__(self, program, *, expected_program_sha256, view_sha256):
        lo, hi, rows, digest = _validated_program(program)
        _hash(expected_program_sha256, "expected_program_sha256", RequestValidationError)
        _v7._need(expected_program_sha256 == digest,
                  "expected_program_sha256 differs from trusted program",
                  RequestValidationError)
        self._view_sha256 = _hash(view_sha256, "view_sha256", RequestValidationError)
        self._program_sha256 = digest
        self._lo, self._hi, self._rows = lo, hi, rows
        self._points = []
        self._point_by_values = {}
        self._objectives = {}
        self._requests = {}
        self._request_by_index = {}

    @property
    def view_sha256(self):
        return self._view_sha256

    @property
    def program_sha256(self):
        return self._program_sha256

    @property
    def point_count(self):
        return len(self._points)

    @property
    def objective_count(self):
        """Number of exact objective vectors cached, across both columns."""
        return len(self._objectives)

    def bind_request(self, c_rep, c_ideal, delta, *, request_index, policy_index):
        """Bind exact external query data; never derive these from a proof.

        The same request index is idempotent only for identical complete input.
        Neither objectives nor the original binary64 threshold are rounded or
        inferred.  Callers converting stored binary64 values must use their
        exact integer ratio before calling this method.
        """
        budget = _v7._Budget(RequestValidationError)
        rep = _v7._vector(c_rep, len(self._lo), "c_rep", budget)
        ideal = _v7._vector(c_ideal, len(self._lo), "c_ideal", budget)
        threshold = _v7._number(delta, "delta", budget)
        rid = _index(request_index, "request_index")
        pid = _index(policy_index, "policy_index")
        canonical = {
            "schema_version": REQUEST_SCHEMA, "view_sha256": self._view_sha256,
            "program_sha256": self._program_sha256, "c_rep": _texts(rep),
            "c_ideal": _texts(ideal), "delta": _v7._render(threshold),
            "request_index": rid, "policy_index": pid,
        }
        digest = _v7._digest(canonical)
        previous = self._request_by_index.get(rid)
        if previous is not None:
            _v7._need(previous.request_sha256 == digest,
                      "request_index was already bound to different input",
                      RequestValidationError)
            return previous
        handle = BoundRequest(self._view_sha256, self._program_sha256, digest, rid, pid)
        self._requests[id(handle)] = (handle, _RequestData(rep, ideal, threshold))
        self._request_by_index[rid] = handle
        return handle

    def _request(self, request):
        record = self._requests.get(id(request))
        _v7._need(type(request) is BoundRequest and record is not None and
                  record[0] is request, "request is not issued by this receiver",
                  RequestValidationError)
        return record[1]

    def _wire(self, wire, keys, schema):
        certificate = strict_json_loads(wire)
        _v7._object(certificate, keys, "certificate", CertificateValidationError)
        _v7._need(type(certificate["schema_version"]) is str and
                  certificate["schema_version"] == schema,
                  "certificate: wrong schema version", CertificateValidationError)
        _v7._need(certificate["view_sha256"] == self._view_sha256 and
                  certificate["program_sha256"] == self._program_sha256,
                  "certificate: wrong trusted view/program binding",
                  CertificateValidationError)
        return certificate

    def _point(self, value, budget):
        point = _v7._vector(value, len(self._lo), "certificate.point", budget, wire=True)
        _v7._need(all(lo <= v <= hi for lo, v, hi in zip(self._lo, point, self._hi)),
                  "certificate point violates an original box", CertificateValidationError)
        for j, (coefficients, low, high) in enumerate(self._rows):
            value = sum((a * x for a, x in zip(coefficients, point)), _ZERO)
            _v7._need(low <= value <= high,
                      "certificate point violates original row " + str(j),
                      CertificateValidationError)
        return point

    def _point_wire(self, values):
        return {"schema_version": POINT_SCHEMA, "view_sha256": self._view_sha256,
                "program_sha256": self._program_sha256, "point": _texts(values)}

    def _store_point(self, values):
        previous = self._point_by_values.get(values)
        if previous is not None:
            return previous, False
        point = _Point(values, _v7._digest(self._point_wire(values)))
        self._points.append(point)
        self._point_by_values[values] = point
        return point, True

    def accept_point(self, wire):
        """Independently validate a rational point and add it to this stream."""
        certificate = self._wire(wire, _POINT_KEYS, POINT_SCHEMA)
        values = self._point(certificate["point"], _v7._Budget(CertificateValidationError))
        point, added = self._store_point(values)
        return {"verified": True, "view_sha256": self._view_sha256,
                "program_sha256": self._program_sha256, "point_sha256": point.digest,
                "added": added, "point_count": self.point_count}

    def export_points(self):
        """Return fresh point certificates; reimporting always rechecks feasibility."""
        return [self._point_wire(point.values) for point in self._points]

    def _dual(self, certificate, budget, request_sha256):
        eta = _v7._vector(certificate["original_row_lower_dual"], len(self._rows),
                          "certificate.original_row_lower_dual", budget, wire=True)
        lam = _v7._vector(certificate["original_row_upper_dual"], len(self._rows),
                          "certificate.original_row_upper_dual", budget, wire=True)
        _v7._need(all(v >= 0 for v in eta + lam),
                  "certificate has a negative original-row dual", CertificateValidationError)
        return _Dual(eta, lam, request_sha256)

    def _weak_dual(self, objective, dual):
        residual = list(objective)
        result = _ZERO
        for (coefficients, low, high), eta, lam in zip(
                self._rows, dual.lower, dual.upper):
            result += low * eta - high * lam
            for i, coefficient in enumerate(coefficients):
                residual[i] += coefficient * (lam - eta)
        return result + sum((min(r * lo, r * hi) for r, lo, hi in
                             zip(residual, self._lo, self._hi)), _ZERO)

    def _objective(self, objective):
        state = self._objectives.get(objective)
        if state is None:
            zeros = (_ZERO,) * len(self._rows)
            dual = _Dual(zeros, zeros, None)
            digest = _v7._digest({"schema_version": OBJECTIVE_SCHEMA,
                                  "view_sha256": self._view_sha256,
                                  "program_sha256": self._program_sha256,
                                  "objective": _texts(objective)})
            state = _ObjectiveState(objective, digest, self._weak_dual(objective, dual), dual)
            self._objectives[objective] = state
        for point in self._points[state.scanned_points:]:
            value = sum((c * x for c, x in zip(objective, point.values)), _ZERO)
            if state.upper is None or value < state.upper:
                state.upper, state.point = value, point
        state.scanned_points = len(self._points)
        return state

    def _column(self, state, delta):
        # These are new dicts/lists; no caller-visible object is a cache entry.
        dual = {"view_sha256": self._view_sha256,
                "program_sha256": self._program_sha256,
                "objective_sha256": state.digest,
                "original_row_lower_dual": _texts(state.dual.lower),
                "original_row_upper_dual": _texts(state.dual.upper)}
        lower_evidence = {
            "kind": "original-row-dual", "evidence_sha256": _v7._digest(dual),
            "origin_request_sha256": state.dual.origin_request_sha256, **dual,
        }
        upper_evidence = None if state.point is None else {
            "kind": "original-feasible-point", "point_sha256": state.point.digest,
            **self._point_wire(state.point.values),
        }
        optimum = None
        if state.upper is not None:
            _v7._need(state.lower <= state.upper,
                      "internal exact bounds are inconsistent", VerificationError)
            if state.lower == state.upper:
                optimum = _v7._render(state.upper)
        decision = "unknown"
        if state.lower > delta:
            decision = "true"
        elif state.upper is not None and state.upper <= delta:
            decision = "false"
        return {
            "objective_sha256": state.digest, "objective": _texts(state.objective),
            "lower_bound": _v7._render(state.lower),
            "upper_bound": None if state.upper is None else _v7._render(state.upper),
            "decision": decision, "optimum": optimum,
            "lower_evidence": lower_evidence, "upper_evidence": upper_evidence,
        }

    def evaluate(self, request, wire=None):
        """Recheck current bounds at the request's original exact threshold.

        A certificate is bound to this full request, not only to its objective.
        Rejected certificates do not mutate any point or objective cache.  A
        successful certificate can supply a point, a dual, or both.  Each dual
        is evaluated separately for c_rep and c_ideal; equality in one column
        never implies optimality in the other.  With no certificate, retained
        valid evidence and the original box's zero-dual bound are used.
        """
        data = self._request(request)
        point = dual = None
        if wire is not None:
            certificate = self._wire(wire, _PARTIAL_KEYS, PARTIAL_SCHEMA)
            _v7._need(certificate["request_sha256"] == request.request_sha256,
                      "certificate: wrong objective/threshold/request binding",
                      CertificateValidationError)
            has_eta = certificate["original_row_lower_dual"] is not None
            has_lam = certificate["original_row_upper_dual"] is not None
            has_point = certificate["point"] is not None
            _v7._need(has_eta == has_lam and (has_point or has_eta),
                      "certificate: provide a point and/or both dual vectors",
                      CertificateValidationError)
            budget = _v7._Budget(CertificateValidationError)
            if has_point:
                point = self._point(certificate["point"], budget)
            if has_eta:
                dual = self._dual(certificate, budget, request.request_sha256)
            # Finish all witness checking and arithmetic before committing.
            lower_rep = None if dual is None else self._weak_dual(data.c_rep, dual)
            lower_ideal = None if dual is None else self._weak_dual(data.c_ideal, dual)
            if point is not None:
                self._store_point(point)
        rep = self._objective(data.c_rep)
        ideal = self._objective(data.c_ideal)
        if dual is not None:
            for state, lower in ((rep, lower_rep), (ideal, lower_ideal)):
                if lower > state.lower:
                    state.lower, state.dual = lower, dual
        return {
            "schema_version": RECEIPT_SCHEMA, "verified": True,
            "view_sha256": self._view_sha256, "program_sha256": self._program_sha256,
            "request_sha256": request.request_sha256,
            "request_index": request.request_index, "policy_index": request.policy_index,
            "delta": _v7._render(data.delta),
            "represented": self._column(rep, data.delta),
            "ideal": self._column(ideal, data.delta),
            "certificate_accepted": wire is not None,
            "point_count": self.point_count, "objective_count": self.objective_count,
            "scope_notice": "Exact bounds for the external original-program robust minimum; "
                            "false does not identify the real effect, and no family capacity "
                            "or statistical validity is certified.",
        }


__all__ = [
    "RationalReceiver", "BoundRequest", "program_input_digest", "encode_wire",
    "strict_json_loads", "VerificationError", "RequestValidationError",
    "CertificateValidationError", "WireFormatError", "PROGRAM_SCHEMA",
    "REQUEST_SCHEMA", "OBJECTIVE_SCHEMA", "POINT_SCHEMA", "PARTIAL_SCHEMA", "RECEIPT_SCHEMA",
]
