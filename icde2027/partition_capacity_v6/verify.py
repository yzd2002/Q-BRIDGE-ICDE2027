"""Independent exact receiver for partition-capacity-v6/1 certificates.

Only the Python standard library is used. This module neither imports nor calls
the producer, its parser, cap/knapsack algorithm, or self-check. The caller must
supply the separately trusted original request; embedded certificate input is
never an authority. Acceptance proves original-program optimality by matching
weak-duality lower and common-point upper bounds. ``contexts`` construction
metadata and ``internal_check`` are explicitly outside mathematical validation.

Importing this file directly works without the package. The package's existing
__init__.py imports its producer; this receiver does not depend on that import.
"""

from fractions import Fraction
import hashlib
import json
import re


CERTIFICATE_SCHEMA = "partition-capacity-v6/1"
INPUT_SCHEMA = "partition-capacity-input/1"
VERIFICATION_SCHEMA = "partition-capacity-verification/1"
UNCHECKED_FIELDS = ("contexts", "internal_check")
_CERTIFICATE_KEYS = frozenset({
    "schema_version", "input_sha256", "canonical_input", "point", "policy",
    "original_row_lower_dual", "original_row_upper_dual", "original_row_values",
    "objective_coefficients", "box_reduced_costs", "capacity",
    "common_point_upper_bound", "policy_at_point", "policy_dual_lower_bound",
    "contexts", "internal_check",
})


class VerificationError(ValueError):
    """The trusted request or supplied certificate cannot be verified."""


class RequestValidationError(VerificationError):
    """The separately supplied trusted request violates the exact input format."""


class CertificateValidationError(VerificationError):
    """The certificate is malformed, answers a different request, or is invalid."""


def _need(condition, message, error=CertificateValidationError):
    if not condition:
        raise error(message)


def _render(value):
    return str(value.numerator) + "/" + str(value.denominator)


def _request_number(value, where):
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, Fraction):
        return Fraction(value.numerator, value.denominator)
    if type(value) is str:
        pieces = value.split("/")
        numerator_ok = bool(re.fullmatch(r"[+-]?[0-9]+", pieces[0]))
        denominator_ok = len(pieces) == 1 or (
            len(pieces) == 2 and bool(re.fullmatch(r"[1-9][0-9]*", pieces[1]))
        )
        if numerator_ok and denominator_ok:
            try:
                return Fraction(int(pieces[0]), int(pieces[1]) if len(pieces) == 2 else 1)
            except (ValueError, OverflowError) as exc:
                raise RequestValidationError(where + ": rational integer cannot be parsed") from exc
    raise RequestValidationError(where + ": require int (not bool), Fraction, or exact integer/rational string")


def _wire_number(value, where):
    _need(type(value) is str, where + ": certificate rationals must be canonical n/d strings")
    try:
        number = _request_number(value, where)
    except RequestValidationError as exc:
        raise CertificateValidationError(str(exc)) from exc
    _need(_render(number) == value, where + ": noncanonical rational string")
    return number


def _object(value, keys, where, error):
    _need(type(value) is dict and set(value) == set(keys),
          where + ": incorrect object keys", error)


def _sequence(value, size, where, error, wire=False):
    legal = type(value) is list if wire else type(value) in (list, tuple)
    _need(legal, where + ": incorrect sequence type", error)
    if size is not None:
        _need(len(value) == size, where + ": incorrect sequence length", error)
    return value


def _request_vector(value, size, where):
    values = _sequence(value, size, where, RequestValidationError)
    return tuple(_request_number(item, where + "[" + str(i) + "]")
                 for i, item in enumerate(values))


def _wire_vector(value, size, where):
    values = _sequence(value, size, where, CertificateValidationError, wire=True)
    return tuple(_wire_number(item, where + "[" + str(i) + "]")
                 for i, item in enumerate(values))


def _same_tree(actual, expected, where):
    """Compare canonical JSON with exact types; bool must not impersonate int."""
    _need(type(actual) is type(expected), where + ": noncanonical JSON type")
    if type(expected) is dict:
        _need(set(actual) == set(expected), where + ": canonical object keys differ")
        for key in expected:
            _same_tree(actual[key], expected[key], where + "." + key)
    elif type(expected) is list:
        _need(len(actual) == len(expected), where + ": canonical array length differs")
        for i, (left, right) in enumerate(zip(actual, expected)):
            _same_tree(left, right, where + "[" + str(i) + "]")
    else:
        _need(actual == expected, where + ": canonical value differs from trusted request")


def _json_types(value, where):
    """Reject non-wire objects and floats, including in opaque auxiliary data."""
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _json_types(item, where + "[" + str(i) + "]")
        return
    if type(value) is dict:
        for key, item in value.items():
            _need(type(key) is str, where + ": JSON object keys must be strings")
            _json_types(item, where + "." + key)
        return
    raise CertificateValidationError(where + ": floats and non-JSON objects are forbidden")


def _sha(canonical):
    encoded = json.dumps(canonical, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _digest_text(value, where, error=CertificateValidationError):
    _need(type(value) is str and bool(re.fullmatch(r"[0-9a-f]{64}", value)),
          where + ": require a lowercase SHA-256 hexadecimal string", error)
    return value


def _trusted_request(program, context_sizes, exact_weights, exact_reference):
    error = RequestValidationError
    _object(program, ("lower", "upper", "rows"), "program", error)
    lower = _request_vector(program["lower"], None, "program.lower")
    n = len(lower)
    _need(n > 0, "program: require at least one coordinate", error)
    upper = _request_vector(program["upper"], n, "program.upper")
    _need(all(0 <= lo <= hi <= 1 for lo, hi in zip(lower, upper)),
          "program: require 0 <= lower <= upper <= 1", error)
    sizes = tuple(_sequence(context_sizes, None, "context_sizes", error))
    _need(bool(sizes) and all(type(s) is int and s > 0 for s in sizes)
          and sum(sizes) == n, "context_sizes: require positive integer partition", error)
    weights = _request_vector(exact_weights, len(sizes), "exact_weights")
    _need(all(w >= 0 for w in weights) and sum(weights) == 1,
          "exact_weights: require exact nonnegative sum one", error)
    refs = _sequence(exact_reference, len(sizes), "exact_reference", error)
    references = tuple(_request_vector(row, sizes[x], "exact_reference[" + str(x) + "]")
                       for x, row in enumerate(refs))
    _need(all(all(p >= 0 for p in row) and sum(row) == 1 for row in references),
          "exact_reference: require exact nonnegative sum one in every context", error)
    owner = [x for x, size in enumerate(sizes) for _ in range(size)]
    seen = set()
    rows = []
    for j, raw in enumerate(_sequence(program["rows"], None, "program.rows", error)):
        where = "program.rows[" + str(j) + "]"
        _object(raw, ("coefficients", "lower", "upper"), where, error)
        coefficients = _request_vector(raw["coefficients"], n, where + ".coefficients")
        lo = _request_number(raw["lower"], where + ".lower")
        hi = _request_number(raw["upper"], where + ".upper")
        _need(lo <= hi, where + ": reversed interval", error)
        _need(all(t >= 0 for t in coefficients), where + ": signed rows are outside schema domain", error)
        support = {i for i, t in enumerate(coefficients) if t > 0}
        _need(len({owner[i] for i in support}) <= 1,
              where + ": mixed-context row is outside schema domain", error)
        _need(not seen.intersection(support), where + ": overlapping rows are outside schema domain", error)
        seen.update(support)
        if not support:
            _need(lo <= 0 <= hi, where + ": zero row excludes zero", error)
        rows.append((coefficients, lo, hi))
    canonical = {
        "schema_version": INPUT_SCHEMA,
        "program": {"lower": [_render(v) for v in lower],
                    "upper": [_render(v) for v in upper],
                    "rows": [{"coefficients": [_render(t) for t in row],
                              "lower": _render(lo), "upper": _render(hi)}
                             for row, lo, hi in rows]},
        "context_sizes": list(sizes), "exact_weights": [_render(v) for v in weights],
        "exact_reference": [[_render(v) for v in row] for row in references],
    }
    return lower, upper, tuple(rows), sizes, weights, references, canonical


def input_digest(program, context_sizes, exact_weights, exact_reference):
    """Compute the version-1 digest from a separately trusted exact request.

    No probability is normalized. This validates syntax and the declared row
    domain, not statistical provenance or general feasibility; a received point
    must independently establish every original feasibility constraint.
    """
    return _sha(_trusted_request(program, context_sizes, exact_weights, exact_reference)[-1])


def verify_certificate(program, context_sizes, exact_weights, exact_reference,
                       certificate, *, expected_input_sha256=None):
    """Return an explicit scoped receipt, or raise VerificationError.

    The first four arguments are the caller's trusted request. The certificate's
    input and digest must match its independently reconstructed canonical form.
    A separately held expected_input_sha256 may additionally pin that request.
    Rationals in the certificate must be reduced n/d strings. The mathematical
    predicate is original feasibility, legal probability, nonnegative original
    row duals, and exact LB = policy-at-point = F(point) = reported values.

    contexts.* (cap/root/price/D and all other construction metadata) and
    internal_check receive JSON-shape validation only. They are NOT certified;
    every successful receipt explicitly identifies these unchecked fields.
    """
    lower, upper, rows, sizes, weights, references, canonical = _trusted_request(
        program, context_sizes, exact_weights, exact_reference)
    digest = _sha(canonical)
    if expected_input_sha256 is not None:
        _digest_text(expected_input_sha256, "expected_input_sha256", RequestValidationError)
        _need(expected_input_sha256 == digest,
              "expected_input_sha256 does not match the trusted request", RequestValidationError)
    _object(certificate, _CERTIFICATE_KEYS, "certificate", CertificateValidationError)
    _json_types(certificate, "certificate")
    _need(certificate["schema_version"] == CERTIFICATE_SCHEMA, "certificate: unsupported schema version")
    _same_tree(certificate["canonical_input"], canonical, "certificate.canonical_input")
    _need(_digest_text(certificate["input_sha256"], "certificate.input_sha256") == digest,
          "certificate digest does not bind the trusted request")
    n, nr = len(lower), len(rows)
    point = _wire_vector(certificate["point"], n, "certificate.point")
    _need(all(lo <= m <= hi for lo, m, hi in zip(lower, point, upper)),
          "certificate point violates an original box constraint")
    eta = _wire_vector(certificate["original_row_lower_dual"], nr, "certificate.original_row_lower_dual")
    lam = _wire_vector(certificate["original_row_upper_dual"], nr, "certificate.original_row_upper_dual")
    _need(all(v >= 0 for v in eta + lam), "negative original moment multiplier")
    policy_rows = _sequence(certificate["policy"], len(sizes), "certificate.policy",
                            CertificateValidationError, wire=True)
    c = []
    common_upper = Fraction(0)
    offset = 0
    for x, size in enumerate(sizes):
        q = _wire_vector(policy_rows[x], size, "certificate.policy[" + str(x) + "]")
        _need(all(v >= 0 for v in q) and sum(q) == 1,
              "certificate policy must be a probability vector in every context")
        local_point = point[offset:offset + size]
        mean = sum((p * m for p, m in zip(references[x], local_point)), Fraction(0))
        common_upper += weights[x] * (max(local_point) - mean)
        c.extend(weights[x] * (qi - pi) for qi, pi in zip(q, references[x]))
        offset += size
    z = c.copy()
    original_row_values = []
    dual_lower = Fraction(0)
    for j, (coefficients, lo, hi) in enumerate(rows):
        value = sum((t * m for t, m in zip(coefficients, point)), Fraction(0))
        _need(lo <= value <= hi, "certificate point violates original moment row " + str(j))
        original_row_values.append(value)
        dual_lower += lo * eta[j] - hi * lam[j]
        for i, t in enumerate(coefficients):
            z[i] += t * (lam[j] - eta[j])
    dual_lower += sum((min(zi * lo, zi * hi) for zi, lo, hi in zip(z, lower, upper)), Fraction(0))
    attained = sum((ci * mi for ci, mi in zip(c, point)), Fraction(0))
    _need(dual_lower == attained == common_upper,
          "exact original weak-dual LB, policy-at-point and common-point upper bound differ")
    # These top-level redundant values are checked, unlike construction metadata.
    for name, expected in (("original_row_values", original_row_values),
                           ("objective_coefficients", c), ("box_reduced_costs", z)):
        _need(_wire_vector(certificate[name], len(expected), "certificate." + name) == tuple(expected),
              "certificate." + name + " differs from original-program recomputation")
    for name in ("capacity", "common_point_upper_bound", "policy_at_point", "policy_dual_lower_bound"):
        _need(_wire_number(certificate[name], "certificate." + name) == common_upper,
              "certificate." + name + " differs from exact recomputation")
    _sequence(certificate["contexts"], len(sizes), "certificate.contexts",
              CertificateValidationError, wire=True)
    _need(type(certificate["internal_check"]) is dict,
          "certificate.internal_check must be a JSON object")
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "verified": True,
        "scope": "original-program-optimality-only",
        "certificate_schema_version": CERTIFICATE_SCHEMA,
        "input_sha256": digest,
        "digest_authority": "trusted-request-and-caller-digest" if expected_input_sha256 is not None else "trusted-request",
        "capacity": _render(common_upper),
        "computed_common_point_upper_bound": _render(common_upper),
        "computed_policy_at_point": _render(attained),
        "computed_original_dual_lower_bound": _render(dual_lower),
        "coordinate_count": n, "context_count": len(sizes), "original_row_count": nr,
        "checked_fields": sorted(_CERTIFICATE_KEYS.difference(UNCHECKED_FIELDS)),
        "unchecked_fields": list(UNCHECKED_FIELDS),
        "unchecked_fields_notice": (
            "contexts.* (including cap, root, prices, D, and local summaries) and "
            "internal_check have no mathematical validation. Only original-program "
            "optimality and the listed checked fields are certified. Statistical "
            "provenance of the caller's trusted request is not established here."
        ),
    }


__all__ = ["verify_certificate", "input_digest", "VerificationError",
           "RequestValidationError", "CertificateValidationError", "VERIFICATION_SCHEMA"]
