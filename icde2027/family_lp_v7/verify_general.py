"""Independent exact receivers for general family and fixed-objective optima.

This module uses only the Python standard library.  It does not propose a
solution, call an optimizer, or import any producer/partition implementation.
All mathematical data come from a separately trusted caller request.  The
caller must also supply the expected digest; a certificate's digest alone
never authorizes its request.

Trusted scalars are exactly int, Fraction, or reduced canonical ``n/d`` text.
Certificate scalars are canonical ``n/d`` text only.  Floats, booleans, implicit
normalization, and unknown fields are refused.  Use strict_json_loads at the
wire boundary: a dict cannot reveal duplicate keys already lost by a caller's
ordinary JSON parser.  Duplicate mathematical rows are allowed.

Resource limits below bound accepted representations, not running time or
intermediate Fraction bit growth.  A service must enforce its own outer time
and memory limits.  This module never changes process-global numeric settings.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import re


FAMILY_INPUT_SCHEMA = "family-capacity-input/1"
FAMILY_CERTIFICATE_SCHEMA = "family-capacity-certificate/1"
FAMILY_VERIFICATION_SCHEMA = "family-capacity-verification/1"
FIXED_INPUT_SCHEMA = "fixed-policy-optimum-input/1"
FIXED_CERTIFICATE_SCHEMA = "fixed-policy-optimum-certificate/1"
FIXED_VERIFICATION_SCHEMA = "fixed-policy-optimum-verification/1"

MAX_COORDINATES = 4096
MAX_ROWS = 4096
MAX_DENSE_COEFFICIENTS = 1_000_000
MAX_RATIONAL_DIGITS = 100_000  # separately for numerator and denominator
MAX_TOTAL_RATIONAL_CHARACTERS = 16_000_000  # separately per request/certificate
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSON_NODES = 2_100_000
MAX_JSON_DEPTH = 32

_RATIONAL_TEXT = re.compile(r"(?:0|-?[1-9][0-9]*)/[1-9][0-9]*\Z", re.ASCII)
_HASH_TEXT = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMON_CERTIFICATE_KEYS = frozenset({
    "schema_version", "input_sha256", "point", "original_row_lower_dual",
    "original_row_upper_dual",
})
_FAMILY_CERTIFICATE_KEYS = _COMMON_CERTIFICATE_KEYS | {"policy", "capacity"}
_FIXED_CERTIFICATE_KEYS = _COMMON_CERTIFICATE_KEYS | {"optimum"}
_ZERO = Fraction(0)


class VerificationError(ValueError):
    """The request, wire representation, or certificate cannot be verified."""


class RequestValidationError(VerificationError):
    """The external request or its separately held digest is invalid."""


class CertificateValidationError(VerificationError):
    """The certificate does not prove the external request's optimum."""


class WireFormatError(VerificationError):
    """Strict JSON decoding failed before request/certificate interpretation."""


def _need(condition, message, error):
    if not condition:
        raise error(message)


def _int_text(value):
    # Decimal(int) and fixed formatting are exact even under a low precision
    # context.  Neither str(huge_int) nor a global int-string guard change is used.
    return format(Decimal(value), "f")


def _render(value):
    return _int_text(value.numerator) + "/" + _int_text(value.denominator)


def _integer_from_digits(text):
    negative = text.startswith("-")
    digits = text[1:] if negative else text
    value = 0
    # Each call to int parses at most nine digits, independent of the process
    # int-string limit; the explicit representation limit was already checked.
    for offset in range(0, len(digits), 9):
        part = digits[offset:offset + 9]
        value = value * (10 ** len(part)) + int(part)
    return -value if negative else value


class _Budget:
    def __init__(self, error):
        self.error = error
        self.characters = 0

    def add(self, count, where):
        self.characters += count
        _need(self.characters <= MAX_TOTAL_RATIONAL_CHARACTERS,
              where + ": aggregate rational representation limit", self.error)


def _trusted_integer_text(value, where, error):
    # 3.322 is a strict upper bound for log2(10).  This inexpensive check avoids
    # formatting arbitrarily large caller-provided integers before rejecting.
    _need(abs(value).bit_length() <= (MAX_RATIONAL_DIGITS * 3322) // 1000 + 1,
          where + ": integer exceeds digit limit", error)
    text = _int_text(value)
    _need(len(text.lstrip("-")) <= MAX_RATIONAL_DIGITS,
          where + ": integer exceeds digit limit", error)
    return text


def _number(value, where, budget, *, wire=False):
    error = budget.error
    if type(value) is str:
        _need(len(value) <= 2 * MAX_RATIONAL_DIGITS + 2,
              where + ": rational exceeds digit limit", error)
        _need(_RATIONAL_TEXT.fullmatch(value) is not None,
              where + ": canonical reduced n/d text required", error)
        ntext, dtext = value.split("/")
        _need(len(ntext.lstrip("-")) <= MAX_RATIONAL_DIGITS and
              len(dtext) <= MAX_RATIONAL_DIGITS,
              where + ": rational exceeds digit limit", error)
        budget.add(len(value), where)
        numerator, denominator = _integer_from_digits(ntext), _integer_from_digits(dtext)
        result = Fraction(numerator, denominator)
        _need(result.numerator == numerator and result.denominator == denominator,
              where + ": rational is not reduced", error)
        return result
    _need(not wire and type(value) in (int, Fraction),
          where + ": exact int/Fraction/n/d required; floats and bools are forbidden", error)
    result = Fraction(value)
    nt = _trusted_integer_text(result.numerator, where, error)
    dt = _trusted_integer_text(result.denominator, where, error)
    budget.add(len(nt) + len(dt) + 1, where)
    return result


def _object(value, keys, where, error):
    _need(type(value) is dict and all(type(key) is str for key in value) and
          set(value) == set(keys), where + ": exact object fields required", error)


def _sequence(value, length, where, error, *, wire=False):
    allowed = (list,) if wire else (list, tuple)
    _need(type(value) in allowed and len(value) == length,
          where + ": sequence length/type mismatch", error)
    return value


def _vector(value, length, where, budget, *, wire=False):
    values = _sequence(value, length, where, budget.error, wire=wire)
    return tuple(_number(v, where + "[" + str(i) + "]", budget, wire=wire)
                 for i, v in enumerate(values))


def _program(program, budget):
    error = budget.error
    _object(program, {"lower", "upper", "rows"}, "program", error)
    lower = program["lower"]
    _need(type(lower) in (list, tuple) and 0 < len(lower) <= MAX_COORDINATES,
          "program.lower: coordinate limit/type", error)
    n = len(lower)
    lo = _vector(lower, n, "program.lower", budget)
    hi = _vector(program["upper"], n, "program.upper", budget)
    _need(all(0 <= l <= u <= 1 for l, u in zip(lo, hi)),
          "program: box must satisfy 0 <= lower <= upper <= 1", error)
    source_rows = program["rows"]
    _need(type(source_rows) in (list, tuple) and len(source_rows) <= MAX_ROWS,
          "program.rows: row limit/type", error)
    _need(n * len(source_rows) <= MAX_DENSE_COEFFICIENTS,
          "program.rows: total dense coefficient limit", error)
    rows = []
    canonical_rows = []
    for j, row in enumerate(source_rows):
        where = "program.rows[" + str(j) + "]"
        _object(row, {"coefficients", "lower", "upper"}, where, error)
        coefficients = _vector(row["coefficients"], n, where + ".coefficients", budget)
        low = _number(row["lower"], where + ".lower", budget)
        high = _number(row["upper"], where + ".upper", budget)
        _need(low <= high, where + ": reversed moment interval", error)
        rows.append((coefficients, low, high))
        canonical_rows.append({"coefficients": [_render(v) for v in coefficients],
                               "lower": _render(low), "upper": _render(high)})
    canonical = {"lower": [_render(v) for v in lo], "upper": [_render(v) for v in hi],
                 "rows": canonical_rows}
    return lo, hi, tuple(rows), canonical


def _family_request(program, context_sizes, weights, reference):
    budget = _Budget(RequestValidationError)
    lo, hi, rows, canonical_program = _program(program, budget)
    _need(type(context_sizes) in (list, tuple) and
          0 < len(context_sizes) <= len(lo) and
          all(type(s) is int and 0 < s <= len(lo) for s in context_sizes) and
          sum(context_sizes) == len(lo), "context_sizes: positive integer partition required",
          RequestValidationError)
    sizes = tuple(context_sizes)
    w = _vector(weights, len(sizes), "weights", budget)
    _need(all(v >= 0 for v in w) and sum(w, _ZERO) == 1,
          "weights: exact probability required; no implicit normalization", RequestValidationError)
    refs = _sequence(reference, len(sizes), "reference", RequestValidationError)
    p = tuple(_vector(row, size, "reference[" + str(x) + "]", budget)
              for x, (row, size) in enumerate(zip(refs, sizes)))
    _need(all(all(v >= 0 for v in row) and sum(row, _ZERO) == 1 for row in p),
          "reference: exact probability in every context required", RequestValidationError)
    canonical = {"schema_version": FAMILY_INPUT_SCHEMA, "program": canonical_program,
                 "context_sizes": list(sizes), "weights": [_render(v) for v in w],
                 "reference": [[_render(v) for v in row] for row in p]}
    return lo, hi, rows, sizes, w, p, canonical


def _fixed_request(program, objective):
    budget = _Budget(RequestValidationError)
    lo, hi, rows, canonical_program = _program(program, budget)
    c = _vector(objective, len(lo), "objective", budget)
    canonical = {"schema_version": FIXED_INPUT_SCHEMA, "program": canonical_program,
                 "objective": [_render(v) for v in c]}
    return lo, hi, rows, c, canonical


def _digest(canonical):
    data = json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def family_input_digest(program, context_sizes, exact_weights, exact_reference):
    """Validate and digest the caller's complete trusted family request."""
    return _digest(_family_request(program, context_sizes, exact_weights, exact_reference)[-1])


def fixed_input_digest(program, exact_objective):
    """Validate and digest the caller's trusted program and exact fixed objective."""
    return _digest(_fixed_request(program, exact_objective)[-1])


def _bound_certificate(certificate, keys, schema, canonical, expected_input_sha256):
    digest = _digest(canonical)
    _need(type(expected_input_sha256) is str and
          _HASH_TEXT.fullmatch(expected_input_sha256) is not None,
          "expected_input_sha256: external lowercase SHA256 required", RequestValidationError)
    _need(expected_input_sha256 == digest,
          "expected_input_sha256 differs from the trusted request", RequestValidationError)
    _object(certificate, keys, "certificate", CertificateValidationError)
    _need(type(certificate["schema_version"]) is str and certificate["schema_version"] == schema,
          "certificate: wrong schema version", CertificateValidationError)
    supplied = certificate["input_sha256"]
    _need(type(supplied) is str and _HASH_TEXT.fullmatch(supplied) is not None and supplied == digest,
          "certificate digest differs from the external trusted request", CertificateValidationError)
    return digest


def _primal_dual(certificate, lo, hi, rows, budget):
    point = _vector(certificate["point"], len(lo), "certificate.point", budget, wire=True)
    _need(all(l <= v <= u for l, v, u in zip(lo, point, hi)),
          "certificate point violates an original box", CertificateValidationError)
    eta = _vector(certificate["original_row_lower_dual"], len(rows),
                  "certificate.original_row_lower_dual", budget, wire=True)
    lam = _vector(certificate["original_row_upper_dual"], len(rows),
                  "certificate.original_row_upper_dual", budget, wire=True)
    _need(all(v >= 0 for v in eta + lam),
          "certificate has a negative original moment multiplier", CertificateValidationError)
    for j, (coefficients, low, high) in enumerate(rows):
        value = sum((a * m for a, m in zip(coefficients, point)), _ZERO)
        _need(low <= value <= high,
              "certificate point violates original moment row " + str(j), CertificateValidationError)
    return point, eta, lam


def _weak_dual(lo, hi, rows, objective, eta, lam):
    reduced = list(objective)
    lower = _ZERO
    for j, (coefficients, low, high) in enumerate(rows):
        lower += low * eta[j] - high * lam[j]
        for i, a in enumerate(coefficients):
            reduced[i] += a * (lam[j] - eta[j])
    lower += sum((min(z * l, z * u) for z, l, u in zip(reduced, lo, hi)), _ZERO)
    return lower


def _receipt(schema, certificate_schema, scope, digest, keys, n, nr):
    return {"schema_version": schema, "verified": True, "scope": scope,
            "certificate_schema_version": certificate_schema, "input_sha256": digest,
            "digest_authority": "trusted-request-and-caller-digest",
            "coordinate_count": n, "original_row_count": nr,
            "checked_fields": sorted(keys), "unchecked_fields": [],
            "scope_notice": "Original-program mathematical optimality only; no statistical "
                            "provenance, transport validity, or infeasibility claim."}


def verify_family_certificate(program, context_sizes, exact_weights, exact_reference,
                              certificate, *, expected_input_sha256):
    """Verify a general family optimum against an external request and digest.

    Overlapping, signed, cross-context and zero moment rows are supported.
    Every context, including zero-weight contexts, must have a feasible point
    and a legal policy.  Success proves exact original weak-dual LB equals
    policy-at-point equals F(point) equals the claimed capacity.
    """
    lo, hi, rows, sizes, weights, references, canonical = _family_request(
        program, context_sizes, exact_weights, exact_reference)
    digest = _bound_certificate(certificate, _FAMILY_CERTIFICATE_KEYS,
                               FAMILY_CERTIFICATE_SCHEMA, canonical, expected_input_sha256)
    budget = _Budget(CertificateValidationError)
    point, eta, lam = _primal_dual(certificate, lo, hi, rows, budget)
    source_policy = _sequence(certificate["policy"], len(sizes), "certificate.policy",
                              CertificateValidationError, wire=True)
    c = []
    upper = _ZERO
    offset = 0
    for x, size in enumerate(sizes):
        q = _vector(source_policy[x], size, "certificate.policy[" + str(x) + "]", budget, wire=True)
        _need(all(v >= 0 for v in q) and sum(q, _ZERO) == 1,
              "certificate policy must be an exact probability in every context", CertificateValidationError)
        local = point[offset:offset + size]
        mean = sum((p * m for p, m in zip(references[x], local)), _ZERO)
        upper += weights[x] * (max(local) - mean)
        c.extend(weights[x] * (qi - pi) for qi, pi in zip(q, references[x]))
        offset += size
    lower = _weak_dual(lo, hi, rows, c, eta, lam)
    attained = sum((ci * mi for ci, mi in zip(c, point)), _ZERO)
    claimed = _number(certificate["capacity"], "certificate.capacity", budget, wire=True)
    _need(lower == attained == upper == claimed,
          "exact original LB, policy-at-point, F(point), and capacity differ", CertificateValidationError)
    result = _receipt(FAMILY_VERIFICATION_SCHEMA, FAMILY_CERTIFICATE_SCHEMA,
                      "general-original-program-family-optimality", digest,
                      _FAMILY_CERTIFICATE_KEYS, len(lo), len(rows))
    result.update(context_count=len(sizes), capacity=_render(upper),
                  computed_original_dual_lower_bound=_render(lower),
                  computed_policy_at_point=_render(attained),
                  computed_common_point_upper_bound=_render(upper))
    return result


def verify_fixed_certificate(program, exact_objective, certificate, *, expected_input_sha256):
    """Verify an arbitrary rational fixed objective's exact original optimum."""
    lo, hi, rows, c, canonical = _fixed_request(program, exact_objective)
    digest = _bound_certificate(certificate, _FIXED_CERTIFICATE_KEYS,
                               FIXED_CERTIFICATE_SCHEMA, canonical, expected_input_sha256)
    budget = _Budget(CertificateValidationError)
    point, eta, lam = _primal_dual(certificate, lo, hi, rows, budget)
    lower = _weak_dual(lo, hi, rows, c, eta, lam)
    attained = sum((ci * mi for ci, mi in zip(c, point)), _ZERO)
    claimed = _number(certificate["optimum"], "certificate.optimum", budget, wire=True)
    _need(lower == attained == claimed,
          "exact original LB, attained fixed objective, and optimum differ", CertificateValidationError)
    result = _receipt(FIXED_VERIFICATION_SCHEMA, FIXED_CERTIFICATE_SCHEMA,
                      "general-original-program-fixed-objective-optimality", digest,
                      _FIXED_CERTIFICATE_KEYS, len(lo), len(rows))
    result.update(optimum=_render(attained), computed_original_dual_lower_bound=_render(lower),
                  computed_attained_objective=_render(attained))
    return result


def strict_json_loads(data):
    """Decode bounded JSON without duplicate keys, floats, or nonfinite tokens.

    This is a wire parser, not an authorization or mathematical check.  Supply
    its result to the appropriate public digest/verification function.  Objects
    already parsed by an ordinary parser cannot be retrospectively checked for
    duplicate keys.
    """
    _need(type(data) in (str, bytes), "JSON must be str or bytes", WireFormatError)
    if type(data) is bytes:
        _need(len(data) <= MAX_JSON_BYTES, "JSON byte limit", WireFormatError)
        try:
            data = data.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise WireFormatError("JSON is not UTF-8") from exc
    else:
        _need(len(data) <= MAX_JSON_BYTES, "JSON length limit", WireFormatError)
        try:
            _need(len(data.encode("utf-8")) <= MAX_JSON_BYTES, "JSON byte limit", WireFormatError)
        except UnicodeError as exc:
            raise WireFormatError("JSON text contains invalid Unicode") from exc

    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value, "duplicate JSON object key", WireFormatError)
            value[key] = item
        return value

    def integer(text):
        _need(text != "-0" and len(text.lstrip("-")) <= MAX_RATIONAL_DIGITS,
              "JSON integer representation limit", WireFormatError)
        return _integer_from_digits(text)

    def forbidden(_text):
        raise WireFormatError("JSON floats and nonfinite tokens are forbidden")

    try:
        result = json.loads(data, object_pairs_hook=pairs, parse_int=integer,
                            parse_float=forbidden, parse_constant=forbidden)
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, WireFormatError):
            raise
        raise WireFormatError("malformed or overly nested JSON") from exc
    pending = [(result, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= MAX_JSON_NODES and depth <= MAX_JSON_DEPTH,
              "JSON node/depth limit", WireFormatError)
        if type(item) is dict:
            pending.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            pending.extend((v, depth + 1) for v in item)
    return result


__all__ = [
    "family_input_digest", "fixed_input_digest", "verify_family_certificate",
    "verify_fixed_certificate", "strict_json_loads", "VerificationError",
    "RequestValidationError", "CertificateValidationError", "WireFormatError",
    "FAMILY_INPUT_SCHEMA", "FAMILY_CERTIFICATE_SCHEMA", "FAMILY_VERIFICATION_SCHEMA",
    "FIXED_INPUT_SCHEMA", "FIXED_CERTIFICATE_SCHEMA", "FIXED_VERIFICATION_SCHEMA",
]
