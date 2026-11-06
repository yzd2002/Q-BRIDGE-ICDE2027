"""Construct exact capacity certificates for disjoint nonnegative moment rows.

This module performs no I/O, imports no solver, and uses Fraction arithmetic.
Its internal checks are producer self-checks, not an independent verifier.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re


CERTIFICATE_VERSION = "partition-capacity-v6/1"
INPUT_VERSION = "partition-capacity-input/1"
_RATIONAL = re.compile(r"[+-]?[0-9]+(?:/[1-9][0-9]*)?\Z")
_ZERO = Fraction(0)
_ONE = Fraction(1)


class InvalidInputError(ValueError):
    """Malformed input, invalid probabilities, or an empty supported program."""


class InfeasibleProgramError(InvalidInputError):
    """The exact constraints establish that the supported program is empty."""


class UnsupportedStructureError(ValueError):
    """Well-formed rows are signed, overlapping, or span multiple contexts."""


class CertificateConstructionError(RuntimeError):
    """A generated certificate failed an exact internal invariant."""


def _fraction(value, path):
    # In particular, do not let Fraction(float) silently bless binary64 input.
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if type(value) is str and _RATIONAL.fullmatch(value):
        if "/" in value:
            numerator, denominator = value.split("/")
            return Fraction(int(numerator), int(denominator))
        return Fraction(int(value))
    raise InvalidInputError(
        f"{path}: expected int, Fraction, or integer/positive-denominator string; "
        "float, bool, decimal strings, and implicit normalization are not accepted"
    )


def _sequence(value, path):
    if not isinstance(value, (list, tuple)):
        raise InvalidInputError(f"{path}: expected a list or tuple")
    return value


def _vector(value, path, size=None):
    values = _sequence(value, path)
    if size is not None and len(values) != size:
        raise InvalidInputError(f"{path}: length {len(values)} != {size}")
    return tuple(_fraction(v, f"{path}[{i}]") for i, v in enumerate(values))


def _keys(value, expected, path):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise InvalidInputError(f"{path}: requires exactly keys {tuple(expected)}")


def _q(value):
    """One JSON-safe encoding for every rational, including integers."""
    return f"{value.numerator}/{value.denominator}"


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right, strict=True)), _ZERO)


def _digest(canonical_input):
    payload = json.dumps(
        canonical_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require(condition, message):
    if not condition:
        raise CertificateConstructionError(message)


@dataclass(frozen=True)
class _Row:
    coefficients: tuple
    lower: Fraction
    upper: Fraction
    support: tuple


@dataclass(frozen=True)
class _Input:
    lower: tuple
    upper: tuple
    rows: tuple
    sizes: tuple
    weights: tuple
    reference: tuple
    coordinates: tuple
    row_context: tuple
    groups: tuple
    canonical: dict


def _parse(program, context_sizes, exact_weights, exact_reference):
    """Copy and validate the caller's request; never change its feasible set."""
    _keys(program, ("lower", "upper", "rows"), "program")
    lower = _vector(program["lower"], "program.lower")
    n = len(lower)
    if n == 0:
        raise InvalidInputError("program.lower: the box must have a coordinate")
    upper = _vector(program["upper"], "program.upper", n)
    for i, (lo, hi) in enumerate(zip(lower, upper, strict=True)):
        if not _ZERO <= lo <= hi <= _ONE:
            raise InvalidInputError(f"box[{i}]: require 0 <= lower <= upper <= 1")
    sizes = tuple(_sequence(context_sizes, "context_sizes"))
    if not sizes or any(type(s) is not int or s <= 0 for s in sizes):
        raise InvalidInputError("context_sizes: require nonempty positive integers")
    if sum(sizes) != n:
        raise InvalidInputError("context_sizes: sum must equal the box dimension")
    weights = _vector(exact_weights, "exact_weights", len(sizes))
    if any(w < 0 for w in weights) or sum(weights, _ZERO) != _ONE:
        raise InvalidInputError("exact_weights: require exact nonnegative sum of one")
    reference_rows = _sequence(exact_reference, "exact_reference")
    if len(reference_rows) != len(sizes):
        raise InvalidInputError("exact_reference: one probability row per context")
    reference = tuple(
        _vector(row, f"exact_reference[{x}]", sizes[x])
        for x, row in enumerate(reference_rows)
    )
    for x, row in enumerate(reference):
        if any(p < 0 for p in row) or sum(row, _ZERO) != _ONE:
            raise InvalidInputError(
                f"exact_reference[{x}]: require exact nonnegative sum of one"
            )
    # Parse every original row before making a structural classification.
    rows = []
    for j, row in enumerate(_sequence(program["rows"], "program.rows")):
        _keys(row, ("coefficients", "lower", "upper"), f"program.rows[{j}]")
        coefficients = _vector(row["coefficients"], f"rows[{j}].coefficients", n)
        lo = _fraction(row["lower"], f"rows[{j}].lower")
        hi = _fraction(row["upper"], f"rows[{j}].upper")
        if lo > hi:
            raise InvalidInputError(f"rows[{j}]: lower exceeds upper")
        rows.append(_Row(coefficients, lo, hi, tuple(i for i, a in enumerate(coefficients) if a)))
    coordinates = []
    owner = []
    start = 0
    for x, size in enumerate(sizes):
        coordinates.append(tuple(range(start, start + size)))
        owner.extend([x] * size)
        start += size
    row_context = []
    groups = [[] for _ in sizes]
    used = {}
    for j, row in enumerate(rows):
        if not row.support:
            if not row.lower <= _ZERO <= row.upper:
                raise InfeasibleProgramError(f"zero row {j}: its interval excludes zero")
            row_context.append(None)
            continue
        if any(a < 0 for a in row.coefficients):
            raise UnsupportedStructureError(f"row {j}: signed coefficients are unsupported")
        contexts = {owner[i] for i in row.support}
        if len(contexts) != 1:
            raise UnsupportedStructureError(f"row {j}: mixed-context support is unsupported")
        for i in row.support:
            if i in used:
                raise UnsupportedStructureError(
                    f"rows {used[i]} and {j}: overlapping positive support at {i}"
                )
            used[i] = j
        x = next(iter(contexts))
        row_context.append(x)
        groups[x].append(j)
    # These exact conditions are sufficient only after the disjoint-domain gate.
    for j, row in enumerate(rows):
        if _dot(row.coefficients, lower) > row.upper:
            raise InfeasibleProgramError(f"row {j}: upper bound is below lower-box mass")
        if _dot(row.coefficients, upper) < row.lower:
            raise InfeasibleProgramError(f"row {j}: lower bound exceeds upper-box mass")
    canonical = {
        "schema_version": INPUT_VERSION,
        "program": {
            "lower": list(map(_q, lower)),
            "upper": list(map(_q, upper)),
            "rows": [
                {"coefficients": list(map(_q, row.coefficients)),
                 "lower": _q(row.lower), "upper": _q(row.upper)}
                for row in rows
            ],
        },
        "context_sizes": list(sizes),
        "exact_weights": list(map(_q, weights)),
        "exact_reference": [list(map(_q, row)) for row in reference],
    }
    return _Input(
        lower, upper, tuple(rows), sizes, weights, reference, tuple(coordinates),
        tuple(row_context), tuple(tuple(g) for g in groups), canonical,
    )


def _mass_at_cap(row, upper, cap):
    return sum((row.coefficients[i] * min(upper[i], cap) for i in row.support), _ZERO)


def _root(row, upper, floor):
    """First feasible cap, scanned exactly across upper-box breakpoints."""
    mass = _mass_at_cap(row, upper, floor)
    if mass >= row.lower:
        return floor
    position = floor
    breakpoints = sorted({upper[i] for i in row.support if upper[i] > floor})
    slope = sum((row.coefficients[i] for i in row.support if upper[i] > floor), _ZERO)
    for endpoint in breakpoints:
        next_mass = mass + slope * (endpoint - position)
        if next_mass >= row.lower:
            _require(slope > 0, "root needs a strictly positive left slope")
            return position + (row.lower - mass) / slope
        mass = next_mass
        position = endpoint
        slope -= sum((row.coefficients[i] for i in row.support if upper[i] == endpoint), _ZERO)
    raise CertificateConstructionError("feasible row has no exact cap root")


def _knapsack(row, lower, capped_upper, reference):
    """Return row entries and a nonnegative capped upper-row price."""
    full_mass = _dot(row.coefficients, capped_upper)
    if row.upper >= full_mass:
        return {i: capped_upper[i] for i in row.support}, _ZERO
    point = {i: lower[i] for i in row.support}
    remaining = row.upper - _dot(row.coefficients, lower)
    order = sorted(row.support, key=lambda i: (-reference[i] / row.coefficients[i], i))
    if remaining == 0:
        return point, max(reference[i] / row.coefficients[i] for i in row.support)
    last_price = None
    for i in order:
        amount = min(capped_upper[i] - lower[i], remaining / row.coefficients[i])
        if amount > 0:
            point[i] += amount
            remaining -= row.coefficients[i] * amount
            last_price = reference[i] / row.coefficients[i]
        if remaining == 0:
            break
    _require(remaining == 0 and last_price is not None, "knapsack budget was not attained")
    return point, last_price


def solve(program, context_sizes, exact_weights, exact_reference):
    """Return a JSON-serializable certificate without I/O or input mutation.

    program = {"lower": [...], "upper": [...], "rows": [
        {"coefficients": [...], "lower": a, "upper": b}, ...]}
    Context coordinates are contiguous in context_sizes order. Exact weights and
    each exact_reference row must already be nonnegative and sum exactly to one.
    Scalars accept int, Fraction, or integer/positive-denominator rational string.
    See README for exceptions, certificate trust boundaries, and prototype scope.
    """
    data = _parse(program, context_sizes, exact_weights, exact_reference)
    n = len(data.lower)
    point = list(data.upper)
    policy = []
    lower_dual = [_ZERO] * len(data.rows)
    upper_dual = [_ZERO] * len(data.rows)
    contexts = []
    total = _ZERO
    for x, indices in enumerate(data.coordinates):
        floor = max(data.lower[i] for i in indices)
        roots = {j: _root(data.rows[j], data.upper, floor) for j in data.groups[x]}
        cap = max([floor, *roots.values()])
        h = list(data.upper)
        p = [_ZERO] * n
        for local, i in enumerate(indices):
            h[i] = min(data.upper[i], cap)
            point[i] = h[i]
            p[i] = data.reference[x][local]
        prices = {}
        for j in data.groups[x]:
            entries, prices[j] = _knapsack(data.rows[j], data.lower, h, p)
            for i, value in entries.items():
                point[i] = value
        residual = {i: p[i] for i in indices}
        for j, price in prices.items():
            for i in data.rows[j].support:
                residual[i] -= price * data.rows[j].coefficients[i]
        initial = {i: max(residual[i], _ZERO) if data.upper[i] >= cap else _ZERO for i in indices}
        q = initial.copy()
        deficit = _ONE - sum(initial.values(), _ZERO)
        _require(deficit >= 0, "initial policy exceeds probability one")
        local_lower = {j: _ZERO for j in data.groups[x]}
        if cap == floor:
            k = next(i for i in indices if data.lower[i] == cap)
            q[k] += deficit
            witness = {"kind": "box_lower", "coordinate": k}
        else:
            j = next(j for j in data.groups[x] if roots[j] == cap)
            row = data.rows[j]
            active = [i for i in row.support if data.upper[i] >= cap]
            active_sum = sum((row.coefficients[i] for i in active), _ZERO)
            _require(active_sum > 0 and prices[j] == 0, "invalid cap lower-row witness")
            local_lower[j] = deficit / active_sum
            for i in active:
                q[i] += local_lower[j] * row.coefficients[i]
            witness = {
                "kind": "moment_lower", "row_index": j,
                "active_coordinates": active, "active_coefficient_sum": _q(active_sum),
            }
        policy.append([_q(q[i]) for i in indices])
        row_records = []
        for j in data.groups[x]:
            row = data.rows[j]
            lower_dual[j] = data.weights[x] * local_lower[j]
            upper_dual[j] = data.weights[x] * prices[j]
            row_records.append({
                "row_index": j,
                "root": _q(roots[j]),
                "mass_at_box_floor": _q(_mass_at_cap(row, data.upper, floor)),
                "mass_at_root": _q(_mass_at_cap(row, data.upper, roots[j])),
                "root_active_coordinates": [i for i in row.support if data.upper[i] >= roots[j]],
                "capped_mass": _q(_dot(row.coefficients, h)),
                "target_mass": _q(min(row.upper, _dot(row.coefficients, h))),
                "point_mass": _q(_dot(row.coefficients, point)),
                "upper_price": _q(prices[j]),
                "lower_price": _q(local_lower[j]),
            })
        reference_mean = sum((p[i] * point[i] for i in indices), _ZERO)
        capacity = cap - reference_mean
        capped_dual = sum((prices[j] * data.rows[j].upper for j in data.groups[x]), _ZERO)
        capped_dual += sum((max(residual[i] * data.lower[i], residual[i] * h[i]) for i in indices), _ZERO)
        contexts.append({
            "context_index": x, "coordinate_start": indices[0], "size": len(indices),
            "weight": _q(data.weights[x]), "box_floor": _q(floor), "cap": _q(cap),
            "cap_witness": witness, "rows": row_records,
            "residual_rewards": [_q(residual[i]) for i in indices],
            "initial_policy": [_q(initial[i]) for i in indices], "deficit": _q(deficit),
            "reference_mean": _q(reference_mean), "capped_box_dual_value": _q(capped_dual),
            "capacity": _q(capacity), "weighted_capacity": _q(data.weights[x] * capacity),
        })
        total += data.weights[x] * capacity
    c = []
    for x, row in enumerate(policy):
        c.extend(data.weights[x] * (_fraction(v, "policy") - p) for v, p in zip(row, data.reference[x], strict=True))
    z = [c[i] + sum((row.coefficients[i] * (upper_dual[j] - lower_dual[j]) for j, row in enumerate(data.rows)), _ZERO) for i in range(n)]
    dual_value = sum((row.lower * lower_dual[j] - row.upper * upper_dual[j] for j, row in enumerate(data.rows)), _ZERO)
    dual_value += sum((min(z[i] * data.lower[i], z[i] * data.upper[i]) for i in range(n)), _ZERO)
    certificate = {
        "schema_version": CERTIFICATE_VERSION,
        "input_sha256": _digest(data.canonical), "canonical_input": data.canonical,
        "point": list(map(_q, point)), "policy": policy,
        "original_row_lower_dual": list(map(_q, lower_dual)),
        "original_row_upper_dual": list(map(_q, upper_dual)),
        "original_row_values": [_q(_dot(row.coefficients, point)) for row in data.rows],
        "objective_coefficients": list(map(_q, c)), "box_reduced_costs": list(map(_q, z)),
        "capacity": _q(total), "common_point_upper_bound": _q(total),
        "policy_at_point": _q(_dot(c, point)), "policy_dual_lower_bound": _q(dual_value),
        "contexts": contexts,
        "internal_check": {"kind": "producer_exact_self_check", "independent_review": False},
    }
    # Bind to the separately parsed caller request, not merely a self-supplied C.
    _self_check(program, context_sizes, exact_weights, exact_reference, certificate)
    return certificate


def _self_check(program, context_sizes, exact_weights, exact_reference, certificate):
    """Exact internal invariants using the original caller request as authority."""
    data = _parse(program, context_sizes, exact_weights, exact_reference)
    _require(certificate["schema_version"] == CERTIFICATE_VERSION, "certificate version")
    _require(certificate["canonical_input"] == data.canonical, "certificate changed caller input")
    _require(certificate["input_sha256"] == _digest(data.canonical), "caller input digest mismatch")
    n = len(data.lower)
    m = _vector(certificate["point"], "certificate.point", n)
    eta = _vector(certificate["original_row_lower_dual"], "certificate.lower_dual", len(data.rows))
    lam = _vector(certificate["original_row_upper_dual"], "certificate.upper_dual", len(data.rows))
    _require(all(v >= 0 for v in (*eta, *lam)), "negative original row dual")
    _require(all(lo <= v <= hi for lo, v, hi in zip(data.lower, m, data.upper, strict=True)), "original box feasibility")
    _require(len(certificate["policy"]) == len(data.sizes), "policy context count")
    _require(len(certificate["contexts"]) == len(data.sizes), "context witness count")
    row_values = []
    for j, row in enumerate(data.rows):
        value = _dot(row.coefficients, m)
        row_values.append(value)
        _require(row.lower <= value <= row.upper, f"original row {j} feasibility")
        if not row.support:
            _require(eta[j] == lam[j] == 0, f"zero row {j} dual must be preserved as zero")
        _require(eta[j] * (value - row.lower) == 0, f"row {j} lower complementarity")
        _require(lam[j] * (row.upper - value) == 0, f"row {j} upper complementarity")
    _require(certificate["original_row_values"] == list(map(_q, row_values)), "original row values")
    c = []
    total = _ZERO
    for x, indices in enumerate(data.coordinates):
        q = _vector(certificate["policy"][x], f"policy[{x}]", data.sizes[x])
        _require(all(v >= 0 for v in q) and sum(q, _ZERO) == 1, f"context {x} probability")
        p = data.reference[x]
        w = data.weights[x]
        cx = certificate["contexts"][x]
        _require((cx["context_index"], cx["coordinate_start"], cx["size"]) == (x, indices[0], len(indices)), "context index metadata")
        _require(cx["weight"] == _q(w), "context weight")
        floor = max(data.lower[i] for i in indices)
        cap = _fraction(cx["cap"], "context.cap")
        _require(cx["box_floor"] == _q(floor) and cap >= floor, "context floor")
        _require(max(m[i] for i in indices) == cap, "attained context cap")
        _require([rr["row_index"] for rr in cx["rows"]] == list(data.groups[x]), "original row order in context")
        prices = {}
        lows = {}
        roots = []
        residual = dict(zip(indices, p, strict=True))
        h = [min(data.upper[i], cap) for i in range(n)]
        for rr in cx["rows"]:
            j = rr["row_index"]
            row = data.rows[j]
            root = _fraction(rr["root"], "row.root")
            at_floor = _mass_at_cap(row, data.upper, floor)
            at_root = _mass_at_cap(row, data.upper, root)
            active = [i for i in row.support if data.upper[i] >= root]
            _require(rr["mass_at_box_floor"] == _q(at_floor) and rr["mass_at_root"] == _q(at_root), "root masses")
            _require(rr["root_active_coordinates"] == active, "root active coordinates")
            _require((root == floor and at_floor >= row.lower) or (root > floor and at_root == row.lower and bool(active)), "first feasible root witness")
            roots.append(root)
            capped_mass = _dot(row.coefficients, h)
            _require(rr["capped_mass"] == _q(capped_mass), "capped row mass")
            _require(rr["target_mass"] == _q(min(row.upper, capped_mass)), "knapsack target mass")
            _require(rr["point_mass"] == _q(row_values[j]) and row_values[j] == min(row.upper, capped_mass), "knapsack attained mass")
            price = _fraction(rr["upper_price"], "row.upper_price")
            low = _fraction(rr["lower_price"], "row.lower_price")
            _require(price >= 0 and low >= 0, "local nonnegative prices")
            _require(lam[j] == w * price and eta[j] == w * low, "original dual context scaling")
            prices[j], lows[j] = price, low
            for i in row.support:
                residual[i] -= price * row.coefficients[i]
        _require(cap == max([floor, *roots]), "cap is maximum row root and floor")
        _require(cx["residual_rewards"] == [_q(residual[i]) for i in indices], "residual rewards")
        initial = {i: max(residual[i], _ZERO) if data.upper[i] >= cap else _ZERO for i in indices}
        deficit = _ONE - sum(initial.values(), _ZERO)
        _require(deficit >= 0 and cx["deficit"] == _q(deficit), "policy deficit")
        _require(cx["initial_policy"] == [_q(initial[i]) for i in indices], "initial policy")
        expected_q = initial.copy()
        witness = cx["cap_witness"]
        if witness["kind"] == "box_lower":
            k = witness["coordinate"]
            _require(cap == floor and k in indices and data.lower[k] == cap, "box lower cap witness")
            _require(all(v == 0 for v in lows.values()), "case A lower prices")
            expected_q[k] += deficit
        elif witness["kind"] == "moment_lower":
            j = witness["row_index"]
            _require(cap > floor and j in prices, "case B witness row ownership")
            row = data.rows[j]
            active = [i for i in row.support if data.upper[i] >= cap]
            active_sum = sum((row.coefficients[i] for i in active), _ZERO)
            _require(_mass_at_cap(row, data.upper, cap) == row.lower and active_sum > 0, "case B minimal cap")
            _require(prices[j] == 0, "case B witness upper price must vanish")
            _require(witness["active_coordinates"] == active and witness["active_coefficient_sum"] == _q(active_sum), "case B active witness")
            _require(all(v == (deficit / active_sum if k == j else 0) for k, v in lows.items()), "case B lower prices")
            for i in active:
                expected_q[i] += lows[j] * row.coefficients[i]
        else:
            raise CertificateConstructionError("unknown cap witness kind")
        _require(tuple(expected_q[i] for i in indices) == q, "D completion policy")
        local_m = tuple(m[i] for i in indices)
        mean = _dot(p, local_m)
        capped_dual = sum((price * data.rows[j].upper for j, price in prices.items()), _ZERO)
        capped_dual += sum((max(residual[i] * data.lower[i], residual[i] * h[i]) for i in indices), _ZERO)
        _require(capped_dual == mean, "capped knapsack exact dual equality")
        _require(cx["reference_mean"] == _q(mean) and cx["capped_box_dual_value"] == _q(capped_dual), "capped objective metadata")
        _require(_dot(q, local_m) == cap, "policy supported on attaining actions")
        capacity = cap - mean
        _require(capacity >= 0 and cx["capacity"] == _q(capacity) and cx["weighted_capacity"] == _q(w * capacity), "context capacity")
        c.extend(w * (qi - pi) for qi, pi in zip(q, p, strict=True))
        total += w * capacity
    z = [c[i] + sum((row.coefficients[i] * (lam[j] - eta[j]) for j, row in enumerate(data.rows)), _ZERO) for i in range(n)]
    for i in range(n):
        _require(z[i] * m[i] == min(z[i] * data.lower[i], z[i] * data.upper[i]), f"original box minimizing condition {i}")
    lower = sum((row.lower * eta[j] - row.upper * lam[j] for j, row in enumerate(data.rows)), _ZERO)
    lower += sum((min(z[i] * data.lower[i], z[i] * data.upper[i]) for i in range(n)), _ZERO)
    attained = _dot(tuple(c), m)
    _require(lower == attained == total, "original weak-duality lower = policy point = F(m*)")
    _require(certificate["objective_coefficients"] == list(map(_q, c)) and certificate["box_reduced_costs"] == list(map(_q, z)), "global objective coefficients")
    for key in ("capacity", "common_point_upper_bound", "policy_at_point", "policy_dual_lower_bound"):
        _require(certificate[key] == _q(total), f"reported {key} exact equality")
