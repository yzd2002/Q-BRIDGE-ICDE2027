"""Independent synthetic tests; no producer, SoPlex, partition core, or real data.

The tiny oracle enumerates rational vertices of the full epigraph/fixed LP and
solves its active-inequality dual, independently of the receiver's box support
formula.  It is deliberately restricted to two through four mean coordinates.
"""
from copy import deepcopy
from decimal import Decimal, Inexact, Rounded, localcontext
from fractions import Fraction as Q
import hashlib
from itertools import combinations
import json
import sys
import unittest
from unittest.mock import patch

from icde2027.family_lp_v7 import verify_general as receiver


def text(q):
    q = Q(q)
    return format(Decimal(q.numerator), "f") + "/" + format(Decimal(q.denominator), "f")


def dot(a, b):
    return sum((Q(x) * Q(y) for x, y in zip(a, b)), Q())


def linear_solve(matrix, rhs):
    n = len(rhs)
    table = [[Q(v) for v in row] + [Q(v)] for row, v in zip(matrix, rhs)]
    for k in range(n):
        pivot = next((i for i in range(k, n) if table[i][k]), None)
        if pivot is None:
            return None
        table[k], table[pivot] = table[pivot], table[k]
        p = table[k][k]
        table[k] = [v / p for v in table[k]]
        for i in range(n):
            if i != k:
                p = table[i][k]
                table[i] = [x - p * y for x, y in zip(table[i], table[k])]
    return tuple(row[-1] for row in table)


def original_inequalities(program, total_variables):
    d = len(program["lower"])
    inequalities = []
    for i, (lo, hi) in enumerate(zip(program["lower"], program["upper"])):
        row = [Q()] * total_variables
        row[i] = Q(1)
        inequalities.append((row, Q(hi), ("box_upper", i)))
        inequalities.append(([-a for a in row], -Q(lo), ("box_lower", i)))
    for j, source in enumerate(program["rows"]):
        row = [Q(v) for v in source["coefficients"]] + [Q()] * (total_variables - d)
        inequalities.append((row, Q(source["upper"]), ("moment_upper", j)))
        inequalities.append(([-a for a in row], -Q(source["lower"]), ("moment_lower", j)))
    return inequalities


def tiny_vertex_oracle(inequalities, objective):
    """Return an optimum and full nonnegative inequality dual; exact only."""
    n = len(objective)
    best = None
    best_point = None
    for selected in combinations(range(len(inequalities)), n):
        point = linear_solve([inequalities[i][0] for i in selected],
                             [inequalities[i][1] for i in selected])
        if point is None or any(dot(row, point) > rhs for row, rhs, _tag in inequalities):
            continue
        value = dot(objective, point)
        if best is None or value < best:
            best, best_point = value, point
    if best is None:
        raise AssertionError("synthetic oracle has no feasible vertex")
    active = [i for i, (row, rhs, _tag) in enumerate(inequalities) if dot(row, best_point) == rhs]
    for selected in combinations(active, n):
        transpose = [[inequalities[i][0][j] for i in selected] for j in range(n)]
        multipliers = linear_solve(transpose, [-Q(v) for v in objective])
        if multipliers is None or any(v < 0 for v in multipliers):
            continue
        full = [Q()] * len(inequalities)
        for i, v in zip(selected, multipliers):
            full[i] = v
        assert -sum((v * row[1] for v, row in zip(full, inequalities)), Q()) == best
        return best_point, best, full
    raise AssertionError("synthetic oracle could not find an exact active dual")


def program(lo, hi, rows=()):
    return {"lower": list(map(Q, lo)), "upper": list(map(Q, hi)),
            "rows": [{"coefficients": list(map(Q, a)), "lower": Q(l), "upper": Q(u)}
                     for a, l, u in rows]}


FIXTURES = [
    ("positive_box", program([Q(2, 3), 0], [1, Q(1, 3)]), [2], [1], [[Q(1, 2)] * 2]),
    ("equality_tie", program([Q(1, 2), 0], [1, 1], [([1, 1], 1, 1)]), [2], [1], [[Q(1, 2)] * 2]),
    ("signed_overlapping", program([0] * 3, [1] * 3, [([1, 1, 0], 1, 1), ([0, 1, 1], 1, 1), ([1, 0, -1], 0, 0)]),
     [3], [1], [[Q(1, 3)] * 3]),
    ("cross_context", program([0] * 4, [1] * 4, [([1, 0, -1, 0], 0, 0), ([0, 1, 0, 1], 1, 1)]),
     [2, 2], [Q(1, 3), Q(2, 3)], [[1, 0], [0, 1]]),
    ("zero_weight", program([0, 0, Q(1, 3)], [1, 1, Q(1, 3)], [([1, -1, 1], 0, Q(1, 2))]),
     [2, 1], [0, 1], [[0, 1], [1]]),
    ("near_coincident", program([0, 0], [1, 1], [([1, 1], 1, 1),
         ([1, 1 + Q(1, 2**60)], 1 + Q(1, 3 * 2**60), 1 + Q(1, 3 * 2**60))]),
     [2], [1], [[Q(1, 2)] * 2]),
    ("fixed_box_zero_row", program([Q(1, 3), Q(2, 3), 0], [Q(1, 3), Q(2, 3), 0],
         [([0, 0, 0], 0, 0), ([1, -1, 0], -Q(1, 3), -Q(1, 3))]),
     [3], [1], [[Q(1, 3)] * 3]),
    ("duplicate_rows", program([0, 0], [1, 1], [([1, -1], Q(1, 3), Q(1, 3))] * 2),
     [2], [1], [[Q(1, 2)] * 2]),
    ("zero_reference", program([0, Q(2, 3)], [Q(1, 3), 1]), [2], [1], [[1, 0]]),
    ("negative_row_upper", program([0, 0], [1, 1], [([-1, 2], -1, Q(1, 4))]),
     [2], [1], [[Q(1, 3), Q(2, 3)]]),
    ("singleton_context", program([0, Q(2, 3), 0], [1, 1, Q(1, 3)], [([1, -1, 0], 0, Q(1, 4))]),
     [1, 2], [Q(1, 4), Q(3, 4)], [[1], [0, 1]]),
    ("strict_difference", program([0, 0], [1, 1], [([1, -1], Q(1, 4), 1)]),
     [2], [1], [[Q(1, 2)] * 2]),
]


def family_oracle_certificate(fixture):
    _name, p, sizes, weights, refs = fixture
    d, k = len(p["lower"]), len(sizes)
    inequalities = original_inequalities(p, d + k)
    objective = []
    offset = 0
    for x, size in enumerate(sizes):
        objective.extend(-Q(weights[x]) * Q(v) for v in refs[x])
        for i in range(offset, offset + size):
            row = [Q()] * (d + k)
            row[i], row[d + x] = Q(1), Q(-1)
            inequalities.append((row, Q(), ("epigraph", i)))
        offset += size
    objective.extend(map(Q, weights))
    point, value, dual = tiny_vertex_oracle(inequalities, objective)
    eta, lam, gamma = [Q()] * len(p["rows"]), [Q()] * len(p["rows"]), [Q()] * d
    for (_row, _rhs, (kind, i)), v in zip(inequalities, dual):
        if kind == "moment_lower": eta[i] += v
        if kind == "moment_upper": lam[i] += v
        if kind == "epigraph": gamma[i] += v
    policy = []
    offset = 0
    for x, size in enumerate(sizes):
        row = [v / Q(weights[x]) for v in gamma[offset:offset + size]] if weights[x] else list(map(Q, refs[x]))
        assert sum(row, Q()) == 1
        policy.append(list(map(text, row)))
        offset += size
    return {"schema_version": receiver.FAMILY_CERTIFICATE_SCHEMA,
            "input_sha256": receiver.family_input_digest(p, sizes, weights, refs),
            "point": list(map(text, point[:d])), "policy": policy,
            "original_row_lower_dual": list(map(text, eta)),
            "original_row_upper_dual": list(map(text, lam)), "capacity": text(value)}, value


def fixed_oracle_certificate(p, objective):
    inequalities = original_inequalities(p, len(objective))
    point, value, dual = tiny_vertex_oracle(inequalities, objective)
    eta, lam = [Q()] * len(p["rows"]), [Q()] * len(p["rows"])
    for (_row, _rhs, (kind, i)), v in zip(inequalities, dual):
        if kind == "moment_lower": eta[i] += v
        if kind == "moment_upper": lam[i] += v
    return {"schema_version": receiver.FIXED_CERTIFICATE_SCHEMA,
            "input_sha256": receiver.fixed_input_digest(p, objective), "point": list(map(text, point)),
            "original_row_lower_dual": list(map(text, eta)),
            "original_row_upper_dual": list(map(text, lam)), "optimum": text(value)}, value


class ExactOracleTests(unittest.TestCase):
    pass


def family_test(fixture):
    def run(self):
        cert, value = family_oracle_certificate(fixture)
        _name, p, sizes, weights, refs = fixture
        result = receiver.verify_family_certificate(p, sizes, weights, refs, cert,
                                                    expected_input_sha256=receiver.family_input_digest(p, sizes, weights, refs))
        self.assertTrue(result["verified"])
        self.assertEqual(result["capacity"], text(value))
        self.assertEqual(result["unchecked_fields"], [])
        if fixture[0] == "near_coincident": self.assertEqual(value, Q(1, 6))
    return run


def fixed_test(fixture, index):
    def run(self):
        p = fixture[1]
        objective = [Q(((i + 1) * (index + 2)) % 7 - 3, i + 1) for i in range(len(p["lower"]))]
        cert, value = fixed_oracle_certificate(p, objective)
        result = receiver.verify_fixed_certificate(p, objective, cert,
                                                   expected_input_sha256=receiver.fixed_input_digest(p, objective))
        self.assertEqual(result["optimum"], text(value))
        self.assertEqual(result["computed_original_dual_lower_bound"], text(value))
    return run


for index, fixture in enumerate(FIXTURES):
    setattr(ExactOracleTests, "test_family_" + fixture[0], family_test(fixture))
    setattr(ExactOracleTests, "test_fixed_" + fixture[0], fixed_test(fixture, index))


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = FIXTURES[0]
        cls.family, _ = family_oracle_certificate(cls.fixture)
        cls.objective = [Q(-2, 3), Q(1, 7)]
        cls.fixed, _ = fixed_oracle_certificate(cls.fixture[1], cls.objective)

    def family_check(self, certificate=None, fixture=None, digest=None):
        _name, p, sizes, weights, refs = self.fixture if fixture is None else fixture
        cert = self.family if certificate is None else certificate
        return receiver.verify_family_certificate(p, sizes, weights, refs, cert,
            expected_input_sha256=receiver.family_input_digest(p, sizes, weights, refs) if digest is None else digest)

    def fixed_check(self, certificate=None, p=None, c=None, digest=None):
        cert = self.fixed if certificate is None else certificate
        actual_p = self.fixture[1] if p is None else p
        actual_c = self.objective if c is None else c
        return receiver.verify_fixed_certificate(actual_p, actual_c, cert,
            expected_input_sha256=receiver.fixed_input_digest(actual_p, actual_c) if digest is None else digest)

    def test_external_digest_is_required(self):
        with self.assertRaises(TypeError):
            receiver.verify_family_certificate(*self.fixture[1:], self.family)
        with self.assertRaises(TypeError):
            receiver.verify_fixed_certificate(self.fixture[1], self.objective, self.fixed)

    def test_external_digest_mismatch_and_shape(self):
        for digest in ["0" * 64, "A" * 64, "", None, 1, True]:
            with self.subTest(digest=digest), self.assertRaises(receiver.RequestValidationError):
                receiver.verify_family_certificate(*self.fixture[1:], self.family, expected_input_sha256=digest)

    def test_certificate_digest_cannot_authorize_other_request(self):
        changed = deepcopy(self.fixture)
        changed[1]["lower"][0] = Q(3, 4)
        other_digest = receiver.family_input_digest(*changed[1:])
        cert = deepcopy(self.family)
        cert["input_sha256"] = other_digest
        with self.assertRaises(receiver.CertificateValidationError):
            self.family_check(cert, changed, other_digest)
        with self.assertRaises(receiver.RequestValidationError):
            self.family_check(cert, self.fixture, other_digest)

    def test_fixed_objective_is_externally_bound(self):
        changed = [v + 1 for v in self.objective]
        with self.assertRaises(receiver.RequestValidationError):
            self.fixed_check(c=changed, digest=self.fixed["input_sha256"])
        cert = deepcopy(self.fixed)
        cert["input_sha256"] = receiver.fixed_input_digest(self.fixture[1], changed)
        with self.assertRaises(receiver.CertificateValidationError):
            self.fixed_check(cert, c=changed)

    def test_reference_and_weights_are_digest_bound(self):
        for location in ("reference", "sizes"):
            changed = deepcopy(self.fixture)
            if location == "reference": changed[4][0] = [Q(1, 3), Q(2, 3)]
            else: changed = (changed[0], changed[1], [1, 1], [Q(1, 2)] * 2, [[1], [1]])
            with self.subTest(location=location), self.assertRaises(receiver.RequestValidationError):
                self.family_check(fixture=changed, digest=self.family["input_sha256"])

    def test_no_unknown_certificate_fields(self):
        for key, value in [("canonical_input", {}), ("cap", "1/1"), ("internal_check", {"passed": True})]:
            bad = {**self.family, key: value}
            with self.subTest(key=key), self.assertRaises(receiver.CertificateValidationError):
                self.family_check(bad)

    def test_missing_fields_and_wrong_schemas(self):
        for key in self.family:
            bad = dict(self.family)
            del bad[key]
            with self.subTest(key=key), self.assertRaises(receiver.CertificateValidationError):
                self.family_check(bad)
        for schema in [receiver.FIXED_CERTIFICATE_SCHEMA, "partition-capacity-v6/1", "family-capacity-certificate/2", 1]:
            with self.subTest(schema=schema), self.assertRaises(receiver.CertificateValidationError):
                self.family_check({**self.family, "schema_version": schema})

    def test_wire_numbers_are_strict_and_reduced(self):
        bad_values = [0, Q(0), 0.0, True, None, "0", "0/2", "-0/1", "01/1", "+1/1", "1/-2", "1/0",
                      "2/4", " 0/1", "0/1\n", "NaN/1", "Infinity", "１/1", "1e0/1", "0/01"]
        for value in bad_values:
            bad = deepcopy(self.family)
            bad["capacity"] = value
            with self.subTest(value=repr(value)), self.assertRaises(receiver.CertificateValidationError):
                self.family_check(bad)

    def test_claim_point_and_policy_tampering(self):
        mutations = [lambda c: c.update(capacity="0/1"),
                     lambda c: c["point"].__setitem__(0, "1/2"),
                     lambda c: c["policy"].__setitem__(0, ["1/2", "1/2"]),
                     lambda c: c["policy"].__setitem__(0, ["2/1", "-1/1"]),
                     lambda c: c["policy"].__setitem__(0, ["1/2", "1/3"])]
        for i, mutate in enumerate(mutations):
            bad = deepcopy(self.family); mutate(bad)
            with self.subTest(i=i), self.assertRaises(receiver.CertificateValidationError):
                self.family_check(bad)

    def test_moment_duals_tampering(self):
        fixture = FIXTURES[-1]
        cert, _ = family_oracle_certificate(fixture)
        for lower, upper in [("-1/1", "0/1"), ("0/1", "-1/1"), ("0/1", "0/1"), ("10/1", "0/1")]:
            bad = deepcopy(cert)
            bad["original_row_lower_dual"] = [lower]
            bad["original_row_upper_dual"] = [upper]
            with self.subTest(lower=lower, upper=upper), self.assertRaises(receiver.CertificateValidationError):
                self.family_check(bad, fixture)

    def test_zero_weight_point_still_checked(self):
        fixture = FIXTURES[4]
        cert, _ = family_oracle_certificate(fixture)
        bad = deepcopy(cert); bad["point"][0] = "1/1"; bad["point"][1] = "0/1"
        with self.assertRaises(receiver.CertificateValidationError): self.family_check(bad, fixture)
        bad = deepcopy(cert); bad["policy"][0] = ["2/1", "-1/1"]
        with self.assertRaises(receiver.CertificateValidationError): self.family_check(bad, fixture)

    def test_near_coincident_equality_cannot_be_rounded(self):
        fixture = FIXTURES[5]
        cert, _ = family_oracle_certificate(fixture)
        bad = deepcopy(cert); bad["point"] = ["1/2", "1/2"]
        with self.assertRaises(receiver.CertificateValidationError): self.family_check(bad, fixture)

    def test_structurally_legal_empty_set_never_claims_infeasible(self):
        p = program([0], [1], [([0], 1, 2)])
        digest = receiver.fixed_input_digest(p, [0])
        cert = {"schema_version": receiver.FIXED_CERTIFICATE_SCHEMA, "input_sha256": digest,
                "point": ["0/1"], "original_row_lower_dual": ["0/1"],
                "original_row_upper_dual": ["0/1"], "optimum": "0/1"}
        with self.assertRaises(receiver.CertificateValidationError):
            receiver.verify_fixed_certificate(p, [0], cert, expected_input_sha256=digest)

    def test_zero_objective_and_two_positive_equality_multipliers(self):
        p = program([0], [1], [([1], Q(1, 3), Q(1, 3))])
        digest = receiver.fixed_input_digest(p, [0])
        cert = {"schema_version": receiver.FIXED_CERTIFICATE_SCHEMA, "input_sha256": digest,
                "point": ["1/3"], "original_row_lower_dual": ["7/1"],
                "original_row_upper_dual": ["7/1"], "optimum": "0/1"}
        receipt = receiver.verify_fixed_certificate(p, [0], cert, expected_input_sha256=digest)
        self.assertEqual(receipt["computed_original_dual_lower_bound"], "0/1")

    def test_trusted_scalar_types_and_no_normalization(self):
        for value in [True, 0.0, float("nan"), float("inf"), Decimal(0), "0.0", "2/4"]:
            p = deepcopy(self.fixture[1]); p["lower"][1] = value
            with self.subTest(value=repr(value)), self.assertRaises(receiver.RequestValidationError):
                receiver.fixed_input_digest(p, self.objective)
        for weights, refs in [([2], [[Q(1, 2)] * 2]), ([0], [[Q(1, 2)] * 2]),
                              ([1], [[1, 1]]), ([1], [[-1, 2]])]:
            with self.subTest(weights=weights, refs=refs), self.assertRaises(receiver.RequestValidationError):
                receiver.family_input_digest(self.fixture[1], [2], weights, refs)

    def test_program_and_context_shape(self):
        mutations = [lambda p: p.update(extra=1), lambda p: p.update(lower=[]),
                     lambda p: p.update(upper=[1]), lambda p: p["lower"].__setitem__(0, -1),
                     lambda p: p["upper"].__setitem__(1, 2),
                     lambda p: p["lower"].__setitem__(1, 1),
                     lambda p: p.update(rows=[{"coefficients": [1], "lower": 0, "upper": 1}]),
                     lambda p: p.update(rows=[{"coefficients": [1, 1], "lower": 2, "upper": 1}])]
        for i, mutate in enumerate(mutations):
            p = deepcopy(self.fixture[1]); mutate(p)
            with self.subTest(i=i), self.assertRaises(receiver.RequestValidationError):
                receiver.fixed_input_digest(p, self.objective)
        for sizes in [[], [0, 2], [True, 1], [1.0, 1], [1], [3], [2, 1]]:
            with self.subTest(sizes=sizes), self.assertRaises(receiver.RequestValidationError):
                receiver.family_input_digest(self.fixture[1], sizes, [1], [[Q(1, 2)] * 2])

    def test_exact_int_fraction_and_text_have_same_digest(self):
        p = program([0], [1], [([-2], -2, 0)])
        encoded = {"lower": ["0/1"], "upper": ["1/1"],
                   "rows": [{"coefficients": ["-2/1"], "lower": "-2/1", "upper": "0/1"}]}
        self.assertEqual(receiver.fixed_input_digest(p, [Q(1, 3)]), receiver.fixed_input_digest(encoded, ["1/3"]))

    def test_digest_layout_independently_recomputed(self):
        p = {"lower": [0], "upper": [1], "rows": []}
        canonical = {"schema_version": "family-capacity-input/1", "program": {"lower": ["0/1"], "upper": ["1/1"], "rows": []},
                     "context_sizes": [1], "weights": ["1/1"], "reference": [["1/1"]]}
        expected = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        self.assertEqual(receiver.family_input_digest(p, [1], [1], [[1]]), expected)
        self.assertNotEqual(expected, receiver.fixed_input_digest(p, [0]))

    def test_does_not_mutate_caller_objects_or_numeric_settings(self):
        before = deepcopy((self.fixture, self.family, self.fixed))
        limit = sys.get_int_max_str_digits()
        with localcontext() as context:
            context.prec = 1; context.traps[Inexact] = True; context.traps[Rounded] = True
            snapshot = context.copy()
            self.family_check(); self.fixed_check()
            self.assertEqual(context.prec, snapshot.prec)
            self.assertEqual(context.flags, snapshot.flags)
        self.assertEqual(before, (self.fixture, self.family, self.fixed))
        self.assertEqual(sys.get_int_max_str_digits(), limit)

    def test_large_allowed_rational_without_global_guard_change(self):
        denominator = 10**99_999 + 1
        c = [Q(1, denominator)]
        p = program([0], [0], [([0], 0, 0)])
        limit = sys.get_int_max_str_digits()
        digest = receiver.fixed_input_digest(p, c)
        cert = {"schema_version": receiver.FIXED_CERTIFICATE_SCHEMA, "input_sha256": digest,
                "point": ["0/1"], "original_row_lower_dual": [text(c[0])],
                "original_row_upper_dual": ["0/1"], "optimum": "0/1"}
        result = receiver.verify_fixed_certificate(p, c, cert, expected_input_sha256=digest)
        self.assertTrue(result["verified"])
        self.assertEqual(sys.get_int_max_str_digits(), limit)

    def test_oversized_rational_is_rejected(self):
        p = program([0], [1])
        for value in ["1/" + "1" * 100_001, 10**100_000, Q(1, 10**100_000)]:
            with self.subTest(kind=type(value).__name__), self.assertRaises(receiver.RequestValidationError):
                receiver.fixed_input_digest(p, [value])
        bad = deepcopy(self.fixed); bad["optimum"] = "1/" + "1" * 100_001
        with self.assertRaises(receiver.CertificateValidationError): self.fixed_check(bad)

    def test_dimension_row_and_coefficient_limits(self):
        with self.assertRaises(receiver.RequestValidationError):
            receiver.fixed_input_digest({"lower": [0] * 4097, "upper": [1] * 4097, "rows": []}, [0] * 4097)
        with self.assertRaises(receiver.RequestValidationError):
            receiver.fixed_input_digest({"lower": [0], "upper": [1], "rows": [{}] * 4097}, [0])
        with self.assertRaises(receiver.RequestValidationError):
            receiver.fixed_input_digest({"lower": [0] * 4096, "upper": [1] * 4096, "rows": [{}] * 245}, [0] * 4096)

    def test_aggregate_character_limit(self):
        with patch.object(receiver, "MAX_TOTAL_RATIONAL_CHARACTERS", 10):
            with self.assertRaises(receiver.RequestValidationError):
                receiver.fixed_input_digest(self.fixture[1], self.objective)

    def test_strict_json_rejects_duplicate_keys_at_any_depth(self):
        for source in ['{"a":1,"a":2}', '{"rows":[{"lower":"0/1","lower":"1/1"}]}']:
            with self.subTest(source=source), self.assertRaises(receiver.WireFormatError):
                receiver.strict_json_loads(source)

    def test_strict_json_rejects_float_nonfinite_and_malformed(self):
        for source in ["0.0", "1e0", "NaN", "Infinity", "-Infinity", "-0", "{", "[1,]", b"\xff"]:
            with self.subTest(source=source), self.assertRaises(receiver.WireFormatError):
                receiver.strict_json_loads(source)
        self.assertEqual(receiver.strict_json_loads(b'{"n":-2,"s":"1/3"}'), {"n": -2, "s": "1/3"})

    def test_strict_json_size_depth_and_nodes(self):
        with patch.object(receiver, "MAX_JSON_BYTES", 5):
            with self.assertRaises(receiver.WireFormatError): receiver.strict_json_loads('"abcdef"')
        with self.assertRaises(receiver.WireFormatError):
            receiver.strict_json_loads("[" * 34 + "0" + "]" * 34)
        with patch.object(receiver, "MAX_JSON_NODES", 3):
            with self.assertRaises(receiver.WireFormatError): receiver.strict_json_loads("[0,1,2]")

    def test_strict_wire_round_trip_and_receipt_scope(self):
        cert = receiver.strict_json_loads(json.dumps(self.family).encode("utf-8"))
        receipt = self.family_check(cert)
        self.assertEqual(receipt["digest_authority"], "trusted-request-and-caller-digest")
        self.assertEqual(receipt["scope"], "general-original-program-family-optimality")
        self.assertEqual(set(receipt["checked_fields"]), set(cert))
        self.assertEqual(receipt["unchecked_fields"], [])


if __name__ == "__main__":
    unittest.main()
