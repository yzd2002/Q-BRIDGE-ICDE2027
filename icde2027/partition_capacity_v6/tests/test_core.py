"""Producer-authored synthetic tests, not an independent mathematical oracle."""

import copy
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import json
import random
import unittest
from unittest.mock import patch

from icde2027.partition_capacity_v6 import (
    CERTIFICATE_VERSION,
    CertificateConstructionError,
    InfeasibleProgramError,
    InvalidInputError,
    UnsupportedStructureError,
    solve,
)
from icde2027.partition_capacity_v6.core import _self_check


def program(lower, upper, rows=()):
    return {"lower": lower, "upper": upper, "rows": [
        {"coefficients": t, "lower": a, "upper": b} for t, a, b in rows
    ]}


def case_b():
    return program([0, 0], [F(1, 3), 1], [([1, 1], 1, 1)]), [2], [1], [[F(1, 2), F(1, 2)]]


def evaluate_original(request, cert):
    """Direct test arithmetic, deliberately no cap/root/fill implementation."""
    raw, sizes, weights, reference = request
    m = list(map(F, cert["point"]))
    q = [list(map(F, row)) for row in cert["policy"]]
    eta = list(map(F, cert["original_row_lower_dual"]))
    lam = list(map(F, cert["original_row_upper_dual"]))
    lo, hi = list(map(F, raw["lower"])), list(map(F, raw["upper"]))
    c = []
    upper = F(0)
    offset = 0
    for x, size in enumerate(sizes):
        assert len(q[x]) == size and sum(q[x]) == 1 and min(q[x]) >= 0
        local = m[offset:offset + size]
        upper += F(weights[x]) * (max(local) - sum(F(p) * v for p, v in zip(reference[x], local, strict=True)))
        c.extend(F(weights[x]) * (a - F(p)) for a, p in zip(q[x], reference[x], strict=True))
        offset += size
    assert all(a <= v <= b for a, v, b in zip(lo, m, hi, strict=True))
    z = c.copy()
    lower = F(0)
    for j, row in enumerate(raw["rows"]):
        t = list(map(F, row["coefficients"]))
        value = sum(a * v for a, v in zip(t, m, strict=True))
        assert F(row["lower"]) <= value <= F(row["upper"])
        assert eta[j] >= 0 and lam[j] >= 0
        lower += F(row["lower"]) * eta[j] - F(row["upper"]) * lam[j]
        z = [r + a * (lam[j] - eta[j]) for r, a in zip(z, t, strict=True)]
    lower += sum(min(r * a, r * b) for r, a, b in zip(z, lo, hi, strict=True))
    assert lower == upper == F(cert["capacity"])
    assert sum(a * v for a, v in zip(c, m, strict=True)) == upper
    return c, upper


class AnalyticExamples(unittest.TestCase):
    def check(self, request, capacity):
        cert = solve(*request)
        self.assertEqual(F(cert["capacity"]), F(capacity))
        evaluate_original(request, cert)
        return cert

    def test_unconstrained_common_zero(self):
        cert = self.check((program([0, 0], [1, 1]), [2], [1], [[F(1, 2), F(1, 2)]]), 0)
        self.assertEqual(cert["point"], ["0/1", "0/1"])
        self.assertEqual(cert["policy"], [["1/2", "1/2"]])

    def test_box_witness_positive_deficit_and_unused_coordinates(self):
        request = (program([F(2, 3), 0, 0], [1, F(1, 3), F(1, 4)]), [3], [1], [[0, F(1, 2), F(1, 2)]])
        cert = self.check(request, F(3, 8))
        self.assertEqual(cert["contexts"][0]["cap_witness"], {"kind": "box_lower", "coordinate": 0})
        self.assertEqual(cert["policy"], [["1/1", "0/1", "0/1"]])
        self.assertEqual(cert["contexts"][0]["deficit"], "1/1")

    def test_moment_witness_positive_deficit(self):
        cert = self.check(case_b(), F(1, 6))
        self.assertEqual(cert["point"], ["1/3", "2/3"])
        self.assertEqual(cert["policy"], [["0/1", "1/1"]])
        self.assertEqual(cert["original_row_lower_dual"], ["1/2"])
        self.assertEqual(cert["contexts"][0]["cap_witness"]["active_coordinates"], [1])

    def test_non_dyadic_nonuniform_coefficients(self):
        request = (program([0, 0], [F(2, 5), 1], [([F(2, 3), F(5, 7)], F(2, 3), F(2, 3))]), [2], [1], [[F(1, 3), F(2, 3)]])
        cert = self.check(request, F(4, 75))
        self.assertEqual(cert["contexts"][0]["cap"], "14/25")
        self.assertEqual(cert["original_row_lower_dual"], ["7/15"])

    def test_budget_ends_exactly_after_coordinate_fill(self):
        request = (program([0, 0, 1], [1, 1, 1], [([1, 1, 0], 0, 1)]), [3], [1], [[F(3, 5), F(2, 5), 0]])
        cert = self.check(request, F(2, 5))
        self.assertEqual(cert["point"], ["1/1", "0/1", "1/1"])
        self.assertEqual(cert["original_row_upper_dual"], ["3/5"])

    def test_fractional_knapsack_fill(self):
        request = (program([0, 0, 0, 1], [1, 1, 1, 1], [([1, 2, 3, 0], 0, F(5, 2))]), [4], [1], [[F(1, 2), F(3, 10), F(1, 5), 0]])
        cert = self.check(request, F(11, 40))
        self.assertEqual(cert["point"], ["1/1", "3/4", "0/1", "1/1"])
        self.assertEqual(cert["original_row_upper_dual"], ["3/20"])
        self.assertEqual(cert["policy"], [["7/20", "0/1", "0/1", "13/20"]])

    def test_zero_budget_includes_fixed_high_ratio_coordinate(self):
        request = (program([F(1, 4), 0, 1], [F(1, 4), 1, 1], [([1, 1, 0], 0, F(1, 4))]), [3], [1], [[F(4, 5), F(1, 5), 0]])
        cert = self.check(request, F(4, 5))
        self.assertEqual(cert["original_row_upper_dual"], ["4/5"])

    def test_ratio_ties_have_index_order(self):
        request = (program([0, 0, 1], [1, 1, 1], [([1, 1, 0], 0, 1)]), [3], [1], [[F(1, 2), F(1, 2), 0]])
        cert = self.check(request, F(1, 2))
        self.assertEqual(cert["point"], ["1/1", "0/1", "1/1"])
        self.assertEqual(cert["original_row_upper_dual"], ["1/2"])

    def test_fixed_box(self):
        request = (program([F(1, 5), F(7, 10)], [F(1, 5), F(7, 10)]), [2], [1], [[F(2, 3), F(1, 3)]])
        self.check(request, F(1, 3))

    def test_final_upper_breakpoint_is_a_valid_root(self):
        request = (program([0, 0], [F(1, 3), F(2, 3)], [([1, 1], 1, 1)]), [2], [1], [[F(1, 2), F(1, 2)]])
        cert = self.check(request, F(1, 6))
        rr = cert["contexts"][0]["rows"][0]
        self.assertEqual(rr["root"], "2/3")
        self.assertEqual(rr["root_active_coordinates"], [1])

    def test_tied_cap_roots_use_first_original_row(self):
        request = (program([0] * 4, [F(1, 3), 1, F(1, 3), 1], [([1, 1, 0, 0], 1, 1), ([0, 0, 1, 1], 1, 1)]), [4], [1], [[F(1, 4)] * 4])
        cert = self.check(request, F(1, 6))
        self.assertEqual(cert["contexts"][0]["cap_witness"]["row_index"], 0)
        self.assertEqual(cert["policy"], [["0/1", "3/4", "0/1", "1/4"]])

    def test_multicontext_weighted_duals_and_capacity(self):
        request = (program([0, 0, F(2, 3), 0, 0], [F(1, 3), 1, 1, F(1, 3), F(1, 4)], [([1, 1, 0, 0, 0], 1, 1)]), [2, 3], [F(2, 7), F(5, 7)], [[F(1, 2), F(1, 2)], [0, F(1, 2), F(1, 2)]])
        cert = self.check(request, F(53, 168))
        self.assertEqual(cert["original_row_lower_dual"], ["1/7"])

    def test_zero_context_weight_preserves_local_certificate(self):
        request = (program([0, 0, 1], [F(1, 3), 1, 1], [([1, 1, 0], 1, 1)]), [2, 1], [0, 1], [[F(1, 2), F(1, 2)], [1]])
        cert = self.check(request, 0)
        self.assertEqual(cert["original_row_lower_dual"], ["0/1"])
        self.assertEqual(cert["contexts"][0]["rows"][0]["lower_price"], "1/2")

    def test_single_action_has_zero_capacity(self):
        request = (program([0], [1], [([F(3, 7)], F(1, 7), F(2, 7))]), [1], [1], [[1]])
        cert = self.check(request, 0)
        self.assertEqual(cert["point"], ["1/3"])

    def test_zero_probability_row_and_full_capped_budget(self):
        request = (program([0, 0, 1], [1, 1, 1], [([1, 1, 0], 0, 2)]), [3], [1], [[0, 0, 1]])
        cert = self.check(request, 0)
        self.assertEqual(cert["original_row_upper_dual"], ["0/1"])

    def test_zero_row_is_preserved_in_original_position(self):
        base = case_b()
        changed = copy.deepcopy(base)
        changed[0]["rows"].insert(0, {"coefficients": [0, 0], "lower": -1, "upper": 2})
        cert = self.check(changed, F(1, 6))
        self.assertEqual(cert["original_row_lower_dual"], ["0/1", "1/2"])
        self.assertEqual(cert["original_row_values"], ["0/1", "1/1"])
        self.assertEqual(cert["contexts"][0]["cap_witness"]["row_index"], 1)
        self.assertNotEqual(cert["input_sha256"], solve(*base)["input_sha256"])


class InputRejections(unittest.TestCase):
    def test_float_bool_decimal_and_ambiguous_strings_rejected(self):
        for value in (0.5, True, Decimal("0.5"), "0.5", "1e-2", " 1/2", "1/2 ", "1/0", "1/-2", "1/02", "nan", None):
            with self.subTest(value=repr(value)):
                request = list(case_b())
                request[0]["lower"][0] = value
                with self.assertRaises(InvalidInputError):
                    solve(*request)

    def test_float_rejected_in_every_numeric_part(self):
        mutations = [
            lambda r: r[0]["upper"].__setitem__(0, 0.5),
            lambda r: r[0]["rows"][0]["coefficients"].__setitem__(0, 1.0),
            lambda r: r[0]["rows"][0].__setitem__("lower", 1.0),
            lambda r: r[0]["rows"][0].__setitem__("upper", 1.0),
            lambda r: r[2].__setitem__(0, 1.0),
            lambda r: r[3][0].__setitem__(0, 0.5),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(location=index):
                request = list(case_b())
                mutate(request)
                with self.assertRaises(InvalidInputError):
                    solve(*request)

    def test_weight_probability_not_normalized_or_negative(self):
        for weights in ([2], [0], [-1], [F(999, 1000)]):
            with self.subTest(weights=weights):
                request = list(case_b())
                request[2] = weights
                with self.assertRaises(InvalidInputError):
                    solve(*request)

    def test_reference_not_normalized_or_negative(self):
        for reference in ([[1, 1]], [[0, 0]], [[-1, 2]], [[F(1, 3), F(1, 3)]]):
            with self.subTest(reference=reference):
                request = list(case_b())
                request[3] = reference
                with self.assertRaises(InvalidInputError):
                    solve(*request)

    def test_dimensions_and_container_shape(self):
        mutations = [
            lambda r: r.__setitem__(1, []),
            lambda r: r.__setitem__(1, [0, 2]),
            lambda r: r.__setitem__(1, [True, 1]),
            lambda r: r.__setitem__(1, [F(2)]),
            lambda r: r.__setitem__(1, [3]),
            lambda r: r.__setitem__(1, "2"),
            lambda r: r.__setitem__(2, []),
            lambda r: r.__setitem__(3, []),
            lambda r: r.__setitem__(3, [[1]]),
            lambda r: r[0].__setitem__("upper", [1]),
            lambda r: r[0]["rows"][0].__setitem__("coefficients", [1]),
            lambda r: r[0].__setitem__("rows", None),
            lambda r: r[0].__setitem__("metadata", "unbound"),
            lambda r: r[0]["rows"][0].__setitem__("metadata", "unbound"),
            lambda r: r[0].pop("rows"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                request = list(case_b())
                mutate(request)
                with self.assertRaises(InvalidInputError):
                    solve(*request)
        with self.assertRaises(InvalidInputError):
            solve(program([], []), [], [], [])

    def test_invalid_boxes_and_reversed_row(self):
        for lo, hi in [([-1, 0], [1, 1]), ([0, 0], [2, 1]), ([1, 0], [0, 1])]:
            with self.subTest(lo=lo, hi=hi):
                with self.assertRaises(InvalidInputError):
                    solve(program(lo, hi), [2], [1], [[1, 0]])
        with self.assertRaises(InvalidInputError):
            solve(program([0, 0], [1, 1], [([1, 1], 2, 1)]), [2], [1], [[1, 0]])

    def test_empty_supported_row_sets(self):
        requests = [
            program([0, 0], [1, 1], [([0, 0], 1, 2)]),
            program([0, 0], [1, 1], [([0, 0], -2, -1)]),
            program([F(1, 2), F(1, 2)], [1, 1], [([1, 1], 0, F(1, 2))]),
            program([0, 0], [F(1, 3), F(1, 3)], [([1, 1], 1, 2)]),
        ]
        for raw in requests:
            with self.subTest(program=raw):
                with self.assertRaises(InfeasibleProgramError):
                    solve(raw, [2], [1], [[1, 0]])

    def test_overlap_counterexample_is_unsupported(self):
        raw = program([0, 0, 0], [1, 1, 1], [([1, 0, 1], 1, 1), ([0, 1, 2], 1, 1)])
        with self.assertRaises(UnsupportedStructureError):
            solve(raw, [3], [1], [[0, 1, 0]])

    def test_signed_counterexample_is_unsupported(self):
        with self.assertRaises(UnsupportedStructureError):
            solve(program([0, 0], [1, 1], [([2, -1], 1, 1)]), [2], [1], [[0, 1]])

    def test_mixed_context_counterexample_is_unsupported(self):
        with self.assertRaises(UnsupportedStructureError):
            solve(program([0] * 4, [1, 0, 1, 0], [([1, 0, 1, 0], 1, 1)]), [2, 2], [F(1, 2)] * 2, [[0, 1], [0, 1]])

    def test_exception_classes_are_distinct(self):
        self.assertFalse(issubclass(UnsupportedStructureError, InvalidInputError))
        self.assertTrue(issubclass(InfeasibleProgramError, InvalidInputError))


class SerializationAndChecks(unittest.TestCase):
    def test_canonical_digest_and_json_roundtrip(self):
        request = case_b()
        cert = solve(*request)
        loaded = json.loads(json.dumps(cert, allow_nan=False))
        self.assertEqual(cert, loaded)
        payload = json.dumps(cert["canonical_input"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
        self.assertEqual(cert["input_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(cert["schema_version"], CERTIFICATE_VERSION)
        _self_check(*request, loaded)

    def test_equivalent_exact_literals_have_same_digest(self):
        base = case_b()
        other = list(case_b())
        other[0]["upper"][0] = "+2/6"
        other[3] = [["2/4", "3/6"]]
        other[2] = ["+01"]
        self.assertEqual(solve(*base), solve(*other))

    def test_request_is_not_mutated_or_aliased(self):
        request = case_b()
        before = copy.deepcopy(request)
        cert = solve(*request)
        self.assertEqual(request, before)
        cert["canonical_input"]["program"]["lower"][0] = "1/9"
        self.assertEqual(request, before)

    def test_tuple_vectors_are_accepted(self):
        request = (program((0, 0), (F(1, 3), 1), [((1, 1), 1, 1)]), (2,), (1,), ((F(1, 2), F(1, 2)),))
        self.assertEqual(solve(*request), solve(*case_b()))

    def test_trusted_request_binding_rejects_self_consistent_replacement(self):
        original = case_b()
        replacement = list(case_b())
        replacement[3] = [[1, 0]]
        forged_for_original = solve(*replacement)
        with self.assertRaises(CertificateConstructionError):
            _self_check(*original, forged_for_original)

    def test_original_zero_rows_cannot_disappear_from_binding(self):
        original = list(case_b())
        original[0]["rows"].append({"coefficients": [0, 0], "lower": 0, "upper": 0})
        with self.assertRaises(CertificateConstructionError):
            _self_check(*original, solve(*case_b()))

    def test_certificate_corruptions_trigger_internal_check(self):
        mutations = [
            lambda c: c.__setitem__("input_sha256", "0" * 64),
            lambda c: c["point"].__setitem__(0, "0/1"),
            lambda c: c["policy"][0].__setitem__(1, "2/1"),
            lambda c: c["original_row_lower_dual"].__setitem__(0, "-1/2"),
            lambda c: c["original_row_lower_dual"].__setitem__(0, "0/1"),
            lambda c: c.__setitem__("capacity", "1/5"),
            lambda c: c["contexts"][0]["rows"][0].__setitem__("root", "1/1"),
            lambda c: c["contexts"][0]["rows"][0].__setitem__("upper_price", "1/1"),
            lambda c: c["contexts"][0]["cap_witness"].__setitem__("active_coordinates", []),
            lambda c: c["contexts"][0].__setitem__("deficit", "0/1"),
            lambda c: c["objective_coefficients"].__setitem__(0, "0/1"),
            lambda c: c["box_reduced_costs"].__setitem__(0, "0/1"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(corruption=index):
                request = case_b()
                cert = solve(*request)
                mutate(cert)
                with self.assertRaises(CertificateConstructionError):
                    _self_check(*request, cert)

    def test_unscaled_original_duals_are_rejected(self):
        request = (program([0, 0, 1], [F(1, 3), 1, 1], [([1, 1, 0], 1, 1)]), [2, 1], [F(1, 3), F(2, 3)], [[F(1, 2), F(1, 2)], [1]])
        cert = solve(*request)
        cert["original_row_lower_dual"][0] = "1/2"
        with self.assertRaises(CertificateConstructionError):
            _self_check(*request, cert)

    def test_internal_algorithm_failure_is_not_returned_as_capacity(self):
        with patch("icde2027.partition_capacity_v6.core._knapsack", return_value=({0: F(0), 1: F(0)}, F(0))):
            with self.assertRaises(CertificateConstructionError):
                solve(*case_b())

    def test_large_exact_denominators_never_round(self):
        epsilon = F(1, 10 ** 80 + 7)
        request = (program([F(1, 2), 0], [F(1, 2), F(1, 2) - epsilon]), [2], [1], [[0, 1]])
        cert = solve(*request)
        self.assertEqual(F(cert["capacity"]), epsilon)


class SyntheticCertificates(unittest.TestCase):
    def test_240_seeded_synthetic_programs(self):
        rng = random.Random(2026110610)
        for case in range(240):
            sizes = [rng.randint(1, 7) for _ in range(rng.randint(1, 4))]
            n = sum(sizes)
            lo = [F(rng.randint(0, 12), 12) for _ in range(n)]
            hi = [F(rng.randint(int(v * 12), 12), 12) for v in lo]
            feasible = [a + (b - a) * F(rng.randint(0, 7), 7) for a, b in zip(lo, hi, strict=True)]
            rows = []
            reference = []
            start = 0
            for size in sizes:
                ids = list(range(start, start + size))
                rng.shuffle(ids)
                while ids:
                    width = rng.randint(1, len(ids))
                    selected, ids = ids[:width], ids[width:]
                    if rng.randrange(4) == 0:
                        continue
                    t = [F(0)] * n
                    for i in selected:
                        t[i] = F(rng.randint(1, 9), rng.choice([3, 5, 7, 11, 13]))
                    at_lo = sum(a * b for a, b in zip(t, lo, strict=True))
                    at_hi = sum(a * b for a, b in zip(t, hi, strict=True))
                    at_witness = sum(a * b for a, b in zip(t, feasible, strict=True))
                    a = at_lo + (at_witness - at_lo) * F(rng.randint(0, 5), 5)
                    b = at_witness + (at_hi - at_witness) * F(rng.randint(0, 5), 5)
                    rows.append((t, a, b))
                raw_p = [rng.randint(0, 5) for _ in range(size)]
                raw_p[rng.randrange(size)] += 1
                reference.append([F(p, sum(raw_p)) for p in raw_p])
                start += size
            if case % 3 == 0:
                rows.insert(rng.randint(0, len(rows)), ([0] * n, -1, 1))
            raw_w = [rng.randint(0, 5) for _ in sizes]
            raw_w[rng.randrange(len(sizes))] += 1
            weights = [F(w, sum(raw_w)) for w in raw_w]
            request = (program(lo, hi, rows), sizes, weights, reference)
            with self.subTest(case=case, sizes=sizes):
                cert = solve(*request)
                c, capacity = evaluate_original(request, cert)
                # Each convex combination is feasible; some points may coincide.
                optimum = list(map(F, cert["point"]))
                for mixing in (F(0), F(1, 3), F(3, 5), F(1)):
                    candidate = [mixing * a + (1 - mixing) * b for a, b in zip(feasible, optimum, strict=True)]
                    value = sum(a * b for a, b in zip(c, candidate, strict=True))
                    self.assertGreaterEqual(value, capacity)
                self.assertEqual(cert, solve(*request))


if __name__ == "__main__":
    unittest.main()
