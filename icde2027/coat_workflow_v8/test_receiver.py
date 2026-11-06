"""Small exact synthetic and wire-tamper cases; no optimizer or real inputs."""
from copy import deepcopy
from fractions import Fraction as Q
from itertools import combinations, product
import json
from pathlib import Path
import subprocess
import sys
import unittest

from icde2027.coat_workflow_v8 import receiver as rx


VIEW = "1" * 64


def text(value):
    value = Q(value)
    return f"{value.numerator}/{value.denominator}"


def program(lower=(0,), upper=(1,), rows=()):
    return {"lower": list(lower), "upper": list(upper), "rows": [
        {"coefficients": list(a), "lower": low, "upper": high}
        for a, low, high in rows]}


def make(source=None, view=VIEW):
    source = program() if source is None else source
    return rx.RationalReceiver(source, expected_program_sha256=rx.program_input_digest(source),
                               view_sha256=view)


def point_wire(receiver, point):
    return rx.encode_wire({"schema_version": rx.POINT_SCHEMA,
                           "view_sha256": receiver.view_sha256,
                           "program_sha256": receiver.program_sha256,
                           "point": [text(x) for x in point]})


def partial_value(receiver, request, point=None, eta=None, lam=None):
    return {"schema_version": rx.PARTIAL_SCHEMA, "view_sha256": receiver.view_sha256,
            "program_sha256": receiver.program_sha256,
            "request_sha256": request.request_sha256,
            "point": None if point is None else [text(x) for x in point],
            "original_row_lower_dual": None if eta is None else [text(x) for x in eta],
            "original_row_upper_dual": None if lam is None else [text(x) for x in lam]}


def partial_wire(receiver, request, **kwargs):
    return rx.encode_wire(partial_value(receiver, request, **kwargs))


def bind(receiver, rep=(1,), ideal=None, delta=0, index=0, policy=0):
    return receiver.bind_request(rep, rep if ideal is None else ideal, delta,
                                 request_index=index, policy_index=policy)


def vertices_2d(source):
    """Enumerate exact intersections of original 2-D halfspaces independently."""
    inequalities = []
    for i in range(2):
        unit = [Q(0), Q(0)]
        unit[i] = Q(1)
        inequalities.append((tuple(unit), Q(source["upper"][i])))
        inequalities.append((tuple(-x for x in unit), -Q(source["lower"][i])))
    for row in source["rows"]:
        a = tuple(Q(x) for x in row["coefficients"])
        inequalities += [(a, Q(row["upper"])), (tuple(-x for x in a), -Q(row["lower"]))]
    points = set()
    for (a, b), (c, d) in combinations(inequalities, 2):
        det = a[0] * c[1] - a[1] * c[0]
        if not det:
            continue
        point = ((b * c[1] - a[1] * d) / det, (a[0] * d - b * c[0]) / det)
        if all(sum(x * y for x, y in zip(row, point)) <= bound
               for row, bound in inequalities):
            points.add(point)
    return sorted(points)


class ReceiverTests(unittest.TestCase):
    def constrained(self):
        return make(program(rows=[((1,), Q(1, 3), 1)]))

    def test_external_program_digest_required(self):
        source = program()
        for value in (None, "A" * 64, "2" * 64):
            with self.subTest(value=value), self.assertRaises(rx.RequestValidationError):
                rx.RationalReceiver(source, expected_program_sha256=value, view_sha256=VIEW)

    def test_bad_trusted_program_and_scalar_types(self):
        for source in (program(lower=(-1,)), program(upper=(2,)),
                       program(lower=(True,)), program(lower=(0.0,)),
                       program(rows=[((1,), 1, 0)])):
            with self.subTest(source=source), self.assertRaises(rx.RequestValidationError):
                make(source)
        receiver = make()
        for scalar in (True, 0.5, "1/2.0", "2/4", "-0/1"):
            with self.subTest(scalar=scalar), self.assertRaises(rx.RequestValidationError):
                bind(receiver, rep=(scalar,))

    def test_program_digest_normalizes_exact_trusted_types(self):
        self.assertEqual(rx.program_input_digest(program(lower=(Q(1, 3),))),
                         rx.program_input_digest(program(lower=("1/3",))))

    def test_point_is_exact_and_deduplicated(self):
        receiver = make()
        first = receiver.accept_point(point_wire(receiver, [Q(1, 3)]))
        second = receiver.accept_point(point_wire(receiver, [Q(1, 3)]))
        self.assertTrue(first["added"])
        self.assertFalse(second["added"])
        self.assertEqual(receiver.point_count, 1)
        value = receiver.evaluate(bind(receiver))["represented"]
        self.assertEqual(value["upper_bound"], "1/3")
        self.assertEqual(value["upper_evidence"]["point"], ["1/3"])

    def test_infeasible_original_box_and_row_refused(self):
        receiver = self.constrained()
        for point in ([-1], [2], [Q(1, 4)]):
            with self.subTest(point=point), self.assertRaises(rx.CertificateValidationError):
                receiver.accept_point(point_wire(receiver, point))
        self.assertEqual(receiver.point_count, 0)

    def test_signed_overlapping_rows_are_all_checked(self):
        source = program((0, 0), (1, 1), [((1, 1), 1, 1), ((-1, 1), 0, Q(1, 3))])
        receiver = make(source)
        receiver.accept_point(point_wire(receiver, [Q(1, 2)] * 2))
        with self.assertRaises(rx.CertificateValidationError):
            receiver.accept_point(point_wire(receiver, [Q(3, 4), Q(1, 4)]))

    def test_negative_duals_and_bad_lengths_refused(self):
        receiver = self.constrained()
        request = bind(receiver)
        for eta, lam in (([-1], [0]), ([0], [-1]), ([], [0]), ([0], [])):
            with self.subTest(eta=eta, lam=lam), self.assertRaises(rx.CertificateValidationError):
                receiver.evaluate(request, partial_wire(receiver, request, eta=eta, lam=lam))

    def test_true_from_partial_lower_does_not_claim_optimum(self):
        receiver = self.constrained()
        request = bind(receiver, delta=Q(1, 4))
        result = receiver.evaluate(request, partial_wire(receiver, request, eta=[1], lam=[0]))
        col = result["represented"]
        self.assertEqual(col["decision"], "true")
        self.assertEqual(col["lower_bound"], "1/3")
        self.assertIsNone(col["upper_bound"])
        self.assertIsNone(col["optimum"])

    def test_false_is_not_optimum_or_real_effect(self):
        receiver = make()
        request = bind(receiver, delta=Q(1, 2))
        result = receiver.evaluate(request, partial_wire(receiver, request, point=[Q(1, 2)]))
        col = result["represented"]
        self.assertEqual(col["decision"], "false")
        self.assertEqual(col["lower_bound"], "0/1")
        self.assertEqual(col["upper_bound"], "1/2")
        self.assertIsNone(col["optimum"])
        # x=1 remains a feasible potential truth above the threshold.
        receiver.accept_point(point_wire(receiver, [1]))
        self.assertEqual(receiver.evaluate(request)["represented"]["decision"], "false")
        self.assertIn("false does not identify the real effect", result["scope_notice"])

    def test_strict_true_and_closed_false_boundary(self):
        receiver = self.constrained()
        request = bind(receiver, delta=Q(1, 3))
        result = receiver.evaluate(request, partial_wire(receiver, request,
                                    point=[Q(1, 2)], eta=[1], lam=[0]))
        self.assertEqual(result["represented"]["decision"], "unknown")
        receiver.accept_point(point_wire(receiver, [Q(1, 3)]))
        result = receiver.evaluate(request)
        self.assertEqual(result["represented"]["decision"], "false")
        self.assertEqual(result["represented"]["optimum"], "1/3")

    def test_zero_threshold_is_not_rounded_or_inferred_from_family_capacity(self):
        receiver = make(program((0, 0), (1, 1), [((1, -1), 0, 0)]))
        request = bind(receiver, rep=(1, -1 + Q(1, 2**60)), ideal=(1, -1), delta=0)
        result = receiver.evaluate(request, partial_wire(receiver, request, point=[Q(1, 3)] * 2))
        self.assertEqual(result["ideal"]["upper_bound"], "0/1")
        self.assertEqual(result["ideal"]["decision"], "false")
        self.assertEqual(Q(result["represented"]["upper_bound"]), Q(1, 3 * 2**60))
        self.assertEqual(result["represented"]["decision"], "unknown")
        self.assertIsNone(result["represented"]["optimum"])

    def test_equality_is_checked_for_each_objective_separately(self):
        receiver = self.constrained()
        request = bind(receiver, rep=(1,), ideal=(-1,))
        result = receiver.evaluate(request, partial_wire(receiver, request,
                                  point=[Q(1, 3)], eta=[1], lam=[0]))
        self.assertEqual(result["represented"]["optimum"], "1/3")
        self.assertEqual(result["represented"]["decision"], "true")
        self.assertIsNone(result["ideal"]["optimum"])
        self.assertEqual(result["ideal"]["decision"], "false")

    def test_zero_objective_exact_optimum_from_any_feasible_point(self):
        receiver = make()
        request = bind(receiver, rep=(0,))
        result = receiver.evaluate(request, partial_wire(receiver, request, point=[Q(2, 3)]))
        self.assertEqual(result["represented"]["optimum"], "0/1")
        self.assertEqual(result["represented"]["decision"], "false")

    def test_nonoptimal_valid_dual_is_a_bound_and_can_be_improved(self):
        receiver = self.constrained()
        request = bind(receiver, delta=Q(1, 4))
        first = receiver.evaluate(request, partial_wire(receiver, request,
                                 point=[Q(1, 2)], eta=[Q(1, 2)], lam=[0]))
        self.assertEqual(first["represented"]["lower_bound"], "1/6")
        self.assertEqual(first["represented"]["decision"], "unknown")
        second = receiver.evaluate(request, partial_wire(receiver, request, eta=[1], lam=[0]))
        self.assertEqual(second["represented"]["decision"], "true")

    def test_threshold_recomparison_uses_cached_bounds_not_boolean(self):
        receiver = self.constrained()
        first = bind(receiver, delta=0)
        receiver.evaluate(first, partial_wire(receiver, first,
                          point=[Q(1, 3)], eta=[1], lam=[0]))
        second = bind(receiver, delta=Q(1, 3), index=1)
        self.assertEqual(receiver.evaluate(first)["represented"]["decision"], "true")
        self.assertEqual(receiver.evaluate(second)["represented"]["decision"], "false")
        self.assertEqual(receiver.objective_count, 1)

    def test_new_shared_point_can_resolve_previous_unknown(self):
        receiver = make()
        request = bind(receiver, delta=Q(1, 2))
        receiver.accept_point(point_wire(receiver, [Q(3, 4)]))
        earlier_receipt = receiver.evaluate(request)
        self.assertEqual(earlier_receipt["represented"]["decision"], "unknown")
        receiver.accept_point(point_wire(receiver, [Q(1, 4)]))
        self.assertEqual(receiver.evaluate(request)["represented"]["decision"], "false")
        self.assertEqual(earlier_receipt["represented"]["decision"], "unknown")
        self.assertEqual(earlier_receipt["represented"]["upper_bound"], "3/4")

    def test_nonzero_dual_not_reused_for_different_objective(self):
        receiver = self.constrained()
        first = bind(receiver)
        receiver.evaluate(first, partial_wire(receiver, first, point=[Q(1, 3)], eta=[1], lam=[0]))
        second = bind(receiver, rep=(2,), delta=Q(1, 2), index=1)
        result = receiver.evaluate(second)["represented"]
        self.assertEqual(result["lower_bound"], "0/1")
        self.assertEqual(result["decision"], "unknown")
        self.assertIsNone(result["lower_evidence"]["origin_request_sha256"])

    def test_wrong_view_or_program_binding(self):
        receiver = make()
        request = bind(receiver)
        for field in ("view_sha256", "program_sha256"):
            value = partial_value(receiver, request, point=[0])
            value[field] = "2" * 64
            with self.subTest(field=field), self.assertRaises(rx.CertificateValidationError):
                receiver.evaluate(request, rx.encode_wire(value))
            point = json.loads(point_wire(receiver, [0]))
            point[field] = "2" * 64
            with self.assertRaises(rx.CertificateValidationError):
                receiver.accept_point(rx.encode_wire(point))

    def test_binding_covers_both_objectives_threshold_and_indices(self):
        receiver = make()
        first = bind(receiver, delta=Q(1, 1000))
        wire = partial_wire(receiver, first, point=[0])
        cases = [dict(rep=(2,)), dict(ideal=(2,)), dict(delta=Q(0.001)),
                 dict(policy=1), dict()]
        for index, change in enumerate(cases, 1):
            values = dict(delta=Q(1, 1000), index=index)
            values.update(change)
            request = bind(receiver, **values)
            with self.subTest(change=change), self.assertRaises(rx.CertificateValidationError):
                receiver.evaluate(request, wire)

    def test_original_binary64_threshold_is_kept_exact(self):
        receiver = make()
        request = bind(receiver, delta=Q(0.001))
        self.assertEqual(receiver.evaluate(request)["delta"], text(Q(0.001)))
        self.assertNotEqual(receiver.evaluate(request)["delta"], "1/1000")

    def test_duplicate_request_index_is_idempotent_or_refused(self):
        receiver = make()
        request = bind(receiver)
        self.assertIs(request, bind(receiver))
        with self.assertRaises(rx.RequestValidationError):
            bind(receiver, delta=1)
        for value in (True, -1, 1.0, 2**63):
            with self.subTest(index=value), self.assertRaises(rx.RequestValidationError):
                bind(receiver, index=value)

    def test_immutable_handle_and_forged_cache_handle_refused(self):
        receiver = make()
        request = bind(receiver)
        with self.assertRaises(AttributeError):
            request.request_sha256 = "0" * 64
        forged = rx.BoundRequest(*request)
        self.assertEqual(forged, request)
        self.assertIsNot(forged, request)
        with self.assertRaises(rx.RequestValidationError):
            receiver.evaluate(forged)
        with self.assertRaises(rx.RequestValidationError):
            make().evaluate(request)

    def test_trusted_source_mutation_does_not_change_internal_view_or_query(self):
        source = program(rows=[((1,), Q(1, 3), 1)])
        receiver = make(source)
        objective = [1]
        request = bind(receiver, rep=objective)
        source["lower"][0] = 1
        source["rows"][0]["lower"] = 1
        objective[0] = -999
        result = receiver.evaluate(request, partial_wire(receiver, request, point=[Q(1, 2)]))
        self.assertEqual(result["represented"]["objective"], ["1/1"])
        self.assertEqual(result["represented"]["upper_bound"], "1/2")

    def test_receipt_mutation_and_export_mutation_cannot_poison_cache(self):
        receiver = make()
        request = bind(receiver)
        result = receiver.evaluate(request, partial_wire(receiver, request, point=[Q(1, 2)]))
        original = deepcopy(result)
        result["represented"]["lower_bound"] = "999/1"
        result["represented"]["upper_evidence"]["point"][0] = "-1/1"
        exported = receiver.export_points()
        exported[0]["point"][0] = "-1/1"
        fresh = receiver.evaluate(request)
        for field in ("represented", "ideal"):
            self.assertEqual(fresh[field], original[field])
        with self.assertRaises(rx.CertificateValidationError):
            receiver.accept_point(rx.encode_wire(exported[0]))
        with self.assertRaises(rx.CertificateValidationError):
            receiver.evaluate(request, rx.encode_wire(result))
        self.assertEqual(receiver.point_count, 1)

    def test_cross_view_export_cannot_be_trusted_as_cached_point(self):
        first = make()
        first.accept_point(point_wire(first, [Q(1, 3)]))
        second = make(view="2" * 64)
        with self.assertRaises(rx.CertificateValidationError):
            second.accept_point(rx.encode_wire(first.export_points()[0]))
        self.assertEqual(second.point_count, 0)
        same_view = make()
        same_view.accept_point(rx.encode_wire(first.export_points()[0]))
        self.assertEqual(same_view.point_count, 1)

    def test_rejected_partial_certificate_is_atomic(self):
        receiver = self.constrained()
        request = bind(receiver)
        with self.assertRaises(rx.CertificateValidationError):
            receiver.evaluate(request, partial_wire(receiver, request,
                              point=[Q(1, 2)], eta=[-1], lam=[0]))
        self.assertEqual(receiver.point_count, 0)
        self.assertEqual(receiver.objective_count, 0)

    def test_no_untrusted_claimed_upper_lower_optimum_boolean_or_family_capacity(self):
        receiver = make()
        request = bind(receiver)
        for field, value in (("upper_bound", "-999/1"), ("lower_bound", "999/1"),
                             ("optimum", "999/1"), ("decision", "true"),
                             ("capacity", "999/1"), ("verified", True), ("cache", {})):
            cert = partial_value(receiver, request, point=[Q(1, 2)])
            cert[field] = value
            with self.subTest(field=field), self.assertRaises(rx.CertificateValidationError):
                receiver.evaluate(request, rx.encode_wire(cert))

    def test_empty_or_incomplete_witness_is_refused(self):
        receiver = self.constrained()
        request = bind(receiver)
        for values in ({}, {"eta": [0]}, {"lam": [0]}):
            with self.subTest(values=values), self.assertRaises(rx.CertificateValidationError):
                receiver.evaluate(request, partial_wire(receiver, request, **values))

    def test_noncanonical_rationals_refused(self):
        receiver = make()
        for scalar in ("2/4", "0/2", "-0/1", "+1/2", "01/2", "1/-2", "1/0", "0.5", 1, True):
            value = json.loads(point_wire(receiver, [0]))
            value["point"] = [scalar]
            with self.subTest(scalar=scalar), self.assertRaises(rx.CertificateValidationError):
                receiver.accept_point(rx.encode_wire(value))

    def test_duplicate_json_keys_float_nan_and_preparsed_dict_refused(self):
        receiver = make()
        valid = point_wire(receiver, [0])
        duplicate = valid[:-1] + ',"point":["0/1"]}'
        for wire in (duplicate, valid.replace('["0/1"]', '[0.0]'),
                     valid.replace('["0/1"]', '[NaN]'), json.loads(valid)):
            with self.subTest(wire=str(wire)[:90]), self.assertRaises(rx.WireFormatError):
                receiver.accept_point(wire)
        with self.assertRaises(rx.WireFormatError):
            rx.strict_json_loads('{"nested":{"a":1,"a":2}}')

    def test_exact_upper_and_lower_evidence_are_self_contained(self):
        receiver = self.constrained()
        request = bind(receiver)
        result = receiver.evaluate(request, partial_wire(receiver, request,
                                  point=[Q(1, 3)], eta=[1], lam=[0]))
        col = result["represented"]
        self.assertEqual(col["lower_evidence"]["original_row_lower_dual"], ["1/1"])
        self.assertEqual(col["lower_evidence"]["origin_request_sha256"], request.request_sha256)
        self.assertEqual(col["upper_evidence"]["point"], ["1/3"])
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_fresh_stdlib_import_does_not_load_producer_or_optimizer(self):
        root = Path(__file__).resolve().parents[2]
        script = (
            "import json, sys\n"
            "from icde2027.coat_workflow_v8 import receiver\n"
            "forbidden = ('icde2027.family_lp_v7', 'numpy', 'scipy', 'highspy', 'ctypes')\n"
            "bad = sorted(name for name in sys.modules if any(name == p or name.startswith(p + '.') "
            "for p in forbidden))\n"
            "print(json.dumps(bad))\n"
            "raise SystemExit(bool(bad))\n"
        )
        result = subprocess.run([sys.executable, "-B", "-S", "-c", script],
                                cwd=root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout), [])

    def test_box_support_uses_original_unequal_endpoints_and_negative_coefficients(self):
        receiver = make(program((Q(1, 4), Q(2, 3)), (Q(1, 2), Q(3, 4))))
        request = bind(receiver, rep=(2, -3))
        result = receiver.evaluate(request, partial_wire(receiver, request, point=[Q(1, 4), Q(3, 4)]))
        self.assertEqual(result["represented"]["optimum"], "-7/4")

    def test_original_upper_row_dual_orientation(self):
        receiver = make(program(rows=[((1,), 0, Q(2, 3))]))
        request = bind(receiver, rep=(-1,))
        result = receiver.evaluate(request, partial_wire(receiver, request,
                                  point=[Q(2, 3)], eta=[0], lam=[1]))
        self.assertEqual(result["represented"]["optimum"], "-2/3")

    def test_synthetic_240_requests_50_objectives_150_pairs(self):
        receiver = make(program((0, 0), (1, 1)))
        receiver.accept_point(point_wire(receiver, [Q(1, 3)] * 2))
        objectives, pairs, count = set(), set(), 0
        for threshold in (Q(0), Q(0.001), Q(0.01)):
            for position in range(80):
                k = position % 50
                objective = (Q(k, 1000), -Q(k, 1000))
                request = bind(receiver, rep=objective, delta=threshold, index=count, policy=position)
                result = receiver.evaluate(request)
                self.assertEqual(result["represented"]["decision"], "false")
                self.assertEqual(result["ideal"]["decision"], "false")
                self.assertEqual(result["request_index"], count)
                objectives.add(objective)
                pairs.add((objective, threshold))
                count += 1
        self.assertEqual((count, len(objectives), len(pairs)), (240, 50, 150))
        self.assertEqual(receiver.objective_count, 50)

    def test_bounds_against_independent_exact_vertex_enumeration(self):
        source = program((0, 0), (1, 1), [((1, 1), Q(1, 2), Q(3, 2)),
                         ((1, -1), -Q(1, 3), Q(2, 3)), ((0, 0), 0, 0)])
        vertices = vertices_2d(source)
        self.assertGreaterEqual(len(vertices), 4)
        for k, objective in enumerate(product((-1, 0, 1), repeat=2)):
            receiver = make(source)
            request = bind(receiver, rep=objective, delta=Q(1, 7))
            exact = min(sum(Q(c) * x for c, x in zip(objective, point)) for point in vertices)
            for point in vertices:
                receiver.accept_point(point_wire(receiver, point))
            for eta, lam in (([0, 0, 0], [0, 0, 0]), ([1, 0, 7], [0, 0, 2]),
                             ([0, 1, 0], [0, 0, 0]), ([0, 0, 0], [0, 1, 0]),
                             ([Q(1, 2), Q(2, 3), 0], [Q(3, 4), Q(4, 5), 0])):
                with self.subTest(objective=objective, eta=eta, lam=lam):
                    col = receiver.evaluate(request, partial_wire(receiver, request,
                                            eta=eta, lam=lam))["represented"]
                    self.assertLessEqual(Q(col["lower_bound"]), exact)
                    self.assertEqual(Q(col["upper_bound"]), exact)
                    if col["optimum"] is not None:
                        self.assertEqual(Q(col["optimum"]), exact)
                    if col["decision"] == "true":
                        self.assertGreater(exact, Q(1, 7))
                    elif col["decision"] == "false":
                        self.assertLessEqual(exact, Q(1, 7))


if __name__ == "__main__":
    unittest.main()
