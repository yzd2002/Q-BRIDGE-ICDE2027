"""Synthetic adapter/runner tests; never open a real numerical study input."""

import copy
from contextlib import ExitStack
from decimal import Inexact, Rounded, localcontext
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from icde2027.partition_capacity_v6 import study as s
from icde2027.partition_capacity_v6.core import solve
from icde2027.partition_capacity_v6.verify import input_digest, verify_certificate


def qp(value):
    value = F(value)
    return [str(value.numerator), str(value.denominator)]


def fixture():
    queries = {family: {"policy": [[0.0, 1.0]], "reference": [[0.5, 0.5]], "objective": [-0.5, 0.5]} for family in s.FAMILIES}
    world = {"world": {"spec": {"id": 0, "k": 1, "a": 2, "unconfounded": False},
                       "target_mean": [[0.25, 0.75]], "randomized_mean": [[0.9, 0.1]]},
             "records": [{"world_id": 0, "rep": r, "sensitivity": "conservative", "r_events_per_cell": 100,
                          "stratum": "synthetic", "rho_declared": 0.0, "radius": 12345.0,
                          "unconfounded": False, "queries": copy.deepcopy(queries)} for r in range(2)]}
    world_hash = "e" * 64
    contexts, programs, occurrences = {}, {}, {}
    original_occurrences, certificates = [], []
    context_ids = []
    for source_record in range(2):
        context = {"world_id": 0, "source_record": source_record, "size": [1, 2],
                   "weights": [qp(1)], "reference": [[qp(F(1, 2)), qp(F(1, 2))]],
                   "original_queries_sha256": s.canonical_hash(queries),
                   "policies": {family: {"policy": [[qp(0), qp(1)]],
                         "ideal_objective": [qp(F(-1, 2)), qp(F(1, 2))],
                         "stored_objective": [qp(F(-1, 2)), qp(F(1, 2))]} for family in s.FAMILIES}}
        cid = s.canonical_hash(context)
        context["context_id"] = cid
        contexts[cid] = context
        context_ids.append(cid)
    for kind in s.KINDS:
        for method in s.METHODS:
            for source_record in range(2):
                metadata = {"world_id": 0, "kind": kind, "method": method, "radius": qp(2)}
                if kind == "finite":
                    metadata.update(source_record=source_record, new_view_digest="d" * 64)
                else:
                    metadata["source_sha256"] = world_hash
                original = {"format": "fresh-mechanism-exact-population-v1", "metadata": metadata,
                            "lower": [qp(F(1, 4)), qp(F(3, 4))], "upper": [qp(F(1, 4)), qp(F(3, 4))],
                            "rows": [], "rhs_lower": [], "rhs_upper": []}
                pid = s.canonical_hash(original)
                if pid not in programs:
                    programs[pid] = {"program_id": pid, "program": original, "size": [1, 2],
                        "weights": [qp(1)], "reference": [[qp(F(1, 2)), qp(F(1, 2))]],
                        "occurrence_ids": [], "old_pool_upper": qp(1), "augmented_upper": qp(F(1, 2))}
                query_ids = {family: s.canonical_hash([pid, source_record, family]) for family in s.FAMILIES}
                occurrence = {"kind": kind, "method": method, "program_id": pid, "query_certificate_ids": query_ids,
                    "world_id": 0, "source_record": source_record, "rep": source_record,
                    "sensitivity": "conservative", "r_events_per_cell": 100, "stratum": "synthetic",
                    "size": [1, 2], "radius": qp(2), "rho_declared": 0.0, "threshold": qp(s.THRESHOLD)}
                oid = s.canonical_hash(occurrence)
                occurrence["occurrence_id"] = oid
                original_occurrences.append(copy.deepcopy(occurrence))
                lower_records = []
                lower = F(source_record, 8)
                for family, cid in query_ids.items():
                    cert_hash = s.canonical_hash([cid, "original-certificate"])
                    certificates.append({"certificate_id": cid, "certificate_sha256": cert_hash, "program_id": pid})
                    lower_records.append({"family": family, "certificate_id": cid, "threshold": qp(s.THRESHOLD),
                        "archive_certificate_sha256": cert_hash, "ideal_objective_lower": qp(lower),
                        "old_stored_objective_lower": qp(lower - F(1, 100)), "encoding_transfer_sanity_lower": qp(lower - F(1, 50))})
                occurrence.update(context_id=context_ids[source_record], lower=qp(lower), lower_records=lower_records,
                    old_pool_upper=qp(1), augmented_upper=qp(F(1, 2)),
                    classification_old_pool=s.ABOVE if lower > s.THRESHOLD else s.UNKNOWN,
                    classification_augmented=s.ABOVE if lower > s.THRESHOLD else s.UNKNOWN)
                occurrences[oid] = occurrence
                programs[pid]["occurrence_ids"].append(oid)
    inventory = {"occurrences": original_occurrences, "certificates": certificates, "program_ids": sorted(programs),
                 "source_files_sha256": {"world.json": world_hash}}
    return programs, contexts, occurrences, inventory, {0: world}, {0: world_hash}


def numerical_fixture():
    programs, contexts, occurrences, inventory, worlds, world_hashes = fixture()
    digests = s.validate_joins(programs, contexts, occurrences, inventory, worlds, world_hashes)
    results = {pid: s.evaluate_program(p, worlds[0], digests[pid]) for pid, p in programs.items()}
    rows = [s.occurrence_result(o, results[o["program_id"]]) for o in occurrences.values()]
    return results, rows


class ExactOutputEncodingTests(unittest.TestCase):
    def setUp(self):
        self.previous_limit = sys.get_int_max_str_digits()
        s.configure_cli_runtime()
        self.assertEqual(sys.get_int_max_str_digits(), 100000)

    def tearDown(self):
        sys.set_int_max_str_digits(self.previous_limit)

    @staticmethod
    def parse_output_integer(text):
        """Independent base-10^9 parsing of output, never int(the_long_string)."""
        negative = text.startswith("-")
        digits = text[1:] if negative else text
        if not digits or any(c not in "0123456789" for c in digits):
            raise AssertionError("non-decimal integer output")
        if len(digits) > 1 and digits[0] == "0":
            raise AssertionError("noncanonical leading zero")
        if negative and digits == "0":
            raise AssertionError("noncanonical negative zero")
        value = 0
        for offset in range(0, len(digits), 9):
            chunk = digits[offset:offset+9]
            value = value * (10 ** len(chunk)) + int(chunk)
        return -value if negative else value

    def assert_exact_output(self, value):
        encoded = s.qp(value)
        for text, integer in zip(encoded, (value.numerator, value.denominator)):
            self.assertTrue(self.parse_output_integer(text) == integer)
            for modulus in (97, 1000003, 2305843009213693951):
                residue = 0
                for digit in text.lstrip("-"):
                    residue = (10 * residue + ord(digit) - ord("0")) % modulus
                if text.startswith("-"):
                    residue = -residue % modulus
                self.assertEqual(residue, integer % modulus)
        self.assertEqual(sys.get_int_max_str_digits(), 100000)
        return encoded

    def test_oversized_positive_negative_zero_and_denominator_outputs(self):
        large = 10 ** 100100 + 17
        for label, value in (("positive", F(large)), ("negative", F(-large)),
                             ("zero", F(0, large)), ("denominator", F(1, large)),
                             ("both", F(large + 2, large))):
            with self.subTest(label=label):
                encoded = self.assert_exact_output(value)
                if value:
                    self.assertTrue(any(len(part.lstrip("-")) > 100000 for part in encoded))
                else:
                    self.assertEqual(encoded, ["0", "1"])

    def test_decimal_low_precision_context_cannot_round_integer_encoding(self):
        large = 10 ** 100100 + 17
        value = F(-(large + 2), large)
        for precision in (1, 2, 7):
            with self.subTest(precision=precision), localcontext() as context:
                context.prec = precision
                context.Emax = 9
                context.Emin = -9
                context.traps[Inexact] = True
                context.traps[Rounded] = True
                context.clear_flags()
                self.assert_exact_output(value)
                self.assertFalse(any(context.flags.values()))

    def test_new_encoding_matches_original_str_bytes_within_input_limit(self):
        for integer in (0, 1, -1, 2**256, -(2**256), 10**100 + 17, 10**99999 + 17):
            for value in (F(integer), F(integer, 97), F(1, abs(integer) + 1)):
                with self.subTest(bits=integer.bit_length()):
                    expected = [str(value.numerator), str(value.denominator)]
                    self.assertEqual(s.qp(value), expected)

    def test_output_formatting_does_not_relax_input_conversion_limit(self):
        large = 10 ** 100100 + 17
        encoded = self.assert_exact_output(F(large))
        with self.assertRaises(ValueError):
            int(encoded[0])
        with self.assertRaises(ValueError):
            s.pair(encoded)
        self.assertEqual(sys.get_int_max_str_digits(), 100000)

    def test_qp_refuses_non_int_components(self):
        class Components:
            pass
        for numerator, denominator in (("1", 1), (1.0, 1), (True, 1), (1, "2"), (1, 2.0)):
            value = Components()
            value.numerator, value.denominator = numerator, denominator
            with self.subTest(types=(type(numerator).__name__, type(denominator).__name__)), self.assertRaises(s.StudyError):
                s.qp(value)

    def test_sum_of_legal_input_pairs_can_have_oversized_output_components(self):
        first, second = 10 ** 60001 + 1, 10 ** 60001 + 3
        values = [F(first - 1, first), F(second - 1, second)]
        original_pairs = [[str(v.numerator), str(v.denominator)] for v in values]
        total = sum((s.pair(encoded) for encoded in original_pairs), F(0))
        encoded = self.assert_exact_output(total)
        self.assertTrue(all(len(part) > 100000 for part in encoded))


class AdapterTests(unittest.TestCase):
    def test_synthetic_complete_joins_without_numerical_calls(self):
        with patch("icde2027.partition_capacity_v6.core.solve", side_effect=AssertionError("forbidden")), patch("icde2027.partition_capacity_v6.verify.verify_certificate", side_effect=AssertionError("forbidden")), patch.object(s, "truth_diagnostic", side_effect=AssertionError("forbidden")):
            values = fixture()
            digests = s.validate_joins(*values)
            self.assertEqual(len(digests), 9)

    def test_sparse_coefficients_rows_and_zero_rows_preserved(self):
        p = next(iter(fixture()[0].values()))
        p = copy.deepcopy(p)
        p["program"]["rows"] = [[], [[0, qp(F(2, 3))], [1, qp(F(5, 7))]]]
        p["program"]["rhs_lower"] = [qp(-1), qp(0)]
        p["program"]["rhs_upper"] = [qp(1), qp(2)]
        p["program_id"] = s.canonical_hash(p["program"])
        request = s.adapt_program(p)
        self.assertEqual(request[0]["rows"][0]["coefficients"], [F(0), F(0)])
        self.assertEqual(request[0]["rows"][1]["coefficients"], [F(2, 3), F(5, 7)])
        self.assertEqual(s.request_digest(request), input_digest(*request))

    def test_scope_uses_metadata_not_population_format(self):
        p = next(p for p in fixture()[0].values() if p["program"]["metadata"]["kind"] == "finite")
        self.assertIn("population", p["program"]["format"])
        self.assertEqual(p["program"]["metadata"]["kind"], "finite")
        s.adapt_program(p)

    def test_pair_requires_exact_canonical_strings(self):
        for value in (["2", "4"], ["-0", "1"], ["1", "-2"], ["1", "0"], [1, 2], ["01", "2"], ["1.0", "2"], [True, "1"], ("1", "2")):
            with self.subTest(value=value), self.assertRaises(s.StudyError):
                s.pair(value)
        self.assertEqual(s.pair(["-1", "3"]), F(-1, 3))

    def test_cli_large_integer_compatibility_is_explicit(self):
        before = sys.get_int_max_str_digits()
        try:
            s.configure_cli_runtime()
            self.assertEqual(sys.get_int_max_str_digits(), 100000)
            value = "1" + "0" * 5000
            fraction = s.pair([value, "3"])
            self.assertEqual(s.qp(fraction), [value, "3"])
        finally:
            sys.set_int_max_str_digits(before)

    def test_json_duplicate_and_nonfinite_rejection(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e400}'):
            with self.subTest(text=text), self.assertRaises(s.StudyError):
                s.strict_json(text)

    def test_original_identity_includes_metadata(self):
        p = copy.deepcopy(next(iter(fixture()[0].values())))
        p["program"]["metadata"]["radius"] = qp(1)
        with self.assertRaises(s.StudyError):
            s.adapt_program(p)

    def test_canonical_digest_is_not_raw_file_sha(self):
        value = {"z": 2, "a": 1}
        raw = json.dumps(value, indent=2).encode()
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(), s.canonical_hash(value))
        self.assertEqual(s.canonical_hash(s.strict_json(raw)), s.canonical_hash(value))

    def test_threshold_binary64_not_decimal(self):
        self.assertEqual(s.THRESHOLD, F(0.001))
        self.assertNotEqual(s.THRESHOLD, F(1, 1000))
        self.assertEqual(s.classify(F(0), s.THRESHOLD, s.THRESHOLD), s.ATMOST)

    def test_join_corruptions_are_rejected(self):
        changes = [
            lambda v: next(iter(v[2].values())).__setitem__("context_id", "0" * 64),
            lambda v: next(iter(v[2].values())).__setitem__("program_id", "0" * 64),
            lambda v: next(iter(v[0].values()))["occurrence_ids"].clear(),
            lambda v: next(iter(v[1].values())).__setitem__("original_queries_sha256", "0" * 64),
            lambda v: next(iter(v[2].values())).__setitem__("threshold", qp(F(1, 1000))),
            lambda v: next(iter(v[2].values()))["lower_records"][0].__setitem__("certificate_id", "0" * 64),
            lambda v: next(iter(v[2].values()))["lower_records"].pop(),
            lambda v: next(iter(v[2].values())).__setitem__("augmented_upper", qp(1)),
            lambda v: next(iter(v[2].values())).__setitem__("lower", qp(F(1, 4))),
            lambda v: v[4][0]["records"][0].__setitem__("rep", 4),
            lambda v: v[4][0]["world"]["spec"].__setitem__("unconfounded", True),
            lambda v: v[5].__setitem__(0, "0" * 64),
            lambda v: v[3]["program_ids"].pop(),
            lambda v: next(iter(v[0].values())).__setitem__("reference", [[qp(1), qp(0)]]),
        ]
        for i, change in enumerate(changes):
            with self.subTest(corruption=i):
                values = fixture()
                change(values)
                with self.assertRaises(s.StudyError):
                    s.validate_joins(*values)

    def test_changed_world_radius_does_not_replace_repaired_radius(self):
        values = fixture()
        self.assertEqual(values[4][0]["records"][0]["radius"], 12345.0)
        s.validate_joins(*values)
        self.assertTrue(all(o["radius"] == qp(2) for o in values[2].values()))

    def test_production_counts_cannot_be_shrunk_for_fixture(self):
        values = fixture()
        with self.assertRaises(s.StudyError):
            s.production_census(*values[:4], values[4])

    def test_duplicate_sparse_coordinate_bool_and_overlap(self):
        for rows in ([[[0, qp(1)], [0, qp(1)]]], [[[True, qp(1)]]], [[[0, qp(1)]], [[0, qp(1)]]], [[[0, qp(-1)]]]):
            with self.subTest(rows=rows):
                p = copy.deepcopy(next(iter(fixture()[0].values())))
                p["program"].update(rows=rows, rhs_lower=[qp(-10)] * len(rows), rhs_upper=[qp(10)] * len(rows))
                p["program_id"] = s.canonical_hash(p["program"])
                with self.assertRaises(s.StudyError):
                    s.adapt_program(p)


class NumericalAndSummaryTests(unittest.TestCase):
    def test_exact_synthetic_program_and_independent_receipt(self):
        results, rows = numerical_fixture()
        self.assertEqual(len(results), 9)
        self.assertEqual(len(rows), 12)
        for result in results.values():
            self.assertTrue(result["receiver_receipt"]["verified"])
            self.assertEqual(result["certificate"]["capacity"], "1/4")
            self.assertTrue(result["truth"]["truth_in_C"])
            self.assertEqual(result["truth"]["new_policy_truth_contrast"], qp(F(1, 4)))

    def test_population_occurrence_baselines_are_not_pooled(self):
        _, rows = numerical_fixture()
        selected = [r for r in rows if r["original_occurrence"]["kind"] == "population" and r["original_occurrence"]["method"] == s.METHODS[0]]
        self.assertEqual(len({r["program_id"] for r in selected}), 1)
        self.assertEqual({tuple(r["capacity_minus_own_three_policy_lower"]) for r in selected}, {tuple(qp(F(1, 4))), tuple(qp(F(1, 8)))})

    def test_external_request_is_passed_to_receiver(self):
        values = fixture()
        p = next(iter(values[0].values()))
        expected = s.request_digest(s.adapt_program(p))
        calls = []
        def receiver(program, sizes, weights, reference, cert, **kwargs):
            self.assertEqual(program, s.adapt_program(p)[0])
            self.assertIsNot(program, cert["canonical_input"]["program"])
            self.assertEqual(kwargs["expected_input_sha256"], expected)
            calls.append(1)
            return verify_certificate(program, sizes, weights, reference, cert, **kwargs)
        with patch("icde2027.partition_capacity_v6.verify.verify_certificate", side_effect=receiver):
            s.evaluate_program(p, values[4][0], expected)
        self.assertEqual(calls, [1])

    def test_wrong_external_digest_fails_before_solve(self):
        values = fixture()
        with patch("icde2027.partition_capacity_v6.core.solve", side_effect=AssertionError("must not solve")):
            with self.assertRaises(s.StudyError):
                s.evaluate_program(next(iter(values[0].values())), values[4][0], "0" * 64)

    def test_corrupt_certificate_is_fatal(self):
        values = fixture()
        p = next(iter(values[0].values()))
        request = s.adapt_program(p)
        cert = solve(*request)
        cert["capacity"] = "1/2"
        with patch("icde2027.partition_capacity_v6.core.solve", return_value=cert):
            with self.assertRaisesRegex(s.StudyError, "receiver rejected"):
                s.evaluate_program(p, values[4][0], s.request_digest(request))

    def test_truth_uses_binary64_target_not_randomized_mean(self):
        values = fixture()
        p = next(iter(values[0].values()))
        p = copy.deepcopy(p)
        p["program"]["lower"] = [qp(F(0.1)), qp(F(0.3))]
        p["program"]["upper"] = copy.deepcopy(p["program"]["lower"])
        p["program_id"] = s.canonical_hash(p["program"])
        world = copy.deepcopy(values[4][0]); world["world"]["target_mean"] = [[0.1, 0.3]]
        result = s.evaluate_program(p, world, s.request_digest(s.adapt_program(p)))
        expected = (F(0.3) - F(0.1)) / 2
        self.assertEqual(s.pair(result["truth"]["new_policy_truth_contrast"]), expected)
        self.assertNotEqual(expected, F(1, 10))
        self.assertEqual(result["truth"]["target_mean_exact_binary64"][0], qp(F(0.1)))

    def test_finite_noncoverage_is_retained_and_population_is_fatal(self):
        values = fixture()
        world = copy.deepcopy(values[4][0]); world["world"]["target_mean"] = [[0.9, 0.1]]
        for kind in s.KINDS:
            p = next(p for p in values[0].values() if p["program"]["metadata"]["kind"] == kind)
            if kind == "finite":
                result = s.evaluate_program(p, world, s.request_digest(s.adapt_program(p)))
                self.assertFalse(result["truth"]["truth_in_C"])
                self.assertFalse(result["truth"]["new_policy_lower_le_truth"])
                occurrence = next(o for o in values[2].values() if o["program_id"] == p["program_id"])
                self.assertTrue(s.occurrence_result(occurrence, result)["false_positive_assertion"])
            else:
                with self.assertRaisesRegex(s.StudyError, "population truth"):
                    s.evaluate_program(p, world, s.request_digest(s.adapt_program(p)))

    def test_truth_membership_checks_moments(self):
        values = fixture()
        p = next(p for p in values[0].values() if p["program"]["metadata"]["kind"] == "finite")
        p = copy.deepcopy(p)
        p["program"].update(lower=[qp(0), qp(0)], upper=[qp(1), qp(1)],
                            rows=[[[0, qp(1)], [1, qp(1)]]], rhs_lower=[qp(F(3, 2))], rhs_upper=[qp(2)])
        p["program_id"] = s.canonical_hash(p["program"])
        result = s.evaluate_program(p, values[4][0], s.request_digest(s.adapt_program(p)))
        self.assertFalse(result["truth"]["truth_in_C"])
        self.assertEqual(result["truth"]["membership_violations"][0]["kind"], "moment")

    def test_threshold_equality_is_at_most_and_distinct_from_zero(self):
        results, rows = numerical_fixture()
        result = copy.deepcopy(next(iter(results.values())))
        result["certificate"]["capacity"] = s.qs(s.THRESHOLD)
        original = next(r["original_occurrence"] for r in rows if s.pair(r["original_occurrence"]["lower"]) == 0)
        row = s.occurrence_result(original, result)
        self.assertEqual(row["classification_new"], s.ATMOST)
        self.assertEqual(row["zero_structure_new"], "R_STRICTLY_POSITIVE")
        self.assertFalse(row["false_positive_assertion"])

    def test_false_positive_uses_unchanged_weak_truth_boundary(self):
        results, rows = numerical_fixture()
        result = copy.deepcopy(next(iter(results.values())))
        result["truth"]["new_policy_truth_contrast"] = qp(s.THRESHOLD)
        row = s.occurrence_result(rows[0]["original_occurrence"], result)
        self.assertTrue(row["false_positive_assertion"])

    def test_summary_preserves_all_transitions_and_signed_pairs(self):
        _, rows = numerical_fixture()
        # Pure summary fixture: introduce both signs without rerunning any source.
        for row in rows:
            method = row["original_occurrence"]["method"]
            row["capacity"] = qp({s.METHODS[0]: F(1, 4), s.METHODS[1]: F(1, 3), s.METHODS[2]: F(1, 4)}[method])
        pairs = s.source_comparisons(rows)
        self.assertEqual(len(pairs), 12)
        self.assertEqual({p["sign"] for p in pairs}, {-1, 0, 1})
        summary = s.summarize(rows, pairs)
        self.assertEqual(sum(r["occurrences"] for r in summary["tables"]["primary"]), 12)
        self.assertEqual(sum(r["positive"] + r["zero"] + r["negative"] for r in summary["tables"]["paired_sources"]), 12)
        self.assertEqual(summary["exploratory_registered_cell_occurrences"], 2)
        self.assertEqual(summary["exploratory_previously_unresolved_denominator"], 1)
        self.assertEqual(summary["exploratory_unresolved_to_above_count"], 1)

    def test_pairing_cannot_silently_drop_a_method(self):
        _, rows = numerical_fixture()
        with self.assertRaises(s.StudyError):
            s.source_comparisons(rows[1:])

    def test_false_positive_is_reported_separately(self):
        _, rows = numerical_fixture()
        rows[0]["false_positive_assertion"] = True
        summary = s.summarize(rows, s.source_comparisons(rows))
        self.assertEqual(summary["false_positive_assertion_ids"], [rows[0]["occurrence_id"]])
        self.assertEqual(summary["finite_coverage_failure_ids"], [])
        self.assertEqual(summary["lower_exceeds_truth_ids"], [])

    def test_summary_zero_categories_are_explicit(self):
        _, rows = numerical_fixture()
        summary = s.summarize(rows, s.source_comparisons(rows))
        for row in summary["tables"]["primary"]:
            self.assertEqual(row["classifications"][s.ATMOST], 0)
            self.assertEqual(row["classifications"][s.UNKNOWN], 0)
            self.assertEqual(row["zero_structure"]["R_EQUALS_ZERO"], 0)
            self.assertEqual(len(row["transitions_augmented"]), 6)

    def test_truth_outside_probability_range_is_an_input_error(self):
        values = fixture()
        p = next(iter(values[0].values()))
        world = copy.deepcopy(values[4][0]); world["world"]["target_mean"] = [[2.0, 0.5]]
        with self.assertRaisesRegex(s.StudyError, "outside"):
            s.evaluate_program(p, world, s.request_digest(s.adapt_program(p)))


class FilesAndGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value)
        else:
            s.write_json(path, value)
        return s.raw_identity(path)

    def gate_fixture(self):
        files = {}
        for p in (*s.NUMERICAL_CODE, s.PLAN, s.INPUT_MAP, s.BASE + "/program_upper_certificates.jsonl"):
            files[p] = {**self.write(p, "synthetic " + p), "roles": ["synthetic"]}
        index = {"schema_version": "partition-capacity-input-index/1", "project_root": str(self.root),
                 "census": {"counts": s.EXPECTED}, "files": files, "actual_runtime": {"files": {}}}
        index_path = s.STUDY + "/synthetic_index.json"
        index_sha = self.write(index_path, index)["sha256"]
        config = s.config_proposal(index_path, index_sha, index)
        config["input_map"]["sha256"] = files[s.INPUT_MAP]["sha256"]
        config_path = s.STUDY + "/config.json"
        config_sha = self.write(config_path, config)["sha256"]
        code = {p: files[p]["sha256"] for p in s.NUMERICAL_CODE}
        reviews = []
        for role in ("implementation", "input_and_methodology"):
            path = s.STUDY + "/reviews/" + role + ".json"
            declaration = {"decision": "GO", "independent_review": True, "role": role,
                           "scope": "registered-fixed-data-partition-capacity-v6", "config_sha256": config_sha,
                           "input_index_sha256": index_sha, "numerical_sources_sha256": code}
            reviews.append({"role": role, "path": path, "sha256": self.write(path, declaration)["sha256"],
                            "decision": "GO", "independent": True, "scope": declaration["scope"]})
        gate = {"schema_version": "partition-capacity-execution-gate/1", "decision": "GO",
                "scope": "registered-fixed-data-partition-capacity-v6", "output": s.RUN_OUTPUT,
                "resources": s.RESOURCES, "config": {"path": config_path, "sha256": config_sha},
                "plan": config["plan"], "input_index": config["input_index"],
                "numerical_sources_sha256": code, "runtime_identity_sha256": s.canonical_hash(index["actual_runtime"]),
                "reviews": reviews}
        gate_path = s.STUDY + "/gate.json"
        self.write(gate_path, gate)
        return config_path, gate_path, config, gate, index

    def gate_context(self, index):
        return patch.dict(s.PINNED, {s.INPUT_MAP: index["files"][s.INPUT_MAP]["sha256"]})

    def validate(self, cp, gp, index):
        with self.gate_context(index), patch.object(os, "sched_getaffinity", return_value={10}), patch.object(sys, "get_int_max_str_digits", return_value=100000), patch.dict(os.environ, {"OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1", "CUDA_VISIBLE_DEVICES":""}):
            return s.validate_gate(self.root, cp, gp, s.RUN_OUTPUT)

    def retained_controls(self, cp, gp, config, gate):
        paths = [cp, gp, *[config[f]["path"] for f in ("plan", "input_map", "input_index")],
                 *[r["path"] for r in gate["reviews"]]]
        return {p: {**s.raw_identity(self.root/p), "roles":["execution_control"]} for p in paths}

    def runtime_controls(self, index):
        stack = ExitStack()
        stack.enter_context(self.gate_context(index))
        stack.enter_context(patch.object(os,"sched_getaffinity",return_value={10}))
        stack.enter_context(patch.object(sys,"get_int_max_str_digits",return_value=100000))
        stack.enter_context(patch.dict(os.environ,{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":""}))
        return stack

    def tiny_control_bundle(self):
        pid,oid="1"*64,"2"*64
        return {"programs":{pid:{"program_id":pid,"program":{"metadata":{"world_id":0}},"occurrence_ids":[oid]}},
                "worlds":{0:{"synthetic":True}},"request_digests":{pid:"3"*64},
                "world_hashes":{0:"4"*64},"occurrences":{oid:{"occurrence_id":oid}}}

    def test_complete_bound_synthetic_gate_accepts(self):
        cp, gp, _, _, index = self.gate_fixture()
        self.validate(cp, gp, index)

    def test_arbitrary_go_string_and_wrong_scope_rejected(self):
        cp, gp, _, gate, index = self.gate_fixture()
        for replacement in ({"decision":"GO"}, {**gate, "scope":"another study"}, {**gate,"reviews":[]}):
            self.write(gp, replacement)
            with self.assertRaises(s.StudyError):
                self.validate(cp, gp, index)

    def test_review_must_bind_current_code_and_request(self):
        cp, gp, _, gate, index = self.gate_fixture()
        declaration = s.load_json(self.root, gate["reviews"][0]["path"])
        declaration["config_sha256"] = "0" * 64
        gate["reviews"][0]["sha256"] = self.write(gate["reviews"][0]["path"], declaration)["sha256"]
        self.write(gp, gate)
        with self.assertRaises(s.StudyError):
            self.validate(cp, gp, index)

    def test_config_cannot_override_registered_count_time_or_cpu(self):
        cp, gp, config, gate, index = self.gate_fixture()
        for field, key, value in (("expected","programs",1), ("resources","wall_seconds",7200), ("resources","cpu",16)):
            modified = copy.deepcopy(config); modified[field][key] = value
            gate["config"]["sha256"] = self.write(cp, modified)["sha256"]
            self.write(gp, gate)
            with self.assertRaises(s.StudyError):
                self.validate(cp, gp, index)

    def test_before_after_source_changes_rejected(self):
        identity = self.write("source.json", {"a":1})
        s.check_files(self.root, {"source.json":identity})
        self.write("source.json", {"a":2})
        with self.assertRaises(s.StudyError):
            s.check_files(self.root, {"source.json":identity})

    def test_gate_source_change_rejected(self):
        cp, gp, _, _, index = self.gate_fixture()
        self.write(s.NUMERICAL_CODE[-1], "changed")
        with self.assertRaises(s.StudyError):
            self.validate(cp, gp, index)

    def test_wrong_paths_symlinks_and_fallback_refused(self):
        self.write("file.json", "{}")
        (self.root/"link.json").symlink_to(self.root/"file.json")
        (self.root/"dirlink").symlink_to(self.root, target_is_directory=True)
        for path in ("../file.json", "/file.json", "./file.json", "a//b", "link.json", "dirlink/file.json", "missing.json", "a\\b"):
            with self.subTest(path=path), self.assertRaises(s.StudyError):
                s.safe_path(self.root, path)

    def test_failure_is_preserved_and_existing_output_refused(self):
        with patch.object(s, "evaluate_program", side_effect=AssertionError("numerical boundary")):
            with self.assertRaises(s.StudyError):
                s.run(self.root, "absent_config.json", "absent_gate.json", s.RUN_OUTPUT)
        out = self.root/s.RUN_OUTPUT
        self.assertTrue((out/"FAILURE.json").is_file())
        self.assertFalse(s.load_json(self.root, s.RUN_OUTPUT+"/manifest.json")["complete"])
        before = (out/"FAILURE.json").read_bytes()
        with self.assertRaises(s.StudyError):
            s.run(self.root, "absent_config.json", "absent_gate.json", s.RUN_OUTPUT)
        self.assertEqual((out/"FAILURE.json").read_bytes(), before)

    def test_preflight_failure_has_no_numeric_activity(self):
        out = s.STUDY+"/preflight_synthetic"
        with patch.object(s, "load_fixed_inputs", side_effect=s.StudyError("bad source")), patch.object(s, "evaluate_program", side_effect=AssertionError("numerical boundary")):
            with self.assertRaises(s.StudyError):
                s.preflight(self.root, out)
        self.assertFalse(s.load_json(self.root, out+"/FAILURE.json")["new_numerical_evaluation_performed"])

    def test_deadline_stops_before_next_program_and_preserves_failure(self):
        cp, gp, config, gate, index = self.gate_fixture()
        programs, contexts, occurrences, inventory, worlds, hashes = fixture()
        digests = s.validate_joins(programs, contexts, occurrences, inventory, worlds, hashes)
        bundle = {"programs":programs,"contexts":contexts,"occurrences":occurrences,
                  "worlds":worlds,"world_hashes":hashes,"request_digests":digests}
        readings = iter([0.0])
        def clock():
            return next(readings, 3601.0)
        with patch.object(s, "validate_gate", return_value=(config,gate,index,self.retained_controls(cp,gp,config,gate))), patch.object(s,"load_fixed_inputs",return_value=(bundle,index)), patch.object(s.time,"monotonic",side_effect=clock), patch.object(s,"evaluate_program",side_effect=AssertionError("deadline must prevent evaluation")):
            with self.assertRaises(s.DeadlineReached):
                s.run(self.root,cp,gp,s.RUN_OUTPUT)
        manifest=s.load_json(self.root,s.RUN_OUTPUT+"/manifest.json")
        self.assertEqual(manifest["status"], "incomplete_deadline")
        self.assertFalse(manifest["new_numerical_evaluation_performed"])
        self.assertTrue(manifest["all_sources_rechecked_at_end"])

    def test_output_index_includes_nested_same_name_files(self):
        out=self.root/"out";out.mkdir()
        nested=out/"source_snapshot/OUTPUT_INDEX.json";nested.parent.mkdir();nested.write_text('{}')
        s.output_index(out)
        self.assertIn("source_snapshot/OUTPUT_INDEX.json", json.loads((out/"OUTPUT_INDEX.json").read_text())["files"])

    def test_first_numerical_call_has_already_logged_manifest_flag(self):
        cp, gp, config, gate, index = self.gate_fixture()
        programs, contexts, occurrences, inventory, worlds, hashes = fixture()
        digests = s.validate_joins(programs, contexts, occurrences, inventory, worlds, hashes)
        bundle = {"programs":programs,"contexts":contexts,"occurrences":occurrences,
                  "worlds":worlds,"world_hashes":hashes,"request_digests":digests}
        def stop_at_first_call(*args):
            manifest=s.load_json(self.root,s.RUN_OUTPUT+"/manifest.json")
            self.assertTrue(manifest["new_numerical_evaluation_performed"])
            self.assertEqual(manifest["programs_completed"], 0)
            raise s.StudyError("synthetic interruption")
        with patch.object(s,"validate_gate",return_value=(config,gate,index,self.retained_controls(cp,gp,config,gate))), patch.object(s,"load_fixed_inputs",return_value=(bundle,index)), patch.object(s,"evaluate_program",side_effect=stop_at_first_call):
            with self.assertRaisesRegex(s.StudyError,"program evaluation failed"):
                s.run(self.root,cp,gp,s.RUN_OUTPUT)

    def test_reviewer_post_validation_control_mutations_are_all_rejected(self):
        original_root = self.root
        for field in ("config", "gate", "index", "implementation", "methodology", "plan", "input_map", "numerical_source"):
            with self.subTest(field=field):
                self.root = original_root / field
                self.root.mkdir()
                cp,gp,config,gate,index=self.gate_fixture()
                candidates={"config":cp,"gate":gp,"index":config["input_index"]["path"],
                            "implementation":gate["reviews"][0]["path"],"methodology":gate["reviews"][1]["path"],
                            "plan":s.PLAN,"input_map":s.INPUT_MAP,"numerical_source":s.NUMERICAL_CODE[-1]}
                changed=candidates[field]
                def load(_root):
                    self.write(changed,{"decision":"NO_GO","injected_after_validation":True})
                    return self.tiny_control_bundle(),index
                with self.runtime_controls(index), patch.object(s,"load_fixed_inputs",side_effect=load), patch.object(s,"evaluate_program") as numerical:
                    with self.assertRaises(s.StudyError):
                        s.run(self.root,cp,gp,s.RUN_OUTPUT)
                    numerical.assert_not_called()
                manifest=s.load_json(self.root,s.RUN_OUTPUT+"/manifest.json")
                self.assertFalse(manifest["new_numerical_evaluation_performed"])
                self.assertFalse(manifest["complete"])
                self.assertFalse(manifest["all_sources_rechecked_at_end"])
        self.root=original_root

    def test_control_parse_and_hash_use_the_same_first_bytes(self):
        cp,gp,_,_,index=self.gate_fixture()
        original_read=Path.read_bytes
        injected=[]
        def read_then_change(path):
            value=original_read(path)
            if path == self.root/gp and not injected:
                injected.append(True)
                path.write_text('{"decision":"NO_GO"}')
            return value
        with patch.object(Path,"read_bytes",read_then_change):
            with self.assertRaises(s.StudyError):
                self.validate(cp,gp,index)
        self.assertEqual(injected,[True])

    def test_control_change_after_snapshot_before_numeric_is_rejected(self):
        cp,gp,_,_,index=self.gate_fixture()
        original_copy=s.shutil.copyfile
        injected=[]
        def copy_then_change(source,destination):
            result=original_copy(source,destination)
            if source == self.root/gp and not injected:
                injected.append(True)
                self.write(gp,{"decision":"NO_GO"})
            return result
        with self.runtime_controls(index), patch.object(s,"load_fixed_inputs",return_value=(self.tiny_control_bundle(),index)), patch.object(s.shutil,"copyfile",side_effect=copy_then_change), patch.object(s,"evaluate_program") as numerical:
            with self.assertRaises(s.StudyError):
                s.run(self.root,cp,gp,s.RUN_OUTPUT)
            numerical.assert_not_called()
        self.assertFalse(s.load_json(self.root,s.RUN_OUTPUT+"/manifest.json")["new_numerical_evaluation_performed"])

    def full_count_control_run(self, trigger=None):
        """Actual 3168/5760 control tokens, no real inputs or mathematical calls."""
        cp,gp,_,_,index=self.gate_fixture()
        programs,occurrences,digests={}, {}, {}
        number=0
        for i in range(3168):
            pid=hashlib.sha256(f"synthetic-program-{i}".encode()).hexdigest()
            own=[]
            for _ in range(10 if i < 288 else 1):
                oid=hashlib.sha256(f"synthetic-occurrence-{number}".encode()).hexdigest();number+=1
                own.append(oid);occurrences[oid]={"occurrence_id":oid,"program_id":pid}
            programs[pid]={"program_id":pid,"program":{"metadata":{"world_id":0}},"occurrence_ids":own}
            digests[pid]="3"*64
        bundle={"programs":programs,"occurrences":occurrences,"request_digests":digests,
                "worlds":{0:{"synthetic":True}},"world_hashes":{0:"4"*64}}
        clock=[0.0];summary_finished=[]
        original_build=s.build_output_index;original_write=s.write_json;original_bytes=Path.write_bytes;original_identity=s.raw_identity
        def summary(*args):
            summary_finished.append(True)
            if trigger=="summary":clock[0]=3601.0
            if trigger=="control_after_numeric":self.write(gp,{"decision":"NO_GO"})
            return {"synthetic_control_tokens":True}
        def build(out):
            result=original_build(out)
            if trigger=="output_index":clock[0]=3601.0
            if trigger=="control_during_index":self.write(gp,{"decision":"NO_GO"})
            if trigger=="code_during_index":self.write(s.NUMERICAL_CODE[-1],"changed numerical source")
            if trigger=="input_during_index":self.write(s.BASE+"/program_upper_certificates.jsonl","changed original input")
            if trigger=="snapshot_during_index":(out/"source_snapshot"/s.NUMERICAL_CODE[-1]).write_text("changed copied source")
            return result
        def write(path,value):
            result=original_write(path,value)
            if trigger=="index_write" and path.name=="OUTPUT_INDEX.json":clock[0]=3601.0
            return result
        def write_bytes(path,value):
            result=original_bytes(path,value)
            if trigger=="terminal_write" and path.name=="manifest.json" and b'"complete": true' in value:clock[0]=3601.0
            if trigger=="control_during_terminal_write" and path.name=="manifest.json" and b'"complete": true' in value:self.write(gp,{"decision":"NO_GO"})
            return result
        def identity(path):
            result=original_identity(path)
            if trigger=="snapshot_recheck" and summary_finished and "source_snapshot" in path.parts:clock[0]=3601.0
            return result
        error=None
        with self.runtime_controls(index), patch.object(s,"load_fixed_inputs",return_value=(bundle,index)), patch.object(s,"evaluate_program",side_effect=lambda p,*_: {"program_id":p["program_id"]}), patch.object(s,"occurrence_result",side_effect=lambda original,_:original), patch.object(s,"source_comparisons",return_value=[]), patch.object(s,"summarize",side_effect=summary), patch.object(s.time,"monotonic",side_effect=lambda:clock[0]), patch.object(s,"build_output_index",side_effect=build), patch.object(s,"write_json",side_effect=write), patch.object(Path,"write_bytes",write_bytes), patch.object(s,"raw_identity",side_effect=identity):
            try:s.run(self.root,cp,gp,s.RUN_OUTPUT)
            except s.StudyError as exc:error=exc
        manifest=s.load_json(self.root,s.RUN_OUTPUT+"/manifest.json")
        self.assertEqual(manifest["programs_completed"],3168)
        self.assertEqual(manifest["occurrences_completed"],5760)
        output=s.load_json(self.root,s.RUN_OUTPUT+"/OUTPUT_INDEX.json")
        for path,expected in output["files"].items():
            self.assertEqual(s.raw_identity(self.root/s.RUN_OUTPUT/path),expected)
        return manifest,error

    def check_postprocessing_deadline(self,stage):
        manifest,error=self.full_count_control_run(stage)
        self.assertIsInstance(error,s.DeadlineReached)
        self.assertEqual(manifest["status"],"incomplete_deadline")
        self.assertFalse(manifest["complete"])

    def test_reviewer_deadline_during_summary_is_incomplete(self):
        self.check_postprocessing_deadline("summary")

    def test_deadline_during_final_snapshot_check_is_incomplete(self):
        self.check_postprocessing_deadline("snapshot_recheck")

    def test_deadline_during_output_index_construction_is_incomplete(self):
        self.check_postprocessing_deadline("output_index")

    def test_deadline_during_index_write_is_incomplete(self):
        self.check_postprocessing_deadline("index_write")

    def test_deadline_during_terminal_manifest_write_is_incomplete(self):
        self.check_postprocessing_deadline("terminal_write")

    def test_control_changed_after_numeric_still_compares_first_identity(self):
        manifest,error=self.full_count_control_run("control_after_numeric")
        self.assertIsInstance(error,s.StudyError)
        self.assertFalse(manifest["complete"])
        self.assertFalse(manifest["all_sources_rechecked_at_end"])

    def test_control_changed_during_terminal_index_is_rejected(self):
        manifest,error=self.full_count_control_run("control_during_index")
        self.assertIsInstance(error,s.StudyError)
        self.assertFalse(manifest["complete"])
        self.assertFalse(manifest["all_sources_rechecked_at_end"])

    def test_control_changed_during_terminal_manifest_is_rejected(self):
        manifest,error=self.full_count_control_run("control_during_terminal_write")
        self.assertIsInstance(error,s.StudyError)
        self.assertFalse(manifest["complete"])
        self.assertFalse(manifest["all_sources_rechecked_at_end"])

    def test_numerical_source_changed_during_index_is_rejected(self):
        manifest,error=self.full_count_control_run("code_during_index")
        self.assertIsInstance(error,s.StudyError)
        self.assertFalse(manifest["complete"])
        self.assertFalse(manifest["all_sources_rechecked_at_end"])

    def test_original_input_changed_during_index_is_rejected(self):
        manifest,error=self.full_count_control_run("input_during_index")
        self.assertIsInstance(error,s.StudyError)
        self.assertFalse(manifest["complete"])
        self.assertFalse(manifest["all_sources_rechecked_at_end"])

    def test_snapshot_changed_during_index_is_rejected(self):
        manifest,error=self.full_count_control_run("snapshot_during_index")
        self.assertIsInstance(error,s.StudyError)
        self.assertFalse(manifest["complete"])

    def test_within_deadline_terminal_manifest_and_index_match(self):
        manifest,error=self.full_count_control_run()
        self.assertIsNone(error)
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["elapsed_wall_seconds"],0)
        final_check=manifest["final_integrity_check"]
        self.assertLessEqual(final_check["started_at_utc"],final_check["completed_at_utc"])
        self.assertLessEqual(final_check["completed_at_utc"],manifest["completed_at_utc"])


if __name__ == "__main__":
    unittest.main()
