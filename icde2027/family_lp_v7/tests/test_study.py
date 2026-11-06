"""Input-only schedule tests; no real study request or solver is executed."""
from collections import Counter
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


study = load("study_under_test", ROOT / "icde2027/family_lp_v7/study.py")
general = load("general_digest_test", ROOT / "icde2027/family_lp_v7/verify_general.py")
old = load("old_digest_test", ROOT / "icde2027/partition_capacity_v6/verify.py")


def family(i):
    return {"program": {"lower": ["0/1", "0/1"], "upper": ["1/1", "1/1"],
                        "rows": [{"coefficients": ["1/1", "1/1"],
                                  "lower": "0/1", "upper": f"{i+1}/10"}]},
            "context_sizes": [2], "exact_weights": ["1/1"],
            "exact_reference": [["1/2", "1/2"]]}


def fixture():
    result = {key: {} for key in ("families", "programs", "fixed", "coat", "family_strata", "fixed_strata")}
    for i in range(6):
        request = {"schema_version": "partition-capacity-input/1", **family(i)}
        rid = study.digest(request)
        result["families"][rid] = {"request": request, "task_id": rid}
        result["family_strata"][rid] = {str(i % 3), "9"}
        pid = study.digest(request["program"])
        result["programs"][pid] = {"program_identity": pid, "program": request["program"]}
        task = {"schema_version": "verified-fixed-task/1", "program_identity": pid,
                "objective": ["1/2", "-1/2"]}
        fid = study.digest(task)
        result["fixed"][fid] = {"task_id": fid, "task": task}
        result["fixed_strata"][fid] = {str(i % 3)}
    for i in range(3):
        request = {"schema_version": "verified-coat-family-task/1", **family(i)}
        rid = study.digest(request)
        result["coat"][rid] = {"request": request, "task_id": rid, "partition_supported": i < 2}
    result["timing_ids"] = sorted(result["families"])[:2]
    return result


class ScheduleTests(unittest.TestCase):
    def test_complete_population_and_three_timing_rounds(self):
        data = fixture()
        queue = study.make_queue(data)
        self.assertEqual(Counter(r["phase"] for r in queue),
                         {"synthetic_quality": 12, "fixed_quality": 6, "coat_quality": 5, "controlled_timing": 12})
        self.assertEqual(len(queue), len({r["call_id"] for r in queue}))
        self.assertEqual([r["sequence"] for r in queue], list(range(len(queue))))
        for rid in data["families"]:
            rows = [r for r in queue if r["phase"] == "synthetic_quality" and r["task_id"] == rid]
            self.assertEqual({r["method"] for r in rows}, {"S", "G_partition"})
        for rid in data["timing_ids"]:
            rows = [r for r in queue if r["phase"] == "controlled_timing" and r["task_id"] == rid]
            self.assertEqual(Counter((r["round"], r["method"]) for r in rows),
                             Counter((n, m) for n in (1, 2, 3) for m in ("S", "G_partition")))

    def test_no_unsupported_specialized_calls(self):
        data = fixture()
        queue = study.make_queue(data)
        for rid, item in data["coat"].items():
            methods = [r["method"] for r in queue if r["namespace"] == "coat" and r["task_id"] == rid]
            self.assertEqual(set(methods), {"S", "G_partition"} if item["partition_supported"] else {"G_general"})

    def test_pair_order_is_balanced(self):
        rows = study.make_queue(fixture())
        for phase in ("synthetic_quality", "coat_quality", "controlled_timing"):
            selected = [r for r in rows if r["phase"] == phase and r["method"] != "G_general"]
            first = Counter(selected[i]["method"] for i in range(0, len(selected), 2))
            self.assertLessEqual(abs(first["S"] - first["G_partition"]), 1)
            for i in range(0, len(selected), 2):
                self.assertEqual(selected[i]["task_id"], selected[i+1]["task_id"])

    def test_randomization_depends_on_inputs_only(self):
        original = fixture()
        changed = copy.deepcopy(original)
        for item in changed["families"].values():
            item.update(old_capacity="999/1", failure=True, runtime=0, favorable=False)
        self.assertEqual(study.make_queue(original), study.make_queue(changed))

    def test_payload_whitelist_excludes_answer_metadata(self):
        data = fixture()
        rid = next(iter(data["families"]))
        data["families"][rid]["old_answer"] = "123/1"
        payload = study.request_for(data, "synthetic", rid)
        self.assertEqual(set(payload), study.FAMILY_KEYS)
        data["families"][rid]["request"]["oracle"] = "123/1"
        with self.assertRaises(ValueError):
            study.request_for(data, "synthetic", rid)

    def test_owner_stratum_round_robin(self):
        ids = ["a", "b", "c", "d", "e", "f"]
        memberships = {"a": {"0", "5"}, "b": {"0"}, "c": {"1"},
                       "d": {"1"}, "e": {"2"}, "f": {"2"}}
        order = study.balanced_ids(ids, memberships, study.SEED)
        self.assertEqual(set(order), set(ids))
        self.assertEqual({min(memberships[x]) for x in order[:3]}, {"0", "1", "2"})
        self.assertEqual(order, study.balanced_ids(list(reversed(ids)), memberships, study.SEED))

    def test_missing_stratum_rejected(self):
        with self.assertRaises(ValueError):
            study.balanced_ids(["missing"], {}, study.SEED)

    def test_all_receiver_identity_schemas(self):
        payload = family(0)
        args = (payload["program"], payload["context_sizes"], payload["exact_weights"], payload["exact_reference"])
        self.assertEqual(study.receiver_digest(payload, "S"), old.input_digest(*args))
        self.assertEqual(study.receiver_digest(payload, "G_partition"), old.input_digest(*args))
        self.assertEqual(study.receiver_digest(payload, "G_general"), general.family_input_digest(*args))
        fixed = {"program": payload["program"], "exact_objective": ["1/2", "-1/2"]}
        self.assertEqual(study.receiver_digest(fixed, "G_fixed"), general.fixed_input_digest(fixed["program"], fixed["exact_objective"]))
        self.assertNotEqual(study.digest(payload), study.receiver_digest(payload, "G_general"))
        self.assertNotEqual(study.digest({"schema_version": "verified-coat-family-task/1", **payload}), study.receiver_digest(payload, "S"))

    def test_input_change_changes_call_identity(self):
        data = fixture()
        before = study.make_queue(data)
        rid = next(iter(data["families"]))
        data["families"][rid]["request"]["program"]["upper"][0] = "9/10"
        after = study.make_queue(data)
        before_ids = {r["call_id"] for r in before if r["task_id"] == rid}
        after_ids = {r["call_id"] for r in after if r["task_id"] == rid}
        self.assertTrue(before_ids.isdisjoint(after_ids))

    def test_json_duplicate_and_nonfinite_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.assertRaises(ValueError):
                study.loads(raw)

    def test_existing_output_and_source_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            study.write_new(path, {"valid": True})
            expected = study.identity(path)
            with self.assertRaises(FileExistsError):
                study.write_new(path, {"valid": False})
            path.write_text("modified")
            with self.assertRaises(ValueError):
                study.bound_file(path, expected)

    def test_truncated_jsonl_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.jsonl"
            path.write_bytes(b'{"sequence":0}')
            with self.assertRaises(ValueError):
                list(study.records(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
