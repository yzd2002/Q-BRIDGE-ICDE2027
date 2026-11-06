"""SYNTHETIC storage fixtures only; never build or replay the real 7.4GB closure."""
from contextlib import contextmanager, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import partition as p
else:
    import partition as p

f = p.f
TOOL_SOURCE = Path(__file__).resolve().parents[2] / "runs/icde2027/analysis/partition_capacity_public_replay_v6"


def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


class Fixture(unittest.TestCase):
    """744 tiny artificial inputs; only the eleven validator/tool sources are real."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="SYNTHETIC_PARTITION_", dir=os.environ.get("V6_PARTITION_ADAPTER_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "baseline"
        self.family = self.root / "family_addon"
        self.tools = self.root / "tools_source"
        self.inputs = self.root / "inputs_source"
        self.control = self.root / "control"
        self.plans = self.root / "plans"
        self.output = self.root / "partition_addon"
        self.restored = self.root / "restored"
        for path in (self.base, self.tools, self.inputs, self.control, self.plans):
            path.mkdir()
        self.a, self.b, self.c = b"SYNTHETIC_BASE\n", b"SYNTHETIC_FAMILY\n", b"SYNTHETIC_NEW\n"
        self.tool_raws = {name: (TOOL_SOURCE / name).read_bytes() for name in p.TOOL_PINS}
        self.boundary, self.vendor = p._validators(self.tool_raws)
        for name, raw in self.tool_raws.items():
            put(self.tools / name, raw)
        self._family()
        self._manifest()
        self.plan_path = self.plans / "FILE_MAP.json"

    def _family(self):
        """Build a tiny genuine family-format addon in the synthetic fixture."""
        family_inputs = self.root / "family_inputs"
        family_replay = self.root / "family_replay"
        family_inputs.mkdir(); family_replay.mkdir()
        base_values = {"physical/a.txt": self.a, "physical/vendor.py": self.tool_raws["vendor_portable_io.py"]}
        for name, raw in base_values.items():
            put(self.base / name, raw)
        base_doc = {"schema": "qbridge-public-file-manifest-v1",
            "metadata_self_exclusions": [f.BASELINE_MANIFEST, "SHA256SUMS"],
            "file_count": len(base_values) + 2,
            "files": [{"path": name, **f.digest(raw)} for name, raw in sorted(base_values.items())]}
        put(self.base / f.BASELINE_MANIFEST, f.encode(base_doc))
        put(self.base / "SHA256SUMS", b"# SYNTHETIC fixture only\n")
        self.base_sha = f.digest((self.base / f.BASELINE_MANIFEST).read_bytes())["sha256"]
        put(family_inputs / "old/family.txt", self.b)
        family_manifest = {"format": "policy-family-portable-inputs-v1", "historical_root": f.HISTORICAL_ROOT,
            "files": {"old/family.txt": {**f.digest(self.b), "roles": ["synthetic"]}},
            "file_count": 1, "total_bytes": len(self.b)}
        manifest_raw = f.encode(family_manifest)
        put(family_replay / f.INPUTS, manifest_raw)
        layout = {"format": "policy-family-portable-layout-v1", "historical_runtime_bytes_included": False,
            "local_input_source": "synthetic/family_inputs", "input_file_count": 1, "input_bytes": len(self.b),
            "input_manifest_sha256": f.digest(manifest_raw)["sha256"],
            "project_input_files": {"old/family.txt": {**f.digest(self.b), "destination": "project/old/family.txt"}},
            "tool_and_evidence_files": {f.INPUTS: {**f.digest(manifest_raw), "destination": "policy_replay/" + f.INPUTS}}}
        layout_raw = f.encode(layout)
        put(family_replay / f.LAYOUT, layout_raw)
        put(family_replay / f.DELIVERY, f.encode({"format": "policy-family-portable-delivery-index-v1",
            "input_manifest": f.digest(manifest_raw), "files": {f.INPUTS: f.digest(manifest_raw), f.LAYOUT: f.digest(layout_raw)}}))
        plan = f.plan(self.base, family_replay)
        fp = self.plans / "family.json"
        fp.write_bytes(f.encode(plan))
        self.family_receipt = f.build(self.base, family_replay, family_inputs, fp,
            f.digest(fp.read_bytes())["sha256"], self.family)
        self.family_sha = self.family_receipt["completion_sha256"]

    def _manifest(self):
        b = self.boundary
        runtime = {f".python/synthetic_{i:03d}.py": f.digest(f"NOT_DISTRIBUTED_{i}".encode()) for i in range(77)}
        self.data = {f"synthetic/input_{i:03d}.txt": self.a if i == 0 else self.b if i == 1 else self.c for i in range(737)}
        logical = {name: f.digest(raw) for name, raw in self.data.items()}
        logical.update(runtime)
        logical.update({b.RUN + "/source_snapshot/" + name: rec for name, rec in runtime.items()})
        self.assertEqual(len(logical), 891)
        evidence_prefix = b.AUDIT_DIR + "/SYNTHETIC_AUDIT_ONLY/"
        gate = f.digest(b"SYNTHETIC_GATE_ONLY")["sha256"]
        reference = {name: f.encode({"SYNTHETIC_ONLY": True, "file": name}) for name in b.REFERENCE_OUTPUTS}
        reference["bound_input_hashes.json"] = f.encode(logical)
        # This artificial admission is for storage-path tests, never mathematical evidence.
        reference["report.json"] = f.encode({"status": "PASS", "complete": True, "production": True,
            "SYNTHETIC_FIXTURE_ONLY_NOT_SCIENTIFIC_EVIDENCE": True,
            "independent_result_audit": True, "all_sources_and_outputs_rechecked": True,
            "parent_expected_gate_sha256": gate, "producer_actual_runtime": {"files": runtime},
            "input_output_files_bound": 891,
            "counts": {k: b.EXPECTED[k] for k in ("programs", "occurrences", "contexts", "historical_certificates")},
            "audit_code": {"audit.py": {"sha256": b.ORIGINAL_AUDIT_SHA}, "math_checks.py": {"sha256": b.ORIGINAL_MATH_SHA}}})
        reference["OUTPUT_HASHES.json"] = f.encode({name: f.digest(raw) for name, raw in reference.items()})
        authority = {"schema_version": "partition-capacity-portable-admission/1", "decision": "GO",
            "scope": "completed-run-and-independent-audit-portable-replay", "run": b.RUN,
            "expected_gate_sha256": gate, "expected_counts": dict(b.EXPECTED),
            "original_auditor_sources": {"audit.py": b.ORIGINAL_AUDIT_SHA, "math_checks.py": b.ORIGINAL_MATH_SHA},
            "evidence": {name: {"path": evidence_prefix + name, **f.digest(raw)} for name, raw in reference.items()}}
        self.data.update({evidence_prefix + name: raw for name, raw in reference.items()})
        self.manifest = b.compose_manifest(logical, runtime, authority, {name: f.digest(raw) for name, raw in self.tool_raws.items()})
        self.assertEqual(len(self.data), 744)
        for name, raw in self.data.items():
            put(self.inputs / name, raw)
        self.manifest_path = self.control / p.MANIFEST
        self._save_manifest()

    def _save_manifest(self):
        self.manifest_raw = json.dumps(self.manifest, indent=1).encode() + b"\n\n"
        self.manifest_path.write_bytes(self.manifest_raw)
        self.manifest_sha = f.digest(self.manifest_raw)["sha256"]

    def make_plan(self):
        self.selected = p.plan(self.base, self.family, self.tools, self.manifest_path,
            self.base_sha, self.family_sha, self.manifest_sha)
        self.plan_path.write_bytes(f.encode(self.selected))
        self.plan_sha = f.digest(self.plan_path.read_bytes())["sha256"]
        return self.selected

    def build(self):
        if not self.plan_path.exists():
            self.make_plan()
        self.receipt = p.build(self.base, self.family, self.tools, self.inputs, self.manifest_path,
            self.base_sha, self.family_sha, self.manifest_sha, self.plan_path, self.plan_sha, self.output)
        return self.receipt

    def restore(self, output=None, **kwargs):
        return p.restore(self.base, self.family, self.output, self.base_sha, self.family_sha,
            self.manifest_sha, kwargs.get("completion_sha", self.receipt["completion_sha256"]), output or self.restored)

    @contextmanager
    def rejected(self, output=None):
        with self.assertRaises((RuntimeError, OSError, ValueError, KeyError, TypeError)):
            yield
        if output:
            self.assertFalse((output / p.COMPLETE).exists(), "Failed operation left a complete marker")

    def resign_map(self, change):
        value = f.parse((self.output / p.MAP).read_bytes())
        change(value)
        raw = f.encode(value)
        (self.output / p.MAP).write_bytes(raw)
        receipt = f.parse((self.output / p.COMPLETE).read_bytes())
        receipt["payload"][p.MAP] = f.digest(raw)
        receipt["plan_sha256"] = f.digest(raw)["sha256"]
        raw_receipt = f.encode(receipt)
        (self.output / p.COMPLETE).write_bytes(raw_receipt)
        self.receipt["completion_sha256"] = f.digest(raw_receipt)["sha256"]


class RoundTripTests(Fixture):
    def test_three_stores_exact_raw_manifest_original_checks_and_permissions(self):
        before = os.umask(0o022)
        try:
            self.build()
            observed = os.umask(0o022)
            self.assertEqual(observed, 0o022)
            result = self.restore()
        finally:
            os.umask(before)
        self.assertTrue(result["complete"])
        map_raw = (self.output / p.MAP).read_bytes()
        self.assertEqual(map_raw, self.plan_path.read_bytes())
        self.assertEqual(f.digest(map_raw)["sha256"], self.plan_sha)
        self.assertEqual(self.receipt["plan_sha256"], self.plan_sha)
        self.assertEqual(self.receipt["payload"][p.MAP], f.digest(map_raw))
        self.assertEqual(result["original_verify_tree"]["files"], 744)
        self.assertEqual(result["counts"]["restored_files"], 756)
        self.assertFalse(result["checks"]["scientific_replay_executed"])
        self.assertFalse(result["historical_runtime_bytes_revalidated"])
        self.assertEqual((self.restored / p.MANIFEST).read_bytes(), self.manifest_raw)
        for name, raw in self.data.items():
            self.assertEqual((self.restored / "project" / name).read_bytes(), raw)
        for name, raw in self.tool_raws.items():
            self.assertEqual((self.restored / "tools" / name).read_bytes(), raw)
        self.assertEqual({r["storage"] for r in self.selected["artifacts"].values()}, set(p.STORES))
        aliases = [self.selected["artifacts"]["project/synthetic/input_%03d.txt" % i]["path"] for i in range(2, 737)]
        self.assertEqual(len(set(aliases)), 1)
        self.assertEqual(len(list((self.output / "objects/sha256").iterdir())), self.selected["counts"]["new_objects"])
        for directory in (self.output, self.restored):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            for path in directory.rglob("*"):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700 if path.is_dir() else 0o600)

    def test_restore_uses_no_original_data_tools_manifest_or_plan_root(self):
        self.build()
        for path in (self.inputs, self.tools, self.control, self.plans, self.root / "family_inputs", self.root / "family_replay"):
            shutil.rmtree(path)
        self.restore()
        self.assertEqual((self.restored / "project/synthetic/input_736.txt").read_bytes(), self.c)

    def test_plan_does_not_read_scientific_payload_or_collect_private_files(self):
        (self.inputs / "synthetic/input_002.txt").unlink()
        put(self.inputs / "private.txt", b"excluded")
        result = self.make_plan()
        self.assertEqual(result["verification"], "metadata_plan_not_payload_verified")
        self.assertNotIn("project/private.txt", result["artifacts"])
        with self.rejected(self.output):
            self.build()

    def test_cli_synthetic_plan_build_restore_exit_zero_and_real_stdout_pins(self):
        script = Path(p.__file__).absolute()
        pins = ["--baseline", str(self.base), "--baseline-manifest-sha256", self.base_sha,
                "--family-addon", str(self.family), "--family-completion-sha256", self.family_sha,
                "--manifest-sha256", self.manifest_sha]
        def run(operation, extra):
            done = subprocess.run([sys.executable, "-B", "-I", "-S", str(script), operation, *pins, *extra],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(done.returncode, 0, done.stderr)
            return json.loads(done.stdout)
        planned = run("plan", ["--replay-tools", str(self.tools), "--manifest", str(self.manifest_path), "--output", str(self.plan_path)])
        built = run("build", ["--replay-tools", str(self.tools), "--manifest", str(self.manifest_path), "--input-root", str(self.inputs),
            "--plan", str(self.plan_path), "--plan-sha256", planned["plan"]["sha256"], "--output", str(self.output)])
        map_raw = (self.output / p.MAP).read_bytes()
        self.assertEqual(map_raw, self.plan_path.read_bytes())
        self.assertEqual(f.digest(map_raw), planned["plan"])
        self.assertEqual(built["plan_sha256"], planned["plan"]["sha256"])
        self.assertEqual(f.parse((self.output / p.COMPLETE).read_bytes())["payload"][p.MAP], planned["plan"])
        self.assertEqual(built["completion_sha256"], f.digest((self.output / p.COMPLETE).read_bytes())["sha256"])
        restored = run("restore", ["--partition", str(self.output), "--partition-completion-sha256", built["completion_sha256"], "--output", str(self.restored)])
        self.assertEqual(restored["completion_sha256"], f.digest((self.restored / p.COMPLETE).read_bytes())["sha256"])


class AdmissionTests(Fixture):
    def test_cli_plan_cannot_write_into_readonly_sources(self):
        forbidden = self.base / "FORBIDDEN_PLAN.json"
        with redirect_stderr(io.StringIO()):
            code = p.main(["plan", "--baseline", str(self.base), "--baseline-manifest-sha256", self.base_sha,
                "--family-addon", str(self.family), "--family-completion-sha256", self.family_sha,
                "--manifest", str(self.manifest_path), "--manifest-sha256", self.manifest_sha,
                "--replay-tools", str(self.tools), "--output", str(forbidden)])
        self.assertEqual(code, 1)
        self.assertFalse(forbidden.exists())

    def test_all_three_external_input_pins_are_mandatory(self):
        for i in range(3):
            with self.subTest(pin=i):
                pins = [self.base_sha, self.family_sha, self.manifest_sha]
                pins[i] = "0" * 64
                with self.rejected():
                    p.plan(self.base, self.family, self.tools, self.manifest_path, *pins)

    def test_changed_baseline_bytes_cannot_be_relabelled_with_a_new_pin(self):
        path = self.base / f.BASELINE_MANIFEST
        path.write_bytes(path.read_bytes() + b"\n")
        self.base_sha = f.digest(path.read_bytes())["sha256"]
        with self.rejected():
            self.make_plan()

    def test_wrong_census_closed_manifest_and_duplicate_json_keys(self):
        original = self.manifest_raw
        for mode in ("census", "unknown", "duplicate"):
            with self.subTest(mode=mode):
                value = json.loads(original)
                if mode == "census": value["file_count"] = 743
                if mode == "unknown": value["unknown"] = True
                raw = f.encode(value)
                if mode == "duplicate": raw = raw.replace(b'"format":', b'"format": "bad", "format":', 1)
                self.manifest_path.write_bytes(raw)
                self.manifest_sha = f.digest(raw)["sha256"]
                with self.rejected(): self.make_plan()

    def test_changed_tool_never_executes(self):
        put(self.tools / "boundary.py", b"raise AssertionError('UNTRUSTED MUST NOT RUN')\n")
        with self.rejected(): self.make_plan()

    def test_hardlinked_source_metadata_is_rejected(self):
        os.link(self.manifest_path, self.control / "other-name.json")
        with self.rejected(): self.make_plan()

    def test_family_extra_member_and_hardlinked_object_are_rejected(self):
        extra = self.family / "extra.txt"
        extra.write_bytes(b"unlisted")
        with self.rejected(): self.make_plan()
        extra.unlink()
        obj = self.family / ("objects/sha256/" + f.digest(self.b)["sha256"])
        os.link(obj, self.root / "family-hardlink")
        with self.rejected(): self.make_plan()


class BuildTests(Fixture):
    def assert_noncanonical_plan_rejected_by_api_and_cli(self, raw):
        self.assertEqual(json.loads(raw), self.selected)
        self.assertNotEqual(raw, f.encode(self.selected))
        self.plan_path.write_bytes(raw)
        self.plan_sha = f.digest(raw)["sha256"]
        with self.assertRaisesRegex(f.BoundaryError, "Plan bytes must use the canonical encoding"):
            self.build()
        self.assertFalse(self.output.exists(), "Noncanonical plan created an output before rejection")
        done = subprocess.run([sys.executable, "-B", "-I", "-S", str(Path(p.__file__).absolute()), "build",
            "--baseline", str(self.base), "--baseline-manifest-sha256", self.base_sha,
            "--family-addon", str(self.family), "--family-completion-sha256", self.family_sha,
            "--replay-tools", str(self.tools), "--manifest", str(self.manifest_path),
            "--manifest-sha256", self.manifest_sha, "--input-root", str(self.inputs),
            "--plan", str(self.plan_path), "--plan-sha256", self.plan_sha,
            "--output", str(self.output)], capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertEqual(done.stdout, "")
        failure = json.loads(done.stderr)
        self.assertEqual(failure["status"], "FAIL")
        self.assertIn("Plan bytes must use the canonical encoding", failure["error"])
        self.assertFalse(self.output.exists(), "CLI created an output for a noncanonical plan")

    def test_reindented_plan_with_valid_external_pin_rejected_before_output(self):
        self.make_plan()
        self.assert_noncanonical_plan_rejected_by_api_and_cli(
            (json.dumps(self.selected, sort_keys=True, indent=4) + "\n").encode())

    def test_compact_plan_with_valid_external_pin_rejected_before_output(self):
        self.make_plan()
        self.assert_noncanonical_plan_rejected_by_api_and_cli(
            json.dumps(self.selected, sort_keys=True, separators=(",", ":")).encode())

    def test_reordered_plan_with_valid_external_pin_rejected_before_output(self):
        self.make_plan()
        ordered = json.loads(f.encode(self.selected))
        reordered = dict(reversed(list(ordered.items())))
        self.assert_noncanonical_plan_rejected_by_api_and_cli(
            (json.dumps(reordered, indent=2) + "\n").encode())

    def test_fifo_source_rejected_without_blocking_read(self):
        self.make_plan()
        leaf = self.inputs / "synthetic/input_002.txt"
        leaf.unlink(); os.mkfifo(leaf, 0o600)
        with self.rejected(self.output): self.build()

    def test_concurrent_target_creation_is_not_overwritten(self):
        self.make_plan()
        original = f._rename_noreplace
        occupied = []
        def collide(parent_fd, old, new):
            os.mkdir(new, 0o700, dir_fd=parent_fd)
            occupied.append(os.stat(new, dir_fd=parent_fd).st_ino)
            return original(parent_fd, old, new)
        with patch.object(f, "_rename_noreplace", collide), self.rejected(self.output): self.build()
        self.assertEqual(occupied, [self.output.stat().st_ino])
        self.assertEqual(list(self.output.iterdir()), [])

    def test_mismatched_same_size_source_fails_before_completion(self):
        self.make_plan()
        (self.inputs / "synthetic/input_002.txt").write_bytes(b"X" * len(self.c))
        with self.rejected(self.output): self.build()

    def test_symlink_leaf_and_component_are_rejected(self):
        self.make_plan()
        leaf = self.inputs / "synthetic/input_002.txt"
        leaf.unlink(); leaf.symlink_to(self.inputs / "synthetic/input_003.txt")
        with self.rejected(self.output): self.build()
        leaf.unlink(); leaf.write_bytes(self.c)
        moved = self.root / "moved-inputs"
        (self.inputs / "synthetic").rename(moved)
        (self.inputs / "synthetic").symlink_to(moved, target_is_directory=True)
        with self.rejected(self.output): self.build()

    def test_hardlinked_new_and_reused_baseline_sources_rejected(self):
        self.make_plan()
        link = self.root / "hardlink"
        for source in (self.inputs / "synthetic/input_002.txt", self.base / "physical/a.txt"):
            os.link(source, link)
            with self.rejected(self.output): self.build()
            link.unlink()

    def test_existing_target_file_directory_and_symlink_preserved(self):
        self.make_plan()
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind):
                if kind == "file": self.output.write_bytes(b"keep")
                elif kind == "directory": self.output.mkdir()
                else: self.output.symlink_to(self.inputs, target_is_directory=True)
                with self.assertRaises((RuntimeError, OSError)): self.build()
                if kind == "file": self.assertEqual(self.output.read_bytes(), b"keep")
                if kind == "directory": self.output.rmdir()
                else: self.output.unlink()

    def test_output_inside_source_refused(self):
        self.make_plan(); self.output = self.inputs / "bad-output"
        with self.rejected(self.output): self.build()

    def test_altered_plan_even_with_new_external_pin_rejected(self):
        self.make_plan()
        self.selected["counts"]["new_objects"] += 1
        self.plan_path.write_bytes(f.encode(self.selected))
        self.plan_sha = f.digest(self.plan_path.read_bytes())["sha256"]
        with self.rejected(self.output): self.build()

    def test_source_changes_during_copy_fail_without_marker(self):
        self.make_plan()
        original = f.ReadSet.read
        changed = False
        def hook(reads, tree, rel, *args, **kwargs):
            nonlocal changed
            result = original(reads, tree, rel, *args, **kwargs)
            if kwargs.get("target") is not None and not changed:
                changed = True
                (self.base / "physical/a.txt").write_bytes(b"CHANGED")
            return result
        with patch.object(f.ReadSet, "read", hook), self.rejected(self.output): self.build()

    def test_failure_after_marker_creation_invalidates_our_marker(self):
        self.make_plan()
        original = f._verify_output
        def fail_after(tree, expected):
            original(tree, expected)
            if p.COMPLETE in expected:
                raise f.BoundaryError("SYNTHETIC post-marker failure")
        with patch.object(f, "_verify_output", fail_after), self.rejected(self.output): self.build()


class RestoreTests(Fixture):
    def test_wrong_completion_pin_and_existing_output_refused(self):
        self.build()
        with self.rejected(self.restored): self.restore(completion_sha="0" * 64)
        self.restored.mkdir()
        with self.assertRaises((RuntimeError, OSError)): self.restore()
        self.assertEqual(list(self.restored.iterdir()), [])

    def test_resigned_unknown_store_traversal_and_extra_schema_refused(self):
        self.build()
        old_map = (self.output / p.MAP).read_bytes()
        old_receipt = (self.output / p.COMPLETE).read_bytes()
        for mode in ("unknown_store", "traversal", "unknown_field"):
            with self.subTest(mode=mode):
                (self.output / p.MAP).write_bytes(old_map)
                (self.output / p.COMPLETE).write_bytes(old_receipt)
                def corrupt(value):
                    row = value["artifacts"]["tools/boundary.py"]
                    if mode == "unknown_store": row["storage"] = "author_machine"
                    elif mode == "traversal": row["path"] = "../outside"
                    else: value["unknown"] = True
                self.resign_map(corrupt)
                with self.rejected(self.restored): self.restore()

    def test_missing_corrupted_and_hardlinked_object_refused(self):
        self.build()
        path = self.output / self.selected["artifacts"]["project/synthetic/input_002.txt"]["path"]
        raw = path.read_bytes()
        path.unlink()
        with self.rejected(self.restored): self.restore()
        path.write_bytes(b"X" * len(raw))
        with self.rejected(self.restored): self.restore()
        path.write_bytes(raw); os.link(path, self.root / "object-hardlink")
        with self.rejected(self.restored): self.restore()

    def test_extra_payload_member_is_rejected(self):
        self.build(); (self.output / "extra").write_bytes(b"unlisted")
        with self.rejected(self.restored): self.restore()

    def test_original_reference_admission_runs_and_cannot_be_silently_skipped(self):
        self.build()
        original = p._validators
        called = []
        def wrapped(raws):
            boundary, vendor = original(raws)
            def reject(root, manifest):
                called.append(True)
                raise f.BoundaryError("SYNTHETIC original-admission rejection")
            boundary.check_reference_admission = reject
            return boundary, vendor
        with patch.object(p, "_validators", wrapped), self.rejected(self.restored): self.restore()
        self.assertEqual(called, [True])

    def test_restore_postmarker_failure_invalidates_completion(self):
        self.build()
        original = f._verify_output
        def fail_after(tree, expected):
            original(tree, expected)
            if p.COMPLETE in expected:
                raise f.BoundaryError("SYNTHETIC restore final failure")
        with patch.object(f, "_verify_output", fail_after), self.rejected(self.restored): self.restore()


if __name__ == "__main__":
    unittest.main()
