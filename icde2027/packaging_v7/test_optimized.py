"""Small, synthetic, standard-library-only storage tests. No production data."""

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("optimized_storage", Path(__file__).with_name("optimized.py"))
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(data):
    return {"bytes": len(data), "sha256": sha(data)}


class OptimizedStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="optimized-storage-test-")
        self.root = Path(self.temporary.name)
        self.run = self.root / "source"
        self.run.mkdir()
        self.payloads = {
            "completion.json": b'{"synthetic": true}\n',
            "quality/a/raw.jsonl.gz": gzip.compress(b"synthetic raw bytes\n" * 97, mtime=42),
            "quality/b/raw.jsonl.gz": gzip.compress(b"synthetic raw bytes\n" * 97, mtime=42),
            "timing/empty": b"",
            "timing/opaque.txt": b"/historical/inaccessible/root/identity-only\n",
            "unicode/caf\u00e9.txt": b"portable UTF-8 name\n",
        }
        for member, data in self.payloads.items():
            target = self.run / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        self.auditor = self.root / "audit.py"
        self.auditor.write_bytes(b"# synthetic auditor; must never be executed\nraise RuntimeError('executed')\n")
        self.report = self.root / "RESULTS.json"
        self.report_value = {
            "status": storage.PASS_STATUS,
            "production_protocol": True,
            "file_count": len(self.payloads),
            "files": {name: identity(data) for name, data in self.payloads.items()},
            "run": "/does/not/exist/identity-only",
            "audit_source_sha256": sha(self.auditor.read_bytes()),
        }
        self.write_report()
        self.bundle = self.root / "bundle"
        self.restored = self.root / "restored"

    def tearDown(self):
        self.temporary.cleanup()

    def write_report(self, data=None):
        self.report.write_bytes(data if data is not None else json.dumps(self.report_value).encode())

    def build(self, **overrides):
        options = dict(run=self.run, audit_results=self.report,
                       audit_sha256=sha(self.report.read_bytes()), auditor=self.auditor,
                       auditor_sha256=sha(self.auditor.read_bytes()), output=self.bundle)
        options.update(overrides)
        return storage.build(**options)

    def restore(self, **overrides):
        options = dict(bundle=self.bundle, output=self.restored)
        if "manifest_sha256" not in overrides:
            options["manifest_sha256"] = sha((self.bundle / "MANIFEST.json").read_bytes())
        options.update(overrides)
        return storage.restore(**options)

    def edit_manifest(self, change):
        path = self.bundle / "MANIFEST.json"
        value = json.loads(path.read_bytes())
        change(value)
        path.write_text(json.dumps(value))

    def reject(self, call):
        with self.assertRaises((storage.BundleError, OSError)):
            call()

    def test_roundtrip_preserves_all_bytes_and_scopes(self):
        result = self.build()
        self.assertEqual(result["objects"], len(self.payloads) - 1)
        self.assertEqual(result["run_bytes"], sum(map(len, self.payloads.values())))
        self.assertEqual(result["object_bytes"], sum(map(len, set(self.payloads.values()))))
        restored = self.restore(manifest_sha256=result["manifest_sha256"])
        self.assertEqual(restored["status"], "RESTORED")
        found = {str(path.relative_to(self.restored / "run")): path.read_bytes()
                 for path in (self.restored / "run").rglob("*") if path.is_file()}
        self.assertEqual(found, self.payloads)
        self.assertEqual((self.restored / storage.AUDIT_MEMBER).read_bytes(), self.report.read_bytes())
        self.assertEqual((self.restored / storage.AUDITOR_MEMBER).read_bytes(), self.auditor.read_bytes())
        self.assertEqual((self.restored / storage.RESTORER_MEMBER).read_bytes(), Path(storage.__file__).read_bytes())
        duplicates = [self.restored / "run" / name for name in self.payloads if name.endswith(".gz")]
        self.assertNotEqual(duplicates[0].stat().st_ino, duplicates[1].stat().st_ino)
        self.assertEqual(gzip.decompress(duplicates[0].read_bytes()), b"synthetic raw bytes\n" * 97)
        self.assertEqual({str(path.relative_to(self.run)): path.read_bytes()
                          for path in self.run.rglob("*") if path.is_file()}, self.payloads)

    def test_streaming_large_member_uses_bounded_reads(self):
        data = bytes(range(256)) * 2051
        (self.run / "larger.bin").write_bytes(data)
        self.payloads["larger.bin"] = data
        self.report_value["files"]["larger.bin"] = identity(data)
        self.report_value["file_count"] += 1
        self.write_report()
        with mock.patch.object(storage, "CHUNK_BYTES", 4096):
            self.build()
            self.restore()
        self.assertEqual((self.restored / "run/larger.bin").read_bytes(), data)

    def test_reformatted_report_bytes_remain_verbatim(self):
        self.write_report((json.dumps(self.report_value, indent=7) + "\n\n").encode())
        self.build()
        self.restore()
        self.assertEqual((self.restored / storage.AUDIT_MEMBER).read_bytes(), self.report.read_bytes())

    def test_bundled_cli_runs_isolated_after_source_relocated(self):
        result = self.build()
        shutil.rmtree(self.run)
        self.report.unlink()
        self.auditor.unlink()
        argv = [sys.executable, "-B", "-I", "-S", str(self.bundle / storage.RESTORER_MEMBER),
                "restore", "--bundle", str(self.bundle), "--manifest-sha256", result["manifest_sha256"],
                "--output", str(self.restored)]
        outcome = subprocess.run(argv, capture_output=True, text=True, timeout=15, cwd=self.root)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        self.assertEqual(json.loads(outcome.stdout)["status"], "RESTORED")
        self.assertEqual((self.restored / "run/quality/a/raw.jsonl.gz").read_bytes(), self.payloads["quality/a/raw.jsonl.gz"])

    def test_build_rejects_external_report_pin_mismatch(self):
        self.reject(lambda: self.build(audit_sha256="0" * 64))
        self.assertFalse(self.bundle.exists())

    def test_build_rejects_external_auditor_pin_mismatch(self):
        self.reject(lambda: self.build(auditor_sha256="0" * 64))
        self.assertFalse(self.bundle.exists())

    def test_build_rejects_nonpassing_report(self):
        self.report_value["status"] = "FAILED"
        self.write_report()
        self.reject(self.build)

    def test_build_rejects_nonproduction_report(self):
        for value in (False, 1, "true"):
            with self.subTest(value=value):
                self.report_value["production_protocol"] = value
                self.write_report()
                self.reject(self.build)

    def test_build_rejects_wrong_report_count_or_auditor(self):
        for key, value in (("file_count", len(self.payloads) + 1), ("file_count", True),
                           ("audit_source_sha256", "0" * 64)):
            with self.subTest(key=key, value=value):
                old = self.report_value[key]
                self.report_value[key] = value
                self.write_report()
                self.reject(self.build)
                self.report_value[key] = old

    def test_build_rejects_duplicate_report_keys(self):
        self.write_report(b'{"status": "a", "status": "b"}')
        self.reject(self.build)

    def test_build_rejects_nonfinite_json(self):
        self.report_value["ignored"] = float("nan")
        self.write_report()
        self.reject(self.build)

    def test_build_rejects_extra_or_missing_run_files(self):
        unexpected = self.run / "unexpected"
        unexpected.write_bytes(b"extra")
        self.reject(self.build)
        unexpected.unlink()
        (self.run / "completion.json").unlink()
        self.reject(self.build)
        self.assertFalse(self.bundle.exists())

    def test_build_rejects_same_size_tampering_without_seal(self):
        path = self.run / "completion.json"
        path.write_bytes(b"!" * path.stat().st_size)
        self.reject(self.build)
        self.assertFalse((self.bundle / "MANIFEST.json").exists())

    def test_build_checks_duplicate_source_bytes_too(self):
        member = self.run / "quality/b/raw.jsonl.gz"
        member.write_bytes(b"!" * member.stat().st_size)
        self.reject(self.build)
        self.assertFalse((self.bundle / "MANIFEST.json").exists())

    def test_build_detects_source_added_during_copy(self):
        original = storage._copy_member
        changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            original(*args, **kwargs)
            if not changed:
                (self.run / "late-extra").write_bytes(b"late")
                changed = True
        with mock.patch.object(storage, "_copy_member", side_effect=mutate):
            self.reject(self.build)
        self.assertFalse((self.bundle / "MANIFEST.json").exists())

    def test_build_rechecks_original_control_identity(self):
        original = storage._copy_member
        changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            original(*args, **kwargs)
            if not changed:
                self.report.write_bytes(self.report.read_bytes() + b" ")
                changed = True
        with mock.patch.object(storage, "_copy_member", side_effect=mutate):
            self.reject(self.build)
        self.assertFalse((self.bundle / "MANIFEST.json").exists())

    def test_build_rejects_unsafe_report_paths(self):
        for unsafe in ("../escape", "/absolute", "a/../b", "a//b", "./x", "a\\b", "C:x", "a\nname"):
            with self.subTest(path=unsafe):
                original = self.report_value["files"]
                self.report_value["files"] = {unsafe: identity(b"x")}
                self.report_value["file_count"] = 1
                self.write_report()
                self.reject(self.build)
                self.report_value["files"] = original

    def test_build_rejects_path_collision(self):
        self.report_value["files"] = {"a": identity(b"x"), "a/b": identity(b"x")}
        self.report_value["file_count"] = 2
        self.write_report()
        self.reject(self.build)

    def test_build_rejects_invalid_file_record(self):
        for bad in ({"bytes": True, "sha256": "0" * 64}, {"bytes": -1, "sha256": "0" * 64},
                    {"bytes": 1, "sha256": "A" * 64}, {"bytes": 1, "sha256": "0" * 64, "extra": 1}):
            with self.subTest(record=bad):
                self.report_value["files"]["completion.json"] = bad
                self.write_report()
                self.reject(self.build)

    def test_build_rejects_conflicting_object_lengths(self):
        paths = [name for name in self.payloads if name.endswith(".gz")]
        self.report_value["files"][paths[0]]["bytes"] += 1
        self.write_report()
        self.reject(self.build)

    def test_build_rejects_symlink_root_ancestor_and_member(self):
        link = self.root / "linked"
        link.symlink_to(self.run, target_is_directory=True)
        self.reject(lambda: self.build(run=link))
        parent_link = self.root / "linked-parent"
        parent_link.symlink_to(self.root, target_is_directory=True)
        self.reject(lambda: self.build(run=parent_link / "source"))
        member = self.run / "completion.json"
        member.unlink()
        member.symlink_to(self.auditor)
        self.reject(self.build)

    def test_build_rejects_unlisted_symlink_directory(self):
        (self.run / "directory-link").symlink_to(self.root, target_is_directory=True)
        self.reject(self.build)

    def test_build_rejects_fifo_member(self):
        member = self.run / "completion.json"
        member.unlink()
        os.mkfifo(member)
        self.reject(self.build)

    def test_build_rejects_control_links_fifo_and_device(self):
        linked = self.root / "linked-report"
        linked.symlink_to(self.report)
        self.reject(lambda: self.build(audit_results=linked))
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        self.reject(lambda: self.build(audit_results=fifo))
        self.reject(lambda: self.build(auditor=Path("/dev/null")))

    def test_build_rejects_existing_and_contained_outputs(self):
        for output in (self.run, self.run / "nested-bundle", self.root, self.report):
            with self.subTest(output=output):
                self.reject(lambda: self.build(output=output))
        self.bundle.mkdir()
        self.reject(self.build)

    def test_build_rejects_dangling_or_ancestor_output_symlinks(self):
        self.bundle.symlink_to(self.root / "absent")
        self.reject(self.build)
        self.assertTrue(self.bundle.is_symlink())
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.reject(lambda: self.build(output=alias / "escape"))
        self.assertFalse((self.root / "escape").exists())

    def test_restore_rejects_external_manifest_pin_mismatch(self):
        self.build()
        self.reject(lambda: self.restore(manifest_sha256="0" * 64))
        self.assertFalse(self.restored.exists())

    def test_restore_rejects_extra_file_and_empty_directory(self):
        self.build()
        extra = self.bundle / "extra"
        extra.write_bytes(b"x")
        self.reject(self.restore)
        extra.unlink()
        extra.mkdir()
        self.reject(self.restore)

    def test_restore_rejects_missing_object(self):
        self.build()
        next((self.bundle / "objects").iterdir()).unlink()
        self.reject(self.restore)

    def test_restore_rejects_unreferenced_object(self):
        self.build()
        (self.bundle / "objects" / ("f" * 64)).write_bytes(b"unlisted")
        self.reject(self.restore)

    def test_restore_rejects_duplicate_manifest_keys(self):
        self.build()
        path = self.bundle / "MANIFEST.json"
        path.write_bytes(b'{"format":"a","format":"b"}')
        self.reject(self.restore)

    def test_restore_detects_source_mutation_during_copy(self):
        self.build()
        original = storage._copy_member
        changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            original(*args, **kwargs)
            if not changed:
                (self.bundle / "late-extra").write_bytes(b"late")
                changed = True
        with mock.patch.object(storage, "_copy_member", side_effect=mutate):
            self.reject(self.restore)
        self.assertFalse((self.restored / "MANIFEST.json").exists())

    def test_restore_rejects_same_size_object_tampering_without_seal(self):
        self.build()
        path = self.bundle / "objects" / sha(self.payloads["completion.json"])
        path.write_bytes(b"!" * path.stat().st_size)
        self.reject(self.restore)
        self.assertFalse((self.restored / "MANIFEST.json").exists())

    def test_restore_rejects_manifest_link_fifo_and_bundle_root_link(self):
        self.build()
        manifest = self.bundle / "MANIFEST.json"
        pin = sha(manifest.read_bytes())
        saved = self.root / "saved-manifest"
        manifest.rename(saved)
        manifest.symlink_to(saved)
        self.reject(lambda: self.restore(manifest_sha256=pin))
        manifest.unlink()
        os.mkfifo(manifest)
        self.reject(lambda: self.restore(manifest_sha256=pin))
        manifest.unlink()
        saved.rename(manifest)
        alias = self.root / "bundle-link"
        alias.symlink_to(self.bundle, target_is_directory=True)
        self.reject(lambda: self.restore(bundle=alias))

    def test_restore_rejects_member_directory_link_and_fifo(self):
        self.build()
        path = self.bundle / "objects" / sha(self.payloads["completion.json"])
        original = path.read_bytes()
        path.unlink()
        path.symlink_to(self.run / "completion.json")
        self.reject(self.restore)
        path.unlink()
        os.mkfifo(path)
        self.reject(self.restore)
        path.unlink()
        path.write_bytes(original)
        tools = self.bundle / "tools"
        tools.rename(self.root / "saved-tools")
        tools.symlink_to(self.root / "saved-tools", target_is_directory=True)
        self.reject(self.restore)

    def test_restore_rejects_control_tampering(self):
        self.build()
        for name in (storage.AUDIT_MEMBER, storage.AUDITOR_MEMBER, storage.RESTORER_MEMBER):
            with self.subTest(name=name):
                path = self.bundle / name
                original = path.read_bytes()
                path.write_bytes(b"!" * len(original))
                self.reject(self.restore)
                path.write_bytes(original)

    def test_restore_rejects_unsafe_manifest_path_with_updated_external_pin(self):
        self.build()
        def change(value):
            value["files"]["../escape"] = value["files"].pop("completion.json")
        self.edit_manifest(change)
        self.reject(self.restore)
        self.assertFalse((self.root / "escape").exists())

    def test_restore_rejects_incomplete_index_with_updated_external_pin(self):
        self.build()
        self.edit_manifest(lambda value: value["objects"].pop(next(iter(value["objects"]))))
        self.reject(self.restore)

    def test_restore_rejects_wrong_totals_with_updated_external_pin(self):
        self.build()
        self.edit_manifest(lambda value: value["totals"].update(run_files=999))
        self.reject(self.restore)

    def test_restore_rejects_audit_disagreement_even_with_consistent_manifest(self):
        self.build()
        def change(value):
            value["files"]["different-name"] = value["files"].pop("completion.json")
        self.edit_manifest(change)
        self.reject(self.restore)

    def test_restore_rejects_rebound_nonproduction_report(self):
        self.build()
        path = self.bundle / storage.AUDIT_MEMBER
        value = json.loads(path.read_bytes())
        value["production_protocol"] = False
        path.write_text(json.dumps(value))
        self.edit_manifest(lambda value: value["controls"].update({storage.AUDIT_MEMBER: identity(path.read_bytes())}))
        self.reject(self.restore)

    def test_restore_rejects_existing_and_overlapping_outputs(self):
        self.build()
        for output in (self.bundle, self.bundle / "nested", self.root):
            with self.subTest(output=output):
                self.reject(lambda: self.restore(output=output))
        self.restored.mkdir()
        self.reject(self.restore)

    def test_restore_rejects_output_symlink_and_parent_symlink(self):
        self.build()
        self.restored.symlink_to(self.root / "absent")
        self.reject(self.restore)
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.reject(lambda: self.restore(output=alias / "escape"))
        self.assertFalse((self.root / "escape").exists())

    def test_control_memory_limit_is_enforced(self):
        with mock.patch.object(storage, "MAX_JSON_BYTES", 8):
            self.reject(self.build)

    def test_cli_requires_external_pin(self):
        outcome = subprocess.run([sys.executable, "-B", "-I", "-S", storage.__file__,
                                  "restore", "--bundle", str(self.root), "--output", str(self.restored)],
                                 capture_output=True, text=True, timeout=15)
        self.assertNotEqual(outcome.returncode, 0)
        self.assertIn("--manifest-sha256", outcome.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
