"""Synthetic boundary tests; no production inputs or mathematical replay."""
from contextlib import contextmanager
import hashlib
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import family as f
else:
    import family as f


def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


class FamilyFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("V6_ADAPTER_TEST_TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.baseline = self.root / "baseline"
        self.replay = self.root / "replay"
        self.inputs = self.root / "inputs"
        self.out = self.root / "addon"
        self.restored = self.root / "restored"
        self.plan_path = self.root / "plan.json"
        for path in (self.baseline, self.replay, self.inputs):
            path.mkdir()
        self.a, self.b = b"existing baseline bytes\n", b"new family bytes\n"
        self.input_bytes = {"data/a.txt": self.a, "data/alias-a.txt": self.a,
                            "results/b.txt": self.b, "results/alias-b.txt": self.b}
        for rel, raw in self.input_bytes.items():
            put(self.inputs / rel, raw)
        put(self.baseline / "physical/a.txt", self.a)
        put(self.baseline / "z_other/a.txt", self.a)
        self.base_doc = {"schema": "qbridge-public-file-manifest-v1",
                         "metadata_self_exclusions": [f.BASELINE_MANIFEST, "SHA256SUMS"],
                         "file_count": 4,
                         "files": [{"path": name, **f.digest(self.a)}
                                   for name in ("physical/a.txt", "z_other/a.txt")]}
        self.manifest = {"format": "policy-family-portable-inputs-v1", "historical_root": f.HISTORICAL_ROOT,
                         "files": {name: {**f.digest(raw), "roles": ["synthetic"]}
                                   for name, raw in self.input_bytes.items()},
                         "file_count": len(self.input_bytes),
                         "total_bytes": sum(len(v) for v in self.input_bytes.values()),
                         "historical_runtime_not_distributed": {"provenance": "not copied"}}
        self.tool_bytes = {"replay.py": b"# synthetic placeholder; never executed\n",
                           "evidence/b.txt": self.b}
        self._metadata()

    def _metadata(self):
        put(self.baseline / f.BASELINE_MANIFEST, f.encode(self.base_doc))
        # Preserve deliberately noncanonical original bytes in the round-trip.
        raw_manifest = json.dumps(self.manifest, indent=1).encode() + b"\n\n"
        tools = {**self.tool_bytes, f.INPUTS: raw_manifest}
        for name, raw in tools.items():
            put(self.replay / name, raw)
        self.layout = {"format": "policy-family-portable-layout-v1",
                       "historical_runtime_bytes_included": False,
                       "local_input_source": "historical_restored/project",
                       "input_file_count": self.manifest["file_count"],
                       "input_bytes": self.manifest["total_bytes"],
                       "input_manifest_sha256": f.digest(raw_manifest)["sha256"],
                       "project_input_files": {name: {"sha256": rec["sha256"], "bytes": rec["bytes"],
                                                      "destination": "project/" + name}
                                               for name, rec in self.manifest["files"].items()},
                       "tool_and_evidence_files": {name: {**f.digest(raw), "destination": "policy_replay/" + name}
                                                   for name, raw in tools.items()}}
        raw_layout = f.encode(self.layout)
        self.delivery = {"format": "policy-family-portable-delivery-index-v1",
                         "input_manifest": f.digest(raw_manifest),
                         "files": {**{name: f.digest(raw) for name, raw in tools.items()},
                                   f.LAYOUT: f.digest(raw_layout)}}
        put(self.replay / f.LAYOUT, raw_layout)
        put(self.replay / f.DELIVERY, f.encode(self.delivery))

    def save_plan(self):
        self.plan = f.plan(self.baseline, self.replay)
        self.plan_path.write_bytes(f.encode(self.plan))
        self.plan_sha = f.digest(self.plan_path.read_bytes())["sha256"]
        return self.plan

    def build(self):
        if not self.plan_path.exists():
            self.save_plan()
        self.receipt = f.build(self.baseline, self.replay, self.inputs, self.plan_path, self.plan_sha, self.out)
        return self.receipt

    def restore(self):
        return f.restore(self.baseline, self.out, self.receipt["completion_sha256"], self.restored)

    @contextmanager
    def rejected(self, output=None):
        with self.assertRaises((f.BoundaryError, OSError, ValueError, KeyError, TypeError)):
            yield
        if output is not None:
            self.assertFalse((output / f.COMPLETE).exists(), "Failed work must not become completed")


class RoundTripTests(FamilyFixture):
    def test_content_addressed_round_trip_preserves_manifest_bytes(self):
        self.build()
        result = self.restore()
        self.assertTrue(result["complete"])
        self.assertEqual(self.plan["counts"]["project_unique_sha256"], 2)
        self.assertEqual(self.plan["counts"]["baseline_project_unique_candidates"], 1)
        for name, raw in self.input_bytes.items():
            self.assertEqual((self.restored / "project" / name).read_bytes(), raw)
        for name in (f.LAYOUT, f.DELIVERY):
            self.assertEqual((self.restored / name).read_bytes(), (self.replay / name).read_bytes())
        self.assertEqual((self.restored / "policy_replay" / f.INPUTS).read_bytes(), (self.replay / f.INPUTS).read_bytes())
        self.assertEqual(len(list((self.out / "objects/sha256").iterdir())), self.plan["counts"]["addon_objects"])
        self.assertFalse(result["checks"]["scientific_replay_executed"])

    def test_new_data_shared_by_tools_is_stored_once(self):
        self.build()
        rows = [self.plan["artifacts"][name] for name in
                ("project/results/b.txt", "project/results/alias-b.txt", "policy_replay/evidence/b.txt")]
        self.assertEqual(len({row["path"] for row in rows}), 1)
        self.assertEqual((self.out / rows[0]["path"]).read_bytes(), self.b)

    def test_restore_requires_no_original_source_tree(self):
        self.build()
        shutil.rmtree(self.inputs)
        shutil.rmtree(self.replay)
        self.plan_path.unlink()
        self.restore()
        self.assertEqual((self.restored / "project/results/b.txt").read_bytes(), self.b)

    def test_metadata_plan_never_reads_candidate_payload(self):
        (self.baseline / "physical/a.txt").write_bytes(b"wrong")
        original = f.Tree.open

        def only_metadata(tree, rel, write=False):
            self.assertIn(rel, (f.BASELINE_MANIFEST, f.LAYOUT, f.INPUTS, f.DELIVERY))
            return original(tree, rel, write)

        with patch.object(f.Tree, "open", only_metadata):
            result = f.plan(self.baseline, self.replay)
        self.assertEqual(result["verification"], "metadata_only_not_content_verified")

    def test_unlisted_private_files_are_not_collected(self):
        put(self.inputs / "private/reviews.txt", b"must remain absent")
        put(self.replay / "restored_other/secret.txt", b"must remain absent")
        put(self.baseline / "private.txt", b"must remain absent")
        self.build()
        self.restore()
        self.assertFalse((self.restored / "project/private").exists())
        self.assertFalse((self.restored / "policy_replay/restored_other").exists())


class BuildBoundaryTests(FamilyFixture):
    def test_input_root_is_explicit_no_author_machine_fallback(self):
        self.save_plan()
        missing = self.root / "missing-input-root"
        with self.rejected(self.out):
            f.build(self.baseline, self.replay, missing, self.plan_path, self.plan_sha, self.out)

    def test_corrupt_baseline_candidate_rejected(self):
        self.save_plan()
        (self.baseline / "physical/a.txt").write_bytes(b"wrong")
        with self.rejected(self.out):
            self.build()

    def test_missing_selected_candidate_does_not_try_other_basename(self):
        self.save_plan()
        (self.baseline / "physical/a.txt").unlink()
        with self.rejected(self.out):
            self.build()

    def test_corrupt_new_input_rejected(self):
        self.save_plan()
        (self.inputs / "results/b.txt").write_bytes(b"wrong")
        with self.rejected(self.out):
            self.build()

    def test_corrupt_tool_rejected(self):
        self.save_plan()
        (self.replay / "replay.py").write_bytes(b"wrong")
        with self.rejected(self.out):
            self.build()

    def test_changed_plan_binding_rejected(self):
        self.save_plan()
        self.plan_sha = "0" * 64
        with self.rejected(self.out):
            self.build()

    def test_changed_metadata_rejected(self):
        self.save_plan()
        put(self.replay / f.DELIVERY, (self.replay / f.DELIVERY).read_bytes() + b" ")
        with self.rejected(self.out):
            self.build()

    def test_source_changed_after_copy_rejected(self):
        original = f._verify_output
        calls = 0

        def change_after_first_check(tree, expected):
            nonlocal calls
            original(tree, expected)
            calls += 1
            if calls == 1:
                (self.baseline / "physical/a.txt").write_bytes(b"changed after preflight")

        with patch.object(f, "_verify_output", change_after_first_check), self.rejected(self.out):
            self.build()

    def test_source_replaced_with_identical_bytes_rejected(self):
        original = f._verify_output

        def replace(tree, expected):
            original(tree, expected)
            path = self.baseline / "physical/a.txt"
            path.unlink()
            path.write_bytes(self.a)

        with patch.object(f, "_verify_output", replace), self.rejected(self.out):
            self.build()

    def test_output_corruption_after_first_verification_rejected(self):
        original = f._verify_output
        calls = 0

        def corrupt(tree, expected):
            nonlocal calls
            original(tree, expected)
            calls += 1
            if calls == 1:
                (self.out / f.ALIASES).write_bytes(b"corrupted output")

        with patch.object(f, "_verify_output", corrupt), self.rejected(self.out):
            self.build()

    def test_existing_output_never_overwritten(self):
        self.out.mkdir()
        (self.out / "sentinel").write_bytes(b"keep")
        with self.rejected(self.out):
            self.build()
        self.assertEqual((self.out / "sentinel").read_bytes(), b"keep")

    def test_symlink_candidate_leaf_rejected(self):
        path = self.baseline / "physical/a.txt"
        path.unlink()
        path.symlink_to(self.baseline / "z_other/a.txt")
        with self.rejected(self.out):
            self.build()

    def test_symlink_candidate_parent_rejected(self):
        shutil.rmtree(self.baseline / "physical")
        (self.baseline / "physical").symlink_to(self.baseline / "z_other", target_is_directory=True)
        with self.rejected(self.out):
            self.build()

    def test_symlink_input_rejected(self):
        path = self.inputs / "results/b.txt"
        path.unlink()
        path.symlink_to(self.inputs / "results/alias-b.txt")
        with self.rejected(self.out):
            self.build()

    def test_symlink_root_rejected(self):
        alias = self.root / "baseline_alias"
        alias.symlink_to(self.baseline, target_is_directory=True)
        with self.rejected():
            f.plan(alias, self.replay)

    def test_symlink_output_parent_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.out = alias / "addon"
        with self.rejected(self.out):
            self.build()

    def test_source_fifo_rejected_without_blocking(self):
        path = self.baseline / "physical/a.txt"
        path.unlink()
        os.mkfifo(path)
        with self.rejected(self.out):
            self.build()

    def test_symlink_existing_output_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.out.symlink_to(outside, target_is_directory=True)
        with self.rejected(self.out):
            self.build()
        self.assertEqual(list(outside.iterdir()), [])

    def test_completion_fsync_failure_removes_complete_marker(self):
        original = f.os.fsync

        def fail_on_directory(fd):
            import stat
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("synthetic completion fsync failure")
            return original(fd)

        with patch.object(f.os, "fsync", fail_on_directory), self.rejected(self.out):
            self.build()


class RestoreBoundaryTests(FamilyFixture):
    def _rebind_tampered_aliases(self, update):
        aliases = json.loads((self.out / f.ALIASES).read_bytes())
        update(aliases)
        raw = f.encode(aliases)
        (self.out / f.ALIASES).write_bytes(raw)
        completed = json.loads((self.out / f.COMPLETE).read_bytes())
        completed["aliases"] = f.digest(raw)
        completed["payload"][f.ALIASES] = f.digest(raw)
        raw_completed = f.encode(completed)
        (self.out / f.COMPLETE).write_bytes(raw_completed)
        self.receipt["completion_sha256"] = f.digest(raw_completed)["sha256"]

    def test_rebound_extra_logical_destination_rejected(self):
        self.build()
        self._rebind_tampered_aliases(lambda aliases: aliases["artifacts"].update(
            {"project/private/unlisted.txt": dict(aliases["artifacts"]["project/data/a.txt"])}))
        with self.rejected(self.restored):
            self.restore()

    def test_rebound_basename_candidate_change_rejected(self):
        self.build()
        self._rebind_tampered_aliases(lambda aliases: aliases["artifacts"]["project/data/a.txt"].update(
            {"path": "z_other/a.txt"}))
        with self.rejected(self.restored):
            self.restore()

    def test_rebound_object_traversal_rejected(self):
        self.build()
        self._rebind_tampered_aliases(lambda aliases: aliases["artifacts"]["project/results/b.txt"].update(
            {"path": "../outside"}))
        with self.rejected(self.restored):
            self.restore()

    def test_corrupt_baseline_during_restore_rejected(self):
        self.build()
        (self.baseline / "physical/a.txt").write_bytes(b"wrong")
        with self.rejected(self.restored):
            self.restore()

    def test_corrupt_addon_object_rejected(self):
        self.build()
        (self.out / ("objects/sha256/" + hashlib.sha256(self.b).hexdigest())).write_bytes(b"wrong")
        with self.rejected(self.restored):
            self.restore()

    def test_tampered_aliases_rejected(self):
        self.build()
        (self.out / f.ALIASES).write_bytes((self.out / f.ALIASES).read_bytes() + b" ")
        with self.rejected(self.restored):
            self.restore()

    def test_wrong_completion_pin_rejected(self):
        self.build()
        self.receipt["completion_sha256"] = "0" * 64
        with self.rejected(self.restored):
            self.restore()

    def test_incomplete_addon_rejected(self):
        self.build()
        (self.out / f.COMPLETE).unlink()
        with self.rejected(self.restored):
            self.restore()

    def test_unexpected_addon_member_rejected(self):
        self.build()
        (self.out / "unlisted.txt").write_bytes(b"unlisted")
        with self.rejected(self.restored):
            self.restore()

    def test_addon_symlink_rejected(self):
        self.build()
        (self.out / "link").symlink_to(self.replay / "replay.py")
        with self.rejected(self.restored):
            self.restore()

    def test_existing_restore_directory_rejected(self):
        self.build()
        self.restored.mkdir()
        with self.rejected(self.restored):
            self.restore()

    def test_restore_root_may_not_contain_baseline(self):
        self.build()
        self.restored = self.baseline / "nested"
        with self.rejected(self.restored):
            self.restore()

    def test_output_directory_symlink_injection_cannot_escape(self):
        self.build()
        outside = self.root / "outside"
        outside.mkdir()
        original = f.Tree.open
        injected = False

        def inject(tree, rel, write=False):
            nonlocal injected
            if tree.path == self.restored and write and rel.startswith("project/") and not injected:
                (self.restored / "project").symlink_to(outside, target_is_directory=True)
                injected = True
            return original(tree, rel, write)

        with patch.object(f.Tree, "open", inject), self.rejected(self.restored):
            self.restore()
        self.assertEqual(list(outside.iterdir()), [])

    def test_restore_source_changed_after_output_verification_rejected(self):
        self.build()
        original = f._verify_output

        def change(tree, expected):
            original(tree, expected)
            (self.out / f.ALIASES).write_bytes(b"source changed late")

        with patch.object(f, "_verify_output", change), self.rejected(self.restored):
            self.restore()

    def test_hardlinked_addon_payload_rejected(self):
        self.build()
        object_path = self.out / ("objects/sha256/" + hashlib.sha256(self.b).hexdigest())
        os.link(object_path, self.root / "object-hardlink")
        with self.rejected(self.restored):
            self.restore()


class MetadataBoundaryTests(FamilyFixture):
    def test_path_traversal_rejected(self):
        for unsafe in ("../x", "/absolute", "a/../b", "a//b", "a/./b", "a\\b", "x:y", ".", "a\nq"):
            with self.subTest(unsafe=unsafe), self.rejected():
                f.relative(unsafe)

    def test_unknown_historical_root_rejected(self):
        self.manifest["historical_root"] = "/unknown/machine"
        self._metadata()
        with self.rejected():
            self.save_plan()

    def test_duplicate_json_key_rejected(self):
        with self.rejected():
            f.parse(b'{"files": {}, "files": {}}')

    def test_nonfinite_json_rejected(self):
        for value in (b'NaN', b'Infinity', b'-Infinity'):
            with self.subTest(value=value), self.rejected():
                f.parse(b'{"v":' + value + b'}')

    def test_manifest_boolean_size_rejected(self):
        self.manifest["files"]["data/a.txt"]["bytes"] = True
        self._metadata()
        with self.rejected():
            self.save_plan()

    def test_runtime_payload_rejected(self):
        self.input_bytes[".python/bin/python"] = b"not distributed"
        self.manifest["files"][".python/bin/python"] = {**f.digest(b"not distributed"), "roles": ["runtime"]}
        self.manifest["file_count"] += 1
        self.manifest["total_bytes"] += len(b"not distributed")
        self._metadata()
        with self.rejected():
            self.save_plan()

    def test_file_directory_prefix_collision_rejected(self):
        self.manifest["files"]["data"] = {**f.digest(b"collision"), "roles": ["test"]}
        self.manifest["file_count"] += 1
        self.manifest["total_bytes"] += len(b"collision")
        self._metadata()
        with self.rejected():
            self.save_plan()

    def test_layout_input_disagreement_rejected(self):
        self.layout["project_input_files"]["data/a.txt"]["destination"] = "project/data/other.txt"
        raw = f.encode(self.layout)
        put(self.replay / f.LAYOUT, raw)
        self.delivery["files"][f.LAYOUT] = f.digest(raw)
        put(self.replay / f.DELIVERY, f.encode(self.delivery))
        with self.rejected():
            self.save_plan()

    def test_delivery_missing_tool_rejected(self):
        del self.delivery["files"]["replay.py"]
        put(self.replay / f.DELIVERY, f.encode(self.delivery))
        with self.rejected():
            self.save_plan()

    def test_duplicate_baseline_path_rejected(self):
        self.base_doc["files"].append(self.base_doc["files"][0])
        self.base_doc["file_count"] += 1
        self._metadata()
        with self.rejected():
            self.save_plan()

    def test_same_sha_with_conflicting_size_rejected(self):
        self.base_doc["files"][1]["bytes"] += 1
        self._metadata()
        with self.rejected():
            self.save_plan()


class CreationAndRenameBoundaryTests(FamilyFixture):
    """Creation binding regressions and an explicitly retained same-UID limit."""

    def test_original_named_mkdir_transition_no_longer_exists(self):
        self.save_plan()
        original = f.os.mkdir
        named_creations = []

        def observe(name, *args, **kwargs):
            if name == self.out.name and kwargs.get("dir_fd") is not None:
                named_creations.append(name)
            return original(name, *args, **kwargs)

        with patch.object(f.os, "mkdir", observe):
            result = self.build()
        self.assertTrue(result["complete"])
        self.assertEqual(named_creations, [])

    def test_replacement_immediately_after_output_binding_is_rejected(self):
        self.save_plan()
        victim = self.root / "preexisting-empty"
        victim.mkdir()
        victim_inode = victim.stat().st_ino
        moved = self.root / "fresh-moved-aside"
        original = f._rename_noreplace
        injected = False

        def replace(parent_fd, old, new):
            nonlocal injected
            original(parent_fd, old, new)
            os.rename(self.out, moved)
            os.rename(victim, self.out)
            injected = True

        with patch.object(f, "_rename_noreplace", replace), self.rejected(self.out):
            self.build()
        self.assertTrue(injected)
        self.assertEqual(self.out.stat().st_ino, victim_inode)
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(list(moved.iterdir()), [])

    def test_late_existing_empty_output_not_overwritten(self):
        self.save_plan()
        original = f._rename_noreplace
        inode = None

        def collide(parent_fd, old, new):
            nonlocal inode
            os.mkdir(new, dir_fd=parent_fd)
            inode = self.out.stat().st_ino
            return original(parent_fd, old, new)

        with patch.object(f, "_rename_noreplace", collide), self.assertRaises(FileExistsError) as caught:
            self.build()
        self.assertEqual(caught.exception.errno, errno.EEXIST)
        self.assertEqual(self.out.stat().st_ino, inode)
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertFalse((self.out / f.COMPLETE).exists())

    def test_actual_renameat2_noreplace_preserves_both_inodes(self):
        stage, existing = self.root / "stage", self.root / "existing"
        stage.mkdir()
        existing.mkdir()
        before = stage.stat().st_ino, existing.stat().st_ino
        with f.Tree(self.root) as parent, self.assertRaises(FileExistsError) as caught:
            f._rename_noreplace(parent.fd, stage.name, existing.name)
        self.assertEqual(caught.exception.errno, errno.EEXIST)
        self.assertEqual((stage.stat().st_ino, existing.stat().st_ino), before)

    def test_missing_renameat2_export_fails_closed(self):
        with patch.object(f.ctypes, "CDLL", return_value=object()), \
                patch.object(f.os, "rename", side_effect=AssertionError("No overwriting rename fallback")), \
                self.rejected(self.out):
            self.build()
        self.assertFalse(self.out.exists())

    def test_unsupported_syscall_errno_and_types_fail_closed(self):
        class Unsupported:
            def __call__(self, *args):
                self.actual_args = args
                f.ctypes.set_errno(errno.ENOSYS)
                return -1

        class Libc:
            renameat2 = Unsupported()

        with patch.object(f.ctypes, "CDLL", return_value=Libc()), \
                patch.object(f.os, "rename", side_effect=AssertionError("No fallback")), \
                self.assertRaises(OSError) as caught:
            self.build()
        self.assertEqual(caught.exception.errno, errno.ENOSYS)
        self.assertEqual(Libc.renameat2.argtypes,
                         (f.ctypes.c_int, f.ctypes.c_char_p, f.ctypes.c_int, f.ctypes.c_char_p, f.ctypes.c_uint))
        self.assertIs(Libc.renameat2.restype, f.ctypes.c_int)
        args = Libc.renameat2.actual_args
        self.assertEqual(args[0], args[2])
        self.assertIsInstance(args[1], bytes)
        self.assertEqual(args[3:], (self.out.name.encode(), 1))
        self.assertFalse(self.out.exists())

    def test_source_root_replacement_during_completion_rejected(self):
        self.save_plan()
        original = f.Tree.write
        moved = self.root / "baseline-moved"

        def replace(tree, rel, raw):
            original(tree, rel, raw)
            if tree.path == self.out and rel == "_COMPLETION_PENDING.json":
                os.rename(self.baseline, moved)
                self.baseline.mkdir()

        with patch.object(f.Tree, "write", replace), self.rejected(self.out):
            self.build()

    def test_output_root_replacement_during_completion_rolls_back_marker(self):
        self.save_plan()
        original = f.os.link
        moved = self.root / "completed-moved-aside"
        victim = self.root / "preexisting-empty"
        victim.mkdir()

        def replace(src, dst, *args, **kwargs):
            original(src, dst, *args, **kwargs)
            if dst == f.COMPLETE:
                os.rename(self.out, moved)
                os.rename(victim, self.out)

        with patch.object(f.os, "link", replace), self.rejected(self.out):
            self.build()
        self.assertFalse((moved / f.COMPLETE).exists())

    def test_held_output_creation_fd_is_the_published_root(self):
        with f.Tree.create(self.out, []) as tree:
            self.assertEqual((os.fstat(tree.fd).st_dev, os.fstat(tree.fd).st_ino),
                             (self.out.stat().st_dev, self.out.stat().st_ino))
            self.assertIsNotNone(tree.container_fd)
            tree.check_root()

    def test_moved_open_child_can_receive_write_before_final_rejection(self):
        # This is a documented excluded same-UID rename threat, NOT an isolation
        # success test. Retain the escaped write as evidence of the limitation.
        self.build()
        original = f.Tree._parent
        outside = self.root / "escaped-output"
        injected = False

        def move_after_open(tree, rel, create=False):
            nonlocal injected
            opened = original(tree, rel, create)
            if tree.path == self.restored and rel == "project/data/a.txt" and create:
                os.rename(self.restored / "project", outside)
                injected = True
            return opened

        with patch.object(f.Tree, "_parent", move_after_open), self.rejected(self.restored):
            self.restore()
        self.assertTrue(injected)
        self.assertEqual((outside / "data/a.txt").read_bytes(), self.a)


if __name__ == "__main__":
    unittest.main(verbosity=2)
