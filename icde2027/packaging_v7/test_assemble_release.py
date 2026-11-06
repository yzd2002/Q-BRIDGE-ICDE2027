"""Synthetic-only tests for the explicit-list release assembler.

Run with a writable temporary directory beneath the assigned evidence area.
Nothing in these fixtures uses scientific outputs or account credentials.
"""

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


SOURCE = Path(__file__).with_name("assemble_release.py")
module_spec = importlib.util.spec_from_file_location("release_under_test", SOURCE)
release = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="release synthetic ")
        self.base = Path(self.temporary.name)
        self.inputs = self.base / "original inputs"
        self.inputs.mkdir()
        self.output = self.base / "new delivery"
        self.spec_path = self.base / "allowlist.json"
        self.pdf = self.inputs / "paper.pdf"
        self.pdf.write_bytes(b"%PDF-1.4\nSYNTHETIC TEST ONLY\n")
        self.data = self.inputs / "data.bin"
        self.data.write_bytes(b"unchanged synthetic bytes\x00\xff\n")
        self.spec = {
            "schema": release.SPEC_SCHEMA,
            "metadata": {"document_status": "draft", "publication_status": "not_published",
                         "paper_pdf": {"destination_relative": "paper/paper.pdf", "sha256": self.sha(self.pdf)},
                         "note": "Synthetic fixture; not a submission or scientific approval."},
            "files": [self.entry(self.pdf, "paper/paper.pdf"), self.entry(self.data, "evidence/data.bin")],
        }
        self.save_spec()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def entry(self, source, target):
        return {"source_path": str(source), "destination_relative": target,
                "bytes": source.stat().st_size, "sha256": self.sha(source)}

    def save_spec(self):
        # Intentionally noncanonical; the copied spec must be byte-identical.
        self.spec_path.write_text(json.dumps(self.spec, ensure_ascii=False, indent=3) + "\n\n", encoding="utf-8")
        self.spec_sha = self.sha(self.spec_path)

    def assemble(self):
        return release.assemble(self.spec_path, self.spec_sha, self.output)

    def verify(self, report):
        return release.verify(self.output, report["manifest"]["sha256"], report["completion"]["sha256"])

    def cli(self, arguments, timeout=5):
        return subprocess.run([sys.executable, "-B", "-I", "-S", str(SOURCE), *map(str, arguments)],
                              capture_output=True, text=True, timeout=timeout, cwd=self.base)

    def test_roundtrip_exact_bytes_and_inventory(self):
        report = self.assemble()
        self.assertEqual(report, self.verify(report))
        self.assertEqual(report["package_regular_file_count"], 5)
        self.assertEqual((self.output / release.SPEC_NAME).read_bytes(), self.spec_path.read_bytes())
        self.assertEqual((self.output / "paper/paper.pdf").read_bytes(), self.pdf.read_bytes())
        self.assertEqual((self.output / "evidence/data.bin").read_bytes(), self.data.read_bytes())
        self.assertFalse((self.output / release.PENDING_NAME).exists())

    def test_self_contained_relocated_cli_no_sources_or_neighbor_imports(self):
        self.spec["files"].append(self.entry(SOURCE, "tools/assemble_release.py"))
        self.save_spec()
        report = self.assemble()
        relocated = self.base / "copied delivery with spaces"
        shutil.copytree(self.output, relocated)
        shutil.rmtree(self.output)
        shutil.rmtree(self.inputs)
        self.spec_path.unlink()
        command = [sys.executable, "-B", "-I", "-S", str(relocated / "tools/assemble_release.py"),
                   "verify", "--bundle", str(relocated),
                   "--expected-manifest-sha256", report["manifest"]["sha256"],
                   "--expected-completion-sha256", report["completion"]["sha256"]]
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, cwd=self.base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), report)

    def test_external_spec_hash_required_before_output(self):
        with self.assertRaisesRegex(release.ReleaseError, "external_sha_mismatch"):
            release.assemble(self.spec_path, "0" * 64, self.output)
        self.assertFalse(self.output.exists())

    def test_source_hash_and_size_preflight(self):
        for field, value in [("sha256", "0" * 64), ("bytes", 999), ("bytes", True)]:
            with self.subTest(field=field, value=value):
                old = self.spec["files"][1][field]
                self.spec["files"][1][field] = value
                self.save_spec()
                with self.assertRaises(release.ReleaseError):
                    self.assemble()
                self.assertFalse(self.output.exists())
                self.spec["files"][1][field] = old

    def test_destinations_traversal_absolute_empty_and_control(self):
        invalid = ["../escape", "/absolute", "a/../b", "a/./b", "a//b", "a/", "", ".",
                   "a\\b", "C:/file", "a\x00b", "a\nb", "//network"]
        for value in invalid:
            with self.subTest(value=repr(value)):
                self.spec["files"][1]["destination_relative"] = value
                self.save_spec()
                with self.assertRaises(release.ReleaseError):
                    self.assemble()
                self.assertFalse(self.output.exists())

    def test_reserved_duplicate_and_prefix_destinations(self):
        for value in [*release.RESERVED, "release_spec.JSON", "FILE_MANIFEST.json/child", "paper/paper.pdf", "paper"]:
            with self.subTest(value=value):
                self.spec["files"][1]["destination_relative"] = value
                self.save_spec()
                with self.assertRaises(release.ReleaseError):
                    self.assemble()
                self.assertFalse(self.output.exists())

    def test_reverse_prefix_order_rejected(self):
        self.spec["files"].append(self.entry(self.data, "evidence/data.bin/child"))
        self.spec["files"].reverse()
        self.save_spec()
        with self.assertRaisesRegex(release.ReleaseError, "prefix"):
            self.assemble()

    def test_source_absolute_normalized_required(self):
        for value in ["relative/file", str(self.inputs) + "/../data", str(self.inputs) + "//data.bin"]:
            self.spec["files"][1]["source_path"] = value
            self.save_spec()
            with self.assertRaises(release.ReleaseError):
                self.assemble()

    def test_source_leaf_symlink_no_target_read(self):
        link = self.inputs / "linked.bin"
        link.symlink_to(self.data)
        self.spec["files"][1]["source_path"] = str(link)
        self.save_spec()
        with mock.patch.object(release, "consume", wraps=release.consume) as reads:
            with self.assertRaisesRegex(release.ReleaseError, "non_regular_or_link"):
                self.assemble()
        self.assertEqual(reads.call_count, 2)  # spec and preceding PDF only
        self.assertFalse(self.output.exists())

    def test_source_ancestor_symlink(self):
        link = self.base / "linked inputs"
        link.symlink_to(self.inputs, target_is_directory=True)
        self.spec["files"][1]["source_path"] = str(link / self.data.name)
        self.save_spec()
        with self.assertRaisesRegex(release.ReleaseError, "non_directory_or_link"):
            self.assemble()

    def test_source_swapped_to_link_or_fifo_between_stat_and_open(self):
        original_data = self.data.read_bytes()
        original_open = release.os.open
        for kind in ("link", "fifo"):
            swapped = False
            def raced_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if path == self.data.name and flags == release.READ_FLAGS and not swapped:
                    swapped = True
                    self.data.unlink()
                    if kind == "link":
                        self.data.symlink_to(self.pdf)
                    else:
                        os.mkfifo(self.data)
                return original_open(path, flags, *args, **kwargs)
            with self.subTest(kind=kind), mock.patch.object(release.os, "open", side_effect=raced_open):
                with mock.patch.object(release, "consume", wraps=release.consume) as reads:
                    with self.assertRaises((OSError, release.ReleaseError)):
                        self.assemble()
                self.assertEqual(reads.call_count, 2)
            self.assertTrue(swapped)
            self.assertFalse(self.output.exists())
            self.data.unlink()
            self.data.write_bytes(original_data)

    def test_source_fifo_socket_and_directory_never_open(self):
        fifo = self.inputs / "fifo"
        os.mkfifo(fifo)
        sockpath = self.inputs / "socket"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            with contextlib.chdir(sockpath.parent):
                sock.bind(sockpath.name)
            for path in (fifo, sockpath, self.inputs):
                with self.subTest(kind=path.name):
                    self.spec["files"][1]["source_path"] = str(path)
                    self.save_spec()
                    result = self.cli(["assemble", "--spec", self.spec_path, "--expected-spec-sha256", self.spec_sha,
                                       "--output", self.output], timeout=3)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("non_regular_or_link", result.stderr)
                    self.assertFalse(self.output.exists())

    def test_spec_symlink_fifo_and_directory_reject_immediately(self):
        link = self.base / "spec link"
        link.symlink_to(self.spec_path)
        fifo = self.base / "spec fifo"
        os.mkfifo(fifo)
        for path in (link, fifo, self.inputs):
            with self.subTest(kind=path.name):
                result = self.cli(["assemble", "--spec", path, "--expected-spec-sha256", self.spec_sha,
                                   "--output", self.output], timeout=3)
                self.assertEqual(result.returncode, 2)
                self.assertIn("non_regular_or_link", result.stderr)

    def test_output_existing_file_directory_and_dangling_link(self):
        for kind in ("file", "directory", "dangling_link"):
            with self.subTest(kind=kind):
                if kind == "file":
                    self.output.write_bytes(b"retain existing")
                elif kind == "directory":
                    self.output.mkdir()
                else:
                    self.output.symlink_to(self.base / "missing")
                with self.assertRaisesRegex(release.ReleaseError, "output_already_exists"):
                    self.assemble()
                if kind == "directory":
                    self.output.rmdir()
                else:
                    self.output.unlink()

    def test_output_ancestor_symlink(self):
        link = self.base / "linked destination"
        link.symlink_to(self.inputs, target_is_directory=True)
        with self.assertRaisesRegex(release.ReleaseError, "non_directory_or_link"):
            release.assemble(self.spec_path, self.spec_sha, link / "output")
        self.assertFalse((self.inputs / "output").exists())

    def test_input_output_overlap_in_both_directions(self):
        with self.assertRaisesRegex(release.ReleaseError, "input_output_overlap"):
            release.assemble(self.spec_path, self.spec_sha, str(self.spec_path) + "/nested")
        with self.assertRaisesRegex(release.ReleaseError, "input_output_overlap"):
            release.assemble(self.spec_path, self.spec_sha, self.base)
        self.spec["files"][1]["source_path"] = str(self.output / "input.bin")
        self.save_spec()
        with self.assertRaisesRegex(release.ReleaseError, "input_output_overlap"):
            self.assemble()

    def test_payload_tamper_missing_extra_and_empty_directory(self):
        report = self.assemble()
        payload = self.output / "evidence/data.bin"
        original = payload.read_bytes()
        payload.write_bytes(b"tampered")
        with self.assertRaises(release.ReleaseError):
            self.verify(report)
        payload.unlink()
        with self.assertRaises(release.ReleaseError):
            self.verify(report)
        payload.write_bytes(original)
        extra = self.output / "extra"
        extra.write_bytes(b"not allowed")
        with self.assertRaises(release.ReleaseError):
            self.verify(report)
        extra.unlink()
        extra.mkdir()
        with self.assertRaises(release.ReleaseError):
            self.verify(report)

    def test_extra_link_fifo_and_socket_rejected_without_open(self):
        report = self.assemble()
        extra = self.output / "unexpected"
        extra.symlink_to(self.data)
        with self.assertRaisesRegex(release.ReleaseError, "inventory_link_or_special_file"):
            self.verify(report)
        extra.unlink()
        os.mkfifo(extra)
        with self.assertRaisesRegex(release.ReleaseError, "inventory_link_or_special_file"):
            self.verify(report)
        extra.unlink()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            with contextlib.chdir(extra.parent):
                sock.bind(extra.name)
            with self.assertRaisesRegex(release.ReleaseError, "inventory_link_or_special_file"):
                self.verify(report)

    def test_metadata_files_symlink_fifo_directory_reject_immediately(self):
        report = self.assemble()
        for name in (release.SPEC_NAME, release.MANIFEST_NAME, release.COMPLETION_NAME):
            path = self.output / name
            data = path.read_bytes()
            external = self.base / (name + ".external")
            external.write_bytes(data)
            for kind in ("link", "fifo", "directory"):
                with self.subTest(name=name, kind=kind):
                    path.unlink()
                    if kind == "link":
                        path.symlink_to(external)
                    elif kind == "fifo":
                        os.mkfifo(path)
                    else:
                        path.mkdir()
                    result = self.cli(["verify", "--bundle", self.output,
                                       "--expected-manifest-sha256", report["manifest"]["sha256"],
                                       "--expected-completion-sha256", report["completion"]["sha256"]], timeout=3)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("non_regular_or_link", result.stderr)
                    if kind == "directory":
                        path.rmdir()
                    else:
                        path.unlink()
                    path.write_bytes(data)

    def test_external_manifest_and_completion_hashes(self):
        report = self.assemble()
        for manifest_sha, completion_sha in [("0" * 64, report["completion"]["sha256"]),
                                              (report["manifest"]["sha256"], "0" * 64)]:
            with self.assertRaisesRegex(release.ReleaseError, "external_sha_mismatch"):
                release.verify(self.output, manifest_sha, completion_sha)

    def test_bundle_root_symlink_rejected(self):
        report = self.assemble()
        link = self.base / "bundle alias"
        link.symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(release.ReleaseError, "non_directory_or_link"):
            release.verify(link, report["manifest"]["sha256"], report["completion"]["sha256"])

    def test_rehashed_manifest_must_still_match_completion_and_spec(self):
        report = self.assemble()
        path = self.output / release.MANIFEST_NAME
        manifest = json.loads(path.read_bytes())
        manifest["metadata"]["document_status"] = "final"
        path.write_bytes(release.encoded(manifest))
        with self.assertRaisesRegex(release.ReleaseError, "manifest_spec_binding_mismatch"):
            release.verify(self.output, self.sha(path), report["completion"]["sha256"])

    def test_spec_bytes_bound_in_manifest_and_completion(self):
        report = self.assemble()
        path = self.output / release.SPEC_NAME
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(release.ReleaseError, "external_sha_mismatch"):
            self.verify(report)

    def test_metadata_paper_pdf_and_status_must_be_bound(self):
        original = json.loads(json.dumps(self.spec))
        for field, value in [("document_status", "unknown"), ("publication_status", ""),
                             ("paper_pdf", {"destination_relative": "missing.pdf", "sha256": "0" * 64})]:
            self.spec = json.loads(json.dumps(original))
            self.spec["metadata"][field] = value
            self.save_spec()
            with self.assertRaises(release.ReleaseError):
                self.assemble()

    def test_duplicate_json_key_and_nonstandard_constants(self):
        self.spec_path.write_bytes(b'{"schema": "a", "schema": "b"}')
        with self.assertRaisesRegex(release.ReleaseError, "duplicate_json_key"):
            release.assemble(self.spec_path, self.sha(self.spec_path), self.output)
        self.spec_path.write_bytes(b'{"schema": NaN}')
        with self.assertRaisesRegex(release.ReleaseError, "invalid_json_constant"):
            release.assemble(self.spec_path, self.sha(self.spec_path), self.output)

    def test_oversize_json_is_rejected_before_read(self):
        with self.spec_path.open("wb") as stream:
            stream.truncate(release.MAX_JSON_BYTES + 1)
        with self.assertRaisesRegex(release.ReleaseError, "json_too_large"):
            release.assemble(self.spec_path, "0" * 64, self.output)
        self.assertFalse(self.output.exists())

    def test_source_modified_after_preflight_leaves_incomplete(self):
        original_copy = release.copy_file
        def changed_copy(fd, entry):
            if entry["source_path"] == str(self.data):
                self.data.write_bytes(b"changed after preflight")
            return original_copy(fd, entry)
        with mock.patch.object(release, "copy_file", side_effect=changed_copy):
            with self.assertRaisesRegex(release.ReleaseError, "identity_mismatch"):
                self.assemble()
        self.assertFalse((self.output / release.COMPLETION_NAME).exists())
        self.assertTrue((self.output / release.FAILURE_NAME).is_file())
        self.assertTrue((self.output / "paper/paper.pdf").is_file())

    def test_exception_and_keyboard_interrupt_retain_partial_without_completion(self):
        for exception in (OSError("synthetic failure"), KeyboardInterrupt()):
            with self.subTest(kind=type(exception).__name__):
                self.output = self.base / ("partial " + type(exception).__name__)
                with mock.patch.object(release, "copy_file", side_effect=exception):
                    with self.assertRaises(type(exception)):
                        self.assemble()
                self.assertFalse((self.output / release.COMPLETION_NAME).exists())
                failure = json.loads((self.output / release.FAILURE_NAME).read_bytes())
                self.assertEqual(failure["status"], "incomplete")
                self.assertEqual(failure["copied_payload_files"], 0)

    def test_sigterm_during_copy_has_no_completion(self):
        script = '''import importlib.util, pathlib, signal, sys
s = importlib.util.spec_from_file_location("assembler", sys.argv[1])
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
def paused_copy(fd, entry):
    m.new_file(fd, "partial.bin", b"partial synthetic bytes")
    pathlib.Path(sys.argv[5]).write_text("ready")
    signal.pause()
m.copy_file = paused_copy
raise SystemExit(m.main(["assemble", "--spec", sys.argv[2], "--expected-spec-sha256", sys.argv[3], "--output", sys.argv[4]]))
'''
        ready = self.base / "ready"
        process = subprocess.Popen([sys.executable, "-B", "-I", "-S", "-c", script, str(SOURCE),
                                    str(self.spec_path), self.spec_sha, str(self.output), str(ready)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 4
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists())
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(stdout, "")
            self.assertIn("interrupted_signal_15", stderr)
            self.assertFalse((self.output / release.COMPLETION_NAME).exists())
            self.assertTrue((self.output / "partial.bin").is_file())
            self.assertTrue((self.output / release.FAILURE_NAME).is_file())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_late_extra_before_final_rehash_prevents_completion(self):
        write = release.new_file
        def injected(fd, relative, data):
            write(fd, relative, data)
            if relative == release.PENDING_NAME:
                write(fd, "unexpected.bin", b"injected before inventory")
        with mock.patch.object(release, "new_file", side_effect=injected):
            with self.assertRaisesRegex(release.ReleaseError, "inventory_mismatch"):
                self.assemble()
        self.assertFalse((self.output / release.COMPLETION_NAME).exists())

    def test_commit_does_not_overwrite_and_catchable_failure_revokes_marker(self):
        with mock.patch.object(release.os, "link", side_effect=FileExistsError("synthetic conflict")):
            with self.assertRaises(FileExistsError):
                self.assemble()
        self.assertFalse((self.output / release.COMPLETION_NAME).exists())
        self.assertTrue((self.output / release.PENDING_NAME).exists())
        self.output = self.base / "unlink interruption"
        original = release.os.unlink
        def fail_pending(path, *args, **kwargs):
            if path == release.PENDING_NAME:
                raise InterruptedError("synthetic interruption")
            return original(path, *args, **kwargs)
        with mock.patch.object(release.os, "unlink", side_effect=fail_pending):
            with self.assertRaises(InterruptedError):
                self.assemble()
        self.assertFalse((self.output / release.COMPLETION_NAME).exists())
        self.assertTrue((self.output / release.FAILURE_NAME).exists())

    def test_hard_kill_commit_window_extra_pending_cannot_verify(self):
        report = self.assemble()
        # This is the on-disk state of the exclusive-link commit before unlink.
        os.link(self.output / release.COMPLETION_NAME, self.output / release.PENDING_NAME)
        with self.assertRaisesRegex(release.ReleaseError, "inventory_mismatch"):
            self.verify(report)

    def test_streaming_large_payload_and_zero_byte_file(self):
        # 8 MiB synthetic input, much larger than the read chunk; no read(-1).
        self.data.write_bytes(b"0123456789abcdef" * (512 * 1024))
        self.spec["files"][1] = self.entry(self.data, "evidence/data.bin")
        empty = self.inputs / "empty"
        empty.write_bytes(b"")
        self.spec["files"].append(self.entry(empty, "evidence/empty"))
        self.save_spec()
        real_consume = release.consume
        read_sizes = []
        def observed(stream, before, **kwargs):
            class LimitedRead:
                def read(self, size=-1):
                    if size <= 0 or size > release.CHUNK_BYTES:
                        raise AssertionError("unbounded read")
                    read_sizes.append(size)
                    return stream.read(size)
                def fileno(self):
                    return stream.fileno()
            return real_consume(LimitedRead(), before, **kwargs)
        with mock.patch.object(release, "consume", side_effect=observed):
            report = self.assemble()
            self.verify(report)
        self.assertGreater(len(read_sizes), 32)
        self.assertEqual((self.output / "evidence/empty").stat().st_size, 0)

    def test_payload_scripts_never_execute_and_errors_do_not_print_contents(self):
        program = self.inputs / "danger.py"
        sentinel = self.base / "must never exist"
        program.write_text("from pathlib import Path\nPath(" + repr(str(sentinel)) + ").write_text('ran')\n")
        self.spec["files"].append(self.entry(program, "scripts/arbitrary.py"))
        self.save_spec()
        report = self.assemble()
        self.verify(report)
        self.assertFalse(sentinel.exists())
        self.data.write_bytes(b"SYNTHETIC_PRIVATE_MARKER_NOT_A_CREDENTIAL")
        result = self.cli(["assemble", "--spec", self.spec_path, "--expected-spec-sha256", self.spec_sha,
                           "--output", self.base / "failed output"])
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("SYNTHETIC_PRIVATE_MARKER", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
