#!/usr/bin/env python3
"""Byte-preserving storage and restoration of an externally pinned audited run.

This module performs no optimization, performance measurement, or execution of
the bundled auditor. It uses Python's standard library on Linux.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


FORMAT = "icde2027.optimized-storage.v1"
PASS_STATUS = "PASS_COMPLETED_RUN_INTEGRITY_AND_ARITHMETIC"
CHUNK_BYTES = 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
AUDIT_MEMBER = "evidence/RESULTS.json"
AUDITOR_MEMBER = "tools/optimized_schedule_v6_audit.py"
RESTORER_MEMBER = "tools/restore_optimized.py"
MANIFEST_MEMBER = "MANIFEST.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class BundleError(ValueError):
    """Invalid input, identity mismatch, or unsafe filesystem member."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def _digest(value: object) -> str:
    _need(isinstance(value, str) and SHA256.fullmatch(value) is not None,
          "expected a lowercase 64-character SHA-256")
    return value


def _relative(value: object) -> str:
    _need(isinstance(value, str) and value != "", "empty or non-string member path")
    _need(not value.startswith("/") and "\\" not in value and ":" not in value,
          "absolute or non-portable member path")
    _need(all(part not in ("", ".", "..") for part in value.split("/")),
          "unsafe member path component")
    _need(all(ord(char) >= 32 and ord(char) != 127 for char in value),
          "control character in member path")
    return value


def _absolute(value: str | Path) -> Path:
    raw = os.fspath(value)
    _need("\x00" not in raw and ".." not in raw.split("/"),
          "unsafe filesystem path")
    return Path(os.path.abspath(raw))


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _need(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _parse_json(data: bytes) -> object:
    try:
        return json.loads(data, object_pairs_hook=_no_duplicate_keys,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              BundleError("non-finite JSON number")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError("invalid JSON") from exc


DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _stamp(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _open_directory(path: Path) -> int:
    """Open every ancestor without following links; never resolve an input link."""
    fd = os.open("/", DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


class Tree:
    """Descriptor-relative operations anchored to a no-follow directory handle."""

    def __init__(self, path: str | Path):
        self.path = _absolute(path)
        self.fd = _open_directory(self.path)

    def __enter__(self) -> Tree:
        return self

    def __exit__(self, *_: object) -> None:
        os.close(self.fd)

    def _parent(self, relative: str, *, create: bool = False) -> tuple[int, str]:
        parts = _relative(relative).split("/")
        fd = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(part, DIR_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd, parts[-1]
        except BaseException:
            os.close(fd)
            raise

    def read(self, relative: str):
        parent, leaf = self._parent(relative)
        try:
            original = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            _need(stat.S_ISREG(original.st_mode), "input member is not a regular file")
            fd = os.open(leaf, FILE_FLAGS, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            _need(stat.S_ISREG(os.fstat(fd).st_mode), "input member is not a regular file")
            _need(_stamp(original) == _stamp(os.fstat(fd)), "input changed while opening")
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise

    def create(self, relative: str):
        parent, leaf = self._parent(relative, create=True)
        try:
            fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
        finally:
            os.close(parent)
        return os.fdopen(fd, "wb")

    def inventory(self) -> tuple[dict[str, tuple], set[str]]:
        files, directories = {}, set()

        def walk(fd: int, prefix: str) -> None:
            with os.scandir(fd) as entries:
                for entry in entries:
                    name = _relative(prefix + entry.name)
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        child = os.open(entry.name, DIR_FLAGS, dir_fd=fd)
                        try:
                            _need(_stamp(info) == _stamp(os.fstat(child)),
                                  "directory changed during traversal")
                            directories.add(name)
                            walk(child, name + "/")
                        finally:
                            os.close(child)
                    else:
                        _need(stat.S_ISREG(info.st_mode),
                              "tree contains a link or non-regular member: " + name)
                        files[name] = _stamp(info)

        walk(self.fd, "")
        return files, directories


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _new_tree(path: str | Path, protected: list[Path]) -> Tree:
    output = _absolute(path)
    _need(output.name != "", "output cannot be filesystem root")
    _need(all(not _overlap(output, item) for item in protected),
          "output overlaps an input path")
    parent = _open_directory(output.parent)
    try:
        # mkdir rejects existing files, directories, and dangling symbolic links.
        os.mkdir(output.name, mode=0o700, dir_fd=parent)
        fd = os.open(output.name, DIR_FLAGS, dir_fd=parent)
    finally:
        os.close(parent)
    tree = object.__new__(Tree)
    tree.path, tree.fd = output, fd
    return tree


def _read_blob(tree: Tree, member: str, expected_sha: str | None = None) -> tuple[bytes, dict]:
    with tree.read(member) as stream:
        before = os.fstat(stream.fileno())
        _need(before.st_size <= MAX_JSON_BYTES, "control file exceeds the memory limit")
        data = stream.read(MAX_JSON_BYTES + 1)
        _need(len(data) <= MAX_JSON_BYTES, "control file exceeds the memory limit")
        _need(_stamp(before) == _stamp(os.fstat(stream.fileno())),
              "control file changed while reading")
    record = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    _need(len(data) == before.st_size, "control file length changed while reading")
    if expected_sha is not None:
        _need(record["sha256"] == _digest(expected_sha), "external SHA-256 mismatch: " + member)
    return data, record


def _input_blob(path: Path, expected_sha: str | None = None) -> tuple[bytes, dict]:
    with Tree(path.parent) as tree:
        return _read_blob(tree, path.name, expected_sha)


def _record(value: object) -> dict:
    _need(isinstance(value, dict) and set(value) == {"bytes", "sha256"},
          "invalid file identity record")
    _need(type(value["bytes"]) is int and value["bytes"] >= 0, "invalid byte count")
    _digest(value["sha256"])
    return value


def _file_map(value: object) -> dict:
    _need(isinstance(value, dict) and bool(value), "missing or empty file map")
    for path, identity in value.items():
        _relative(path)
        _record(identity)
        parent = path
        while "/" in parent:
            parent = parent.rsplit("/", 1)[0]
            _need(parent not in value, "file/directory path collision")
    return value


def _audit(data: bytes, auditor_sha: str) -> dict:
    report = _parse_json(data)
    _need(isinstance(report, dict), "audit report must be an object")
    _need(report.get("status") == PASS_STATUS, "audit did not pass")
    _need(report.get("production_protocol") is True, "audit is not a production audit")
    files = _file_map(report.get("files"))
    _need(type(report.get("file_count")) is int and report["file_count"] == len(files),
          "audit file count mismatch")
    _need(report.get("audit_source_sha256") == _digest(auditor_sha),
          "audit report does not bind the supplied auditor")
    # The report's historical absolute paths are opaque identities, never inputs.
    return files


def _object_map(files: dict) -> dict:
    objects = {}
    for record in files.values():
        digest = record["sha256"]
        item = {"bytes": record["bytes"]}
        _need(digest not in objects or objects[digest] == item,
              "one object hash has contradictory byte counts")
        objects[digest] = item
    return objects


def _totals(files: dict, objects: dict) -> dict:
    return {"run_files": len(files), "run_bytes": sum(v["bytes"] for v in files.values()),
            "objects": len(objects), "object_bytes": sum(v["bytes"] for v in objects.values())}


def _parents(paths: set[str]) -> set[str]:
    parents = set()
    for path in paths:
        while "/" in path:
            path = path.rsplit("/", 1)[0]
            parents.add(path)
    return parents


def _copy_member(source: Tree, member: str, identity: dict,
                 destination: Tree | None = None, target: str | None = None,
                 expected_stamp: tuple | None = None) -> None:
    """Hash and optionally copy the original bytes in a single bounded-memory pass."""
    digest, length = hashlib.sha256(), 0
    with ExitStack() as stack:
        reader = stack.enter_context(source.read(member))
        before = os.fstat(reader.fileno())
        _need(before.st_size == identity["bytes"], "file length mismatch: " + member)
        if expected_stamp is not None:
            _need(_stamp(before) == expected_stamp, "file changed after inventory: " + member)
        writer = stack.enter_context(destination.create(target)) if destination is not None else None
        while True:
            chunk = reader.read(CHUNK_BYTES)
            if not chunk:
                break
            length += len(chunk)
            _need(length <= identity["bytes"], "file grew while reading: " + member)
            digest.update(chunk)
            if writer is not None:
                writer.write(chunk)
        _need(_stamp(before) == _stamp(os.fstat(reader.fileno())),
              "file changed while reading: " + member)
        _need(length == identity["bytes"] and digest.hexdigest() == identity["sha256"],
              "file SHA-256 mismatch: " + member)
        if writer is not None:
            writer.flush()
            os.fsync(writer.fileno())


def _write_blob(tree: Tree, member: str, data: bytes) -> None:
    with tree.create(member) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _manifest(data: bytes) -> dict:
    value = _parse_json(data)
    _need(isinstance(value, dict) and set(value) == {"format", "files", "objects", "controls", "totals"},
          "invalid manifest fields")
    _need(value["format"] == FORMAT, "unsupported bundle format")
    files = _file_map(value["files"])
    expected_objects = _object_map(files)
    objects = value["objects"]
    _need(isinstance(objects, dict), "invalid object index")
    for digest, record in objects.items():
        _digest(digest)
        _need(isinstance(record, dict) and set(record) == {"bytes"} and
              type(record["bytes"]) is int and record["bytes"] >= 0,
              "invalid object record")
    _need(objects == expected_objects, "object index does not exactly cover run files")
    controls = value["controls"]
    _need(isinstance(controls, dict) and set(controls) == {AUDIT_MEMBER, AUDITOR_MEMBER, RESTORER_MEMBER},
          "invalid control index")
    for record in controls.values():
        _record(record)
    totals = value["totals"]
    _need(isinstance(totals, dict) and all(type(item) is int for item in totals.values()) and
          totals == _totals(files, objects), "manifest totals mismatch")
    return value


def build(*, run: str | Path, audit_results: str | Path, audit_sha256: str,
          auditor: str | Path, auditor_sha256: str, output: str | Path) -> dict:
    run_path, audit_path, auditor_path = map(_absolute, (run, audit_results, auditor))
    report, report_record = _input_blob(audit_path, _digest(audit_sha256))
    auditor_data, auditor_record = _input_blob(auditor_path, _digest(auditor_sha256))
    files = _audit(report, auditor_sha256)
    script_path = _absolute(__file__)
    script, script_record = _input_blob(script_path)
    objects = _object_map(files)
    controls = {AUDIT_MEMBER: report_record, AUDITOR_MEMBER: auditor_record,
                RESTORER_MEMBER: script_record}
    manifest = {"format": FORMAT, "files": files, "objects": objects,
                "controls": controls, "totals": _totals(files, objects)}
    manifest_data = _json_bytes(manifest)
    _manifest(manifest_data)
    with Tree(run_path) as source:
        before, _ = source.inventory()
        _need(set(before) == set(files), "run file set differs from audited allowlist")
        with _new_tree(output, [run_path, audit_path, auditor_path, script_path]) as dest:
            copied = set()
            for member in sorted(files):
                identity = files[member]
                digest = identity["sha256"]
                first = digest not in copied
                _copy_member(source, member, identity, dest if first else None,
                             "objects/" + digest if first else None, before[member])
                copied.add(digest)
            after, _ = source.inventory()
            _need(before == after, "run changed during build")
            for path, original, record in ((audit_path, report, report_record),
                                           (auditor_path, auditor_data, auditor_record),
                                           (script_path, script, script_record)):
                data, identity = _input_blob(path, record["sha256"])
                _need(data == original and identity == record, "control file changed during build")
            _write_blob(dest, AUDIT_MEMBER, report)
            _write_blob(dest, AUDITOR_MEMBER, auditor_data)
            _write_blob(dest, RESTORER_MEMBER, script)
            # The manifest is the last member: a failed build has no success seal.
            _write_blob(dest, MANIFEST_MEMBER, manifest_data)
    return {"status": "BUILT", "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
            **manifest["totals"]}


def restore(*, bundle: str | Path, manifest_sha256: str, output: str | Path) -> dict:
    bundle_path = _absolute(bundle)
    with Tree(bundle_path) as source:
        before, directories = source.inventory()
        # Inventory precedes opening the manifest, rejecting FIFO/link inputs early.
        data, manifest_record = _read_blob(source, MANIFEST_MEMBER, _digest(manifest_sha256))
        manifest = _manifest(data)
        expected = {MANIFEST_MEMBER, *manifest["controls"],
                    *("objects/" + digest for digest in manifest["objects"])}
        _need(set(before) == expected and directories == _parents(expected),
              "bundle has missing or unexpected files/directories")
        control_data = {}
        for member, identity in manifest["controls"].items():
            original, actual = _read_blob(source, member, identity["sha256"])
            _need(actual == identity, "control size mismatch")
            control_data[member] = original
        files = _audit(control_data[AUDIT_MEMBER], manifest["controls"][AUDITOR_MEMBER]["sha256"])
        _need(files == manifest["files"], "manifest disagrees with original audit allowlist")
        script_path = _absolute(__file__)
        with _new_tree(output, [bundle_path, script_path]) as dest:
            for member in sorted(files):
                identity = files[member]
                object_name = "objects/" + identity["sha256"]
                _copy_member(source, object_name, identity, dest, "run/" + member,
                             before[object_name])
            for member, original in control_data.items():
                _write_blob(dest, member, original)
            after, after_directories = source.inventory()
            _need(before == after and directories == after_directories,
                  "bundle changed during restoration")
            _write_blob(dest, MANIFEST_MEMBER, data)
    return {"status": "RESTORED", "manifest_sha256": manifest_record["sha256"],
            **manifest["totals"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build", help="store an externally pinned, completed audited run")
    for name in ("run", "audit-results", "audit-sha256", "auditor", "auditor-sha256", "output"):
        create.add_argument("--" + name, required=True)
    recover = commands.add_parser("restore", help="restore a bundle using an external manifest SHA-256")
    for name in ("bundle", "manifest-sha256", "output"):
        recover.add_argument("--" + name, required=True)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    try:
        result = build(**arguments) if command == "build" else restore(**arguments)
    except (BundleError, OSError, RecursionError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

