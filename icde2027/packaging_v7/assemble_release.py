#!/usr/bin/env python3
"""Explicit-list release copying and byte verification; no scientific execution.

This standalone POSIX/stdlib utility never discovers source files recursively.
External SHA-256 pins are required; an internally consistent package is not proof
of scientific correctness, privacy review, publication, or submission.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import stat
import sys


SPEC_NAME = "RELEASE_SPEC.json"
MANIFEST_NAME = "FILE_MANIFEST.json"
COMPLETION_NAME = "COMPLETION.json"
FAILURE_NAME = "ASSEMBLY_FAILURE.json"
PENDING_NAME = ".COMPLETION.pending"
RESERVED = {SPEC_NAME, MANIFEST_NAME, COMPLETION_NAME, FAILURE_NAME, PENDING_NAME}
SPEC_SCHEMA = "icde2027.explicit-release-spec.v1"
MANIFEST_SCHEMA = "icde2027.release-file-manifest.v1"
COMPLETION_SCHEMA = "icde2027.release-completion.v1"
CHUNK_BYTES = 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
HEX = re.compile(r"[0-9a-f]{64}\Z")
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


class ReleaseError(Exception):
    """A content-free error code suitable for logs."""


def require(condition, code):
    if not condition:
        raise ReleaseError(code)


def sha_string(value):
    require(isinstance(value, str) and HEX.fullmatch(value) is not None, "invalid_sha256")
    return value


def nonnegative_int(value):
    require(type(value) is int and value >= 0, "invalid_byte_count")
    return value


def path_parts(value, *, absolute):
    require(isinstance(value, str) and value != "", "invalid_path")
    require(not any(ord(c) < 32 or ord(c) == 127 for c in value), "invalid_path")
    require("\\" not in value and ":" not in value, "invalid_path")
    require(value.startswith("/") == absolute, "invalid_path")
    parts = value.split("/")[1:] if absolute else value.split("/")
    require(parts and all(p not in ("", ".", "..") for p in parts), "invalid_path")
    return parts


def absolute_path(value):
    # CLI relative paths are converted lexically; symlinks are never resolved.
    require(isinstance(value, (str, os.PathLike)), "invalid_path")
    value = os.fspath(value)
    require(isinstance(value, str), "invalid_path")
    if not value.startswith("/"):
        path_parts(value, absolute=False)
        value = os.getcwd().rstrip("/") + "/" + value
    path_parts(value, absolute=True)
    return value


def relative_path(value, *, payload=False):
    parts = path_parts(value, absolute=False)
    if payload:
        require(parts[0].casefold() not in {x.casefold() for x in RESERVED}, "reserved_destination")
    return value


def beneath_or_equal(a, b):
    return a == b or a.startswith(b.rstrip("/") + "/")


def same_object(a, b):
    return (a.st_dev, a.st_ino, stat.S_IFMT(a.st_mode)) == (b.st_dev, b.st_ino, stat.S_IFMT(b.st_mode))


def file_stamp(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


@contextlib.contextmanager
def open_directory(path):
    """Open each absolute component without following links, anchored by fds."""
    parts = path_parts(path, absolute=True)
    fd = os.open("/", DIR_FLAGS)
    try:
        for name in parts:
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode), "non_directory_or_link")
            child = os.open(name, DIR_FLAGS, dir_fd=fd)
            try:
                require(same_object(before, os.fstat(child)), "directory_changed")
            except BaseException:
                os.close(child)
                raise
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextlib.contextmanager
def parent_at(root_fd, relative, *, create=False):
    parts = path_parts(relative, absolute=False)
    fd = os.dup(root_fd)
    try:
        for name in parts[:-1]:
            if create:
                try:
                    os.mkdir(name, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode), "non_directory_or_link")
            child = os.open(name, DIR_FLAGS, dir_fd=fd)
            try:
                require(same_object(before, os.fstat(child)), "directory_changed")
            except BaseException:
                os.close(child)
                raise
            os.close(fd)
            fd = child
        yield fd, parts[-1]
    finally:
        os.close(fd)


@contextlib.contextmanager
def regular_at(root_fd, relative):
    with parent_at(root_fd, relative) as (parent, name):
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode), "non_regular_or_link")
        fd = os.open(name, READ_FLAGS, dir_fd=parent)
        try:
            current = os.fstat(fd)
            require(stat.S_ISREG(current.st_mode) and same_object(before, current), "file_changed_before_open")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                yield stream, current
        finally:
            os.close(fd)


@contextlib.contextmanager
def regular_absolute(path):
    parts = path_parts(path, absolute=True)
    root_fd = os.open("/", DIR_FLAGS)
    try:
        with regular_at(root_fd, "/".join(parts)) as opened:
            yield opened
    finally:
        os.close(root_fd)


def consume(stream, before, *, capture=False, sink=None):
    if capture:
        require(before.st_size <= MAX_JSON_BYTES, "json_too_large")
    hasher = hashlib.sha256()
    size = 0
    chunks = [] if capture else None
    while True:
        block = stream.read(CHUNK_BYTES)
        if not block:
            break
        size += len(block)
        require(size <= before.st_size, "source_changed_during_read")
        hasher.update(block)
        if capture:
            chunks.append(block)
        if sink is not None:
            sink.write(block)
    require(file_stamp(before) == file_stamp(os.fstat(stream.fileno())) and size == before.st_size,
            "source_changed_during_read")
    result = {"bytes": size, "sha256": hasher.hexdigest()}
    return result, b"".join(chunks) if capture else None


def identity_bytes(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assert_identity(actual, expected):
    require(actual == {"bytes": expected["bytes"], "sha256": expected["sha256"]}, "identity_mismatch")


def read_pinned_absolute(path, expected_sha):
    sha_string(expected_sha)
    with regular_absolute(path) as (stream, before):
        identity, data = consume(stream, before, capture=True)
    require(identity["sha256"] == expected_sha, "external_sha_mismatch")
    return identity, data


def read_pinned_at(root_fd, relative, expected_sha):
    sha_string(expected_sha)
    with regular_at(root_fd, relative) as (stream, before):
        identity, data = consume(stream, before, capture=True)
    require(identity["sha256"] == expected_sha, "external_sha_mismatch")
    return identity, data


def no_duplicate_keys(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def parse_json(data):
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=no_duplicate_keys,
                          parse_constant=lambda _: (_ for _ in ()).throw(ReleaseError("invalid_json_constant")))
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ReleaseError("invalid_json") from error


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def validate_spec(spec):
    require(isinstance(spec, dict) and set(spec) == {"schema", "metadata", "files"}, "invalid_spec_fields")
    require(spec["schema"] == SPEC_SCHEMA, "invalid_spec_schema")
    metadata = spec["metadata"]
    require(isinstance(metadata, dict), "invalid_metadata")
    require(metadata.get("document_status") in ("draft", "final"), "invalid_document_status")
    require(isinstance(metadata.get("publication_status"), str) and metadata["publication_status"].strip() != "",
            "invalid_publication_status")
    paper = metadata.get("paper_pdf")
    require(isinstance(paper, dict) and set(paper) == {"destination_relative", "sha256"}, "invalid_paper_identity")
    relative_path(paper["destination_relative"], payload=True)
    sha_string(paper["sha256"])
    require(paper["destination_relative"].lower().endswith(".pdf"), "invalid_paper_identity")
    require(isinstance(spec["files"], list) and len(spec["files"]) > 0, "empty_or_invalid_allowlist")
    entries = {}
    for entry in spec["files"]:
        require(isinstance(entry, dict) and set(entry) == {"source_path", "destination_relative", "bytes", "sha256"},
                "invalid_file_entry")
        path_parts(entry["source_path"], absolute=True)
        dest = relative_path(entry["destination_relative"], payload=True)
        nonnegative_int(entry["bytes"])
        sha_string(entry["sha256"])
        require(dest not in entries, "duplicate_destination")
        entries[dest] = entry
    for dest in entries:
        parts = dest.split("/")
        require(not any("/".join(parts[:i]) in entries for i in range(1, len(parts))), "destination_prefix_conflict")
    require(paper["destination_relative"] in entries and
            entries[paper["destination_relative"]]["sha256"] == paper["sha256"], "paper_not_bound_to_allowlist")
    return entries


def new_file(root_fd, relative, data):
    with parent_at(root_fd, relative, create=True) as (parent, name):
        fd = os.open(name, WRITE_FLAGS, 0o644, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(fd)
        finally:
            os.close(fd)


def copy_file(root_fd, entry):
    # Source type is checked before any destination content read or source open.
    with regular_absolute(entry["source_path"]) as (source, before):
        require(before.st_size == entry["bytes"], "identity_mismatch")
        with parent_at(root_fd, entry["destination_relative"], create=True) as (parent, name):
            fd = os.open(name, WRITE_FLAGS, 0o644, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb", closefd=False) as sink:
                    actual, _ = consume(source, before, sink=sink)
                    sink.flush()
                    os.fsync(fd)
                assert_identity(actual, entry)
            finally:
                os.close(fd)


def inventory(root_fd):
    files, directories = set(), set()

    def visit(fd, prefix):
        with os.scandir(fd) as children:
            for child in children:
                rel = prefix + child.name
                relative_path(rel)
                before = child.stat(follow_symlinks=False)
                if stat.S_ISREG(before.st_mode):
                    files.add(rel)
                elif stat.S_ISDIR(before.st_mode):
                    directories.add(rel)
                    subfd = os.open(child.name, DIR_FLAGS, dir_fd=fd)
                    try:
                        require(same_object(before, os.fstat(subfd)), "directory_changed")
                        visit(subfd, rel + "/")
                    finally:
                        os.close(subfd)
                else:
                    raise ReleaseError("inventory_link_or_special_file")
    visit(root_fd, "")
    return files, directories


def check_tree(root_fd, expected):
    wanted_dirs = set()
    for rel in expected:
        parts = rel.split("/")
        wanted_dirs.update("/".join(parts[:i]) for i in range(1, len(parts)))
    files, directories = inventory(root_fd)
    require(files == set(expected) and directories == wanted_dirs, "inventory_mismatch")
    for rel, item in expected.items():
        with regular_at(root_fd, rel) as (stream, before):
            actual, _ = consume(stream, before)
        assert_identity(actual, item)
    # Repeat names/types after all reads to detect additions during validation.
    require(inventory(root_fd) == (files, directories), "inventory_changed")


def expected_manifest(spec, spec_identity):
    files = [{"path": e["destination_relative"], "bytes": e["bytes"], "sha256": e["sha256"]}
             for e in spec["files"]]
    files.append({"path": SPEC_NAME, **spec_identity})
    return {"schema": MANIFEST_SCHEMA, "spec": {"path": SPEC_NAME, **spec_identity},
            "metadata": spec["metadata"], "files": sorted(files, key=lambda e: e["path"])}


def expected_completion(manifest_identity, spec_identity, entries):
    return {"schema": COMPLETION_SCHEMA, "status": "byte_verified",
            "manifest": {"path": MANIFEST_NAME, **manifest_identity},
            "spec": {"path": SPEC_NAME, **spec_identity},
            "payload_file_count": len(entries), "payload_bytes": sum(e["bytes"] for e in entries.values()),
            "package_regular_file_count": len(entries) + 3}


def verify(bundle, expected_manifest_sha256, expected_completion_sha256):
    """Read only the package: never open source_path identities in its spec."""
    bundle = absolute_path(bundle)
    with open_directory(bundle) as root_fd:
        completion_id, completion_data = read_pinned_at(root_fd, COMPLETION_NAME, expected_completion_sha256)
        manifest_id, manifest_data = read_pinned_at(root_fd, MANIFEST_NAME, expected_manifest_sha256)
        completion, manifest = parse_json(completion_data), parse_json(manifest_data)
        require(isinstance(completion, dict) and isinstance(completion.get("spec"), dict), "invalid_completion")
        spec_sha = completion["spec"].get("sha256")
        spec_id, spec_data = read_pinned_at(root_fd, SPEC_NAME, spec_sha)
        spec = parse_json(spec_data)
        entries = validate_spec(spec)
        require(manifest_data == encoded(expected_manifest(spec, spec_id)), "manifest_spec_binding_mismatch")
        require(completion_data == encoded(expected_completion(manifest_id, spec_id, entries)), "completion_binding_mismatch")
        expected = {e["path"]: {"bytes": e["bytes"], "sha256": e["sha256"]} for e in manifest["files"]}
        expected[MANIFEST_NAME], expected[COMPLETION_NAME] = manifest_id, completion_id
        check_tree(root_fd, expected)
    return {"status": "byte_verified", "manifest": manifest_id, "completion": completion_id,
            "spec": spec_id, "payload_file_count": len(entries), "package_regular_file_count": len(expected),
            "package_bytes": sum(e["bytes"] for e in expected.values()),
            "scope": "explicit-list byte identity only; no scientific, privacy, publication, or submission approval"}


def assemble(spec_path, expected_spec_sha256, output):
    spec_path, output = absolute_path(spec_path), absolute_path(output)
    require(not beneath_or_equal(spec_path, output) and not beneath_or_equal(output, spec_path), "input_output_overlap")
    spec_id, spec_data = read_pinned_absolute(spec_path, expected_spec_sha256)
    spec = parse_json(spec_data)
    entries = validate_spec(spec)
    for entry in entries.values():
        source = entry["source_path"]
        require(not beneath_or_equal(source, output) and not beneath_or_equal(output, source), "input_output_overlap")
    # Validate output parent and absence BEFORE the potentially large source pass.
    output_parts = path_parts(output, absolute=True)
    root = os.open("/", DIR_FLAGS)
    try:
        with parent_at(root, "/".join(output_parts)) as (parent_fd, output_name):
            try:
                os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ReleaseError("output_already_exists")
            # All sources are separately type/size/hash checked before creation.
            for entry in entries.values():
                with regular_absolute(entry["source_path"]) as (stream, before):
                    require(before.st_size == entry["bytes"], "identity_mismatch")
                    actual, _ = consume(stream, before)
                assert_identity(actual, entry)
            os.mkdir(output_name, 0o755, dir_fd=parent_fd)
            root_fd = os.open(output_name, DIR_FLAGS, dir_fd=parent_fd)
            completed = 0
            pending_identity = None
            try:
                root_identity = os.fstat(root_fd)
                new_file(root_fd, SPEC_NAME, spec_data)
                for entry in entries.values():
                    copy_file(root_fd, entry)
                    completed += 1
                manifest = expected_manifest(spec, spec_id)
                manifest_data = encoded(manifest)
                require(len(manifest_data) <= MAX_JSON_BYTES, "json_too_large")
                manifest_id = identity_bytes(manifest_data)
                new_file(root_fd, MANIFEST_NAME, manifest_data)
                expected = {e["path"]: {"bytes": e["bytes"], "sha256": e["sha256"]} for e in manifest["files"]}
                expected[MANIFEST_NAME] = manifest_id
                completion_data = encoded(expected_completion(manifest_id, spec_id, entries))
                completion_id = identity_bytes(completion_data)
                # The pending file is not a completion marker. Rehash everything
                # before committing it; hard-link creation never overwrites an
                # already present destination, unlike an ordinary rename.
                new_file(root_fd, PENDING_NAME, completion_data)
                pending_identity = os.stat(PENDING_NAME, dir_fd=root_fd, follow_symlinks=False)
                expected[PENDING_NAME] = completion_id
                check_tree(root_fd, expected)
                require(same_object(root_identity, os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)),
                        "output_root_changed")
                os.link(PENDING_NAME, COMPLETION_NAME, src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)
                os.unlink(PENDING_NAME, dir_fd=root_fd)
                os.fsync(root_fd)
            except BaseException as error:
                # Keep partial files; never delete any source or prior output.
                # Revoke only this attempt's own marker on a catchable commit
                # failure. A hard kill leaves a pending/extra file or an absent
                # marker until commit; verify always requires the exact tree.
                if pending_identity is not None:
                    try:
                        marker = os.stat(COMPLETION_NAME, dir_fd=root_fd, follow_symlinks=False)
                        if same_object(pending_identity, marker):
                            os.unlink(COMPLETION_NAME, dir_fd=root_fd)
                    except OSError:
                        pass
                code = str(error) if isinstance(error, ReleaseError) else type(error).__name__
                try:
                    new_file(root_fd, FAILURE_NAME, encoded({"status": "incomplete", "error_code": code,
                             "copied_payload_files": completed, "expected_payload_files": len(entries)}))
                except BaseException:
                    pass
                raise
            finally:
                os.close(root_fd)
    finally:
        os.close(root)
    return {"status": "byte_verified", "manifest": manifest_id, "completion": completion_id,
            "spec": spec_id, "payload_file_count": len(entries), "package_regular_file_count": len(entries) + 3,
            "package_bytes": sum(e["bytes"] for e in entries.values()) + spec_id["bytes"] + manifest_id["bytes"] + completion_id["bytes"],
            "scope": "explicit-list byte identity only; no scientific, privacy, publication, or submission approval"}


def interrupt_handler(signum, _frame):
    raise ReleaseError("interrupted_signal_" + str(signum))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assemble_parser = sub.add_parser("assemble", help="copy an externally pinned explicit list to a new directory")
    assemble_parser.add_argument("--spec", required=True)
    assemble_parser.add_argument("--expected-spec-sha256", required=True)
    assemble_parser.add_argument("--output", required=True)
    verify_parser = sub.add_parser("verify", help="verify original package bytes without executing package scripts")
    verify_parser.add_argument("--bundle", required=True)
    verify_parser.add_argument("--expected-manifest-sha256", required=True)
    verify_parser.add_argument("--expected-completion-sha256", required=True)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler)
    try:
        if args.command == "assemble":
            report = assemble(args.spec, args.expected_spec_sha256, args.output)
        else:
            report = verify(args.bundle, args.expected_manifest_sha256, args.expected_completion_sha256)
    except BaseException as error:
        code = str(error) if isinstance(error, ReleaseError) else type(error).__name__
        print(json.dumps({"status": "failed", "error_code": code}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
