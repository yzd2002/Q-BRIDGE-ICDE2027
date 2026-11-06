"""Explicit-list family addon planner, builder and restorer (Python 3.11+, stdlib).

No scientific module is imported or executed. Paths are opened relative to pinned
directory descriptors with O_NOFOLLOW at every component. Requires stable, trusted,
exclusively controlled source/output directories and ancestors. Directory FDs are
not isolation against a same-UID actor that can rename open directories. A
completion record is written only after checks; interrupted directories are incomplete.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys

HISTORICAL_ROOT = "/home/chaojin/yzd/ICDE 2027"
BASELINE_MANIFEST = "FILE_MANIFEST.json"
LAYOUT = "PUBLIC_REPLAY_LAYOUT.json"
DELIVERY = "DELIVERY_INDEX.json"
INPUTS = "input_manifest.json"
ALIASES = "ALIASES.json"
COMPLETE = "COMPLETED.json"
SCHEMA = "qbridge-family-addon-v1"
CHUNK = 1024 * 1024


class BoundaryError(RuntimeError):
    """A provenance, byte identity, or filesystem boundary failed."""


def require(condition, message):
    if not condition:
        raise BoundaryError(message)


def relative(value):
    require(isinstance(value, str) and value and not any(
        ch in value for ch in ("\\", ":", "\x00", "\n", "\r")), "Invalid relative path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and value != "."
            and p.as_posix() == value, "Unsafe relative path: " + value)
    return p.parts


def identity(record):
    require(isinstance(record, dict), "Identity must be an object")
    sha, size = record.get("sha256"), record.get("bytes")
    require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha), "Invalid SHA256")
    require(type(size) is int and size >= 0, "Invalid byte count")
    return {"sha256": sha, "bytes": size}


def digest(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def key(record):
    rec = identity(record)
    return rec["sha256"], rec["bytes"]


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def parse(raw):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, "Duplicate JSON key: " + name)
            result[name] = value
        return result

    def invalid(value):
        raise BoundaryError("Nonfinite JSON constant: " + value)

    result = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    require(isinstance(result, dict), "Expected JSON object")
    return result


def absolute(path):
    p = Path(path).absolute()
    require(".." not in p.parts, "Root traversal rejected")
    return p


def signature(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_mode)


def _open_absolute_directory(path):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        return fd
    except BaseException:
        os.close(fd)
        raise


def _rename_noreplace(parent_fd, old, new):
    """Linux renameat2(RENAME_NOREPLACE); no check-then-overwriting-rename fallback.

    Flag 1 is the Linux UAPI value from <linux/fs.h>. ctypes is stdlib; this calls
    the existing libc export, adding no package or executable dependency.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    require(rename is not None, "Linux libc renameat2 is required; no unsafe fallback")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    result = rename(parent_fd, os.fsencode(old), parent_fd, os.fsencode(new), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), new)


class Tree:
    """Pinned directory with descriptor-relative reads and exclusive writes."""

    def __init__(self, path):
        self.path = absolute(path)
        self.fd = _open_absolute_directory(self.path)
        st = os.fstat(self.fd)
        self.root_identity = st.st_dev, st.st_ino
        self.container_fd = None

    @classmethod
    def _from_open_fd(cls, path, fd, container_fd):
        """Adopt the already opened creation FD; never select another path root."""
        tree = cls.__new__(cls)
        tree.path, tree.fd, tree.container_fd = absolute(path), fd, container_fd
        st = os.fstat(fd)
        tree.root_identity = st.st_dev, st.st_ino
        return tree

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        os.close(self.fd)
        if self.container_fd is not None:
            os.close(self.container_fd)

    def check_root(self):
        if self.container_fd is not None:
            st = os.stat(self.path.name, dir_fd=self.container_fd, follow_symlinks=False)
            require(stat.S_ISDIR(st.st_mode) and (st.st_dev, st.st_ino) == self.root_identity,
                    "Created root entry was replaced: " + str(self.path))
        fd = _open_absolute_directory(self.path)
        try:
            st = os.fstat(fd)
            require((st.st_dev, st.st_ino) == self.root_identity, "Root was replaced: " + str(self.path))
        finally:
            os.close(fd)

    @classmethod
    def create(cls, path, sources):
        path = absolute(path)
        for source in sources:
            require(not path.is_relative_to(source.path) and not source.path.is_relative_to(path),
                    "Output must be separate from every source tree")
        with cls(path.parent) as parent:
            # Refuse existing targets early. The atomic rename also enforces this
            # if an entry appears later; ordinary rename could overwrite an empty dir.
            try:
                os.stat(path.name, dir_fd=parent.fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(errno.EEXIST, "Output already exists", str(path))
            parent.check_root()
            stage = ".family-create-" + secrets.token_hex(16)
            os.mkdir(stage, 0o700, dir_fd=parent.fd)
            child_fd = None
            tree = None
            try:
                child_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent.fd)
                st = os.fstat(child_fd)
                require(st.st_uid == os.geteuid() and stat.S_IMODE(st.st_mode) & 0o077 == 0,
                        "Creation staging directory must be private to the current user")
                tree = cls._from_open_fd(parent.path / stage, child_fd, os.dup(parent.fd))
                child_fd = None
                tree.check_root()
                parent.check_root()
                _rename_noreplace(parent.fd, stage, path.name)
                tree.path = path
                tree.check_root()
                parent.check_root()
                return tree
            except BaseException:
                if tree is not None:
                    tree.__exit__()
                elif child_fd is not None:
                    os.close(child_fd)
                # No recursive cleanup: leave an incomplete private staging root
                # for diagnosis. A concurrent writer may have changed its binding.
                raise

    def _parent(self, rel, create=False):
        parts = relative(rel)
        fd = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, 0o755, dir_fd=fd)
                    except FileExistsError:
                        pass
                new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = new
            return fd, parts[-1]
        except BaseException:
            os.close(fd)
            raise

    def open(self, rel, write=False):
        self.check_root()
        fd, leaf = self._parent(rel, create=write)
        try:
            flags = os.O_NOFOLLOW | (os.O_WRONLY | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY)
            opened = os.open(leaf, flags | os.O_NONBLOCK, 0o644, dir_fd=fd)
        finally:
            os.close(fd)
        try:
            require(stat.S_ISREG(os.fstat(opened).st_mode), "Nonregular file: " + rel)
            return os.fdopen(opened, "wb" if write else "rb")
        except BaseException:
            os.close(opened)
            raise

    def write(self, rel, raw):
        with self.open(rel, write=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.check_root()

    def files(self):
        """Inventory only an output/addon tree; never collect source trees."""
        self.check_root()
        found = set()

        def visit(fd, prefix):
            for name in os.listdir(fd):
                rel = prefix + name
                relative(rel)
                st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(st.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        visit(child, rel + "/")
                    finally:
                        os.close(child)
                else:
                    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1,
                            "Nonregular, linked or symlink output: " + rel)
                    found.add(rel)

        visit(self.fd, "")
        self.check_root()
        return found


class ReadSet:
    """Hash every selected physical file; recheck bytes and stat identity at end."""

    def __init__(self):
        self.records = {}

    def read(self, tree, rel, expected=None, *, retain=False, target=None, destination=None):
        with tree.open(rel) as source:
            before = signature(os.fstat(source.fileno()))
            h, count, chunks = hashlib.sha256(), 0, []
            with ExitStack() as stack:
                output = stack.enter_context(target.open(destination, write=True)) if target else None
                while block := source.read(CHUNK):
                    h.update(block)
                    count += len(block)
                    if retain:
                        chunks.append(block)
                    if output:
                        output.write(block)
                if output:
                    output.flush()
                    os.fsync(output.fileno())
            require(signature(os.fstat(source.fileno())) == before, "File changed during read: " + rel)
        tree.check_root()
        if target is not None:
            target.check_root()
        actual = {"sha256": h.hexdigest(), "bytes": count}
        if expected is not None:
            require(actual == identity(expected), "File bytes mismatch: " + rel)
        token = tree.path, rel
        observation = (tree, before, actual)
        if token in self.records:
            require(self.records[token][1:] == observation[1:], "Source identity changed: " + rel)
        else:
            self.records[token] = observation
        return b"".join(chunks) if retain else actual

    def verify_all(self):
        for (_, rel), (tree, _, expected) in list(self.records.items()):
            self.read(tree, rel, expected)
        for tree in {record[0] for record in self.records.values()}:
            tree.check_root()


def _no_prefix_collisions(paths):
    names = set(paths)
    for value in names:
        parts = relative(value)
        require(all("/".join(parts[:i]) not in names for i in range(1, len(parts))),
                "File/directory path collision: " + value)


def _plan_documents(raws):
    baseline, layout, manifest, delivery = (parse(raws[name]) for name in
                                           (BASELINE_MANIFEST, LAYOUT, INPUTS, DELIVERY))
    require(baseline.get("schema") == "qbridge-public-file-manifest-v1", "Baseline schema")
    require(isinstance(baseline.get("files"), list), "Baseline file list")
    candidates, listed, hash_sizes = {}, {}, {}
    for row in baseline["files"]:
        require(isinstance(row, dict) and set(row) == {"path", "sha256", "bytes"}, "Baseline record schema")
        name = row["path"]
        relative(name)
        require(name not in listed, "Duplicate baseline physical path")
        rec = identity(row)
        sha, size = key(rec)
        require(sha not in hash_sizes or hash_sizes[sha] == size, "Conflicting baseline SHA byte counts")
        hash_sizes[sha] = size
        listed[name] = rec
        candidates.setdefault(key(rec), []).append(name)
    exclusions = baseline.get("metadata_self_exclusions")
    require(exclusions == [BASELINE_MANIFEST, "SHA256SUMS"], "Baseline metadata exclusions")
    require(baseline.get("file_count") == len(listed) + len(exclusions), "Baseline file census")
    _no_prefix_collisions(listed)
    require(manifest.get("format") == "policy-family-portable-inputs-v1", "Input schema")
    require(manifest.get("historical_root") == HISTORICAL_ROOT, "Unknown historical root")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Empty input allowlist")
    for rel, rec in files.items():
        relative(rel)
        require(not rel.startswith(".python/"), "Historical runtime bytes are provenance only")
        require(isinstance(rec, dict) and set(rec) == {"sha256", "bytes", "roles"}
                and isinstance(rec["roles"], list) and rec["roles"]
                and all(isinstance(role, str) for role in rec["roles"])
                and len(rec["roles"]) == len(set(rec["roles"])), "Input role schema")
        identity(rec)
    require(manifest.get("file_count") == len(files), "Input file census")
    require(manifest.get("total_bytes") == sum(v["bytes"] for v in files.values()), "Input byte census")
    require(layout.get("format") == "policy-family-portable-layout-v1", "Layout schema")
    require(layout.get("historical_runtime_bytes_included") is False, "Runtime bytes cannot be included")
    relative(layout["local_input_source"])  # informational only; never used for fallback
    require(layout.get("input_manifest_sha256") == digest(raws[INPUTS])["sha256"], "Layout manifest binding")
    require(layout.get("input_file_count") == len(files)
            and layout.get("input_bytes") == manifest["total_bytes"], "Layout input census")
    require(layout.get("project_input_files") == {
        rel: {**identity(rec), "destination": "project/" + rel} for rel, rec in files.items()},
        "Layout input allowlist differs from manifest")
    tools = layout.get("tool_and_evidence_files")
    require(isinstance(tools, dict) and INPUTS in tools, "Missing replay-tool allowlist")
    for rel, rec in tools.items():
        relative(rel)
        require(set(rec) == {"sha256", "bytes", "destination"}
                and rec["destination"] == "policy_replay/" + rel, "Tool destination/schema")
        identity(rec)
    require(identity(tools[INPUTS]) == digest(raws[INPUTS]), "Tool manifest binding")
    require(delivery.get("format") == "policy-family-portable-delivery-index-v1", "Delivery schema")
    require(delivery.get("input_manifest") == digest(raws[INPUTS]), "Delivery manifest binding")
    require(delivery.get("files") == {**{rel: identity(rec) for rel, rec in tools.items()},
                                     LAYOUT: digest(raws[LAYOUT])}, "Delivery allowlist/binding mismatch")

    artifacts = {}

    def add(destination, rec, origin, source):
        relative(destination)
        require(destination not in artifacts, "Duplicate destination")
        sha, size = key(rec)
        require(sha not in hash_sizes or hash_sizes[sha] == size, "Conflicting SHA byte counts")
        hash_sizes[sha] = size
        available = candidates.get((sha, size), [])
        artifacts[destination] = {
            **identity(rec), "origin": {"tree": origin, "path": source},
            "storage": "baseline" if available else "addon",
            "path": min(available) if available else "objects/sha256/" + sha,
        }

    for rel, rec in sorted(files.items()):
        add("project/" + rel, rec, "input", rel)
    for rel, rec in sorted(tools.items()):
        add("policy_replay/" + rel, rec, "replay", rel)
    for name in (LAYOUT, DELIVERY):
        add(name, digest(raws[name]), "replay", name)
    _no_prefix_collisions(artifacts)
    project_unique = {key(rec) for rec in files.values()}
    reused = project_unique & set(candidates)
    new = project_unique - reused
    objects = {row["path"]: identity(row) for row in artifacts.values() if row["storage"] == "addon"}
    return {
        "format": SCHEMA, "verification": "metadata_only_not_content_verified",
        "metadata": {name: digest(raws[name]) for name in sorted(raws)},
        "artifacts": artifacts,
        "counts": {
            "project_logical_files": len(files), "project_logical_bytes": manifest["total_bytes"],
            "project_unique_sha256": len(project_unique), "baseline_project_unique_candidates": len(reused),
            "addon_project_unique_sha256": len(new), "addon_project_unique_bytes": sum(size for _, size in new),
            "tool_and_evidence_files": len(tools), "descriptor_files": 2,
            "restored_files": len(artifacts), "addon_objects": len(objects),
            "addon_object_bytes": sum(rec["bytes"] for rec in objects.values()),
        },
        "scope": "Explicit family inputs and reviewed tools only; no science replay or publication."
    }


def _read_metadata(baseline, replay, reads):
    raws = {BASELINE_MANIFEST: reads.read(baseline, BASELINE_MANIFEST, retain=True)}
    raws.update({name: reads.read(replay, name, retain=True) for name in (LAYOUT, INPUTS, DELIVERY)})
    return raws


def plan(baseline_root, replay_root):
    """Read four metadata files only; candidate file bytes are NOT verified."""
    with Tree(baseline_root) as baseline, Tree(replay_root) as replay:
        reads = ReadSet()
        result = _plan_documents(_read_metadata(baseline, replay, reads))
        reads.verify_all()
        return result


def _load_pinned(path, expected_sha, reads, stack):
    require(isinstance(expected_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha), "Expected SHA256 required")
    path = absolute(path)
    tree = stack.enter_context(Tree(path.parent))
    raw = reads.read(tree, path.name, retain=True)
    require(digest(raw)["sha256"] == expected_sha, "Pinned document SHA256 mismatch")
    return raw


def _payload(plan_doc):
    return {row["path"]: identity(row) for row in plan_doc["artifacts"].values()
            if row["storage"] == "addon"}


def _verify_output(tree, expected):
    require(tree.files() == set(expected), "Output inventory mismatch")
    reads = ReadSet()
    for rel, rec in sorted(expected.items()):
        reads.read(tree, rel, rec)
    tree.check_root()


def _complete(tree, receipt, source_roots=()):
    """Publish the marker atomically; an I/O/check failure removes our marker."""
    pending = "_COMPLETION_PENDING.json"
    raw = encode(receipt)
    linked = False

    def check_roots():
        for root in (tree, *source_roots):
            root.check_root()

    try:
        check_roots()
        tree.write(pending, raw)
        ReadSet().read(tree, pending, digest(raw))
        check_roots()
        # link is exclusive even if another process has created COMPLETED.json.
        os.link(pending, COMPLETE, src_dir_fd=tree.fd, dst_dir_fd=tree.fd, follow_symlinks=False)
        linked = True
        os.unlink(pending, dir_fd=tree.fd)
        os.fsync(tree.fd)
        check_roots()
    except BaseException:
        if linked:
            os.unlink(COMPLETE, dir_fd=tree.fd)
        raise


def build(baseline_root, replay_root, input_root, plan_path, plan_sha256, output):
    """Build a new addon after verifying all referenced physical source bytes."""
    with ExitStack() as stack:
        baseline, replay, inputs = [stack.enter_context(Tree(p)) for p in (baseline_root, replay_root, input_root)]
        reads = ReadSet()
        pinned = parse(_load_pinned(plan_path, plan_sha256, reads, stack))
        actual = _plan_documents(_read_metadata(baseline, replay, reads))
        require(pinned == actual, "Plan no longer matches source metadata")
        sources, new_sources = {}, {}
        for row in actual["artifacts"].values():
            if row["storage"] == "baseline":
                tree, rel = baseline, row["path"]
            else:
                tree = inputs if row["origin"]["tree"] == "input" else replay
                rel = row["origin"]["path"]
                new_sources.setdefault(row["path"], (tree, rel, identity(row)))
            sources[(tree.path, rel)] = (tree, rel, identity(row))
        for tree, rel, expected in sources.values():
            reads.read(tree, rel, expected)
        target = stack.enter_context(Tree.create(output, [baseline, replay, inputs]))
        target.write(ALIASES, encode(actual))
        for destination, (tree, rel, expected) in sorted(new_sources.items()):
            reads.read(tree, rel, expected, target=target, destination=destination)
        payload = {ALIASES: digest(encode(actual)), **_payload(actual)}
        _verify_output(target, payload)
        reads.verify_all()
        _verify_output(target, payload)
        receipt = {
            "format": SCHEMA + "-completion", "operation": "build", "complete": True,
            "plan_sha256": plan_sha256, "aliases": payload[ALIASES], "payload": payload,
            "counts": actual["counts"], "referenced_physical_sources_checked": len(sources),
            "filesystem_assumption": "Stable, trusted, exclusively controlled directories and ancestors; no hostile same-UID renames",
            "output_creation": "Private 0700 staging, retained directory FD, Linux renameat2(RENAME_NOREPLACE)",
            "checks": {"source_bytes_before_and_after": True, "output_bytes_and_membership": True,
                       "scientific_replay_executed": False},
        }
        target.check_root()
        _complete(target, receipt, tuple({baseline, replay, inputs, *(r[0] for r in reads.records.values())}))
        return {**receipt, "completion_sha256": digest(encode(receipt))["sha256"]}


def restore(baseline_root, addon_root, completion_sha256, output):
    """Expand only baseline/addon objects; never consult origin/source-machine paths."""
    with ExitStack() as stack:
        baseline, addon = [stack.enter_context(Tree(p)) for p in (baseline_root, addon_root)]
        reads = ReadSet()
        complete_raw = reads.read(addon, COMPLETE, retain=True)
        require(digest(complete_raw)["sha256"] == completion_sha256, "Completion SHA256 mismatch")
        receipt = parse(complete_raw)
        require(receipt.get("format") == SCHEMA + "-completion" and receipt.get("operation") == "build"
                and receipt.get("complete") is True, "Addon is not completed")
        raw_plan = reads.read(addon, ALIASES, receipt["aliases"], retain=True)
        selected = parse(raw_plan)
        require(selected.get("format") == SCHEMA, "Addon schema")
        artifacts = selected["artifacts"]

        def physical(row):
            identity(row)
            relative(row["path"])
            require(row["storage"] in ("baseline", "addon"), "Unknown storage")
            if row["storage"] == "addon":
                require(row["path"] == "objects/sha256/" + row["sha256"], "Invalid object path")
            return baseline if row["storage"] == "baseline" else addon

        raws = {BASELINE_MANIFEST: reads.read(baseline, BASELINE_MANIFEST, selected["metadata"][BASELINE_MANIFEST], retain=True)}
        for name, destination in ((LAYOUT, LAYOUT), (DELIVERY, DELIVERY), (INPUTS, "policy_replay/" + INPUTS)):
            row = artifacts[destination]
            raws[name] = reads.read(physical(row), row["path"], row, retain=True)
        canonical = _plan_documents(raws)
        require(selected == canonical, "Aliases differ from original allowlists or canonical physical references")
        payload = {ALIASES: digest(raw_plan), **_payload(canonical)}
        require(receipt.get("payload") == payload and receipt.get("counts") == canonical["counts"], "Addon completion payload binding")
        require(addon.files() == set(payload) | {COMPLETE}, "Addon has missing or unexpected files")
        checked = set()
        for row in artifacts.values():
            tree = physical(row)
            token = tree.path, row["path"]
            if token not in checked:
                reads.read(tree, row["path"], row)
                checked.add(token)
        target = stack.enter_context(Tree.create(output, [baseline, addon]))
        for destination, row in sorted(artifacts.items()):
            reads.read(physical(row), row["path"], row, target=target, destination=destination)
        expected = {destination: identity(row) for destination, row in artifacts.items()}
        _verify_output(target, expected)
        reads.verify_all()
        _verify_output(target, expected)
        final = {"format": SCHEMA + "-completion", "operation": "restore", "complete": True,
                 "addon_completion_sha256": completion_sha256,
                 "input_manifest": canonical["metadata"][INPUTS], "counts": canonical["counts"],
                 "filesystem_assumption": "Stable, trusted, exclusively controlled directories and ancestors; no hostile same-UID renames",
                 "output_creation": "Private 0700 staging, retained directory FD, Linux renameat2(RENAME_NOREPLACE)",
                 "checks": {"source_bytes_before_and_after": True, "output_bytes_and_membership": True,
                            "scientific_replay_executed": False, "source_machine_fallback": False},
                 "restored_payload": expected}
        target.check_root()
        _complete(target, final, (baseline, addon))
        return {**final, "completion_sha256": digest(encode(final))["sha256"]}


def _write_new(path, raw):
    path = absolute(path)
    with Tree(path.parent) as tree:
        tree.write(path.name, raw)
        tree.check_root()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Metadata-only estimate, no payload hashing")
    planning.add_argument("--baseline", required=True)
    planning.add_argument("--replay", required=True)
    planning.add_argument("--output", required=True)
    building = commands.add_parser("build", help="Verify selected bytes and create new addon")
    building.add_argument("--baseline", required=True)
    building.add_argument("--replay", required=True)
    building.add_argument("--input-root", required=True)
    building.add_argument("--plan", required=True)
    building.add_argument("--plan-sha256", required=True)
    building.add_argument("--output", required=True)
    restoring = commands.add_parser("restore", help="Restore from explicit baseline and addon only")
    restoring.add_argument("--baseline", required=True)
    restoring.add_argument("--addon", required=True)
    restoring.add_argument("--completion-sha256", required=True)
    restoring.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(args.baseline, args.replay)
            raw = encode(result)
            _write_new(args.output, raw)
            summary = {"status": "METADATA_PLAN_ONLY", "payload_bytes_verified": False,
                       "plan": digest(raw), "counts": result["counts"]}
        elif args.command == "build":
            result = build(args.baseline, args.replay, args.input_root, args.plan, args.plan_sha256, args.output)
            summary = {k: v for k, v in result.items() if k != "payload"}
        else:
            result = restore(args.baseline, args.addon, args.completion_sha256, args.output)
            summary = {k: v for k, v in result.items() if k != "restored_payload"}
        print(encode(summary).decode(), end="")
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "complete": False,
                          "error": type(exc).__name__ + ": " + str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
