"""Fixed V6 partition storage adapter; no solver or mathematical replay.

Reuse the frozen family storage primitives under their stable, trusted,
exclusively controlled directory assumption. This is not isolation against a
hostile same-UID actor renaming open directories. Three stores, one flat map.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import os
from pathlib import Path
import stat
import sys
import types

FAMILY_SHA = "b29ffa5889338d07f8a98963fbd13e93ee9d707a90b03355ef3adcd3f33247bb"
SCHEMA = "qbridge-partition-addon-v1"
MANIFEST = "INPUT_MANIFEST.json"
MAP = "FILE_MAP.json"
COMPLETE = "COMPLETED.json"
STORES = ("baseline", "family_addon", "partition")
TOOL_PINS = {
    "ADAPTATION.json": ("7817dcc9dfdd4f0bfa3975fcb8c7b31c4b2df25434331f193d8e97ef5e04f2eb", 5691),
    "adapted/audit.py": ("84971bd83de5f32f1df9cd2e24ad5ab0b76013d3f45a30bd9a2f7d58ab82c0f2", 35763),
    "adapted/math_checks.py": ("924c0c170404fc334e432bd89bdc2f5a5590965210914206c2c8f25f1d759f10", 26333),
    "boundary.py": ("91458a9bdfe71a7f48afca3a5bf37ba2ae17f87b8d34e055ecb5e6af91ea0dc5", 18946),
    "original/audit.py": ("27bba827e475e2775792d6193b46fba14049d248f82faa689f19ef7c42ff354f", 35632),
    "original/math_checks.py": ("924c0c170404fc334e432bd89bdc2f5a5590965210914206c2c8f25f1d759f10", 26333),
    "prepare_manifest.py": ("b89127ec87b78ae75673c09899c991b2162e95f9de02fea799e7ee1f8a29e9b9", 3242),
    "prepare_sources.py": ("e20f0cf400529389bc4b76a33da2df9fa98805bb7acd6c5e47f63093da8a37bb", 8205),
    "replay.py": ("aa8f603077c2347192b628604b5f2e237ab1b87c61fbdc4c31b9b383daea07c8", 6854),
    "restore.py": ("5213af4341e5572633e120490869870f773255f7432331a9f18c1c66d7f02645", 1628),
    "vendor_portable_io.py": ("ea2fbc8cbc0fd9f1e2deb302691ce2afdfb2592f167e2337de04cc9905dcbdd2", 12658),
}


def _frozen_family():
    path = Path(__file__).absolute().with_name("family.py")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError("Frozen family source must be a single-link regular file")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError("Frozen family source changed")
    if hashlib.sha256(raw).hexdigest() != FAMILY_SHA:
        raise RuntimeError("Frozen family storage source SHA256 mismatch")
    module = types.ModuleType("_partition_frozen_family")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


f = _frozen_family()
require = f.require


class Tree(f.Tree):
    """Also refuse multiply linked source files, before any read or copy."""

    def open(self, rel, write=False):
        stream = super().open(rel, write)
        try:
            require(os.fstat(stream.fileno()).st_nlink == 1, "Hardlinked file refused: " + rel)
            return stream
        except BaseException:
            stream.close()
            raise


@contextmanager
def private_umask():
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def _keys(value, wanted, label):
    require(type(value) is dict and set(value) == set(wanted), "Closed schema: " + label)


def _same(a, b):
    # Preserve JSON scalar types: True is not integer 1.
    return f.encode(a) == f.encode(b)


def _pinned(tree, name, expected_sha, reads):
    require(isinstance(expected_sha, str) and f.re.fullmatch("[0-9a-f]{64}", expected_sha), "External SHA256 required")
    raw = reads.read(tree, name, retain=True)
    require(f.digest(raw)["sha256"] == expected_sha, "External pin mismatch: " + name)
    return raw


def _document(path, sha, reads, stack):
    path = f.absolute(path)
    tree = stack.enter_context(Tree(path.parent))
    return _pinned(tree, path.name, sha, reads)


def _baseline(raw):
    value = f.parse(raw)
    allowed = {"schema", "metadata_self_exclusions", "exclusion_explanation", "file_count", "files"}
    require(set(value) <= allowed and set(value) >= allowed - {"exclusion_explanation"}, "Baseline schema keys")
    if "exclusion_explanation" in value:
        require(type(value["exclusion_explanation"]) is str, "Baseline explanation type")
    require(value["schema"] == "qbridge-public-file-manifest-v1" and
            value["metadata_self_exclusions"] == [f.BASELINE_MANIFEST, "SHA256SUMS"], "Baseline format")
    require(type(value["files"]) is list and type(value["file_count"]) is int and
            value["file_count"] == len(value["files"]) + 2, "Baseline census")
    entries = {}
    for row in value["files"]:
        _keys(row, ("path", "sha256", "bytes"), "baseline row")
        f.relative(row["path"])
        require(row["path"] not in entries and row["path"] not in value["metadata_self_exclusions"], "Duplicate baseline path")
        entries[row["path"]] = f.identity(row)
    f._no_prefix_collisions(entries)
    return entries


def _existing(baseline, addon, baseline_sha, family_sha, reads):
    """Validate the original family descriptors, without restoring its tree."""
    base_raw = _pinned(baseline, f.BASELINE_MANIFEST, baseline_sha, reads)
    base = _baseline(base_raw)
    receipt_raw = _pinned(addon, COMPLETE, family_sha, reads)
    receipt = f.parse(receipt_raw)
    _keys(receipt, ("aliases", "checks", "complete", "counts", "filesystem_assumption", "format",
                   "operation", "output_creation", "payload", "plan_sha256", "referenced_physical_sources_checked"), "family completion")
    require(receipt["format"] == f.SCHEMA + "-completion" and receipt["operation"] == "build"
            and receipt["complete"] is True, "Family addon not complete")
    require(_same(receipt["checks"], {"source_bytes_before_and_after": True,
            "output_bytes_and_membership": True, "scientific_replay_executed": False}), "Family checks")
    alias_raw = reads.read(addon, f.ALIASES, f.identity(receipt["aliases"]), retain=True)
    selected = f.parse(alias_raw)
    require(receipt["plan_sha256"] == f.digest(alias_raw)["sha256"], "Original family plan pin")
    require(type(selected.get("artifacts")) is dict, "Family artifacts missing")
    raw = {f.BASELINE_MANIFEST: base_raw}
    for name, destination in ((f.LAYOUT, f.LAYOUT), (f.DELIVERY, f.DELIVERY), (f.INPUTS, "policy_replay/" + f.INPUTS)):
        row = selected["artifacts"][destination]
        _keys(row, ("sha256", "bytes", "origin", "storage", "path"), "family artifact")
        f.relative(row["path"])
        rec = f.identity(row)
        if row["storage"] == "baseline":
            require(base.get(row["path"]) == rec, "Family metadata outside baseline allowlist")
            tree = baseline
        else:
            require(row["storage"] == "addon" and row["path"] == "objects/sha256/" + rec["sha256"], "Family object path")
            require(receipt["payload"].get(row["path"]) == rec, "Family metadata outside pinned payload")
            tree = addon
        raw[name] = reads.read(tree, row["path"], rec, retain=True)
    canonical = f._plan_documents(raw)
    require(_same(selected, canonical), "Family aliases differ from original canonical plan")
    payload = {f.ALIASES: f.digest(alias_raw), **f._payload(canonical)}
    require(_same(receipt["payload"], payload) and _same(receipt["counts"], canonical["counts"]), "Family payload/count binding")
    require(type(receipt["referenced_physical_sources_checked"]) is int and receipt["referenced_physical_sources_checked"] > 0,
            "Family source census")
    require(addon.files() == set(payload) | {COMPLETE}, "Family member mismatch")
    return base, f._payload(canonical), {
        "baseline_manifest": f.digest(base_raw), "family_completion": f.digest(receipt_raw),
        "family_aliases": f.digest(alias_raw),
    }


def _validators(raws):
    """Execute only the two frozen non-mathematical validation modules."""
    require(set(raws) == set(TOOL_PINS), "Fixed eleven tools required")
    for name, (sha, size) in TOOL_PINS.items():
        require(f.digest(raws[name]) == {"sha256": sha, "bytes": size}, "Frozen replay tool changed: " + name)
    vendor = types.ModuleType("vendor_portable_io")
    vendor.__file__ = "<pinned vendor_portable_io.py>"
    exec(compile(raws["vendor_portable_io.py"], vendor.__file__, "exec"), vendor.__dict__)
    boundary = types.ModuleType("_partition_pinned_boundary")
    boundary.__file__ = "<pinned boundary.py>"
    missing = object()
    prior = sys.modules.get("vendor_portable_io", missing)
    try:
        sys.modules["vendor_portable_io"] = vendor
        exec(compile(raws["boundary.py"], boundary.__file__, "exec"), boundary.__dict__)
    finally:
        if prior is missing:
            del sys.modules["vendor_portable_io"]
        else:
            sys.modules["vendor_portable_io"] = prior
    return boundary, vendor


def _code(reads, stack):
    tree = stack.enter_context(Tree(Path(__file__).absolute().parent))
    raw = reads.read(tree, Path(__file__).name, retain=True)
    _pinned(tree, "family.py", FAMILY_SHA, reads)
    return {"partition.py": f.digest(raw), "family.py": {"sha256": FAMILY_SHA, "bytes": 28841}}


def _canonical(base, family_objects, metadata, manifest_raw, tools_raw, implementation):
    boundary, vendor = _validators(tools_raw)
    manifest = boundary.validate_manifest(f.parse(manifest_raw))
    require(manifest["file_count"] == 744, "Only the fixed 744-input V6 closure is supported")
    tools = {name: {"sha256": sha, "bytes": size} for name, (sha, size) in TOOL_PINS.items()}
    require(_same(manifest["tool_files"], tools), "Original eleven tool pins changed")
    candidates, hash_sizes = {}, {}
    for store, records in (("baseline", base), ("family_addon", family_objects)):
        for path, rec in sorted(records.items()):
            token = f.key(rec)
            require(token[0] not in hash_sizes or hash_sizes[token[0]] == token[1], "Conflicting SHA size")
            hash_sizes[token[0]] = token[1]
            candidates.setdefault(token, (store, path))
    artifacts = {}
    for origin, prefix, records in (("input", "project/", manifest["files"]), ("replay", "tools/", tools)):
        for name, rec in sorted(records.items()):
            f.relative(name)
            token = f.key(rec)
            require(token[0] not in hash_sizes or hash_sizes[token[0]] == token[1], "Conflicting SHA size")
            hash_sizes[token[0]] = token[1]
            store, path = candidates.get(token, ("partition", "objects/sha256/" + token[0]))
            artifacts[prefix + name] = {**f.identity(rec), "origin": {"tree": origin, "path": name}, "storage": store, "path": path}
    f._no_prefix_collisions([*artifacts, MANIFEST])
    input_keys = {f.key(r) for r in manifest["files"].values()}
    tool_keys = {f.key(r) for r in tools.values()}
    new_inputs = input_keys - set(candidates)
    new_tools = tool_keys - set(candidates) - input_keys
    objects = {row["path"]: f.identity(row) for row in artifacts.values() if row["storage"] == "partition"}
    result = {"format": SCHEMA, "verification": "metadata_plan_not_payload_verified",
        "metadata": {**metadata, "input_manifest": f.digest(manifest_raw)}, "implementation": implementation,
        "artifacts": artifacts, "counts": {
            "project_logical_files": 744, "project_logical_bytes": manifest["total_bytes"],
            "tool_logical_files": 11, "tool_logical_bytes": sum(r["bytes"] for r in tools.values()),
            "project_unique_contents": len(input_keys), "new_input_objects": len(new_inputs),
            "new_input_bytes": sum(size for sha, size in new_inputs), "additional_new_tool_objects": len(new_tools),
            "additional_new_tool_bytes": sum(size for sha, size in new_tools), "new_objects": len(objects),
            "new_object_bytes": sum(r["bytes"] for r in objects.values()), "raw_manifest_bytes": len(manifest_raw),
            "restored_files": 756, "restored_bytes": manifest["total_bytes"] + sum(r["bytes"] for r in tools.values()) + len(manifest_raw)},
        "scope": "Fixed V6 partition input/tool storage only; no science replay or source-machine fallback."}
    return result, manifest, boundary, vendor


def _metadata(stack, baseline_root, family_root, replay_root, manifest_path, baseline_sha, family_sha, manifest_sha, reads):
    baseline, addon, replay = [stack.enter_context(Tree(p)) for p in (baseline_root, family_root, replay_root)]
    base, objects, metadata = _existing(baseline, addon, baseline_sha, family_sha, reads)
    raw = _document(manifest_path, manifest_sha, reads, stack)
    tools = {name: reads.read(replay, name, {"sha256": sha, "bytes": size}, retain=True)
             for name, (sha, size) in TOOL_PINS.items()}
    result = _canonical(base, objects, metadata, raw, tools, _code(reads, stack))
    return baseline, addon, replay, raw, result


def plan(baseline_root, family_root, replay_root, manifest_path, baseline_sha256, family_sha256, manifest_sha256):
    with ExitStack() as stack:
        reads = f.ReadSet()
        *_, result = _metadata(stack, baseline_root, family_root, replay_root, manifest_path,
                              baseline_sha256, family_sha256, manifest_sha256, reads)
        reads.verify_all()
        return result[0]


def _payload(plan_doc, manifest_raw):
    return {MAP: f.digest(f.encode(plan_doc)), MANIFEST: f.digest(manifest_raw),
        **{row["path"]: f.identity(row) for row in plan_doc["artifacts"].values() if row["storage"] == "partition"}}


def _remove_own_marker(tree, expected):
    try:
        with tree.open(COMPLETE) as stream:
            before = os.fstat(stream.fileno())
            require(f.digest(stream.read()) == expected, "Cannot invalidate a replaced completion marker")
        tree.check_root()
        current = os.stat(COMPLETE, dir_fd=tree.fd, follow_symlinks=False)
        require((current.st_dev, current.st_ino) == (before.st_dev, before.st_ino), "Completion inode changed")
        os.unlink(COMPLETE, dir_fd=tree.fd)
        os.fsync(tree.fd)
    except FileNotFoundError:
        pass


def _finish(tree, receipt, expected, reads):
    roots = tuple({record[0] for record in reads.records.values()})
    f._verify_output(tree, expected)
    reads.verify_all()
    f._verify_output(tree, expected)
    marker = f.digest(f.encode(receipt))
    written = False
    try:
        f._complete(tree, receipt, roots)
        written = True
        final = {**expected, COMPLETE: marker}
        f._verify_output(tree, final)
        reads.verify_all()
        f._verify_output(tree, final)
    except BaseException:
        if written:
            _remove_own_marker(tree, marker)
        raise
    return {**receipt, "completion_sha256": marker["sha256"]}


def build(baseline_root, family_root, replay_root, input_root, manifest_path, baseline_sha256,
          family_sha256, manifest_sha256, plan_path, plan_sha256, output):
    with private_umask(), ExitStack() as stack:
        reads = f.ReadSet()
        plan_raw = _document(plan_path, plan_sha256, reads, stack)
        selected = f.parse(plan_raw)
        baseline, addon, replay, raw, (actual, _, _, _) = _metadata(stack, baseline_root, family_root,
            replay_root, manifest_path, baseline_sha256, family_sha256, manifest_sha256, reads)
        require(_same(selected, actual), "Plan no longer matches pinned source metadata/implementation")
        require(plan_raw == f.encode(actual), "Plan bytes must use the canonical encoding returned by plan")
        inputs = stack.enter_context(Tree(input_root))
        sources, new = {}, {}
        for row in actual["artifacts"].values():
            if row["storage"] == "baseline":
                tree, path = baseline, row["path"]
            elif row["storage"] == "family_addon":
                tree, path = addon, row["path"]
            else:
                tree = inputs if row["origin"]["tree"] == "input" else replay
                path = row["origin"]["path"]
                new.setdefault(row["path"], (tree, path, f.identity(row)))
            sources[(tree.path, path)] = tree, path, f.identity(row)
        for tree, path, rec in sources.values():
            reads.read(tree, path, rec)
        roots = list({baseline, addon, replay, inputs, *(record[0] for record in reads.records.values())})
        target = stack.enter_context(Tree.create(output, roots))
        target.write(MAP, plan_raw)
        target.write(MANIFEST, raw)
        for dest, (tree, path, rec) in sorted(new.items()):
            reads.read(tree, path, rec, target=target, destination=dest)
        payload = _payload(actual, raw)
        receipt = {"format": SCHEMA + "-completion", "operation": "build", "complete": True,
            "plan_sha256": plan_sha256, "metadata": actual["metadata"], "implementation": actual["implementation"],
            "payload": payload, "counts": actual["counts"],
            "checks": {"source_bytes_before_and_after": True, "output_bytes_and_membership": True,
                       "scientific_replay_executed": False},
            "filesystem_assumption": "Stable trusted exclusive directories; no hostile same-UID renames"}
        return _finish(target, receipt, payload, reads)


def _physical(row, stores, base, family_objects, partition_payload):
    _keys(row, ("sha256", "bytes", "origin", "storage", "path"), "partition artifact")
    _keys(row["origin"], ("tree", "path"), "artifact origin")
    require(row["origin"]["tree"] in ("input", "replay"), "Unknown origin")
    f.relative(row["origin"]["path"])
    f.relative(row["path"])
    rec = f.identity(row)
    store = row["storage"]
    require(store in STORES, "Unknown physical store")
    allowed = {"baseline": base, "family_addon": family_objects, "partition": partition_payload}[store]
    require(allowed.get(row["path"]) == rec, "Physical reference outside pinned allowlist")
    if store != "baseline":
        require(row["path"] == "objects/sha256/" + rec["sha256"], "Object path differs from SHA256")
    return stores[store], row["path"], rec


def restore(baseline_root, family_root, partition_root, baseline_sha256, family_sha256,
            manifest_sha256, completion_sha256, output):
    with private_umask(), ExitStack() as stack:
        reads = f.ReadSet()
        baseline, addon, partition = [stack.enter_context(Tree(p)) for p in (baseline_root, family_root, partition_root)]
        base, family_objects, metadata = _existing(baseline, addon, baseline_sha256, family_sha256, reads)
        completion_raw = _pinned(partition, COMPLETE, completion_sha256, reads)
        receipt = f.parse(completion_raw)
        _keys(receipt, ("format", "operation", "complete", "plan_sha256", "metadata", "implementation", "payload", "counts", "checks", "filesystem_assumption"), "partition completion")
        require(receipt["format"] == SCHEMA + "-completion" and receipt["operation"] == "build" and receipt["complete"] is True, "Partition addon not complete")
        require(_same(receipt["checks"], {"source_bytes_before_and_after": True,
            "output_bytes_and_membership": True, "scientific_replay_executed": False}), "Partition checks")
        manifest_raw = _pinned(partition, MANIFEST, manifest_sha256, reads)
        map_raw = reads.read(partition, MAP, f.identity(receipt["payload"][MAP]), retain=True)
        require(f.digest(map_raw)["sha256"] == receipt["plan_sha256"], "Partition plan pin")
        selected = f.parse(map_raw)
        _keys(selected, ("format", "verification", "metadata", "implementation", "artifacts", "counts", "scope"), "partition map")
        require(type(selected["artifacts"]) is dict, "Partition artifacts missing")
        stores = dict(zip(STORES, (baseline, addon, partition)))
        tool_raws = {}
        for name, (sha, size) in TOOL_PINS.items():
            row = selected["artifacts"]["tools/" + name]
            require(f.identity(row) == {"sha256": sha, "bytes": size}, "Frozen tool map changed")
            tree, path, rec = _physical(row, stores, base, family_objects, receipt["payload"])
            tool_raws[name] = reads.read(tree, path, rec, retain=True)
        canonical, manifest, boundary, vendor = _canonical(base, family_objects, metadata,
            manifest_raw, tool_raws, _code(reads, stack))
        require(_same(selected, canonical), "Partition map differs from canonical pinned manifest")
        payload = _payload(canonical, manifest_raw)
        require(_same(receipt["payload"], payload) and _same(receipt["counts"], canonical["counts"])
                and _same(receipt["metadata"], canonical["metadata"])
                and _same(receipt["implementation"], canonical["implementation"]), "Partition completion binding")
        require(partition.files() == set(payload) | {COMPLETE}, "Partition missing/unexpected members")
        checked = set()
        for row in canonical["artifacts"].values():
            tree, path, rec = _physical(row, stores, base, family_objects, payload)
            if (tree.path, path) not in checked:
                reads.read(tree, path, rec)
                checked.add((tree.path, path))
        roots = list({baseline, addon, partition, *(record[0] for record in reads.records.values())})
        target = stack.enter_context(Tree.create(output, roots))
        target.write(MANIFEST, manifest_raw)
        for dest, row in sorted(canonical["artifacts"].items()):
            tree, path, rec = _physical(row, stores, base, family_objects, payload)
            reads.read(tree, path, rec, target=target, destination=dest)
        boundary.check_tools(target.path / "tools", manifest)
        original_tree_check = vendor.verify_tree(target.path / "project", manifest)
        boundary.check_reference_admission(target.path / "project", manifest)
        expected = {MANIFEST: f.digest(manifest_raw), **{dest: f.identity(row) for dest, row in canonical["artifacts"].items()}}
        result = {"format": SCHEMA + "-completion", "operation": "restore", "complete": True,
            "partition_completion_sha256": completion_sha256, "metadata": canonical["metadata"],
            "implementation": canonical["implementation"], "counts": canonical["counts"],
            "original_verify_tree": original_tree_check, "restored_payload": expected,
            "historical_runtime_bytes_revalidated": False, "historical_runtime_identities_omitted": 77,
            "historical_runtime_snapshot_copies_omitted": 77,
            "checks": {"source_bytes_before_and_after": True, "output_bytes_and_membership": True,
                       "original_boundary_admission": True, "source_machine_fallback": False, "scientific_replay_executed": False}}
        return _finish(target, result, expected, reads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for operation in ("plan", "build", "restore"):
        p = commands.add_parser(operation)
        p.add_argument("--baseline", required=True)
        p.add_argument("--baseline-manifest-sha256", required=True)
        p.add_argument("--family-addon", required=True)
        p.add_argument("--family-completion-sha256", required=True)
        p.add_argument("--manifest-sha256", required=True)
        p.add_argument("--output", required=True)
        if operation in ("plan", "build"):
            p.add_argument("--replay-tools", required=True)
            p.add_argument("--manifest", required=True)
        if operation == "build":
            p.add_argument("--input-root", required=True)
            p.add_argument("--plan", required=True)
            p.add_argument("--plan-sha256", required=True)
        if operation == "restore":
            p.add_argument("--partition", required=True)
            p.add_argument("--partition-completion-sha256", required=True)
    args = parser.parse_args(argv)
    common = (args.baseline, args.family_addon)
    pins = (args.baseline_manifest_sha256, args.family_completion_sha256, args.manifest_sha256)
    try:
        if args.command == "plan":
            path = f.absolute(args.output)
            source_roots = [f.absolute(p) for p in (args.baseline, args.family_addon, args.replay_tools)]
            source_roots += [f.absolute(args.manifest).parent, Path(__file__).absolute().parent]
            require(all(not path.is_relative_to(p) for p in source_roots), "Plan output must be outside readonly sources")
            result = plan(*common, args.replay_tools, args.manifest, *pins)
            raw = f.encode(result)
            with private_umask(), Tree(path.parent) as parent:
                parent.write(path.name, raw)
            result = {"status": "METADATA_PLAN_ONLY", "payload_bytes_verified": False,
                      "plan": f.digest(raw), "counts": result["counts"]}
        elif args.command == "build":
            result = build(*common, args.replay_tools, args.input_root, args.manifest, *pins,
                           args.plan, args.plan_sha256, args.output)
            result = {k: v for k, v in result.items() if k != "payload"}
        else:
            result = restore(*common, args.partition, *pins, args.partition_completion_sha256, args.output)
            result = {k: v for k, v in result.items() if k != "restored_payload"}
        print(f.encode(result).decode(), end="")
        return 0
    except Exception as exc:
        print(f.encode({"status": "FAIL", "error": type(exc).__name__ + ": " + str(exc)}).decode(), file=sys.stderr, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
