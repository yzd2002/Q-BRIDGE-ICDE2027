"""Registered fixed-input study adapter; preflight never solves a real program.

Only the gated ``run`` command evaluates actual inputs. Synthetic tests exercise
the pure adapter/evaluation/summary functions separately. No legacy solver is
imported. See the study's RUNNER_INTERFACE.md for the gate and index schemas.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import sysconfig
import time
import traceback


VERSION = "partition-capacity-study-v6/1"
BASE = "runs/icde2027/analysis/policy_family_capacity_v6/run_01"
AUDIT = "runs/icde2027/analysis/policy_family_capacity_v6_result_audit"
STUDY = "runs/icde2027/analysis/partition_capacity_study_v6"
PLAN = STUDY + "/PLAN_02.md"
INTERFACE = STUDY + "/RUNNER_INTERFACE.md"
INPUT_MAP = "runs/icde2027/analysis/revision6_review/PARTITION_CAPACITY_INPUT_MAP.md"
RUN_OUTPUT = STUDY + "/run_02"
OLD_CONFIG = "runs/icde2027/analysis/policy_family_capacity_v6/config.json"
WORLD_PATTERN = "runs/icde2027/fresh_worlds_v1/worlds/world_{world_id:03d}.json"
METHODS = ("observation_consistency", "randomized_projected", "fusion_projected")
FAMILIES = ("metadata", "randomized_heuristic", "robust_design_selection")
KINDS = ("finite", "population")
SENSITIVITIES = ("true_boundary", "conservative")
BUDGETS = (100, 1000)
THRESHOLD = Fraction(1152921504606847, 1152921504606846976)
ABOVE = "EXISTS_ROBUST_POLICY_ABOVE_THRESHOLD"
ATMOST = "ROBUST_FAMILY_CAPACITY_AT_MOST_THRESHOLD"
UNKNOWN = "UNRESOLVED"
EXPECTED = {"programs": 3168, "occurrences": 5760, "contexts": 960,
            "worlds": 48, "scientific_sources": 214, "primary_groups": 24,
            "occurrences_per_primary_group": 240, "finite_programs": 2880,
            "population_programs": 288, "historical_certificates": 11640}
RESOURCES = {"cpu": 10, "threads": 1, "wall_seconds": 3600,
             "integer_string_digits": 100000, "deadline": "safe_program_boundary"}
NUMERICAL_CODE = (
    "icde2027/__init__.py", "icde2027/partition_capacity_v6/__init__.py",
    "icde2027/partition_capacity_v6/core.py", "icde2027/partition_capacity_v6/verify.py",
    "icde2027/partition_capacity_v6/study.py",
)
PINNED = {
    BASE + "/program_upper_certificates.jsonl": "7521725432ec7f52855f298ea4e6b013b79534f8a45bd021b17e6107daeb53b5",
    BASE + "/occurrence_bounds.jsonl": "b1cd4338d9baf2e2d3705dd0389d8bd09ec36130518be795ef18ac6bc395c725",
    BASE + "/normalized_contexts.jsonl": "9499e66d2af9fef4c7e2ae010e53eea25cd5b11c87bfda22f3fb2066efbe070f",
    BASE + "/manifest.json": "30f2d3c02002adc16d34a67f2b0b9608537c4a2ad5a1ed564cd13d262d9aef7b",
    BASE + "/output_index.json": "f67652014b574742b1770d7801459b258d67e6676d47e97aafb5acd9656897a5",
    BASE + "/input_inventory.json": "f3eabe46ae4da325b280d839be3ee9f7bd5986cd7471d41e04f80159dff24fdf",
    AUDIT + "/audit_run_03/report.json": "b4b2c44d94ec815eccefddc3b5729169906277b19d555b755812da18c7df6770",
    AUDIT + "/audit_run_03/bound_input_hashes.json": "79e6074a5a4fa0338314a6973672abb9290bf724218647641dca87608a333a51",
    AUDIT + "/audit_run_03/output_hashes.json": "b8a88d6d845864d8a38ce125d8511dbf53479025b0587c7f4f2a4ce7cdcba127",
    AUDIT + "/FINAL_INDEX.json": "dd4d9f6db423f0f2e63cb2a7dd0ce907eb73c8c1eb17a866eb6fe2360d7f2ecd",
    INPUT_MAP: "7f60e2bc7864c08ff29d186fd2173f7521ab2b87d5519892a279c2cde59ca46e",
}
_INTEGER = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class StudyError(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class DeadlineReached(StudyError):
    pass


def require(condition, message, details=None):
    if not condition:
        raise StudyError(message, details)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def raw_identity(path):
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            length += len(block)
    return {"sha256": digest.hexdigest(), "bytes": length}


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key", {"key": key})
        result[key] = value
    return result


def _json_float(text):
    value = float(text)
    require(math.isfinite(value), "nonfinite JSON number")
    return value


def strict_json(text):
    return json.loads(text, object_pairs_hook=_object_pairs, parse_float=_json_float,
                      parse_constant=lambda text: (_ for _ in ()).throw(StudyError("nonfinite JSON constant")))


def safe_path(root, relative, *, file=True):
    require(type(relative) is str and relative and "\\" not in relative,
            "path must be an explicit POSIX relative path")
    parts = relative.split("/")
    require(not relative.startswith("/") and all(p not in ("", ".", "..") for p in parts),
            "absolute, empty, dot, or traversal path refused", {"path": relative})
    current = root
    for part in parts:
        current = current / part
        require(not current.is_symlink(), "symlink refused", {"path": relative})
    require(current.absolute().is_relative_to(root), "path outside explicit root")
    if file:
        require(current.is_file(), "missing regular input file", {"path": relative})
    return current


def project_root(value):
    root = Path(value).absolute()
    require(root == root.resolve() and root.is_dir(), "project root must be a real absolute directory")
    return root


def historical_relative(root, absolute):
    path = Path(absolute)
    require(path.is_absolute() and path.is_relative_to(root), "historical path does not name this explicit root")
    return path.relative_to(root).as_posix()


def load_json(root, relative):
    return strict_json(safe_path(root, relative).read_text(encoding="utf-8"))


def read_jsonl(root, relative):
    with safe_path(root, relative).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), "blank JSONL member", {"path": relative, "line": number})
            value = strict_json(line)
            require(type(value) is dict, "JSONL member must be an object")
            yield value


def pair(value):
    require(type(value) is list and len(value) == 2 and
            all(type(v) is str and _INTEGER.fullmatch(v) for v in value), "invalid exact rational pair")
    numerator, denominator = map(int, value)
    require(denominator > 0, "nonpositive exact denominator")
    fraction = Fraction(numerator, denominator)
    require([str(fraction.numerator), str(fraction.denominator)] == value,
            "noncanonical rational pair")
    return fraction


def qp(value):
    components = (value.numerator, value.denominator)
    require(all(type(component) is int for component in components),
            "exact output components must already be integers")
    # Decimal(int) preserves the entire integer independently of context
    # precision. Fixed-point formatting changes representation only; the
    # separate 100000-digit input conversion guard remains unchanged.
    return [format(Decimal(component), "f") for component in components]


def qs(value):
    return f"{value.numerator}/{value.denominator}"


def probability(values, size):
    require(type(values) is list and len(values) == size, "probability dimension mismatch")
    result = [pair(value) for value in values]
    require(all(v >= 0 for v in result) and sum(result) == 1, "probability must sum exactly to one")
    return result


def adapt_program(record):
    """Decode only the unchanged exact program/family; no optimizer or receiver."""
    original = record["program"]
    require(canonical_hash(original) == record["program_id"], "original program canonical hash mismatch")
    require(set(original) == {"format", "lower", "upper", "rows", "rhs_lower", "rhs_upper", "metadata"},
            "unexpected original program schema")
    require(original["format"] == "fresh-mechanism-exact-population-v1", "unknown original format")
    size = record["size"]
    require(type(size) is list and len(size) == 2 and all(type(v) is int and v > 0 for v in size), "invalid K/A")
    k, a = size
    n = k * a
    lo, hi = [pair(v) for v in original["lower"]], [pair(v) for v in original["upper"]]
    require(len(lo) == len(hi) == n and all(0 <= l <= u <= 1 for l, u in zip(lo, hi, strict=True)), "invalid original box")
    require(len(original["rows"]) == len(original["rhs_lower"]) == len(original["rhs_upper"]), "row RHS dimension mismatch")
    rows, used = [], set()
    for sparse, low, high in zip(original["rows"], original["rhs_lower"], original["rhs_upper"], strict=True):
        dense, seen = [Fraction(0)] * n, set()
        require(type(sparse) is list, "sparse row must be a list")
        for item in sparse:
            require(type(item) is list and len(item) == 2 and type(item[0]) is int and 0 <= item[0] < n, "invalid sparse coordinate")
            i, value = item
            require(i not in seen, "duplicate sparse coordinate")
            seen.add(i)
            dense[i] = pair(value)
        support = {i for i, value in enumerate(dense) if value}
        require(all(v >= 0 for v in dense), "unsupported signed moment row")
        require(not (used & support), "unsupported overlapping moment rows")
        require(len({i // a for i in support}) <= 1, "unsupported mixed-context row")
        used |= support
        low, high = pair(low), pair(high)
        require(low <= high, "reversed row bounds")
        require(sum(t * l for t, l in zip(dense, lo, strict=True)) <= high and
                sum(t * u for t, u in zip(dense, hi, strict=True)) >= low, "empty supported program")
        rows.append({"coefficients": dense, "lower": low, "upper": high})
    weights = probability(record["weights"], k)
    require(type(record["reference"]) is list and len(record["reference"]) == k, "reference context dimension")
    reference = [probability(row, a) for row in record["reference"]]
    require(original["metadata"]["kind"] in KINDS, "scope must be metadata.kind")
    return {"lower": lo, "upper": hi, "rows": rows}, [a] * k, weights, reference


def request_digest(request):
    """Adapter digest; independent receiver must reconstruct it again at run time."""
    p, sizes, weights, reference = request
    canonical = {"schema_version": "partition-capacity-input/1", "program": {
        "lower": [qs(v) for v in p["lower"]], "upper": [qs(v) for v in p["upper"]],
        "rows": [{"coefficients": [qs(v) for v in row["coefficients"]],
                  "lower": qs(row["lower"]), "upper": qs(row["upper"])} for row in p["rows"]]},
        "context_sizes": sizes, "exact_weights": [qs(v) for v in weights],
        "exact_reference": [[qs(v) for v in row] for row in reference]}
    return canonical_hash(canonical)


def keyed(records, field):
    result = {}
    for record in records:
        key = record[field]
        require(type(key) is str and _SHA.fullmatch(key) and key not in result, "missing/duplicate/malformed identity", {"field": field})
        result[key] = record
    return result


def classify(lower, upper, threshold):
    require(lower <= upper, "old lower exceeds upper")
    return ABOVE if lower > threshold else ATMOST if upper <= threshold else UNKNOWN


def validate_joins(programs, contexts, occurrences, inventory, worlds, world_hashes):
    """Validate every original join. This performs no new capacity/truth evaluation."""
    inv_occ = keyed(inventory["occurrences"], "occurrence_id")
    certs = keyed(inventory["certificates"], "certificate_id")
    require(set(programs) == set(inventory["program_ids"]) and len(inventory["program_ids"]) == len(programs), "program inventory membership")
    require(set(occurrences) == set(inv_occ), "occurrence inventory membership")
    reverse = defaultdict(set)
    contexts_used = set()
    family_digests = {}
    context_slots = set()
    for context_id, context in contexts.items():
        require(context_id == canonical_hash({k: v for k, v in context.items() if k != "context_id"}), "context canonical identity")
        world_id, source_record = context["world_id"], context["source_record"]
        require(type(world_id) is int and world_id in worlds and type(source_record) is int, "context world/record identity")
        world = worlds[world_id]
        require(0 <= source_record < len(world["records"]), "source_record outside world")
        require((world_id, source_record) not in context_slots, "duplicate world configuration context")
        context_slots.add((world_id, source_record))
        spec = world["world"]["spec"]
        require(spec["id"] == world_id and spec["unconfounded"] is False and context["size"] == [spec["k"], spec["a"]], "world header/size mismatch")
        old_queries = world["records"][source_record]["queries"]
        require(context["original_queries_sha256"] == canonical_hash(old_queries), "original query identity mismatch")
        require(set(context["policies"]) == set(FAMILIES) == set(old_queries), "three original policy families required")
        k, a = context["size"]
        weights = probability(context["weights"], k)
        require(len(context["reference"]) == k, "context reference rows")
        reference = [probability(p, a) for p in context["reference"]]
        for family in FAMILIES:
            saved = context["policies"][family]
            require(len(saved["policy"]) == k, "old policy rows")
            policy = [probability(p, a) for p in saved["policy"]]
            objective = [weights[x] * (policy[x][i] - reference[x][i]) for x in range(k) for i in range(a)]
            require([pair(v) for v in saved["ideal_objective"]] == objective, "old normalized ideal objective mismatch")
            require(len(saved["stored_objective"]) == k * a, "old stored objective dimension")
            for value in saved["stored_objective"]:
                pair(value)
    for occurrence_id, occurrence in occurrences.items():
        original = inv_occ[occurrence_id]
        require(occurrence_id == canonical_hash({k: v for k, v in original.items() if k != "occurrence_id"}), "original occurrence canonical identity")
        require(all(k in occurrence and canonical_bytes(occurrence[k]) == canonical_bytes(v) for k, v in original.items()), "original occurrence fields altered")
        pid, cid = occurrence["program_id"], occurrence["context_id"]
        require(pid in programs and cid in contexts, "dangling program/context join")
        reverse[pid].add(occurrence_id)
        contexts_used.add(cid)
        p, context = programs[pid], contexts[cid]
        metadata = p["program"]["metadata"]
        require(all(occurrence[key] == context[key] for key in ("world_id", "source_record", "size")), "occurrence context labels")
        require(all(p[key] == context[key] for key in ("size", "weights", "reference")), "shared program normalized family mismatch")
        require(metadata["kind"] == occurrence["kind"] and metadata["method"] == occurrence["method"] and
                metadata["world_id"] == occurrence["world_id"] and metadata["radius"] == occurrence["radius"], "program metadata scope/labels/radius mismatch")
        require(occurrence["kind"] in KINDS and occurrence["method"] in METHODS and occurrence["sensitivity"] in SENSITIVITIES and occurrence["r_events_per_cell"] in BUDGETS,
                "undeclared occurrence group")
        require(type(occurrence["rep"]) is int and occurrence["rep"] in range(5), "undeclared repetition")
        record = worlds[occurrence["world_id"]]["records"][occurrence["source_record"]]
        require(all(canonical_bytes(record[key]) == canonical_bytes(occurrence[key]) for key in ("world_id", "rep", "sensitivity", "r_events_per_cell", "stratum", "rho_declared")), "world source-record labels mismatch")
        require(record["unconfounded"] is False, "unexpected unconfounded record")
        if occurrence["kind"] == "finite":
            require(metadata["source_record"] == occurrence["source_record"], "finite source_record mismatch")
        else:
            require(metadata["source_sha256"] == world_hashes[occurrence["world_id"]], "population world raw hash mismatch")
        require(pair(occurrence["threshold"]) == THRESHOLD, "saved threshold changed")
        require(set(occurrence["query_certificate_ids"]) == set(FAMILIES), "query certificate family members")
        lowers = occurrence["lower_records"]
        require(len(lowers) == 3 and {r["family"] for r in lowers} == set(FAMILIES), "exactly three occurrence-specific lower records required")
        for lower in lowers:
            family, certificate_id = lower["family"], lower["certificate_id"]
            require(certificate_id == occurrence["query_certificate_ids"][family] and lower["threshold"] == occurrence["threshold"], "old certificate/threshold join mismatch")
            require(certificate_id in certs and certs[certificate_id]["program_id"] == pid and
                    lower["archive_certificate_sha256"] == certs[certificate_id]["certificate_sha256"], "old certificate original identity mismatch")
            for field in ("ideal_objective_lower", "old_stored_objective_lower", "encoding_transfer_sanity_lower"):
                pair(lower[field])
        combined = max([Fraction(0), *[pair(r["ideal_objective_lower"]) for r in lowers]])
        require(pair(occurrence["lower"]) == combined, "occurrence-specific combined lower changed")
        for field in ("old_pool_upper", "augmented_upper"):
            require(occurrence[field] == p[field], "occurrence/program upper mismatch")
        require(occurrence["classification_old_pool"] == classify(combined, pair(p["old_pool_upper"]), THRESHOLD), "old pool class mismatch")
        require(occurrence["classification_augmented"] == classify(combined, pair(p["augmented_upper"]), THRESHOLD), "old augmented class mismatch")
    require(contexts_used == set(contexts), "unused or missing normalized context")
    for pid, p in programs.items():
        require(set(p["occurrence_ids"]) == reverse[pid] and len(p["occurrence_ids"]) == len(reverse[pid]), "bidirectional program occurrence membership")
        family_digests[pid] = request_digest(adapt_program(p))
    return family_digests


def production_census(programs, contexts, occurrences, inventory, worlds):
    """Immutable production cardinalities: config cannot shrink this study."""
    kinds = Counter(p["program"]["metadata"]["kind"] for p in programs.values())
    groups = Counter((o["kind"], o["method"], o["sensitivity"], o["r_events_per_cell"]) for o in occurrences.values())
    require(len(programs) == 3168 and len(contexts) == 960 and len(occurrences) == 5760,
            "registered program/context/occurrence census mismatch")
    require(set(worlds) == set(range(48)) and all(len(w["records"]) == 20 for w in worlds.values()), "registered 48-world/20-configuration membership")
    require(kinds == {"finite": 2880, "population": 288}, "scope program census")
    require(all(len(p["occurrence_ids"]) == (1 if p["program"]["metadata"]["kind"] == "finite" else 10) for p in programs.values()), "registered occurrence reuse count")
    require(set(groups) == set(itertools.product(KINDS, METHODS, SENSITIVITIES, BUDGETS)) and set(groups.values()) == {240}, "all 24 primary groups must have 240 occurrences")
    require(Counter(tuple(p["size"]) for p in programs.values()) == {(2, 4): 1056, (8, 8): 1056, (16, 16): 1056}, "registered size census")
    require(len(inventory["source_files_sha256"]) == 214 and len(inventory["certificates"]) == 11640, "registered source/certificate census")
    return {"counts": EXPECTED.copy(), "primary_groups": [{"group": list(g), "occurrences": groups[g]} for g in sorted(groups)],
            "world_record_counts": {str(i): len(w["records"]) for i, w in sorted(worlds.items())}}


def runtime_identity(root):
    """Actual process runtime; historical identities remain separate provenance."""
    executable = Path(sys.executable).resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    origins = {executable}
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if origin:
            path = Path(origin).resolve()
            if path.is_relative_to(stdlib) and "site-packages" not in path.parts and path.is_file():
                origins.add(path)
    files = {}
    for path in sorted(origins):
        require(path.is_relative_to(root), "actual runtime source outside project environment", {"path": str(path)})
        files[path.relative_to(root).as_posix()] = raw_identity(path)
    return {"python_version": sys.version, "executable": executable.relative_to(root).as_posix(),
            "implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag,
            "files": files, "integer_string_digits": sys.get_int_max_str_digits()}


def check_files(root, files):
    for relative, expected in sorted(files.items()):
        current = raw_identity(safe_path(root, relative))
        require(current["sha256"] == expected["sha256"] and current["bytes"] == expected["bytes"], "source bytes changed", {"path": relative, "expected": expected, "observed": current})


def load_fixed_inputs(root):
    """Read/hash/schema/join only; neither solve nor verify nor truth contrast."""
    files = {}
    def bind(relative, digest=None, role="direct_input"):
        actual = raw_identity(safe_path(root, relative))
        require(digest is None or actual["sha256"] == digest, "pinned input hash mismatch", {"path": relative})
        if relative in files:
            require(all(files[relative][k] == actual[k] for k in actual), "input changed during binding")
            files[relative]["roles"] = sorted(set(files[relative]["roles"]) | {role})
        else:
            files[relative] = {**actual, "roles": [role]}
    for relative, digest in PINNED.items():
        bind(relative, digest)
    manifest = load_json(root, BASE + "/manifest.json")
    inventory = load_json(root, BASE + "/input_inventory.json")
    output_index = load_json(root, BASE + "/output_index.json")
    audit = load_json(root, AUDIT + "/audit_run_03/report.json")
    audit_inputs = load_json(root, AUDIT + "/audit_run_03/bound_input_hashes.json")
    audit_outputs = load_json(root, AUDIT + "/audit_run_03/output_hashes.json")
    audit_final = load_json(root, AUDIT + "/FINAL_INDEX.json")
    require(manifest["status"] == "complete" and manifest["complete"] is True and manifest["optimizer_calls"] == 0 and manifest["all_sources_rechecked_at_end"] is True, "old run not complete")
    require(manifest["programs"] == 3168 and manifest["occurrences"] == 5760 and manifest["contexts"] == 960 and manifest["source_closure_files"] == 214, "old manifest counts")
    require(manifest["input_inventory_sha256"] == canonical_hash(inventory), "inventory canonical identity (not raw SHA) mismatch")
    require(manifest["output_index_sha256"] == files[BASE + "/output_index.json"]["sha256"], "old output index raw identity")
    require(audit["status"] == "PASS" and audit["complete"] is True and audit["independent_review"] is True and
            historical_relative(root, audit["scope"]) == BASE, "final independent audit status/scope")
    require(audit_final["status"] == "PASS", "audit final index status")
    bind(AUDIT + "/audit.py", audit["audit_code_sha256"], "historical_audit_code")
    for relative in (AUDIT + "/audit_run_03/report.json", AUDIT + "/audit_run_03/bound_input_hashes.json", AUDIT + "/audit_run_03/output_hashes.json", AUDIT + "/audit.py"):
        entry = audit_final["artifacts"][relative[len(AUDIT) + 1:]]
        require(all(entry[k] == files[relative][k] for k in ("sha256", "bytes")), "audit final artifact binding")
    require(audit_outputs["report.json"]["sha256"] == files[AUDIT + "/audit_run_03/report.json"]["sha256"] and
            audit_outputs["bound_input_hashes.json"]["sha256"] == files[AUDIT + "/audit_run_03/bound_input_hashes.json"]["sha256"], "audit output ledger binding")
    for relative, digest in inventory["source_files_sha256"].items():
        bind(relative, digest, "original_scientific_source")
    require(canonical_hash(inventory["source_files_sha256"]) == inventory["source_closure_sha256"] == manifest["payload"]["source_closure_sha256"], "scientific source closure identity")
    for relative, digest in inventory["direct_inputs_sha256"].items():
        bind(relative, digest, "historical_direct_input")
    for relative in (OLD_CONFIG, BASE + "/source_snapshot_index.json", BASE + "/execution_gate.json",
                     AUDIT + "/audit_run_03/preexisting_radius_repairs_bound.json"):
        if relative.startswith(BASE + "/"):
            expected = output_index[relative[len(BASE) + 1:]]["sha256"]
        elif relative == OLD_CONFIG:
            expected = manifest["payload"]["reviewed_files_sha256"][relative]
        else:
            expected = audit_outputs["preexisting_radius_repairs_bound.json"]["sha256"]
        bind(relative, expected, "historical_provenance")
    old_config = load_json(root, OLD_CONFIG)
    require(old_config["direct_inputs_sha256"] == inventory["direct_inputs_sha256"], "historical config direct input identity")
    old_gate = historical_relative(root, manifest["gate_path"])
    bind(old_gate, manifest["gate_sha256"], "historical_provenance")
    require(load_json(root, old_gate) == load_json(root, BASE + "/execution_gate.json"), "old gate copies must match as objects, not necessarily raw bytes")
    for relative, entry in list(files.items()):
        if relative in (INPUT_MAP, AUDIT + "/FINAL_INDEX.json") or relative.startswith(AUDIT + "/"):
            continue
        original_absolute = str(root / relative)
        require(original_absolute in audit_inputs and all(audit_inputs[original_absolute][k] == entry[k] for k in ("sha256", "bytes")), "consumed historical input absent/different in independent ledger", {"path": relative})
        if relative.startswith(BASE + "/") and relative[len(BASE) + 1:] in output_index:
            require(all(output_index[relative[len(BASE) + 1:]][k] == entry[k] for k in ("sha256", "bytes")), "old output raw index mismatch")
    programs = keyed(read_jsonl(root, BASE + "/program_upper_certificates.jsonl"), "program_id")
    contexts = keyed(read_jsonl(root, BASE + "/normalized_contexts.jsonl"), "context_id")
    occurrences = keyed(read_jsonl(root, BASE + "/occurrence_bounds.jsonl"), "occurrence_id")
    world_ids = {p["program"]["metadata"]["world_id"] for p in programs.values()}
    worlds, world_hashes = {}, {}
    for world_id in sorted(world_ids):
        require(type(world_id) is int and world_id in range(48), "undeclared target world")
        relative = WORLD_PATTERN.format(world_id=world_id)
        require(relative in inventory["source_files_sha256"], "world outside original scientific source inventory")
        worlds[world_id] = load_json(root, relative)
        world_hashes[world_id] = files[relative]["sha256"]
    requests = validate_joins(programs, contexts, occurrences, inventory, worlds, world_hashes)
    census = production_census(programs, contexts, occurrences, inventory, worlds)
    for relative in (*NUMERICAL_CODE, PLAN, INPUT_MAP, INTERFACE):
        bind(relative, PINNED.get(relative), "runtime_code" if relative in NUMERICAL_CODE else "protocol")
    runtime = runtime_identity(root)
    for relative, entry in runtime["files"].items():
        bind(relative, entry["sha256"], "actual_runtime")
    check_files(root, files)
    index = {"schema_version": "partition-capacity-input-index/1", "project_root": str(root), "files": files,
             "scientific_source_count": 214, "census": census, "program_request_sha256": requests,
             "program_ids": sorted(programs), "occurrence_ids": sorted(occurrences),
             "context_ids": sorted(contexts), "world_ids": sorted(worlds),
             "historical_runtime_provenance": manifest["payload"]["environment"],
             "actual_runtime": runtime, "prior_interpretation_limits": audit["interpretation_limits"],
             "new_numerical_evaluation_performed": False}
    bundle = {"programs": programs, "contexts": contexts, "occurrences": occurrences,
              "worlds": worlds, "world_hashes": world_hashes, "request_digests": requests}
    return bundle, index


def truth_diagnostic(record, request, certificate, world):
    """Run-only diagnostic using binary64 target_mean, never decimal-string truth."""
    p, sizes, weights, reference = request
    raw = world["world"]["target_mean"]
    require(type(raw) is list and len(raw) == len(sizes), "truth context dimension")
    truth = []
    for values, size in zip(raw, sizes, strict=True):
        require(type(values) is list and len(values) == size, "truth action dimension")
        for value in values:
            require(type(value) in (int, float) and math.isfinite(float(value)), "truth must be a finite JSON numeric value")
            exact = Fraction(float(value))
            require(0 <= exact <= 1, "target mean outside [0,1]")
            truth.append(exact)
    violations = []
    for i, (value, lo, hi) in enumerate(zip(truth, p["lower"], p["upper"], strict=True)):
        if not lo <= value <= hi:
            violations.append({"kind": "box", "coordinate": i, "value": qp(value), "lower": qp(lo), "upper": qp(hi)})
    for j, row in enumerate(p["rows"]):
        value = sum(t * v for t, v in zip(row["coefficients"], truth, strict=True))
        if not row["lower"] <= value <= row["upper"]:
            violations.append({"kind": "moment", "row_index": j, "value": qp(value), "lower": qp(row["lower"]), "upper": qp(row["upper"])})
    contrast, offset = Fraction(0), 0
    for x, size in enumerate(sizes):
        policy = [Fraction(v) for v in certificate["policy"][x]]
        contrast += weights[x] * sum((q - r) * truth[offset + i] for i, (q, r) in enumerate(zip(policy, reference[x], strict=True)))
        offset += size
    capacity = Fraction(certificate["capacity"])
    result = {"source_field": "world.target_mean", "encoding": "Fraction(float(JSON_numeric_value))",
              "target_mean_exact_binary64": [qp(v) for v in truth], "truth_in_C": not violations,
              "membership_violations": violations, "new_policy_truth_contrast": qp(contrast),
              "new_policy_lower_le_truth": capacity <= contrast, "lower_minus_truth": qp(capacity - contrast)}
    require(not (record["program"]["metadata"]["kind"] == "population" and violations), "population truth is outside original C", result)
    require(bool(violations) or capacity <= contrast, "feasible truth violates certified policy lower", result)
    return result


def evaluate_program(record, world, expected_request_digest):
    """Numerical boundary; call only after gate, or with synthetic test records."""
    from .core import solve
    from .verify import verify_certificate
    request = adapt_program(record)
    digest = request_digest(request)
    require(digest == expected_request_digest, "external original request digest changed")
    certificate = solve(*request)
    try:
        receipt = verify_certificate(*request, certificate, expected_input_sha256=expected_request_digest)
    except Exception as exc:
        raise StudyError("independent receiver rejected certificate", {"program_id": record["program_id"], "certificate": certificate, "error": repr(exc)}) from exc
    require(receipt.get("verified") is True and receipt.get("scope") == "original-program-optimality-only" and receipt.get("input_sha256") == digest, "independent receipt binding")
    require(all(receipt.get(field) == certificate["capacity"] for field in
                ("capacity", "computed_original_dual_lower_bound", "computed_common_point_upper_bound")),
            "independent receipt exact capacity equality")
    capacity = Fraction(certificate["capacity"])
    require(capacity <= pair(record["old_pool_upper"]) and capacity <= pair(record["augmented_upper"]), "new exact capacity exceeds an old upper")
    truth = truth_diagnostic(record, request, certificate, world)
    return {"program_id": record["program_id"], "original_program_record": record,
            "kernel_input_sha256": digest, "certificate": certificate, "receiver_receipt": receipt,
            "truth": truth, "policy_selection_semantics": "full_simplex_policy_selected_on_fixed_inference_view; not_original_design_policy"}


def occurrence_result(original, result):
    capacity = Fraction(result["certificate"]["capacity"])
    require(pair(original["lower"]) <= capacity <= pair(original["old_pool_upper"]) and capacity <= pair(original["augmented_upper"]), "per-occurrence old bounds do not enclose new capacity", {"occurrence_id": original["occurrence_id"]})
    threshold = pair(original["threshold"])
    new_class = ABOVE if capacity > threshold else ATMOST
    return {"occurrence_id": original["occurrence_id"], "program_id": original["program_id"],
            "context_id": original["context_id"], "original_occurrence": original,
            "capacity": qp(capacity), "threshold": original["threshold"],
            "classification_new": new_class, "zero_structure_new": "R_EQUALS_ZERO" if capacity == 0 else "R_STRICTLY_POSITIVE",
            "transition_old_pool": [original["classification_old_pool"], new_class],
            "transition_augmented": [original["classification_augmented"], new_class],
            "capacity_minus_own_three_policy_lower": qp(capacity - pair(original["lower"])),
            "truth": result["truth"], "policy_certificate_program_id": original["program_id"],
            "false_positive_assertion": capacity > threshold and pair(result["truth"]["new_policy_truth_contrast"]) <= threshold,
            "policy_selection_semantics": result["policy_selection_semantics"]}


def source_comparisons(rows):
    groups = defaultdict(dict)
    for row in rows:
        old = row["original_occurrence"]
        key = (old["world_id"], old["source_record"], old["kind"])
        require(old["method"] not in groups[key], "duplicate paired source member")
        groups[key][old["method"]] = row
    result = []
    for key, members in sorted(groups.items()):
        require(set(members) == set(METHODS), "missing paired source method")
        for left, right in itertools.combinations(METHODS, 2):
            a, b = members[left], members[right]
            original = a["original_occurrence"]
            difference = pair(a["capacity"]) - pair(b["capacity"])
            result.append({"world_id": key[0], "source_record": key[1], "kind": key[2],
                "sensitivity": original["sensitivity"], "r_events_per_cell": original["r_events_per_cell"],
                "size": original["size"], "stratum": original["stratum"],
                "left_method": left, "right_method": right,
                "left_occurrence_id": a["occurrence_id"], "right_occurrence_id": b["occurrence_id"],
                "new_capacity_difference": qp(difference), "sign": (difference > 0) - (difference < 0),
                "own_old_lower_difference": qp(pair(original["lower"]) - pair(b["original_occurrence"]["lower"])),
                "old_pool_upper_difference": qp(pair(original["old_pool_upper"]) - pair(b["original_occurrence"]["old_pool_upper"])),
                "augmented_upper_difference": qp(pair(original["augmented_upper"]) - pair(b["original_occurrence"]["augmented_upper"])),
                "new_policy_truth_difference": qp(pair(a["truth"]["new_policy_truth_contrast"]) - pair(b["truth"]["new_policy_truth_contrast"])),
                "new_class_pair": [a["classification_new"], b["classification_new"]]})
    return result


def summarize(rows, comparisons):
    """Producer summary arithmetic, intentionally separate for a later auditor."""
    def declared_counts(values, declared):
        counts = Counter(values)
        require(set(counts) <= set(declared), "unexpected summary classification")
        return {value: counts[value] for value in declared}

    transitions = [old + " -> " + new for old in (ABOVE, ATMOST, UNKNOWN) for new in (ABOVE, ATMOST)]
    tables = {}
    dimensions = {"primary": ("kind", "method", "sensitivity", "r_events_per_cell"),
                  "by_size": ("kind", "method", "sensitivity", "r_events_per_cell", "size"),
                  "by_stratum": ("kind", "method", "sensitivity", "r_events_per_cell", "stratum"),
                  "by_world": ("world_id", "kind", "method", "sensitivity", "r_events_per_cell")}
    for name, fields in dimensions.items():
        groups = defaultdict(list)
        for row in rows:
            old = row["original_occurrence"]
            key = tuple(tuple(old[f]) if isinstance(old[f], list) else old[f] for f in fields)
            groups[key].append(row)
        output = []
        for key, members in sorted(groups.items()):
            capacities = [pair(v["capacity"]) for v in members]
            output.append({"group": dict(zip(fields, key, strict=True)), "occurrences": len(members),
                "distinct_programs": len({r["program_id"] for r in members}),
                "distinct_worlds": len({r["original_occurrence"]["world_id"] for r in members}),
                "classifications": declared_counts((r["classification_new"] for r in members), (ABOVE, ATMOST, UNKNOWN)),
                "zero_structure": declared_counts((r["zero_structure_new"] for r in members), ("R_EQUALS_ZERO", "R_STRICTLY_POSITIVE")),
                "transitions_old_pool": declared_counts((" -> ".join(r["transition_old_pool"]) for r in members), transitions),
                "transitions_augmented": declared_counts((" -> ".join(r["transition_augmented"]) for r in members), transitions),
                "truth_in_C": sum(r["truth"]["truth_in_C"] for r in members),
                "truth_not_in_C": sum(not r["truth"]["truth_in_C"] for r in members),
                "lower_exceeds_truth": sum(not r["truth"]["new_policy_lower_le_truth"] for r in members),
                "false_positive_assertions": sum(r["false_positive_assertion"] for r in members),
                "capacity_sum": qp(sum(capacities)), "capacity_mean": qp(sum(capacities) / len(members)),
                "capacity_min": qp(min(capacities)), "capacity_max": qp(max(capacities))})
        tables[name] = output
    pairs = defaultdict(list)
    for row in comparisons:
        key = tuple(row[f] for f in ("kind", "sensitivity", "r_events_per_cell", "left_method", "right_method"))
        pairs[key].append(row)
    tables["paired_sources"] = [{"group": list(key), "pairs": len(values),
        "positive": sum(r["sign"] > 0 for r in values), "zero": sum(r["sign"] == 0 for r in values),
        "negative": sum(r["sign"] < 0 for r in values),
        "difference_sum": qp(sum((pair(r["new_capacity_difference"]) for r in values), Fraction(0)))}
        for key, values in sorted(pairs.items())]
    primary_exploration = [r for r in rows if (r["original_occurrence"]["kind"], r["original_occurrence"]["method"], r["original_occurrence"]["sensitivity"], r["original_occurrence"]["r_events_per_cell"]) == ("finite", "fusion_projected", "conservative", 100)]
    return {"schema_version": VERSION, "occurrences": len(rows), "programs": len({r["program_id"] for r in rows}),
            "tables": tables, "exploratory_unresolved_to_above_count": sum(r["transition_augmented"] == [UNKNOWN, ABOVE] for r in primary_exploration),
            "exploratory_registered_cell_occurrences": len(primary_exploration),
            "exploratory_previously_unresolved_denominator": sum(r["original_occurrence"]["classification_augmented"] == UNKNOWN for r in primary_exploration),
            "finite_coverage_failure_ids": [r["occurrence_id"] for r in rows if r["original_occurrence"]["kind"] == "finite" and not r["truth"]["truth_in_C"]],
            "lower_exceeds_truth_ids": [r["occurrence_id"] for r in rows if not r["truth"]["new_policy_lower_le_truth"]],
            "false_positive_assertion_ids": [r["occurrence_id"] for r in rows if r["false_positive_assertion"]],
            "interpretation": "retrospective fixed dependent inputs; no p-values, independent-sample claim, speed claim, or real-effect upper bound"}


def json_document(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    path.write_bytes(json_document(value))


def configure_cli_runtime():
    # Explicit process-local CLI setting, never an import-time core mutation.
    sys.set_int_max_str_digits(RESOURCES["integer_string_digits"])


def config_proposal(index_relative, index_sha, index):
    return {"schema_version": "partition-capacity-study-config/1",
            "plan": {"path": PLAN, "sha256": index["files"][PLAN]["sha256"]},
            "input_map": {"path": INPUT_MAP, "sha256": PINNED[INPUT_MAP]},
            "input_index": {"path": index_relative, "sha256": index_sha},
            "expected": EXPECTED, "resources": RESOURCES, "output": RUN_OUTPUT}


def preflight(root, output):
    require(output.startswith(STUDY + "/") and output != RUN_OUTPUT, "preflight output must be a fresh study subdirectory, not run_01")
    out = safe_path(root, output, file=False)
    require(not out.exists(), "preflight output already exists")
    out.mkdir(parents=True)
    try:
        _, index = load_fixed_inputs(root)
        write_json(out / "INPUT_INDEX.json", index)
        write_json(out / "CENSUS.json", index["census"])
        proposed = config_proposal(output + "/INPUT_INDEX.json", raw_identity(out / "INPUT_INDEX.json")["sha256"], index)
        write_json(out / "CONFIG_PROPOSAL.json", proposed)
        return index
    except Exception as exc:
        write_json(out / "FAILURE.json", {"status": "failed_preflight", "new_numerical_evaluation_performed": False,
                   "error": repr(exc), "details": getattr(exc, "details", {}), "traceback": traceback.format_exc()})
        raise


def validate_gate(root, config_relative, gate_relative, output):
    controls = {}

    def read_control(relative, expected_sha=None):
        # Parse exactly the bytes whose identity is retained. Never parse and
        # later hash separate reads of a mutable authority file.
        payload = safe_path(root, relative).read_bytes()
        identity = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                    "roles": ["execution_control"]}
        require(expected_sha is None or identity["sha256"] == expected_sha,
                "bound control file changed", {"path": relative})
        require(relative not in controls or controls[relative] == identity,
                "control changed within validation", {"path": relative})
        controls[relative] = identity
        return payload

    gate = strict_json(read_control(gate_relative))
    config = strict_json(read_control(config_relative))
    require(set(config) == {"schema_version", "plan", "input_map", "input_index", "expected", "resources", "output"}, "unexpected production config schema")
    require(config["schema_version"] == "partition-capacity-study-config/1" and config["expected"] == EXPECTED and config["resources"] == RESOURCES and config["output"] == output == RUN_OUTPUT, "fixed production protocol cannot be overridden")
    require(gate.get("schema_version") == "partition-capacity-execution-gate/1" and gate.get("decision") == "GO" and gate.get("scope") == "registered-fixed-data-partition-capacity-v6", "missing or wrong execution gate decision/scope")
    require(gate.get("output") == output and gate.get("resources") == RESOURCES,
            "gate output/resource mismatch")
    require(gate.get("config") == {"path": config_relative, "sha256": controls[config_relative]["sha256"]}, "gate config raw binding")
    require(gate.get("plan") == config["plan"] and config["plan"]["path"] == PLAN, "gate plan binding")
    require(gate.get("input_index") == config["input_index"] and config["input_map"] == {"path": INPUT_MAP, "sha256": PINNED[INPUT_MAP]}, "gate/index/map binding")
    for field in ("plan", "input_map", "input_index"):
        expected = config[field]
        payload = read_control(expected["path"], expected["sha256"])
        if field == "input_index":
            index = strict_json(payload)
    require(index["schema_version"] == "partition-capacity-input-index/1" and index["project_root"] == str(root) and index["census"]["counts"] == EXPECTED, "preflight input index scope/census")
    require(gate.get("numerical_sources_sha256") == {p: index["files"][p]["sha256"] for p in NUMERICAL_CODE}, "all numerical source identities must be bound")
    require(gate.get("runtime_identity_sha256") == canonical_hash(index["actual_runtime"]), "actual runtime identity binding")
    reviews = gate.get("reviews", [])
    require(type(reviews) is list and {r.get("role") for r in reviews} == {"implementation", "input_and_methodology"} and len(reviews) == 2, "two independent review scopes required")
    for review in reviews:
        require(review.get("decision") == "GO" and review.get("independent") is True and review.get("scope") == gate["scope"], "review decision/independence/scope")
        declaration = strict_json(read_control(review["path"], review["sha256"]))
        require(declaration.get("decision") == "GO" and declaration.get("independent_review") is True and declaration.get("role") == review["role"] and declaration.get("scope") == gate["scope"], "bound review declaration")
        require(declaration.get("config_sha256") == gate["config"]["sha256"] and declaration.get("input_index_sha256") == config["input_index"]["sha256"] and declaration.get("numerical_sources_sha256") == gate["numerical_sources_sha256"], "review must bind this exact config/input/code")
    check_files(root, index["files"])
    check_files(root, controls)
    require(os.sched_getaffinity(0) == {10} and sys.get_int_max_str_digits() == 100000, "run requires registered CPU/digit setting")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(variable) == "1", "run requires one numerical thread", {"variable": variable})
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run requires no GPU visibility")
    return config, gate, index, controls


def build_output_index(out):
    members = {}
    for path in sorted(out.rglob("*")):
        require(not path.is_symlink(), "output symlink refused")
        if path.is_file() and path != out / "OUTPUT_INDEX.json":
            members[path.relative_to(out).as_posix()] = raw_identity(path)
    return {"schema_version": "partition-capacity-output-index/1", "files": members}


def output_index(out):
    write_json(out / "OUTPUT_INDEX.json", build_output_index(out))


def check_deadline(started, stage):
    elapsed = time.monotonic() - started
    if elapsed >= 3600:
        raise DeadlineReached("fixed whole-run allowance exhausted", {"stage": stage, "elapsed_wall_seconds": elapsed})
    return elapsed


def run(root, config_relative, gate_relative, output):
    require(output == RUN_OUTPUT, "only the registered production output is allowed")
    out = safe_path(root, output, file=False)
    require(not out.exists(), "existing output refused; no resume or overwrite")
    out.mkdir(parents=True)
    started = time.monotonic()
    manifest = {"schema_version": VERSION, "status": "preparing", "complete": False,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "resources": RESOURCES,
                "thread_settings": {key: os.environ.get(key) for key in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
                "workspace_state": "non-git workspace; all consumed code/input bytes are individually hash-bound",
                "new_numerical_evaluation_performed": False, "programs_completed": 0,
                "occurrences_completed": 0, "all_sources_rechecked_at_end": False}
    write_json(out / "manifest.json", manifest)
    index = None
    bound_files = None
    try:
        config, gate, index, controls = validate_gate(root, config_relative, gate_relative, output)
        files = {p: dict(v) for p, v in index["files"].items()}
        for relative, identity in controls.items():
            if relative in files:
                require(all(files[relative][k] == identity[k] for k in ("sha256", "bytes")),
                        "control/input identity conflict", {"path": relative})
                files[relative]["roles"] = sorted(set(files[relative]["roles"]) | set(identity["roles"]))
            else:
                files[relative] = dict(identity)
        bound_files = files
        bundle, fresh_index = load_fixed_inputs(root)
        require(index == fresh_index, "preflight index differs from current complete input binding")
        check_files(root, controls)
        write_json(out / "input_index.json", index)
        write_json(out / "execution_gate.json", gate)
        write_json(out / "config.json", config)
        write_json(out / "validated_control_identities.json", controls)
        for relative, identity in sorted(files.items()):
            source = safe_path(root, relative)
            destination = out / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            require(raw_identity(destination) == {k: identity[k] for k in ("sha256", "bytes")}, "source snapshot hash mismatch")
        write_json(out / "source_snapshot_index.json", files)
        # A control/source may change after its own snapshot was copied. Check
        # the retained original identities again before any numerical call.
        check_files(root, files)
        manifest["status"] = "running"
        write_json(out / "manifest.json", manifest)
        all_rows = []
        with (out / "program_certificates.jsonl").open("w", encoding="utf-8") as certificates, (out / "occurrences.jsonl").open("w", encoding="utf-8") as occurrences, (out / "progress.jsonl").open("w", encoding="utf-8") as progress:
            for pid in sorted(bundle["programs"]):
                if time.monotonic() - started >= 3600:
                    raise DeadlineReached("fixed one-hour allowance exhausted at safe program boundary", {"next_program_id": pid})
                record = bundle["programs"][pid]
                world_id = record["program"]["metadata"]["world_id"]
                try:
                    if not manifest["new_numerical_evaluation_performed"]:
                        manifest["new_numerical_evaluation_performed"] = True
                        write_json(out / "manifest.json", manifest)
                    result = evaluate_program(record, bundle["worlds"][world_id], bundle["request_digests"][pid])
                    result["source_identity"] = {"programs_file": BASE + "/program_upper_certificates.jsonl", "programs_file_sha256": files[BASE + "/program_upper_certificates.jsonl"]["sha256"], "world_file": WORLD_PATTERN.format(world_id=world_id), "world_file_sha256": bundle["world_hashes"][world_id]}
                    records = [occurrence_result(bundle["occurrences"][oid], result) for oid in sorted(record["occurrence_ids"])]
                except Exception as exc:
                    raise StudyError("program evaluation failed", {"program_id": pid, "error": repr(exc), "details": getattr(exc, "details", {})}) from exc
                certificates.write(canonical_bytes(result).decode("ascii") + "\n")
                for row in records:
                    occurrences.write(canonical_bytes(row).decode("ascii") + "\n")
                all_rows.extend(records)
                manifest["programs_completed"] += 1
                manifest["occurrences_completed"] += len(records)
                progress.write(json.dumps({"program_id": pid, "programs_completed": manifest["programs_completed"], "occurrences_completed": manifest["occurrences_completed"]}) + "\n")
                certificates.flush(); occurrences.flush(); progress.flush()
                if manifest["programs_completed"] % 25 == 0:
                    write_json(out / "manifest.json", manifest)
            if time.monotonic() - started >= 3600:
                raise DeadlineReached("fixed allowance exhausted after final program; summaries incomplete")
        require(manifest["programs_completed"] == 3168 and manifest["occurrences_completed"] == 5760, "incomplete registered study")
        comparisons = source_comparisons(all_rows)
        with (out / "source_comparisons.jsonl").open("w", encoding="utf-8") as stream:
            for row in comparisons:
                stream.write(canonical_bytes(row).decode("ascii") + "\n")
        write_json(out / "summary.json", summarize(all_rows, comparisons))
        # Hash the entire output tree while the on-disk manifest is incomplete.
        # The large index construction is part of the same one-hour allowance.
        final_index = build_output_index(out)
        integrity_started = datetime.now(timezone.utc).isoformat()
        check_files(root, files)
        for relative, identity in files.items():
            require(raw_identity(out / "source_snapshot" / relative) == {k: identity[k] for k in ("sha256", "bytes")}, "final source snapshot mismatch")
        integrity_completed = datetime.now(timezone.utc).isoformat()
        elapsed = check_deadline(started, "after_summary_sources_snapshots_and_output_index")
        completed = {**manifest, "status": "complete", "complete": True,
                     "all_sources_rechecked_at_end": True,
                     "final_integrity_check": {
                         "started_at_utc": integrity_started,
                         "completed_at_utc": integrity_completed,
                         "scope": "all_initially_bound_sources_and_copied_snapshots_after_output_index_construction"},
                     "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                     "elapsed_wall_seconds": elapsed}
        completed_bytes = json_document(completed)
        final_index["files"]["manifest.json"] = {
            "sha256": hashlib.sha256(completed_bytes).hexdigest(), "bytes": len(completed_bytes)}
        write_json(out / "OUTPUT_INDEX.json", final_index)
        # Index construction may be lengthy. A revoked control must not become
        # complete merely because its last check preceded that postprocessing.
        check_files(root, controls)
        check_deadline(started, "before_complete_manifest")
        (out / "manifest.json").write_bytes(completed_bytes)
        # If even the small terminal write crosses the cap, the exception path
        # replaces it with incomplete_deadline and refreshes the failure index.
        check_files(root, controls)
        check_deadline(started, "after_terminal_write")
        manifest = completed
        return manifest
    except Exception as exc:
        manifest.update(status="incomplete_deadline" if isinstance(exc, DeadlineReached) else "failed", complete=False,
                        failed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_seconds=time.monotonic() - started)
        write_json(out / "FAILURE.json", {"error": repr(exc), "details": getattr(exc, "details", {}), "traceback": traceback.format_exc()})
        if bound_files is not None:
            try:
                check_files(root, bound_files)
                manifest["all_sources_rechecked_at_end"] = True
            except Exception as source_exc:
                write_json(out / "SOURCE_RECHECK_FAILURE.json", {"error": repr(source_exc), "details": getattr(source_exc, "details", {})})
        write_json(out / "manifest.json", manifest)
        output_index(out)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out", required=True, help="explicit fresh path relative to project root")
    parser.add_argument("--config")
    parser.add_argument("--gate")
    args = parser.parse_args(argv)
    configure_cli_runtime()
    root = project_root(args.project_root)
    if args.mode == "preflight":
        require(args.config is None and args.gate is None, "preflight does not consume execution permission")
        result = preflight(root, args.out)
        print(json.dumps({"status": "PREFLIGHT_ONLY", "counts": result["census"]["counts"], "new_numerical_evaluation_performed": False}))
    else:
        require(args.config and args.gate, "run requires explicit reviewed config and gate")
        result = run(root, args.config, args.gate, args.out)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
