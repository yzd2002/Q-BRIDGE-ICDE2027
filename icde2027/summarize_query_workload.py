"""Reconcile the complete registered execution grids without selecting outcomes.

This reads completed measurements only; it neither reruns timing nor changes
the registered support rule. Independent full proof replay is a separate audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-10, abs_tol=2e-9)


def range_summary(values):
    values = list(values)
    return {"median": median(values), "min": min(values), "max": max(values), "repeats": values}


def validate_method(summary, n, prefixes, common_seconds, stage):
    require(summary["queries"] == n, "Incorrect query count")
    rows = summary["per_query"]
    require([x["query"] for x in rows] == list(range(1, n + 1)), "Missing or duplicate query record")
    require(all(math.isfinite(x["query_seconds"]) and x["query_seconds"] >= 0 and
                math.isfinite(x["serialization_seconds"]) and x["serialization_seconds"] >= 0
                for x in rows), "Invalid measured duration")
    require(all(x["cumulative_seconds"] >= p for x, p in zip(rows, [0.] + [r["cumulative_seconds"] for r in rows[:-1]])),
            "Nonmonotone cumulative duration")
    expected_prefixes = {str(p): rows[p - 1]["cumulative_seconds"] for p in prefixes if p <= n}
    require(summary["prefixes"] == expected_prefixes, "Prefix total differs from measured query")
    serving = sum(x["query_seconds"] + x["serialization_seconds"] for x in rows)
    serving += sum(summary["input_stage_total_seconds"].values())
    if stage == "query_engine":
        serving += common_seconds
    require(close(serving, summary["cumulative_seconds"]), "Serving accounting does not sum")
    require(close(rows[-1]["cumulative_seconds"], serving), "Last query total differs")
    require(close(sum(x["serialization_seconds"] for x in rows), summary["serialization_total_seconds"]),
            "Serialization accounting does not sum")
    replay = summary["replay"]
    require(replay["proofs_valid"] + replay["proofs_missing"] == n, "Incomplete proof accounting")
    require(sum(replay["statuses"].values()) == sum(replay["routes"].values()) == n,
            "Incomplete result or route accounting")
    require(len(replay["lower_bounds"]) == n, "Incomplete lower-bound sequence")
    require(dict(Counter(x["status"] for x in rows)) == replay["statuses"], "Status disagreement")
    return serving


def summarize_run(directory, expected_stage):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    require(manifest["status"] == "complete", f"Run is not complete: {directory}")
    require(manifest["stage"] == expected_stage, "Wrong run stage")
    reg = read(directory / "protocol.json")
    require(digest(directory / "protocol.json") == manifest["protocol_sha256"], "Protocol hash mismatch")
    for name, expected in manifest["source_sha256"].items():
        require(digest(directory / "source_snapshot" / name) == expected, f"Source snapshot mismatch: {name}")
    cfg = reg[expected_stage]
    if expected_stage == "query_engine":
        expected = {f"d{d}_{s}_{q}_r{r}": (d, s, q, r)
                    for d, s, q, r in itertools.product(cfg["dimensions"], cfg["structures"], cfg["streams"], range(reg["repetitions"]))}
    else:
        expected = {f"{s}_r{r}": (s, r) for s, r in itertools.product(cfg["randomized_designs"], range(reg["repetitions"]))}
    completed = read(directory / "completed_conditions.json")
    require(len(completed) == len(expected) == manifest["completed_conditions"], "Grid size mismatch")
    require({x["name"] for x in completed} == set(expected), "Grid has missing or unexpected conditions")
    conditions = []; flat = []; proofs = Counter(); status_counts = Counter(); groups = defaultdict(list)
    for condition in completed:
        name = condition["name"]; loc = directory / name
        require(condition == read(loc / "condition.json"), f"Condition records differ: {name}")
        require(digest(loc / "queries.npy") == condition["queries_sha256"], f"Query file changed: {name}")
        require(sorted(condition["method_order"]) == sorted(cfg["methods"]), f"Method grid differs: {name}")
        if expected_stage == "query_engine":
            require((condition["d"], condition["structure"], condition["stream"], condition["rep"]) == expected[name],
                    "Dimension/structure/stream/repeat mismatch")
            n = cfg["max_queries_by_dimension"][str(condition["d"])]; warm = "highs_warm"
            group_key = (condition["d"], condition["structure"], condition["stream"])
        else:
            require((condition["design"], condition["rep"]) == expected[name], "Design/repeat mismatch")
            n = cfg["queries"]; warm = "materialized_highs_warm"; group_key = (condition["design"],)
        methods = {m: read(loc / f"{m}_summary.json") for m in cfg["methods"]}
        reference = methods[warm]; compact = {}
        for method, measured in methods.items():
            require(measured["method"] == method, "Method name mismatch")
            validate_method(measured, n, reg["query_prefixes"], condition.get("common_view_seconds", 0.), expected_stage)
            replay = measured["replay"]
            require((loc / replay["proof_file"]).stat().st_size == replay["compressed_bytes"], "Proof-file size mismatch")
            comparisons = condition["comparisons"][method]
            ratio = measured["cumulative_seconds"] / reference["cumulative_seconds"]
            disagreements = sum((a["status"] == "certified") != (b["status"] == "certified")
                                for a, b in zip(measured["per_query"], reference["per_query"]))
            delta = [a-b for a, b in zip(replay["lower_bounds"], reference["replay"]["lower_bounds"])
                     if a is not None and b is not None]
            require(close(ratio, comparisons["time_ratio_to_warm"]), "Paired ratio mismatch")
            require(disagreements == comparisons["decision_disagreements_to_warm"], "Decision comparison mismatch")
            for key, val in [("mean", sum(delta)/len(delta) if delta else None),
                             ("min", min(delta) if delta else None), ("max", max(delta) if delta else None)]:
                saved = comparisons[f"{key}_lower_minus_warm"]
                require((val is None and saved is None) or (val is not None and saved is not None and close(val, saved)),
                        f"Bound comparison mismatch: {key}")
            proofs["valid"] += replay["proofs_valid"]; proofs["missing"] += replay["proofs_missing"]
            status_counts.update(replay["statuses"])
            keep = {k: v for k, v in measured.items() if k not in {"per_query", "replay"}}
            keep["replay"] = {k: v for k, v in replay.items() if k != "lower_bounds"}
            keep["comparison_to_warm"] = comparisons
            keep["maximum_absolute_bound_difference"] = max(map(abs, delta), default=None)
            keep["mean_absolute_bound_difference"] = sum(map(abs, delta))/len(delta) if delta else None
            compact[method] = keep
            for prefix, seconds in measured["prefixes"].items():
                flat.append({"stage": expected_stage, "condition": name, "dimension": condition.get("d", cfg.get("actions")),
                             "structure": condition.get("structure", "source_to_answer"), "stream": condition.get("stream", condition.get("design")),
                             "repeat": condition["rep"], "method": method, "queries": int(prefix), "seconds": seconds,
                             "ratio_to_warm": seconds / reference["prefixes"][prefix]})
        item = {**condition, "methods": compact}; conditions.append(item); groups[group_key].append(item)
    group_summaries = []
    for key, values in groups.items():
        values.sort(key=lambda x: x["rep"])
        require([v["rep"] for v in values] == list(range(reg["repetitions"])), "Unpaired repeat group")
        result = {"condition_key": list(key), "methods": {}}
        for method in cfg["methods"]:
            byprefix = {}
            for prefix in values[0]["methods"][method]["prefixes"]:
                byprefix[prefix] = {"seconds": range_summary(v["methods"][method]["prefixes"][prefix] for v in values),
                                   "ratio_to_warm": range_summary(v["methods"][method]["prefixes"][prefix] / v["methods"][warm]["prefixes"][prefix] for v in values)}
            result["methods"][method] = {"prefixes": byprefix}
        group_summaries.append(result)
    return {"stage": expected_stage, "run_directory": str(directory.relative_to(ROOT)), "manifest": manifest,
            "protocol": reg, "conditions": conditions, "groups": group_summaries,
            "proof_totals": dict(proofs), "status_totals": dict(status_counts)}, flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, default=ROOT / "runs/icde2027/query_engine_v1")
    parser.add_argument("--end-to-end", type=Path, default=ROOT / "runs/icde2027/query_end_to_end_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/icde2027/query_workload_summary_v1")
    args = parser.parse_args()
    engine, engine_rows = summarize_run(args.engine, "query_engine")
    e2e, e2e_rows = summarize_run(args.end_to_end, "end_to_end")
    require(engine["manifest"]["protocol_sha256"] == e2e["manifest"]["protocol_sha256"], "Stages used different protocols")
    primary = engine["protocol"]["query_engine"]["primary_timing_cell"]
    paired = sorted([c for c in engine["conditions"] if (c["d"], c["structure"], c["stream"]) ==
                     (primary["dimension"], primary["structure"], primary["stream"])], key=lambda c: c["rep"])
    q = str(primary["queries"])
    ratios = [c["methods"]["prepared_cache"]["prefixes"][q] / c["methods"]["highs_warm"]["prefixes"][q] for c in paired]
    require(len(ratios) == engine["protocol"]["repetitions"], "Incomplete primary timing repeats")
    primary_result = {"cell": primary, "ratios_prepared_to_warm": ratios, "median_ratio": median(ratios),
                      "all_three_at_most_point8": all(r <= .8 for r in ratios), "p_value": None,
                      "inference_scope": "Three paired timing repetitions; no statistical significance claim."}
    payload = {"query_engine": engine, "end_to_end": e2e, "primary": primary_result,
               "scope": "Complete-grid measurement reconciliation; optimizer-independent full proof replay is reported separately."}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    rows = engine_rows + e2e_rows
    with (args.out / "all_prefixes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"primary": primary_result, "query_proofs": engine["proof_totals"], "end_to_end_proofs": e2e["proof_totals"]}, indent=2))


if __name__ == "__main__":
    main()
