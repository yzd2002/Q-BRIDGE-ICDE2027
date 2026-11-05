"""Render the complete registered query study, never partial timing results.

Run after ``summarize_query_workload.py`` has reconciled both completed stages.
The figures use observed cumulative prefix times; they do not extrapolate a
workload.  All uncertainty marks are ranges of three paired timing repetitions,
not confidence intervals.  This script checks input completeness and arithmetic;
it does not issue the independent proof-audit verdict.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "icde2027/paper"
DEFAULT_SUMMARY = ROOT / "runs/icde2027/query_workload_summary_v1/summary.json"
ENGINE_METHODS = (
    "highs_cold", "highs_warm", "disjoint_or_global", "components",
    "box_shortcut", "dual_cache", "prepared_cache",
)
E2E_METHODS = (
    "rebuild_highs_cold", "materialized_highs_cold", "materialized_highs_warm",
    "materialized_components", "materialized_prepared_cache",
)
LABELS = {
    "highs_cold": "Cold LP", "highs_warm": "Hot LP",
    "disjoint_or_global": "Disjoint/global", "components": "Components",
    "box_shortcut": "Box shortcut", "dual_cache": "Dual reuse",
    "prepared_cache": "Prepared reuse", "rebuild_highs_cold": "Rebuild Cold LP",
    "materialized_highs_cold": "Cold LP", "materialized_highs_warm": "Hot LP",
    "materialized_components": "Components", "materialized_prepared_cache": "Prepared reuse",
}
STRUCTURES = ("disjoint", "mixed", "connected")
STREAMS = ("nearby_nonproportional", "unrelated")
BLUE = "#0072B2"
GREEN = "#008B72"
ORANGE = "#D58400"
PURPLE = "#7655A5"
INK = "#25313B"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-10, abs_tol=2e-9)


def finite(value, *, positive=False):
    return isinstance(value, (int, float)) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def range_summary(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(v) for v in values), "Invalid numeric repetition")
    return {"median": median(values), "min": min(values), "max": max(values), "repeats": values}


def validate_range(stored, values, description):
    expected = range_summary(values)
    require(len(stored["repeats"]) == len(values), f"Wrong repetition count: {description}")
    for key in ("median", "min", "max"):
        require(close(stored[key], expected[key]), f"Incorrect {key}: {description}")
    require(all(close(a, b) for a, b in zip(stored["repeats"], values)), f"Unpaired repeats: {description}")


def stage_key(condition, stage):
    return ((condition["d"], condition["structure"], condition["stream"])
            if stage == "query_engine" else (condition["design"],))


def validate_summary(summary_path, csv_path):
    """Reject an incomplete, changed, or internally inconsistent formal grid."""
    require(summary_path.is_file(), f"Completed summary is not available: {summary_path}")
    require(csv_path.is_file(), f"Complete prefix table is not available: {csv_path}")
    payload = read(summary_path)
    require({"query_engine", "end_to_end", "primary"} <= payload.keys(), "Missing formal summary stage")
    reg = payload["query_engine"]["protocol"]
    require(payload["end_to_end"]["protocol"] == reg, "Stage protocols differ")
    require(reg["repetitions"] == 3, "This registered figure requires three paired repeats")
    cfg = reg["query_engine"]
    require(cfg["dimensions"] == [64, 512, 2048], "Unexpected dimension grid")
    require(cfg["structures"] == list(STRUCTURES) and cfg["streams"] == list(STREAMS), "Unexpected workload grid")
    require(cfg["methods"] == list(ENGINE_METHODS), "Unexpected seven-plan registry")
    require(cfg["max_queries_by_dimension"] == {"64": 1000, "512": 1000, "2048": 100}, "Unregistered query counts")
    require(reg["query_prefixes"] == [1, 10, 100, 1000], "Unregistered timing prefixes")
    require(reg["end_to_end"]["methods"] == list(E2E_METHODS), "Unexpected source-to-answer plans")
    require(reg["end_to_end"]["randomized_designs"] == ["uniform", "skew"], "Unexpected randomization grid")
    require(reg["end_to_end"]["queries"] == 100, "Unregistered source-to-answer query count")
    require(cfg["primary_timing_cell"] == {"dimension": 512, "structure": "connected",
                                          "stream": "nearby_nonproportional", "queries": 1000},
            "Registered primary cell changed")
    expected_rows = {}
    for stage, methods, warm, expected_count in (
        ("query_engine", ENGINE_METHODS, "highs_warm", 54),
        ("end_to_end", E2E_METHODS, "materialized_highs_warm", 6),
    ):
        data = payload[stage]
        manifest = data["manifest"]
        require(data["stage"] == manifest["stage"] == stage, "Incorrect stage identity")
        require(manifest["status"] == "complete", f"Formal stage is not complete: {stage}")
        directory = ROOT / data["run_directory"]
        saved_manifest = read(directory / "manifest.json")
        require(saved_manifest == manifest, f"Source manifest changed: {stage}")
        require(digest(directory / "protocol.json") == manifest["protocol_sha256"], "Source protocol hash differs")
        require(read(directory / "protocol.json") == reg, "Source protocol content differs")
        conditions = data["conditions"]
        require(len(conditions) == manifest["completed_conditions"] == expected_count,
                f"Incomplete formal condition grid: {stage}")
        grid = (list(itertools.product(cfg["dimensions"], STRUCTURES, STREAMS)) if stage == "query_engine"
                else [(x,) for x in reg[stage]["randomized_designs"]])
        expected = {(key, rep) for key in grid for rep in range(3)}
        observed = {(stage_key(c, stage), c["rep"]) for c in conditions}
        require(observed == expected and len(observed) == len(conditions), f"Unpaired or duplicate condition: {stage}")
        require(len({c["name"] for c in conditions}) == len(conditions), "Duplicate condition name")
        proofs = Counter()
        statuses = Counter()
        for condition in conditions:
            n = cfg["max_queries_by_dimension"][str(condition["d"])] if stage == "query_engine" else reg[stage]["queries"]
            prefixes = [q for q in reg["query_prefixes"] if q <= n]
            measured = condition["methods"]
            require(set(measured) == set(methods), f"Missing plan in {condition['name']}")
            for method in methods:
                result = measured[method]
                require(result["method"] == method and result["queries"] == n, "Plan/query identity mismatch")
                require(set(result["prefixes"]) == {str(q) for q in prefixes}, "Missing measured prefix")
                times = [result["prefixes"][str(q)] for q in prefixes]
                require(all(finite(v, positive=True) for v in times) and times == sorted(times), "Invalid cumulative timing")
                require(close(times[-1], result["cumulative_seconds"]), "Final prefix does not equal total")
                replay = result["replay"]
                require(replay["proofs_valid"] + replay["proofs_missing"] == n, "Incomplete proof count")
                require(sum(replay["routes"].values()) == sum(replay["statuses"].values()) == n, "Incomplete response routes")
                for key in ("replay_checker_initialization_seconds", "replay_decode_seconds", "replay_verify_seconds", "gzip_write_seconds"):
                    require(finite(replay[key]), f"Invalid recorded replay cost: {key}")
                for key in ("maximum_bank_entries", "final_prepared_terms_python_bytes", "process_peak_rss_kib_not_method_isolated"):
                    require(finite(result[key]), f"Invalid recorded memory field: {key}")
                comp = result["comparison_to_warm"]
                require(0 <= comp["decision_disagreements_to_warm"] <= n, "Invalid decision-disagreement count")
                require(close(comp["time_ratio_to_warm"], result["cumulative_seconds"] / measured[warm]["cumulative_seconds"]),
                        "Final paired timing ratio differs")
                proofs.update(valid=replay["proofs_valid"], missing=replay["proofs_missing"])
                statuses.update(replay["statuses"])
                for q, seconds in zip(prefixes, times):
                    identity = (stage, condition["name"], method, q)
                    expected_rows[identity] = {
                        "dimension": condition.get("d", reg["end_to_end"]["actions"]),
                        "structure": condition.get("structure", "source_to_answer"),
                        "stream": condition.get("stream", condition.get("design")),
                        "repeat": condition["rep"], "seconds": seconds,
                        "ratio_to_warm": seconds / measured[warm]["prefixes"][str(q)],
                    }
        require(dict(proofs) == data["proof_totals"], "Stage proof totals differ")
        require(dict(statuses) == data["status_totals"], "Stage status totals differ")
        groups = data["groups"]
        require(len(groups) == len(grid) and {tuple(g["condition_key"]) for g in groups} == set(grid),
                "Incomplete aggregate grid")
        for group in groups:
            key = tuple(group["condition_key"])
            paired = sorted((c for c in conditions if stage_key(c, stage) == key), key=lambda c: c["rep"])
            require(set(group["methods"]) == set(methods), "Incomplete aggregate plan grid")
            for method in methods:
                prefix_keys = set(paired[0]["methods"][method]["prefixes"])
                require(set(group["methods"][method]["prefixes"]) == prefix_keys, "Incomplete aggregate prefix grid")
                for prefix, result in group["methods"][method]["prefixes"].items():
                    values = [c["methods"][method]["prefixes"][prefix] for c in paired]
                    ratios = [v / c["methods"][warm]["prefixes"][prefix] for v, c in zip(values, paired)]
                    validate_range(result["seconds"], values, f"{stage}/{key}/{method}/{prefix}")
                    validate_range(result["ratio_to_warm"], ratios, f"paired ratio {stage}/{key}/{method}/{prefix}")
    require(payload["query_engine"]["manifest"]["protocol_sha256"] == payload["end_to_end"]["manifest"]["protocol_sha256"],
            "Formal stages used different registered protocols")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(expected_rows) == 1476, "Expected all 1,476 measured prefix rows")
    seen = set()
    for row in rows:
        key = (row["stage"], row["condition"], row["method"], int(row["queries"]))
        require(key in expected_rows and key not in seen, "Unexpected or duplicate CSV prefix")
        seen.add(key)
        expected = expected_rows[key]
        for field in ("dimension", "repeat"):
            require(int(row[field]) == expected[field], f"CSV mismatch: {field}")
        for field in ("structure", "stream"):
            require(row[field] == expected[field], f"CSV mismatch: {field}")
        for field in ("seconds", "ratio_to_warm"):
            require(close(row[field], expected[field]), f"CSV mismatch: {field}")
    primary = payload["primary"]
    require(primary["cell"] == cfg["primary_timing_cell"], "Primary cell identity differs")
    pair = paired_conditions(payload, "query_engine", (512, "connected", "nearby_nonproportional"))
    ratios = [c["methods"]["prepared_cache"]["prefixes"]["1000"] / c["methods"]["highs_warm"]["prefixes"]["1000"] for c in pair]
    require(len(primary["ratios_prepared_to_warm"]) == 3 and
            all(close(a, b) for a, b in zip(primary["ratios_prepared_to_warm"], ratios)), "Primary paired ratios differ")
    require(close(primary["median_ratio"], median(ratios)), "Incorrect primary median")
    require(primary["all_three_at_most_point8"] == all(r <= .8 for r in ratios), "Incorrect registered timing verdict")
    require(primary["p_value"] is None, "No registered significance test for three timings")
    return payload


def paired_conditions(payload, stage, key):
    return sorted((c for c in payload[stage]["conditions"] if stage_key(c, stage) == key), key=lambda c: c["rep"])


def group_index(payload, stage):
    return {tuple(g["condition_key"]): g for g in payload[stage]["groups"]}


def cost_summary(conditions, method, warm):
    """Add measured producer/consumer costs within each repeat, then summarize."""
    def costs(result):
        replay = result["replay"]
        initial = replay["replay_checker_initialization_seconds"]
        decode = replay["replay_decode_seconds"]
        verify = replay["replay_verify_seconds"]
        consumer = initial + decode + verify
        producer = result["cumulative_seconds"]
        return {"producer_seconds": producer, "checker_initialization_seconds": initial,
                "json_decode_seconds": decode, "verify_seconds": verify,
                "consumer_seconds": consumer, "joint_seconds": producer + consumer,
                "gzip_write_seconds_separate": replay["gzip_write_seconds"]}
    records = [costs(c["methods"][method]) for c in conditions]
    references = [costs(c["methods"][warm]) for c in conditions]
    result = {key: range_summary(r[key] for r in records) for key in records[0]}
    result["producer_ratio_to_hot"] = range_summary(r["producer_seconds"] / w["producer_seconds"] for r, w in zip(records, references))
    result["joint_ratio_to_hot"] = range_summary(r["joint_seconds"] / w["joint_seconds"] for r, w in zip(records, references))
    return result


def route_and_bound_summary(conditions, method):
    routes = Counter()
    statuses = Counter()
    bounds_min = []
    bounds_max = []
    absolute = []
    proofs = Counter()
    disagreements = 0
    for condition in conditions:
        result = condition["methods"][method]
        replay = result["replay"]
        routes.update(replay["routes"])
        statuses.update(replay["statuses"])
        proofs.update(valid=replay["proofs_valid"], missing=replay["proofs_missing"])
        comp = result["comparison_to_warm"]
        disagreements += comp["decision_disagreements_to_warm"]
        if comp["min_lower_minus_warm"] is not None:
            bounds_min.append(comp["min_lower_minus_warm"])
            bounds_max.append(comp["max_lower_minus_warm"])
            absolute.append(result["maximum_absolute_bound_difference"])
    return {"responses": sum(routes.values()), "routes": dict(routes), "statuses": dict(statuses),
            "saved_dual_hits": routes["dual_reuse"], "non_hit_responses": sum(routes.values()) - routes["dual_reuse"],
            "solver_fallbacks": routes["solver"], "box_shortcuts": routes["box_shortcut"],
            "materialization_refusals": routes["materialization_refusal"],
            "decision_disagreements_to_hot": disagreements, "recorded_replay": dict(proofs),
            "lower_minus_hot_min": min(bounds_min, default=None), "lower_minus_hot_max": max(bounds_max, default=None),
            "maximum_absolute_bound_difference": max(absolute, default=None)}


def make_facts(payload, summary_path, csv_path):
    engine = group_index(payload, "query_engine")
    e2e = group_index(payload, "end_to_end")
    cells = []
    for d, structure, stream in itertools.product((64, 512, 2048), STRUCTURES, STREAMS):
        ratios = engine[d, structure, stream]["methods"]["prepared_cache"]["prefixes"]["100"]["ratio_to_warm"]
        cells.append({"dimension": d, "structure": structure, "stream": stream,
                      "queries": 100, "prepared_to_hot": ratios})
    whole_table = []
    for structure, stream in itertools.product(STRUCTURES, STREAMS):
        group = engine[512, structure, stream]
        whole_table.append({"dimension": 512, "structure": structure, "stream": stream, "queries": 1000,
                            "methods": {m: group["methods"][m]["prefixes"]["1000"] for m in ENGINE_METHODS}})
    costs = []
    for stage, key, name, methods, warm in (
        ("query_engine", (512, "connected", "nearby_nonproportional"), "Primary engine", ENGINE_METHODS, "highs_warm"),
        ("end_to_end", ("uniform",), "Uniform source-to-answer", E2E_METHODS, "materialized_highs_warm"),
        ("end_to_end", ("skew",), "Skew source-to-answer", E2E_METHODS, "materialized_highs_warm"),
    ):
        paired = paired_conditions(payload, stage, key)
        costs.append({"stage": stage, "condition_key": list(key), "label": name,
                      "queries": paired[0]["methods"][warm]["queries"],
                      "methods": {m: cost_summary(paired, m, warm) for m in methods}})
    prepared_records = [c["methods"]["prepared_cache"] for c in payload["query_engine"]["conditions"]]
    prepared_records += [c["methods"]["materialized_prepared_cache"] for c in payload["end_to_end"]["conditions"]]
    proof_totals = Counter()
    for stage in ("query_engine", "end_to_end"):
        proof_totals.update(payload[stage]["proof_totals"])
    return {
        "sources": [{"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "sha256": digest(p)}
                    for p in (summary_path, csv_path)],
        "protocol_sha256": payload["query_engine"]["manifest"]["protocol_sha256"],
        "complete_conditions": {s: len(payload[s]["conditions"]) for s in ("query_engine", "end_to_end")},
        "prefix_rows": 1476, "timing_uncertainty": "Three paired repetitions; median and minimum/maximum, not confidence intervals.",
        "primary": payload["primary"],
        "cumulative_panels": {"engine_primary": engine[512, "connected", "nearby_nonproportional"],
                              "end_to_end_uniform": e2e["uniform",], "end_to_end_skew": e2e["skew",]},
        "boundary_cells_q100": cells,
        "boundary_no_slower_cells": sum(c["prepared_to_hot"]["median"] <= 1 for c in cells),
        "boundary_worst_cell": max(cells, key=lambda c: c["prepared_to_hot"]["median"]),
        "d512_q1000_all_seven_plans": whole_table,
        "producer_consumer_costs": costs,
        "consumer_accounting": "Checker initialization + measured JSON decoding + proof verification; added within repeat before ratios or medians. Gzip writing is separate. Gzip reading/decompression and network transfer are not included in this measured joint total.",
        "prepared_engine_routes_and_bounds": route_and_bound_summary(payload["query_engine"]["conditions"], "prepared_cache"),
        "prepared_end_to_end_routes_and_bounds": route_and_bound_summary(payload["end_to_end"]["conditions"], "materialized_prepared_cache"),
        "primary_prepared_routes_and_bounds": route_and_bound_summary(paired_conditions(payload, "query_engine", (512, "connected", "nearby_nonproportional")), "prepared_cache"),
        "recorded_replay_proof_totals": dict(proof_totals),
        "independent_audit_verdict": "Not supplied by this plotting/measurement-reconciliation script.",
        "prepared_memory": {
            "maximum_bank_entries": max(r["maximum_bank_entries"] for r in prepared_records),
            "final_prepared_terms_python_bytes": range_summary(r["final_prepared_terms_python_bytes"] for r in prepared_records),
            "process_peak_rss_kib_not_method_isolated": range_summary(r["process_peak_rss_kib_not_method_isolated"] for r in prepared_records),
            "scope": "Prepared-term Python object payload at run end, excluding shared view endpoints; not a native-memory or isolated peak-RSS measurement.",
        },
    }


def plot_assets(facts, directory):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .6, "grid.linewidth": .4,
                         "figure.facecolor": "white", "axes.facecolor": "white", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    styles = {
        "highs_cold": (BLUE, "o", "-"), "highs_warm": (PURPLE, "s", "--"),
        "dual_cache": (ORANGE, "^", "-."), "prepared_cache": (GREEN, "D", "-"),
        "components": ("#7A7D7F", "v", "--"), "rebuild_highs_cold": (INK, "P", ":"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65))
    fig.subplots_adjust(left=.080, right=.99, bottom=.225, top=.70, wspace=.29)
    panels = [("engine_primary", "(a) Primary: 512 cells\nConnected, nearby", ("highs_cold", "highs_warm", "dual_cache", "prepared_cache")),
              ("end_to_end_uniform", "(b) Source-to-answer\nUniform randomization", E2E_METHODS),
              ("end_to_end_skew", "(c) Source-to-answer\nSkewed randomization", E2E_METHODS)]
    all_min = []
    all_max = []
    for ax, (key, title, methods) in zip(axes, panels):
        group = facts["cumulative_panels"][key]
        prefixes = sorted(int(q) for q in group["methods"][methods[0]]["prefixes"])
        for method in methods:
            base = method.removeprefix("materialized_")
            color, marker, linestyle = styles[base]
            records = [group["methods"][method]["prefixes"][str(q)]["seconds"] for q in prefixes]
            mid = [r["median"] for r in records]
            lo = [r["min"] for r in records]
            hi = [r["max"] for r in records]
            all_min.extend(lo)
            all_max.extend(hi)
            ax.fill_between(prefixes, lo, hi, color=color, alpha=.13, linewidth=0)
            ax.plot(prefixes, mid, color=color, marker=marker, linestyle=linestyle, lw=1.35,
                    ms=4.2, markeredgewidth=.55, markeredgecolor="white", label=LABELS[method])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(prefixes, [str(q) for q in prefixes])
        ax.set_xlim(prefixes[0] / 1.25, prefixes[-1] * 1.25)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Queries, Q (log scale)", fontsize=8)
        ax.set_title(title, fontsize=8.5, weight="bold", pad=7, linespacing=1.28)
        ax.tick_params(axis="both", labelsize=8, pad=2)
        ax.grid(axis="y", which="major", color="#D9E0E5", alpha=.85)
        ax.set_axisbelow(True)
    limits = (min(all_min) / 1.4, max(all_max) * 1.5)
    for ax in axes:
        ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Producer time (s; log)", fontsize=8.5)
    handles = [Line2D([0], [0], color=color, marker=marker, linestyle=ls, lw=1.35, ms=4,
                      label=LABELS[m]) for m, (color, marker, ls) in styles.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.015), ncol=3,
               frameon=False, fontsize=8, handlelength=2.1, columnspacing=1.5, labelspacing=.5)
    fig.text(.54, .025, "3 paired repeats; all producer first-use costs included; receiver costs reported separately", ha="center", fontsize=7.6)
    for ext in ("pdf", "png"):
        fig.savefig(directory / f"query_cumulative_v2.{ext}", dpi=220, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)

    values = np.array([c["prepared_to_hot"]["median"] for c in facts["boundary_cells_q100"]]).reshape(3, 6)
    require(np.all(values > 0), "The ratio color scale requires positive measured times")
    logvalues = np.log2(values)
    span = max(1., float(np.max(np.abs(logvalues))))
    cmap = LinearSegmentedColormap.from_list("prepared_hot_ratio", [GREEN, "#F8FAF9", ORANGE])
    fig, ax = plt.subplots(figsize=(7.1, 2.30))
    fig.subplots_adjust(left=.105, right=.81, bottom=.25, top=.77)
    im = ax.imshow(logvalues, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-span, vcenter=0., vmax=span))
    ax.set_yticks(range(3), ["64 cells", "512 cells", "2,048 cells"], fontsize=8.5)
    ax.set_xticks(range(6), ["Nearby", "Unrelated"] * 3, fontsize=8)
    ax.tick_params(length=0, pad=6)
    for pos, label in zip((.5, 2.5, 4.5), ("Disjoint", "Mixed", "Connected")):
        ax.text(pos, -.88, label, ha="center", va="center", fontsize=9, weight="bold")
    for i in range(3):
        for j in range(6):
            value = values[i, j]
            color = "white" if abs(logvalues[i, j]) > .69 * span else INK
            ax.text(j, i, f"{value:.2f}x", ha="center", va="center", fontsize=9.5, color=color)
    for x in np.arange(-.5, 6, 1):
        ax.axvline(x, color="white", lw=1.2)
    for y in np.arange(-.5, 3, 1):
        ax.axhline(y, color="white", lw=1.2)
    for x in (1.5, 3.5):
        ax.axvline(x, color="white", lw=3)
    worst = np.unravel_index(np.argmax(values), values.shape)
    ax.add_patch(Rectangle((worst[1] - .47, worst[0] - .47), .94, .94, fill=False, edgecolor=INK, lw=1.0))
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([.85, .25, .023, .52])
    bar = fig.colorbar(im, cax=cax)
    ticks = [-span, 0., span] if span < 2.5 else [-span, -span / 2, 0., span / 2, span]
    bar.set_ticks(ticks, labels=[f"{2**t:.2g}x" for t in ticks])
    bar.ax.tick_params(labelsize=8, length=2, pad=3)
    bar.set_label("Prepared / Hot (log ratio)", fontsize=8, labelpad=7)
    bar.outline.set_linewidth(.5)
    fig.text(.105, .94, "All 18 workloads at Q = 100", fontsize=9, weight="bold")
    fig.text(.105, .045, "Green < 1: faster     White = 1: equal     Orange > 1: slower; outlined cell is worst", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(directory / f"query_boundary_v2.{ext}", dpi=220, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


def seconds(value):
    return f"{value:.3g}"


def tex_tables(facts):
    lines = [r"\begingroup\fontsize{8}{9.4}\selectfont\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}", r"\toprule",
             "Structure / stream & " + " & ".join(LABELS[m] for m in ENGINE_METHODS) + r"\\", r"\midrule"]
    for row in facts["d512_q1000_all_seven_plans"]:
        label = row["structure"].capitalize() + " / " + ("nearby" if row["stream"] == STREAMS[0] else "unrelated")
        values = [row["methods"][m]["seconds"]["median"] for m in ENGINE_METHODS]
        cells = [r"\textbf{" + seconds(v) + "}" if v == min(values) else seconds(v) for v in values]
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\endgroup"])
    full_table = "\n".join(lines) + "\n"
    lines = [r"\begingroup\fontsize{8}{9.4}\selectfont\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}", r"\toprule",
             r"Workload & Plan & Producer & Checker init. & JSON decode & Verify & Joint & Joint / Hot\\", r"\midrule"]
    for group in facts["producer_consumer_costs"]:
        engine = group["stage"] == "query_engine"
        hot = "highs_warm" if engine else "materialized_highs_warm"
        prepared = "prepared_cache" if engine else "materialized_prepared_cache"
        workload = ("Primary, 1,000 queries" if engine else group["condition_key"][0].capitalize() + ", 100 queries")
        for index, method in enumerate((hot, prepared)):
            costs = group["methods"][method]
            cells = [seconds(costs[k]["median"]) for k in ("producer_seconds", "checker_initialization_seconds",
                                                         "json_decode_seconds", "verify_seconds", "joint_seconds")]
            cells.append(f"{costs['joint_ratio_to_hot']['median']:.3f}")
            lines.append((workload if index == 0 else "") + " & " + LABELS[method] + " & " + " & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\endgroup"])
    return full_table, "\n".join(lines) + "\n"


def performance_text(facts):
    primary = facts["primary"]
    primary_cost = facts["producer_consumer_costs"][0]["methods"]
    prep = primary_cost["prepared_cache"]
    hot = primary_cost["highs_warm"]
    ratio_text = ", ".join(f"{v:.4f}" for v in primary["ratios_prepared_to_warm"])
    verdict = ("All three meet the registered $\\leq0.8$ timing condition." if primary["all_three_at_most_point8"]
               else "The registered timing target fails: not all three ratios are $\\leq0.8$.")
    worst = facts["boundary_worst_cell"]
    worst_stream = "nearby" if worst["stream"] == STREAMS[0] else "unrelated"
    source_sentences = []
    for group in facts["producer_consumer_costs"][1:]:
        methods = group["methods"]
        p = methods["materialized_prepared_cache"]
        h = methods["materialized_highs_warm"]
        r = methods["rebuild_highs_cold"]
        design = "uniform" if group["condition_key"][0] == "uniform" else "skewed"
        source_sentences.append(
            f"With {design} randomization, 100 measured queries take {seconds(p['producer_seconds']['median'])}\\,s "
            f"for materialized Prepared reuse, {seconds(h['producer_seconds']['median'])}\\,s for materialized Hot LP "
            f"and {seconds(r['producer_seconds']['median'])}\\,s for rebuilding; the median paired Prepared/Hot ratio is "
            f"{p['producer_ratio_to_hot']['median']:.3f}.")
    engine_diag = facts["prepared_engine_routes_and_bounds"]
    e2e_diag = facts["prepared_end_to_end_routes_and_bounds"]
    hits = engine_diag["saved_dual_hits"] + e2e_diag["saved_dual_hits"]
    total = engine_diag["responses"] + e2e_diag["responses"]
    misses = total - hits
    fallback = engine_diag["solver_fallbacks"] + e2e_diag["solver_fallbacks"]
    box = engine_diag["box_shortcuts"] + e2e_diag["box_shortcuts"]
    refusal = engine_diag["materialization_refusals"] + e2e_diag["materialization_refusals"]
    disagreements = engine_diag["decision_disagreements_to_hot"] + e2e_diag["decision_disagreements_to_hot"]
    low_values = [d["lower_minus_hot_min"] for d in (engine_diag, e2e_diag) if d["lower_minus_hot_min"] is not None]
    high_values = [d["lower_minus_hot_max"] for d in (engine_diag, e2e_diag) if d["lower_minus_hot_max"] is not None]
    def math_number(value):
        rendered = f"{value:.4g}"
        if "e" not in rendered:
            return rendered
        mantissa, exponent = rendered.split("e")
        return mantissa + r"\times10^{" + str(int(exponent)) + "}"

    bound_text = (f"Across finite matched answers, Prepared-minus-Hot lower bounds range from ${math_number(min(low_values))}$ "
                  f"to ${math_number(max(high_values))}$ in success-probability units." if low_values
                  else "No finite paired bounds were available for comparing lower-bound tightness.")
    proofs = facts["recorded_replay_proof_totals"]
    memory = facts["prepared_memory"]
    joint_ratios = [g["methods"]["prepared_cache" if g["stage"] == "query_engine" else "materialized_prepared_cache"]
                    ["joint_ratio_to_hot"]["median"] for g in facts["producer_consumer_costs"]]
    return rf"""% Generated from both complete registered stages by make_query_assets.py.
% Figure data retain every repeat; independent audit findings are not inferred here.
\begin{{figure*}}[t]
\centering\includegraphics[width=\textwidth]{{figures/query_cumulative_v2.pdf}}
\caption{{Cumulative producer time, including first-use costs. Lines show medians over three paired repetitions; shaded bands show the minimum and maximum, not confidence intervals. Both axes use logarithmic scales and time is in seconds. In (b)--(c), all plans except Rebuild Cold LP materialize once. Only measured prefixes are plotted: the source-to-answer runs stop at 100 queries. Consumer proof checking is reported separately in Table~\ref{{tab:query-consumer}}.}}
\label{{fig:query-cumulative}}
\end{{figure*}}

\textbf{{Registered timing result.}} Both execution stages completed: 54 controlled conditions with seven plans and six source-to-answer conditions with five plans. For the registered 512-cell connected nearby-query workload, Prepared reuse takes a median {seconds(prep['producer_seconds']['median'])}\,s for 1,000 queries, against {seconds(hot['producer_seconds']['median'])}\,s for Hot LP. The three paired time ratios are {ratio_text} (median {primary['median_ratio']:.4f}). {verdict} Figure~\ref{{fig:query-cumulative}} retains the first query, initialization, all later misses, and response serialization. Three timings support a descriptive comparison, not a significance test.

\textbf{{Workload boundary.}} At $Q=100$, Prepared reuse is no slower than Hot LP in {facts['boundary_no_slower_cells']} of 18 cells. The largest median paired ratio is {worst['prepared_to_hot']['median']:.3f}, for {worst['dimension']:,} cells with {worst['structure']} structure and {worst_stream} objectives. The artifact retains the full grid, all seven-plan timings and every repeat; the 2,048-cell grid stops at its registered 100-query cap. These are execution fixtures, not coverage or policy-utility experiments.

\textbf{{Source-to-answer cost.}} {' '.join(source_sentences)} These totals include reading, validating and aggregating five million encoded events, source intervals, feasible-view construction, execution and response serialization. The comparison with materialized Hot LP isolates the remaining execution benefit after shared input work is retained.

\textbf{{Reuse and bound tightness.}} Prepared reuse serves {hits:,}/{total:,} queries from saved duals; the remaining {misses:,} comprise {fallback:,} solver fallbacks, {box:,} box shortcuts and {refusal:,} refusals. There are {disagreements:,} certification disagreements with Hot LP. {bound_text} A hit proves enough for the threshold, not necessarily the tightest bound. Across all plans, recorded replay verifies {proofs['valid']:,} proofs with {proofs['missing']:,} no-bound responses; independent audit is a separate check.

\begin{{table*}}[t]
\centering
\caption{{Measured producer and consumer costs in seconds. Each entry is a three-repeat median. Joint costs add producer time, fresh checker initialization, JSON decoding and proof verification within each repeat before taking the median; Joint/Hot is the median paired joint-cost ratio. Thus the displayed component medians need not sum exactly to the joint median. Compression, compressed-file reading/decompression and unmeasured network transfer are excluded from the joint total.}}
\label{{tab:query-consumer}}\input{{query_consumer_table}}
\end{{table*}}

\textbf{{Cost beyond the producer.}} Table~\ref{{tab:query-consumer}} charges the receiving checker explicitly. Adding its measured work changes the Prepared/Hot comparison to {joint_ratios[0]:.3f} for the primary engine workload, {joint_ratios[1]:.3f} for uniform source-to-answer and {joint_ratios[2]:.3f} for skewed source-to-answer. These joint ratios prevent a producer-only result from concealing work moved to a consumer. Compressed-proof writing is recorded separately; these are not full deployment-latency measurements. The maximum saved-dual bank contains {memory['maximum_bank_entries']:,} entries. The largest final prepared-term Python payload is {memory['final_prepared_terms_python_bytes']['max'] / 1024:.1f}\,KiB, excluding shared view endpoints. This payload is not native solver RSS; the process-wide high-water mark is not isolated by plan.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--prefix-csv", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate complete evidence without writing paper assets")
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    csv_path = args.prefix_csv.resolve() if args.prefix_csv else summary_path.with_name("all_prefixes.csv")
    performance_replaced = False
    try:
        payload = validate_summary(summary_path, csv_path)
        facts = make_facts(payload, summary_path, csv_path)
        if args.check_only:
            print("Complete evidence checked: 54 + 6 conditions, 1,476 prefix rows; no paper files changed.")
            return
        # Stage the complete output set before replacing any current paper file.
        with tempfile.TemporaryDirectory(prefix=".query-assets-", dir=PAPER) as temporary:
            directory = Path(temporary)
            plot_assets(facts, directory)
            grid, consumer = tex_tables(facts)
            (directory / "query_grid_table.tex").write_text(grid)
            (directory / "query_consumer_table.tex").write_text(consumer)
            (directory / "query_figure_data.json").write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")
            (directory / "performance_results_v2.tex").write_text(performance_text(facts))
            (PAPER / "figures").mkdir(exist_ok=True)
            # Publish the text that references the figures last, so a failed
            # figure/table write cannot turn the placeholder into broken links.
            paths = sorted(directory.iterdir(), key=lambda p: (p.name == "performance_results_v2.tex", p.name))
            for path in paths:
                destination = PAPER / "figures" / path.name if path.suffix in {".pdf", ".png"} else PAPER / path.name
                os.replace(path, destination)
                performance_replaced = performance_replaced or path.name == "performance_results_v2.tex"
        print("Generated two query figures, two tables, performance text and provenance data from both complete stages.")
        print("Independent audit and rendered-paper review remain separate checks.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        state = "Performance text was replaced." if performance_replaced else "Existing performance text was not changed."
        parser.exit(2, f"Query asset generation stopped: {error}\n{state}\n")


if __name__ == "__main__":
    main()
