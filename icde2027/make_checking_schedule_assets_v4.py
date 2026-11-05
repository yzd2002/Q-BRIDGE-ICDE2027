"""Generate the complete S/P factorial presentation after independent admission.

No partial-result plotting, timing, query execution, or result-based selection.
The saved facts retain every registered contrast, group, round and prefix.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/icde2027/checking_schedule_v4"
AUDIT = ROOT / "runs/icde2027/audit/checking_schedule_v4_result_audit.json"
PAPER = ROOT / "icde2027/paper"
GROUPS = ("original_main", "threshold_engine", "source_derived")
PROTOCOL_SHA256 = "c58f5c2e26f589a49270793468bcc118b76a0c7b499697b7259382ea1605c8c5"
NAMES = {"original_main": "Primary streams", "threshold_engine": "Engine grid",
         "source_derived": "Source-derived views"}
CONTRAST_NAMES = {
    "s1_p0/s0_p0": r"Screening, $P=0$", "s1_p1/s0_p1": r"Screening, $P=1$",
    "s0_p1/s0_p0": r"Exact retention, $S=0$", "s1_p1/s1_p0": r"Exact retention, $S=1$",
    "s1_p1/original_prepared": "Both / original Prepared", "s1_p1/screened_auto": "Both / Auto bridge",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def gm(values):
    require(values and all(math.isfinite(x) and x > 0 for x in values), "Invalid cost ratio")
    return math.exp(math.fsum(math.log(x) for x in values) / len(values))


def verified_data():
    manifest = json.loads((RUN / "manifest.json").read_text())
    require(manifest.get("status") == "complete", "Complete formal run required before paper assets")
    require(manifest.get("final_integrity_check") == "passed", "Final runtime/input check missing")
    require(AUDIT.is_file(), "Separate independent result audit is required")
    audit = json.loads(AUDIT.read_text())
    require(all(audit.get(k) is True for k in ("complete", "passed", "independent_review", "core_same_evidence_and_paths")),
            "Independent result audit is incomplete")
    require(audit.get("production_checker_imported_or_used") is False, "Independent checker is required")
    completion = json.loads((RUN / "completion.json").read_text())
    expected_completion = {k: manifest.get(k) for k in
                           ("status", "finished_at_utc", "arm_streams", "responses", "final_integrity_check")}
    require(completion == expected_completion, "Manifest and completion differ")
    for key, count in (("arm_streams", 1134), ("responses", 162000)):
        require(type(completion.get(key)) is int and completion[key] == count, "Incomplete production census")
    times = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in
             (manifest["started_at_utc"], manifest["finished_at_utc"], audit["completed_at_utc"])]
    require(all(t.tzinfo is not None for t in times) and times[0] <= times[1] <= times[2],
            "Production/audit UTC order is invalid")
    for k, count in (("conditions_checked", 189), ("arm_streams_checked", 1134),
                     ("wire_responses_checked", 162000), ("native_responses_checked", 162000)):
        require(type(audit.get(k)) is int and audit[k] == count, "Incomplete independent census: " + k)
    pins = audit["sources_sha256"]
    for rel, digest in pins.items():
        path = Path(rel)
        require(not path.is_absolute() and ".." not in path.parts and
                (ROOT / path).resolve().is_relative_to(ROOT), "Audit path escapes project")
        require(sha(ROOT / path) == digest, "Audited input/result changed: " + rel)
    required = {str((RUN / name).relative_to(ROOT)) for name in
                ("manifest.json", "completion.json", "summary.json", "protocol.json", "gate.json", "design.md")}
    require(sha(RUN / "protocol.json") == PROTOCOL_SHA256, "Not the registered protocol")
    reg = json.loads((RUN / "protocol.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    require(len(summary["all_records"]) == 189, "Incomplete condition ledger")
    require(tuple(reg["input_counts"]) == GROUPS, "Registered group identity changed")
    require(reg["timing_rounds"] == 3, "Registered timing rounds changed")
    require(len(manifest["runtime_sha256"]) == 16 and len(manifest["snapshot_files"]) == 20,
            "Incomplete approved runtime/snapshot inventory")
    for rel, digest in manifest["runtime_sha256"].items():
        require(pins.get(rel) == digest, "Runtime not bound by independent audit: " + rel)
        required.add(rel)
        name = str((RUN / "source_snapshot" / rel).relative_to(ROOT))
        require(pins.get(name) == digest, "Runtime snapshot not bound: " + rel)
        required.add(name)
    for rel, identity in manifest["snapshot_files"].items():
        name = str((RUN / rel).relative_to(ROOT))
        require(pins.get(name) == identity["sha256"], "Snapshot not bound: " + rel)
        require((RUN / rel).stat().st_size == identity["bytes"], "Snapshot size changed")
        required.add(name)
    for entry in reg["inputs"]:
        for name, digest in entry["sha256"].items():
            rel = str(Path(entry["directory"]) / name)
            require(pins.get(rel) == digest, "Original input not bound: " + rel)
            required.add(rel)
        for repeat in range(3):
            cell = RUN / f"round_{repeat}" / entry["id"]
            for name in ("order.json", "condition.json"):
                required.add(str((cell / name).relative_to(ROOT)))
            for method in reg["methods"]:
                for suffix in (".jsonl.gz", "_native.jsonl.gz", "_summary.json"):
                    required.add(str((cell / (method + suffix)).relative_to(ROOT)))
    require(required <= set(pins), "Independent audit omits required inputs, streams, ledgers or snapshots")
    return reg, summary, audit, pins


def compile_facts(reg, summary, audit, pins):
    contrasts = [f"{a}/{b}" for a, b in reg["factor_contrasts"] + reg["bridge_contrasts"]]
    records = summary["all_records"]
    inputs = {entry["id"]: entry for entry in reg["inputs"]}
    require(len(inputs) == 63, "Registered input set is incomplete")
    expected_keys = {(i, key) for i in range(3) for key in inputs}
    require(len(records) == len(expected_keys) and
            {(r["timing_round"], r["input_id"]) for r in records} == expected_keys,
            "Condition ledger omits or duplicates a registered input/round")
    for record in records:
        entry = inputs[record["input_id"]]
        n = entry["query_shape"][0]
        require(record["group"] == entry["group"], "Input assigned to another group")
        require(set(record["methods"]) == set(reg["methods"]), "Registered method missing or added")
        expected_prefixes = {str(q) for q in reg["query_prefixes"] if q <= n}
        for method in reg["methods"]:
            arm = record["methods"][method]
            require(type(arm["queries"]) is int and arm["queries"] == n, "Arm query count differs")
            require(set(arm["joint_prefixes"]) == expected_prefixes, "Missing or unexpected planned prefix")
    rows = []
    prefixes = []
    core_diagnostics = []
    comparisons = []
    per_input_comparisons = []
    for group in GROUPS:
        for repeat in range(3):
            selected = [r for r in records if r["group"] == group and r["timing_round"] == repeat]
            require(len(selected) == reg["input_counts"][group], "Incomplete group/round")
            require(len({r["input_id"] for r in selected}) == len(selected), "Duplicated input")
            expected = summary["groups"][group][repeat]
            require([r["input_id"] for r in selected] == expected["input_ids"], "Summary input order differs")
            for contrast in contrasts:
                a, b = contrast.split("/")
                values = [r["methods"][a]["joint_seconds"] / r["methods"][b]["joint_seconds"] for r in selected]
                aggregate = gm(values)
                require(math.isclose(aggregate, expected[contrast]["geometric_mean"], rel_tol=1e-12),
                        "Independent reaggregation differs: " + contrast)
                require(all(math.isclose(x, y, rel_tol=1e-12) for x, y in
                            zip(values, expected[contrast]["per_input"], strict=True)), "Per-input ratio differs")
                rows.append({"group": group, "round": repeat, "contrast": contrast,
                             "geometric_mean": aggregate, "minimum": min(values), "maximum": max(values),
                             "inputs_no_slower": sum(x <= 1 for x in values), "inputs": len(values),
                             "per_input": dict(zip(expected["input_ids"], values, strict=True))})
                for q in reg["query_prefixes"]:
                    available = [r for r in selected if inputs[r["input_id"]]["query_shape"][0] >= q]
                    if available:
                        ratios = [r["methods"][a]["joint_prefixes"][str(q)] /
                                  r["methods"][b]["joint_prefixes"][str(q)] for r in available]
                        prefixes.append({"group": group, "round": repeat, "contrast": contrast,
                                         "queries": q, "geometric_mean": gm(ratios),
                                         "per_input": {r["input_id"]: x for r, x in zip(available, ratios, strict=True)}})
            for method in reg["core_methods"]:
                totals = defaultdict(float)
                decisions = Counter()
                for r in selected:
                    m = r["methods"][method]
                    decisions.update(m["decisions"])
                    for key, value in m["diagnostic_totals"].items():
                        totals[key] += value
                core_diagnostics.append({"group": group, "round": repeat, "method": method,
                                         "diagnostic_totals": dict(totals), "decisions": dict(decisions),
                                         "producer_seconds": sum(r["methods"][method]["producer_seconds"] for r in selected),
                                         "receiver_seconds": sum(r["methods"][method]["persistent_replay"]["total_consumer_seconds"] for r in selected),
                                         "joint_seconds": sum(r["methods"][method]["joint_seconds"] for r in selected)})
            for method in reg["methods"]:
                counts = Counter()
                decisions = Counter()
                refusals = 0
                for r in selected:
                    require(r["comparison"]["core_same_evidence_and_paths"] is True, "Core equivalence failed")
                    counts.update(r["comparison"]["relative_to_s1_p1"][method])
                    arm = r["methods"][method]
                    decisions.update(arm["decisions"])
                    refusals += arm["refusals"]
                    per_input_comparisons.append({"input_id": r["input_id"], "group": group,
                        "round": repeat, "method": method, "queries": arm["queries"],
                        "decisions": arm["decisions"], "refusals": arm["refusals"],
                        **r["comparison"]["relative_to_s1_p1"][method]})
                comparisons.append({"group": group, "round": repeat, "method": method,
                                    "decisions": dict(decisions), "refusals": refusals, **dict(counts)})
    target = reg["primary"]
    target_names = {f"{a}/{b}" for a, b in target["contrasts"]}
    supported = all(r["geometric_mean"] <= target["maximum_ratio"] for r in rows
                    if r["group"] == target["group"] and r["contrast"] in target_names)
    require(summary["both_factors_meet_main_engineering_target"] is supported and
            audit["engineering_target_supported"] is supported, "Registered engineering conclusion differs")
    interaction = [{"group": g, "round": i,
                    "mean_log_interaction": summary["groups"][g][i]["mean_log_interaction"]}
                   for g in GROUPS for i in range(3)]
    for row in interaction:
        values = {r["contrast"]: r["geometric_mean"] for r in rows
                  if (r["group"], r["round"]) == (row["group"], row["round"])}
        require(math.isclose(row["mean_log_interaction"],
                             math.log(values["s1_p1/s0_p1"]) - math.log(values["s1_p0/s0_p0"]),
                             rel_tol=1e-12, abs_tol=1e-15), "Factor interaction differs")
    return {"status": "complete", "schema": "checking-schedule-paper-v4", "sources_sha256": {
                str((RUN / name).relative_to(ROOT)): sha(RUN / name)
                for name in ("manifest.json", "completion.json", "summary.json", "protocol.json")}
            | {str(AUDIT.relative_to(ROOT)): sha(AUDIT)}, "audit_bound_files": len(pins),
            "generator_sha256": sha(Path(__file__)), "contrast_rows": rows, "prefix_rows": prefixes,
            "core_diagnostics": core_diagnostics, "decision_comparisons": comparisons,
            "per_input_decision_comparisons": per_input_comparisons,
            "interactions": interaction, "engineering_target_supported": supported,
            "responses": 162000, "core_evidence_and_native_paths_equal": True,
            "scope": "Three timing rounds on fixed, already inspected inputs; no significance or new workload generalization."}


def draw(facts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(7.12, 2.75), sharey=True)
    figure.subplots_adjust(left=.205, right=.99, bottom=.23, top=.76, wspace=.17)
    panels = (("(a) Screening effect", ("s1_p0/s0_p0", "s1_p1/s0_p1"), ("P off", "P retained")),
              ("(b) Exact-term retention", ("s0_p1/s0_p0", "s1_p1/s1_p0"), ("S off", "S on")))
    values = [r["geometric_mean"] for r in facts["contrast_rows"] if "original_prepared" not in r["contrast"]
              and "screened_auto" not in r["contrast"]]
    xlo, xhi = min(.8, min(values))*.9, max(1, max(values))*1.1
    for ax, (title, contrasts, labels) in zip(axes, panels, strict=True):
        ax.axvline(1, color="#677580", linewidth=.8)
        ax.axvline(.8, color="#A8AFB5", linestyle=":", linewidth=.8)
        for j, (contrast, label, color, marker) in enumerate(zip(contrasts, labels, ("#537EAE", "#DE963D"), ("o", "s"), strict=True)):
            for i, group in enumerate(GROUPS):
                points = [r["geometric_mean"] for r in facts["contrast_rows"]
                          if r["contrast"] == contrast and r["group"] == group]
                require(len(points) == 3, "A timing round is missing from the plot")
                y = i + (-.15 if j == 0 else .15)
                ax.plot([min(points), max(points)], [y, y], color=color, linewidth=1.2)
                ax.scatter(points, [y-.045, y, y+.045], color=color, marker=marker, s=19,
                           label=label if i == 0 else None, edgecolor="white", linewidth=.35, zorder=3)
        ax.set_xlim(xlo, xhi); ax.set_ylim(2.5, -.5)
        ax.set_yticks(range(3), [NAMES[g] for g in GROUPS])
        ax.set_xlabel("Joint-cost ratio (lower is faster)")
        ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold", y=1.22, pad=1)
        ax.legend(loc="lower left", bbox_to_anchor=(-.02, 1.02), ncol=2,
                  frameon=False, fontsize=7.4, handletextpad=.3, columnspacing=.8)
        ax.grid(axis="x", color="#E3E7EA", linewidth=.6)
        ax.tick_params(length=0)
        for edge in ax.spines.values():
            edge.set_visible(False)
    figure.text(.58, .025, "Dots: three round-wise geometric means. Lines: range, not confidence intervals.",
                ha="center", fontsize=7.1, color="#43515C")
    for ext in ("pdf", "png"):
        figure.savefig(PAPER / f"figures/checking_schedule_v4.{ext}", dpi=220, bbox_inches="tight", pad_inches=.03)
    plt.close(figure)


def prose_and_table(facts):
    def values(contrast, group):
        return [r["geometric_mean"] for r in facts["contrast_rows"] if r["contrast"] == contrast and r["group"] == group]
    rows = [r"\begin{tabular}{lrrr}", r"\toprule",
            r"Contrast & Primary & Engine & Source-derived \\", r"\midrule"]
    for contrast, name in CONTRAST_NAMES.items():
        cells = []
        for group in GROUPS:
            x = values(contrast, group)
            cells.append(f"{median(x):.3f} [{min(x):.3f}, {max(x):.3f}]")
        rows.append(name + " & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "checking_schedule_table_v4.tex").write_text("\n".join(rows) + "\n")
    def count_range(xs):
        return str(min(xs)) if min(xs) == max(xs) else f"{min(xs)}--{max(xs)}"
    bridge_rows = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
                   r"Bridge & Inputs & True & False & Unknown & No proof & Both gains & Both loses \\",
                   r"\midrule"]
    for method, label in (("original_prepared", "Original Prepared"), ("screened_auto", "Auto")):
        for group in GROUPS:
            selected = [r for r in facts["decision_comparisons"] if r["method"] == method and r["group"] == group]
            require(len(selected) == 3, "Incomplete bridge capability reporting")
            cells = [count_range([r["decisions"].get(key, 0) for r in selected]) for key in ("true", "false", "unknown")]
            cells += [count_range([r["refusals"] for r in selected]),
                      count_range([r.get("s1_p1_gains_decided", 0) for r in selected]),
                      count_range([r.get("s1_p1_loses_decided", 0) for r in selected])]
            bridge_rows.append(label + " & " + NAMES[group] + " & " + " & ".join(cells) + r" \\")
    bridge_rows += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "checking_schedule_bridge_answers_v4.tex").write_text("\n".join(bridge_rows) + "\n")
    screening = ", ".join(f"${v:.3f}$" for v in values("s1_p1/s0_p1", "original_main"))
    retention = ", ".join(f"${v:.3f}$" for v in values("s1_p1/s1_p0", "original_main"))
    no_retention_screen = values("s1_p0/s0_p0", "original_main")
    no_screen_retention = values("s0_p1/s0_p0", "original_main")
    primary = {(row["round"], row["method"]): row["diagnostic_totals"]
               for row in facts["core_diagnostics"] if row["group"] == "original_main"}
    require(len(primary) == 12, "Primary diagnostic factorial grid is incomplete")
    all_checks = [primary[(r, "s0_p1")]["dual_candidates_exact_checked"] for r in range(3)]
    screened_checks = [primary[(r, "s1_p1")]["dual_candidates_exact_checked"] for r in range(3)]
    screen_time = [primary[(r, "s1_p1")]["screen_seconds"] for r in range(3)]
    saved_check_time = [primary[(r, "s0_p1")]["verify_seconds"] -
                       primary[(r, "s1_p1")]["verify_seconds"] for r in range(3)]
    def integer_range(xs):
        require(all(float(x).is_integer() for x in xs), "Noninteger diagnostic count")
        return count_range([int(x) for x in xs])
    mechanism = (r"\textbf{Caching changes whether screening pays.} "
                 f"Without retained exact terms, screening has primary ratios "
                 f"${min(no_retention_screen):.3f}$--${max(no_retention_screen):.3f}$; "
                 f"after retention, the same screening loses (Fig.~\\ref{{fig:checking-factors}}). "
                 f"Retention without screening gives ratios ${min(no_screen_retention):.3f}$--"
                 f"${max(no_screen_retention):.3f}$. Across the three primary inputs in each round, "
                 f"screening reduces exact dual checks from {integer_range(all_checks)} to "
                 f"{integer_range(screened_checks)}. With retained terms it saves "
                 f"{min(saved_check_time):.2f}--{max(saved_check_time):.2f}\\,s of producer verification, "
                 f"but screening itself costs {min(screen_time):.2f}--{max(screen_time):.2f}\\,s. "
                 "These component timers explain the observed tradeoff; they are not added again "
                 "to joint cost. A filter that avoids checks can therefore become uneconomic "
                 "after those checks have been made cheaper. The broader groups also lose from "
                 "screening under both retention settings; the evidence does not support "
                 "screening every workload by default.")
    conclusion = ("Both factors meet the registered primary engineering target." if facts["engineering_target_supported"]
                  else "The joint requirement that both factors meet the registered primary engineering target is not supported.")
    text = r"""% Generated after the complete independent 162,000-response audit.
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/checking_schedule_v4.pdf}
\caption{Isolating checking costs while preserving the same evidence and solver paths. Each dot is one timing round's equal-weight geometric-mean joint-cost ratio over all inputs in that group; the line shows its three-round range. Screening is compared at each fixed exact-term retention setting, and retention at each fixed screening setting. The $.8$ reference applies to the two registered primary contrasts only. Values below one indicate less total producer-and-receiver work; no interval is a statistical confidence interval.}
\label{fig:checking-factors}
\end{figure*}
\begin{table*}[t]
\centering\small
\caption{All registered factor and bridge comparisons. Entries are median [minimum, maximum] of the three round-wise geometric-mean joint-cost ratios. Original Prepared is lower-only and cannot prove false; Auto changes the simplex strategy. Their cost ratios are descriptive. Source-derived serving starts from saved views, excluding source ingestion.}
\label{tab:checking-factors}\input{checking_schedule_table_v4}
\end{table*}
\begin{table*}[t]
\centering\small
\caption{Bridge answer capabilities on the same fixed inputs. Each cell is the observed range over three timing rounds, or one count when unchanged. ``Both'' means $S_1P_1$; gained/lost counts are decided answers relative to the listed bridge. No-proof refusals are included in unknown. These are repeated executions, not additional independent queries. Full per-input differences are retained.}
\label{tab:checking-bridge-answers}\input{checking_schedule_bridge_answers_v4}
\end{table*}

All 1,134 plan-streams and 162,000 responses complete, with independent numerical replay and complete source bindings. The four core plans return identical evidence, decisions, routes and native-solver paths. Figure~\ref{fig:checking-factors} therefore isolates checking work rather than a change in which queries get answered. On the primary streams, screening with retained terms has round-wise ratios SCREENING; retaining exact terms with screening has ratios RETENTION. CONCLUSION Table~\ref{tab:checking-factors} retains all six contrasts and both broader groups. The artifact includes every per-input ratio, prefix, factor interaction and decision comparison, including slow inputs and bridge losses. These are fixed-workload timing observations, not a significance or deployment-generalization claim.

MECHANISM

The bridges do not guarantee identical answer capabilities (Table~\ref{tab:checking-bridge-answers}). Original Prepared retains its older proposal/history path as well as its lower-only interface; cost comparisons must be read alongside its unknown answers. Auto changes the simplex strategy. Neither bridge isolates a factorial effect.
"""
    text = text.replace("SCREENING", screening).replace("RETENTION", retention).replace("CONCLUSION", conclusion).replace("MECHANISM", mechanism)
    (PAPER / "checking_schedule_results_v4.tex").write_text(text)


def main():
    reg, summary, audit, pins = verified_data()
    facts = compile_facts(reg, summary, audit, pins)
    draw(facts)
    prose_and_table(facts)
    (PAPER / "checking_schedule_facts_v4.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps({"status": "complete", "responses": facts["responses"],
                      "engineering_target_supported": facts["engineering_target_supported"]}))


if __name__ == "__main__":
    main()
