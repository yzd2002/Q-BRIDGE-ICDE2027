"""Write manuscript assets from the completed, independently checked diagnostic.

This performs no inference or optimization and never changes prior manuscripts.
The public plotting input remains the data-only *_figure_data.json file.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "runs/icde2027/analysis"
PAPER = ROOT / "icde2027/paper"
AUDIT = ROOT / "runs/icde2027/reviews/sensitivity_budget_v4_independent_verify.json"
GEOMETRY_AUDIT = ROOT / "runs/icde2027/reviews/fusion_geometry_v4_independent.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result_path = ANALYSIS / "sensitivity_budget_v4_results.json"
    losses_path = ANALYSIS / "sensitivity_budget_v4_loss_witnesses.json"
    figure_path = ANALYSIS / "sensitivity_budget_v4_figure_data.json"
    data = json.loads(result_path.read_text())
    audit = json.loads(AUDIT.read_text())
    losses = json.loads(losses_path.read_text())
    figure = json.loads(figure_path.read_text())
    if any(d["status"] != "complete" for d in (data, audit, losses)):
        raise RuntimeError("Both evidence and independent arithmetic audit must be complete")
    if audit["input_results_sha256"] != sha(result_path):
        raise RuntimeError("Results differ from independently checked bytes")
    if audit["loss_witnesses_sha256"] != sha(losses_path):
        raise RuntimeError("Exact loss witnesses differ from independently checked bytes")
    for rel, digest in audit["input_hashes"].items():
        if sha(ROOT / rel) != digest:
            raise RuntimeError(f"Audited input changed: {rel}")
    if (audit["worlds"], audit["query_records"], audit["proofs_checked"],
            audit["exact_identities_checked"]) != (48, 2880, 8640, 2880):
        raise RuntimeError("Incomplete diagnostic audit")
    if figure["source_sha256"] != sha(result_path) or figure["loss_witnesses_sha256"] != sha(losses_path):
        raise RuntimeError("Figure inputs do not bind this diagnostic")
    if figure["generator_sha256"] != sha(ANALYSIS / "sensitivity_budget_v4_render.py"):
        raise RuntimeError("Figure generator changed after rendering")
    conditions = [g for g in data["groups"] if g["family"] == "metadata"]
    if conditions != figure["conditions"] or len(conditions) != 4:
        raise RuntimeError("Figure must show all four metadata conditions")
    independent = {(g["sensitivity"], g["budget"]): g for g in audit["groups"] if g["family"] == "metadata"}
    mapping = {"queries": "queries", "worlds": "worlds", "observation_certified": "obs",
               "randomized_certified": "rand", "fusion_certified": "fusion",
               "fusion_extra_vs_obs": "extra", "fusion_lost_vs_obs": "lost",
               "mean_added_moment_dual_gain": "mean_gain",
               "mean_obs_confidence_budget_penalty": "mean_penalty",
               "mean_saved_dual_minus_full_obs": "mean_net"}
    for g in conditions:
        checked = independent[g["sensitivity"], g["r_events_per_cell"]]
        for source_key, checked_key in mapping.items():
            if not math.isclose(g[source_key], checked[checked_key], rel_tol=1e-12, abs_tol=1e-16):
                raise RuntimeError(f"Independent summary mismatch: {source_key}")
    low_losses = [r for r in losses["records"] if r["r_events_per_cell"] == 100]
    if len(low_losses) != 2 or not all(r["exact_point_feasible"] and r["box_and_saved_dual_equal"]
                                      and r["exact_fusion_optimum_at_most_threshold"] for r in low_losses):
        raise RuntimeError("Both low-budget losses require exact optimality witnesses")
    geometry = json.loads(GEOMETRY_AUDIT.read_text())
    if not all(geometry.get(k) is True for k in ("complete", "passed", "independent_review")):
        raise RuntimeError("The minimizing-face diagnostic requires independent verification")
    if geometry["production_analysis_solver_or_checker_imported_or_used"] is not False:
        raise RuntimeError("Minimizing-face audit was not independent")
    if (geometry["worlds"], geometry["query_records"]) != (48, 2880):
        raise RuntimeError("Incomplete minimizing-face census")
    for rel, digest in geometry["sources_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise RuntimeError(f"Geometry input changed: {rel}")
    geometry_groups = {(g["sensitivity"], g["r_events_per_cell"]): g
                       for g in geometry["groups"] if g["family"] == "metadata"}

    for ext in ("pdf", "png"):
        shutil.copy2(ANALYSIS / f"sensitivity_budget_v4.{ext}", PAPER / f"figures/sensitivity_budget_v4.{ext}")
    rows = [r"\begin{tabular}{llrrr}", r"\toprule",
            r"Radius & $n_R/d$ & OBS & R & Fusion \\", r"\midrule"]
    for g in conditions:
        radius = r"$2$" if g["sensitivity"] == "conservative" else r"$\Lambda^\star$ (oracle)"
        rows.append(f"{radius} & {g['r_events_per_cell']:,} & {g['observation_certified']} & "
                    f"{g['randomized_certified']} & {g['fusion_certified']} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "sensitivity_table_v4.tex").write_text("\n".join(rows) + "\n")
    text = r"""% Generated from all four metadata conditions and independently checked exact arithmetic.
\textbf{The radius changes what the same evidence supports.}
Across the 48 confounded worlds, $\Lambda^\star$ ranges from RMIN to RMAX; the registered conservative rule therefore uses $\Lambda=2$ throughout. With identical metadata policies and counts at each budget, the low-budget oracle comparison certifies OORACLE queries from OBS and FORACLE from fusion, versus OCONSERVATIVE and FCONSERVATIVE under $\Lambda=2$ (Fig.~\ref{fig:sensitivity}). This is an oracle sensitivity diagnostic: truth supplies $\Lambda^\star$. It does not identify a usable radius from observed data. The 240 query instances still represent 48 independent worlds.

\textbf{Added constraints compete with confidence allocation.}
Let $B_{O,\alpha}$ be the exact box minimum for full-budget OBS, $B_{O,\alpha/2}$ its half-budget counterpart inside fusion, and $D_F$ the exact value of the saved fusion dual. Every saved record satisfies
\begin{align}
 D_F-B_{O,\alpha}
 &=\underbrace{D_F-B_{O,\alpha/2}}_{\text{achieved constraint gain}}
 -\underbrace{(B_{O,\alpha}-B_{O,\alpha/2})}_{\text{budget cost}}.\label{eq:budget-diagnostic}
\end{align}
At the lower budget and oracle radius, the mean gain is GAIN~pp and the mean cost COST~pp, leaving NET~pp. These are changes in bounds, not true policy effects. The saved dual need not attain the fused optimum. For the two lost certifications, however, the half-budget box minimizer satisfies every group constraint and exactly matches the dual. Their loss is therefore due to the budget split, with no missed optimization gain. The third loss at the larger budget lacks this particular optimality witness; all three remain reported.

\textbf{Why a tighter radius can make groups redundant.}
For the lower-budget metadata queries, Proposition~\ref{prop:box-face} proves zero optimal constraint gain in ZEROORACLE/240 oracle-radius views, versus ZEROCONSERVATIVE/240 at $\Lambda=2$. The corresponding larger-budget counts are ZEROORACLELARGE and ZEROCONSERVATIVELARGE. Thus many narrow observational boxes already satisfy the transferred group information at their worst point for the query. The other views have strictly positive optimal constraint gain, but that alone proves neither $G>P$ nor certification above $\delta$. This exact geometric diagnosis concerns these saved programs, without estimating how often it occurs in a new population.

The randomized budgets use different sampled streams, not nested extensions. All three policy families and complete strata are retained. The robust-design objective itself changes in 278 of 480 radius-paired records, so its radius comparison cannot isolate the declaration alone. For metadata queries, exact projection verifies a nonzero component outside the group row space in every world; this describes missing directions without attributing a causal share of failure to geometry.
"""
    conservative, _, oracle, _ = conditions
    substitutions = {
        "RMIN": f"{data['true_radius_minmax'][0]:.4f}", "RMAX": f"{data['true_radius_minmax'][1]:.4f}",
        "OORACLE": str(oracle["observation_certified"]), "FORACLE": str(oracle["fusion_certified"]),
        "OCONSERVATIVE": str(conservative["observation_certified"]),
        "FCONSERVATIVE": str(conservative["fusion_certified"]),
        "GAIN": f"{100 * oracle['mean_added_moment_dual_gain']:.4f}",
        "COST": f"{100 * oracle['mean_obs_confidence_budget_penalty']:.4f}",
        "NET": f"{100 * oracle['mean_saved_dual_minus_full_obs']:.4f}",
        "ZEROORACLELARGE": str(geometry_groups["true_boundary", 1000]["categories"].get("zero_optimal_constraint_gain", 0)),
        "ZEROCONSERVATIVELARGE": str(geometry_groups["conservative", 1000]["categories"].get("zero_optimal_constraint_gain", 0)),
        "ZEROORACLE": str(geometry_groups["true_boundary", 100]["categories"].get("zero_optimal_constraint_gain", 0)),
        "ZEROCONSERVATIVE": str(geometry_groups["conservative", 100]["categories"].get("zero_optimal_constraint_gain", 0)),
    }
    for key in sorted(substitutions, key=len, reverse=True):
        text = text.replace(key, substitutions[key])
    (PAPER / "sensitivity_results_v4.tex").write_text(text)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in
               (result_path, losses_path, figure_path, AUDIT, GEOMETRY_AUDIT,
                ANALYSIS / "fusion_geometry_v4_results.json",
                ANALYSIS / "sensitivity_budget_v4.pdf", ANALYSIS / "sensitivity_budget_v4.png")}
    facts = {"status": "complete", "schema": "sensitivity-paper-assets-v4",
             "sources_sha256": sources, "generator_sha256": sha(Path(__file__)),
             "conditions": conditions, "independent_worlds": 48,
             "scope": data["scope"], "low_budget_loss_optimality_witnesses": len(low_losses),
             "minimizing_face_groups": geometry["groups"],
             "all_zero_objectives_in_geometry_diagnostic": geometry["all_zero_objectives"],
             "all_losses_have_exact_box_optimum_witness": losses["all_losses_have_exact_box_optimum_witness"]}
    (PAPER / "sensitivity_facts_v4.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps({"status": "complete", "conditions": len(conditions), "sources": len(sources)}))


if __name__ == "__main__":
    main()
