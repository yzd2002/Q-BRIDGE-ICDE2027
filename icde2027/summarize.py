"""Rebuild manuscript statistics from every registered trial, without selection."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .experiment import load_worlds, population_bounds

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/icde2027"
OUT = RUNS / "analysis"
METHODS = ("fusion", "observation_standalone", "moment_standalone",
           "observation_matched", "moment_matched")


def proportion(successes, total):
    interval = binomtest(successes, total).proportion_ci(method="exact")
    return {"successes": int(successes), "n": total, "rate": successes/total,
            "ci95": [interval.low, interval.high]}


def summarize_group(rows, config, bootstrap=False):
    result = {"id": config["id"], "config": config, "n": len(rows),
              "truth_mean": float(np.mean([r["true_contrast"] for r in rows])),
              "truth_sd": float(np.std([r["true_contrast"] for r in rows], ddof=1)),
              "truth_min": min(r["true_contrast"] for r in rows),
              "truth_max": max(r["true_contrast"] for r in rows),
              "all_truth_in_model": all(r["truth_in_model"] for r in rows),
              "rescue": proportion(sum(r["decision_rescue"] for r in rows), len(rows)),
              "methods": {}}
    for name in METHODS:
        raw = [r["results"][name] for r in rows]
        successful = [r["lower_bound"] for r in raw if r["lower_bound"] is not None]
        all_values = [r["results"][name]["lower_bound"] if r["results"][name]["lower_bound"] is not None
                      else r["deterministic_lower_bound"] for r in rows]
        result["methods"][name] = {
            "mean_with_fallback": float(np.mean(all_values)),
            "sd_with_fallback": float(np.std(all_values, ddof=1)),
            "mean_successful": float(np.mean(successful)) if successful else None,
            "successful_n": len(successful),
            "refusal": proportion(sum(r["refuses"] for r in raw),len(rows)),
            "certification": proportion(sum(r["certifies"] for r in raw),len(rows)),
            "no_false_assertion": proportion(sum(r["covers_target"] for r in raw),len(rows)),
            "max_primal_dual_gap": max((r["duality_gap"] or 0 for r in raw),default=0),
            "status_counts": dict(Counter(r["status"] for r in raw)),
        }
    gain = np.array([r["paired_improvement"] for r in rows])
    result["paired_gain"] = {"mean": float(gain.mean()), "sd": float(gain.std(ddof=1))}
    if bootstrap:
        rng = np.random.default_rng(902700)
        boot = gain[rng.integers(0,len(gain),(10000,len(gain)))].mean(axis=1)
        result["paired_gain"]["bootstrap_ci95"] = np.quantile(boot,[.025,.975]).tolist()
        result["rescue"]["p_one_sided_gt_0_10"] = float(binomtest(result["rescue"]["successes"],
                                                            len(rows),p=.10,alternative="greater").pvalue)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    output = {"scope":"All registered confirmation results, excluding exploratory pilot. Refusal is no assertion; separate refusal rates and deterministic-fallback means are reported.",
              "source_sha256":{}, "stages":{}}
    for stage in ("primary", "secondary", "coverage"):
        folder = RUNS / f"aggregate_{stage}_v1"
        manifest = json.loads((folder/"manifest.json").read_text())
        rows = [json.loads(line) for line in (folder/"trials.jsonl").read_text().splitlines()]
        assert manifest["status"] == "complete"
        expected = {(c["id"],s) for c in manifest["configs"] for s in manifest["seeds"]}
        actual = [(r["config_id"],r["seed"]) for r in rows]
        assert set(actual) == expected and len(actual)==len(expected)==manifest["completed_trials"]
        groups = defaultdict(list)
        for row in rows:
            groups[row["config_id"]].append(row)
        summaries = [summarize_group(groups[c["id"]],c,stage=="primary") for c in manifest["configs"]]
        if stage == "primary":
            pvals = np.array([s["rescue"]["p_one_sided_gt_0_10"] for s in summaries])
            order = np.argsort(pvals)
            adjusted = np.maximum.accumulate(pvals[order]*np.arange(len(pvals),0,-1))
            for i,p in zip(order,adjusted):
                summaries[i]["rescue"]["p_holm"] = float(min(1,p))
        output["stages"][stage] = summaries
        for filename in ("manifest.json","trials.jsonl"):
            p = folder/filename
            output["source_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()

    worlds = load_worlds(ROOT/"icde2027/data/worlds.json")
    output["population"] = {}
    for key,w in worlds.items():
        curve = []
        for factor in np.linspace(1,2,61):
            radius = factor*w["lambda_star"]
            raw = population_bounds(w,w["external_policy"],w["external_reference"],radius)
            curve.append({"factor":float(factor),"radius":radius,
                          **{k:v["lower_bound"] for k,v in raw.items()}})
        raw = population_bounds(w,w["external_policy"],w["external_reference"],2.)
        output["population"][key] = {"lambda_star":w["lambda_star"],"curve":curve,
            "at_radius_two":{k:v["lower_bound"] for k,v in raw.items()},
            "true_contrast":float(np.sum(w["target_weights"][:,None]*(w["external_policy"]-w["external_reference"])*w["target_mean"])),
            "nontransported_direct_contrast":float(np.sum(w["target_weights"][:,None]*(w["external_policy"]-w["external_reference"])*w["randomized_mean"])),
            "max_cell_transport_error":float(np.max(np.abs(w["target_mean"]-w["randomized_mean"]))) }

    output["systems"] = {}
    for filename in ("aggregation","solvers"):
        path = RUNS/"systems_v4"/(filename+".json")
        raw = json.loads(path.read_text())
        output["systems"][filename] = raw
        output["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/"summary.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    with (OUT/"all_conditions.csv").open("w") as file:
        writer=csv.writer(file)
        writer.writerow(["stage","condition","n","truth_mean","fusion_mean","observation_mean","moment_mean",
                         "rescue_count","rescue_ci_low","rescue_ci_high","fusion_refusal","fusion_false_assertion","paired_gain"])
        for stage,summaries in output["stages"].items():
            for s in summaries:
                writer.writerow([stage,s["id"],s["n"],s["truth_mean"],
                    *[s["methods"][m]["mean_with_fallback"] for m in METHODS[:3]],
                    s["rescue"]["successes"],*s["rescue"]["ci95"],
                    s["methods"]["fusion"]["refusal"]["successes"],
                    s["n"]-s["methods"]["fusion"]["no_false_assertion"]["successes"],s["paired_gain"]["mean"]])
    lines=["# Registered results", "", "Generated from complete per-trial records. No pilot trials included.", "",
           "| World / policy | Truth (mean) | OBS LCB | Moment LCB | Fusion LCB (SD) | Rescue / 100 [exact 95% CI] | Holm p |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in output["stages"]["primary"]:
        ms=s["methods"]; r=s["rescue"]
        lines.append(f'| {s["id"]} | {s["truth_mean"]:.6f} | {ms["observation_standalone"]["mean_with_fallback"]:.6f} | {ms["moment_standalone"]["mean_with_fallback"]:.6f} | {ms["fusion"]["mean_with_fallback"]:.6f} ({ms["fusion"]["sd_with_fallback"]:.6f}) | {r["successes"]} [{r["ci95"][0]:.4f}, {r["ci95"][1]:.4f}] | {r["p_holm"]:.3g} |')
    lines += ["", "All secondary conditions: `all_conditions.csv`. All method-specific coverage/refusal and paired bootstrap intervals: `summary.json`."]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print("Complete registered studies summarized:", {k:sum(s['n'] for s in v) for k,v in output['stages'].items()})
    print("Written",OUT)


if __name__ == "__main__":
    main()
