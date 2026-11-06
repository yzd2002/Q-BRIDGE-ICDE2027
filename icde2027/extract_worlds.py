"""Export original finite experimental laws without editing historical files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "amber/src"), str(ROOT / "amber/scripts")]


def extract():
    from r55_q1q3 import build_tables as controlled
    from r73_eb_tau import build_tables as surface, sup_bias_lp, tau_of
    _, small = controlled()
    large, _ = surface()
    old = json.loads((ROOT / "amber/results/r55_q1q3_v3.json").read_text())
    _, _, shifted_large = sup_bias_lp(large, tau_of(large, large["mR1"]), return_argmax=True)
    worlds = []
    for name, table, randomized, labels in [
        ("controlled_2x2", small, np.array(old["pop"]["law"]), np.zeros(2, dtype=int)),
        ("kuairec_8x8", large, shifted_large, large["q_item"]),
    ]:
        q = table["q"]
        if q.ndim == 1:
            q = np.broadcast_to(q, table["mR1"].shape).copy()
        item = {
            "name": name,
            "p_observation": table["P"].tolist(),
            "target_mean": table["mR1"].tolist(),
            "randomized_mean": randomized.tolist(),
            "randomized_q": q.tolist(),
            "target_weights": table["px"].tolist(),
            "moment_nu": table["nu"].tolist(),
            "groups": labels.tolist(),
            "external_reference": table["eo"].tolist(),
            "external_policy": table["pi"].tolist(),
            "lambda_star": float(np.max(np.maximum(table["rstar"], 1/table["rstar"]))),
            "sampling_unit": "one independently generated binary event; X is a fixed type, not an additional repeated real user",
            "reference_scope": "fixed external policy specification equal numerically to legacy e_o; no e_o supplied to the new evaluator",
        }
        if "rewards" in table:
            item["reward_surface"] = table["rewards"].tolist()
            item["reward_sha256"] = hashlib.sha256(np.ascontiguousarray(table["rewards"]).tobytes()).hexdigest()
        worlds.append(item)
    source_names = ["amber/src/amber/qbridge/synthetic.py", "amber/scripts/r55_q1q3.py",
                    "amber/scripts/r73_eb_tau.py", "amber/scripts/run_qbridge_kuairec.py",
                    "amber/results/r55_q1q3_v3.json", "amber/configs/kuairec_user_order.json"]
    return {"description": "Read-only export of pre-existing experimental worlds, not new observed evidence.",
            "sources": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names},
            "worlds": worlds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite an existing export")
    result = extract()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exported {len(result['worlds'])} historical worlds to {args.out}")
