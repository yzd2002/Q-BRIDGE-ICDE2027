"""Predeclared aggregate-certificate experiments using exact count sampling."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy import sparse

from .aggregate import conditional_box, evaluate_counts, make_group_moments, policy_objective, solve_bound


def json_write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_worlds(path):
    raw = json.loads(Path(path).read_text())
    worlds = {}
    array_keys = ["p_observation", "target_mean", "randomized_mean", "randomized_q", "target_weights",
                  "moment_nu", "groups", "external_reference", "external_policy"]
    for entry in raw["worlds"]:
        world = dict(entry)
        for key in array_keys:
            world[key] = np.asarray(world[key], dtype=int if key == "groups" else float)
        px = world["target_weights"]
        q = world["randomized_q"]
        mean = world["randomized_mean"]
        world["p_randomized"] = px[:, None, None] * q[:, :, None] * np.stack([1-mean, mean], axis=2)
        world["moment_rows"] = make_group_moments(world["moment_nu"], world["groups"])
        residual = world["moment_rows"] @ (world["target_mean"]-mean).ravel()
        if np.max(np.abs(residual), initial=0) > 1e-10:
            raise ValueError("Historical selected-moment law does not preserve declared rows")
        world["transport_residual_max"] = float(np.max(np.abs(residual), initial=0))
        worlds[world["name"]] = world
    return worlds


def draw_counts(rng, distribution, n):
    if not np.isclose(distribution.sum(), 1, atol=1e-12):
        raise ValueError("invalid population law")
    return rng.multinomial(int(n), distribution.ravel()).reshape(distribution.shape)


def learn_policy(obs_design, rand_design):
    """A fixed Laplace-smoothed learner; receives no population nuisances/truth."""
    action_count = obs_design.sum(2) + .5
    reference = action_count / action_count.sum(1, keepdims=True)
    reward_estimate = (rand_design[:, :, 1] + .5) / (rand_design.sum(2) + 1.)
    scores = 4 * (reward_estimate - reward_estimate.mean(1, keepdims=True))
    unnormalized = reference * np.exp(np.clip(scores, -.75, .75))
    return unnormalized / unnormalized.sum(1, keepdims=True), reference


def population_bounds(world, policy, reference, radius, *, rows=None, rho=0):
    c = policy_objective(world["target_weights"], policy, reference)
    p = world["p_observation"][:, :, 1] / world["p_observation"].sum(2)
    lower, upper = conditional_box(p.ravel(), p.ravel(), radius)
    rows = world["moment_rows"] if rows is None else rows
    tau = rows @ world["randomized_mean"].ravel()
    empty, rhs = sparse.csr_matrix((0,len(c))), np.zeros(0)
    outputs = {
        "observation": solve_bound(c, lower, upper, empty, rhs, rhs),
        "moment": solve_bound(c, np.zeros(len(c)), np.ones(len(c)), rows, tau-rho, tau+rho),
        "fusion": solve_bound(c, lower, upper, rows, tau-rho, tau+rho),
    }
    return {key: value.to_dict() for key,value in outputs.items()}


def trial(world, config, seed):
    begin = time.perf_counter()
    sequences = np.random.SeedSequence(seed).spawn(4)
    n_obs, n_rand = config["n_observation"], config["n_randomized"]
    obs = draw_counts(np.random.default_rng(sequences[0]), world["p_observation"], n_obs)
    random_mean = np.clip(world["randomized_mean"] + config.get("randomized_mean_shift",0), 0, 1)
    random_law = world["target_weights"][:,None,None]*world["randomized_q"][:,:,None]*np.stack([1-random_mean,random_mean],axis=2)
    rand = draw_counts(np.random.default_rng(sequences[1]), random_law, n_rand)
    mode = config.get("policy", "external")
    design = {}
    if mode == "learned":
        n_do, n_dr = config["n_design_observation"], config["n_design_randomized"]
        do = draw_counts(np.random.default_rng(sequences[2]), world["p_observation"], n_do)
        dr = draw_counts(np.random.default_rng(sequences[3]), world["p_randomized"], n_dr)
        policy, reference = learn_policy(do, dr)
        design = {"n_observation": n_do, "n_randomized": n_dr,
                  "observation_counts": do.tolist(), "randomized_counts": dr.tolist()}
    else:
        policy, reference = world["external_policy"].copy(), world["external_reference"].copy()
    if mode == "null":
        policy = reference.copy()
    if mode == "nontrivial_null":
        if policy.shape != (2,2):
            raise ValueError("this registered boundary query requires the 2x2 world")
        weights = world["target_weights"]
        means = world["target_mean"]
        action_difference = means[:,1]-means[:,0]
        direction = np.array([[-1.,1.],[1.,-1.]])
        direction[1] *= weights[0]*action_difference[0]/(weights[1]*action_difference[1])
        negative = direction < 0
        step = .5*float(np.min(reference[negative]/-direction[negative]))
        policy = reference+step*direction
    rows = world["moment_rows"]
    radius = config.get("radius", 2.0)
    if isinstance(radius, str):
        radius = float(radius.split("*")[0]) * world["lambda_star"]
    c = policy_objective(world["target_weights"], policy, reference)
    truth = float(c @ world["target_mean"].ravel())
    results = evaluate_counts(obs, rand, target_weights=world["target_weights"], policy=policy,
        reference=reference, moment_rows=rows, randomized_q=world["randomized_q"], radius=radius,
        alpha=config.get("alpha",.05), transport_budget=config.get("rho",0.))
    alpha = config.get("alpha", .05)
    threshold = config.get("threshold", .001)
    for name, result in results.items():
        if name == "diagnostics":
            continue
        value = result["lower_bound"]
        result["certifies"] = value is not None and value > threshold
        result["covers_target"] = value is None or value <= truth
        result["refuses"] = value is None
    trivial_lower = float(np.minimum(c,0).sum())
    baseline_values = [results[name]["lower_bound"] for name in ["observation_standalone", "moment_standalone"]]
    best_single = max(trivial_lower if value is None else value for value in baseline_values)
    fused = results["fusion"]["lower_bound"]
    residual = float(np.max(np.abs(rows @ (world["target_mean"]-random_mean).ravel()),initial=0))
    return {
        "config_id": config["id"], "world": world["name"], "seed": seed, "policy_mode":mode,
        "radius": radius, "alpha":alpha, "threshold":threshold,
        "target_weights": world["target_weights"].tolist(), "policy":policy.tolist(), "reference":reference.tolist(),
        "true_contrast":truth, "transport_residual_max":residual,
        "truth_in_model": bool(radius >= world["lambda_star"]-1e-12 and residual <= config.get("rho",0)+1e-10),
        "observation_counts":obs.tolist(), "randomized_counts":rand.tolist(), "design":design,
        "results":results,
        "decision_rescue": bool(fused is not None and fused > threshold and best_single <= threshold),
        "best_standalone_lcb": best_single,
        "deterministic_lower_bound": trivial_lower,
        "paired_improvement": (trivial_lower if fused is None else fused)-best_single,
        "seconds": time.perf_counter()-begin,
    }


_WORLDS = None
def _initialize(path):
    global _WORLDS
    _WORLDS = load_worlds(path)


def _task(job):
    config, seed = job
    return trial(_WORLDS[config["world"]], config, seed)


def run(args):
    if args.out.exists():
        raise SystemExit("Refusing to overwrite existing run directory")
    protocol = json.loads(args.protocol.read_text())
    configs = protocol[args.stage]["configs"]
    seeds = protocol[args.stage]["seeds"]
    args.out.mkdir(parents=True)
    source_dir = Path(__file__).parent
    record = {"protocol_sha256":hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "worlds_sha256":hashlib.sha256(args.worlds.read_bytes()).hexdigest(),
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.glob("*.py")},
        "python":platform.python_version(), "numpy":np.__version__, "platform":platform.platform(),
        "stage":args.stage, "workers":args.workers, "configs":configs, "seeds":seeds,
        "sampling":"Exact multinomial sufficient counts of iid binary events; not raw-row throughput measurement",
        "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), "status":"running"}
    json_write(args.out/"manifest.json",record)
    jobs = [(config,seed) for config in configs for seed in seeds]
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_initialize,initargs=(str(args.worlds),)) as executor:
        with (args.out/"trials.jsonl").open("w") as file:
            for result in executor.map(_task,jobs,chunksize=4):
                file.write(json.dumps(result,allow_nan=False)+"\n")
                file.flush()
                completed += 1
                if completed % 100 == 0 or completed == len(jobs):
                    print(f"completed {completed}/{len(jobs)}",flush=True)
    record["status"]="complete"
    record["completed_trials"]=completed
    record["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    json_write(args.out/"manifest.json",record)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--worlds",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--stage",choices=["pilot","primary","secondary","coverage"],required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    run(parser.parse_args())
