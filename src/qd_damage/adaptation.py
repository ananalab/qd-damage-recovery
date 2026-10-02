"""Damage adaptation: ITE and baselines, on real simulated episodes of the damaged robot.

Usage:
    python -m qd_damage.adaptation --runs results/repertoires/me_seed0 ...   (the oracle must exist)

For each repertoire, damage and method (ITE, top-k, random, best-intact), `n_repeats` repetitions (episodes are
noisy: random starts). Each trial is ONE episode on the damaged robot. The ITE prior (and the best-intact choice)
is the intact fitness re-evaluated by the oracle. The quality of the recommended gait (the best one observed) is
measured with the oracle mean over several episodes, which is less noisy than a single trial.

Outputs in results/adaptation/<run>/: trials.csv (one trial per row), summary.csv and adaptation.json.
Performances are percentages (F - F0) / (F_ref - F0), with F0 the return of a standing robot
(configs/adaptation.yaml) and F_ref the oracle under the damage or the intact oracle.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import numpy as np

from qd_damage.baselines import run_best_intact, run_random, run_top_k
from qd_damage.config import load_run_config, load_yaml, performance_reference
from qd_damage.envs import damage_scale, load_damages
from qd_damage.ite import Normalizer, run_ite
from qd_damage.oracle import load_genotypes, make_evaluator
from qd_damage.repertoires import REPO_ROOT, git_commit

METHODS = ("ite", "top_k", "random", "best_intact")


def write_csv(path: Path, rows: list):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_adaptation(
    run_dir: Path,
    oracle_root: Path = REPO_ROOT / "results" / "oracle",
    out_root: Path = REPO_ROOT / "results" / "adaptation",
    scenarios=None,
    n_repeats=None,
    methods=METHODS,
    ite_overrides=None,
):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    common = load_run_config(run["algo"], run["profile"])["common"]
    ite_cfg = {**load_yaml("adaptation")["ite"], **(ite_overrides or {})}  # overrides: kappa sweep on dev runs
    max_trials = ite_cfg["max_trials"]
    n_repeats = n_repeats or ite_cfg["n_repeats"]
    damages = load_damages()

    oracle = np.load(Path(oracle_root) / run_dir.name / "oracle.npz")
    oracle_names = list(oracle["scenarios"])
    chosen = [s for s in damages["scenarios"]
              if s["name"] != "intact" and (scenarios is None or s["name"] in scenarios)]

    env, policy_network, evaluate = make_evaluator(common)
    params, filled, data = load_genotypes(run_dir, policy_network, env.observation_size)
    descriptors = data["descriptors"][filled]
    # Prior = intact fitness re-evaluated by the oracle (mean over its episodes) rather than the fitness stored
    # in the repertoire: the stored value comes from one noisy episode and MAP-Elites keeps the lucky ones, so
    # elites are over-estimated. Re-evaluating in simulation is legitimate: the repertoire is built offline,
    # before the damage.
    prior = oracle["fitness"][oracle_names.index("intact")].mean(-1)[filled]
    f0, scale = performance_reference(common["env"]["episode_length"])
    normalizer = Normalizer(f0=f0, scale=scale)
    oracle_intact = float(prior.max())

    rows, summary, t0 = [], [], time.time()
    key = jax.random.key(run["seed"] * 1000 + 7)
    for scenario in chosen:
        scale_vec = damage_scale(scenario, damages, env.action_size)
        true_mean = oracle["fitness"][oracle_names.index(scenario["name"])].mean(-1)[filled]  # (n,)
        oracle_best = float(true_mean.max())

        def pct(fitness):
            return (100 * (fitness - f0) / (oracle_best - f0), 100 * (fitness - f0) / (oracle_intact - f0))

        for method in methods:
            for rep in range(n_repeats):
                key, method_key = jax.random.split(key)
                keys = iter(jax.random.split(method_key, max_trials + 1))

                def trial_fn(i, _keys=keys):
                    one = jax.tree.map(lambda x: x[i:i + 1], params)
                    f, _ = evaluate(one, next(_keys), scale_vec)
                    return float(f[0])

                if method == "ite":
                    hist = run_ite(descriptors, prior, trial_fn, normalizer, kappa=ite_cfg["kappa"],
                                   alpha=ite_cfg["alpha"], lengthscale=ite_cfg["lengthscale"],
                                   noise_variance=ite_cfg["noise_variance"],
                                   signal_variance=ite_cfg["signal_variance"], max_trials=max_trials)
                elif method == "top_k":
                    hist = run_top_k(prior, trial_fn, max_trials)
                elif method == "random":
                    hist = run_random(len(prior), trial_fn, max_trials,
                                      np.random.default_rng(run["seed"] * 100 + rep))
                else:
                    hist = run_best_intact(prior, trial_fn)

                for t in range(len(hist.cells)):
                    rec = hist.best_cell(t + 1)
                    pct_oracle, pct_intact = pct(true_mean[rec])
                    rows.append({
                        "run": run_dir.name, "algo": run["algo"], "seed": run["seed"],
                        "scenario": scenario["name"], "method": method, "rep": rep, "trial": t + 1,
                        "cell": int(filled[hist.cells[t]]), "observed": hist.observed[t],
                        "recommended_cell": int(filled[rec]), "recommended_true": float(true_mean[rec]),
                        "pct_oracle": pct_oracle, "pct_intact": pct_intact,
                        "stop_trial": hist.stop_trial if method == "ite" else "",
                    })
                # Only ITE has a stopping rule; the baselines are scored after all their trials.
                stopped = method == "ite" and hist.stop_trial is not None
                stop = hist.stop_trial if stopped else len(hist.cells)
                rec = hist.best_cell(stop)
                pct_oracle, pct_intact = pct(true_mean[rec])
                summary.append({
                    "run": run_dir.name, "algo": run["algo"], "seed": run["seed"],
                    "scenario": scenario["name"], "method": method, "rep": rep,
                    "trials_used": stop, "stopped_by_criterion": stopped,
                    "recommended_true": float(true_mean[rec]),
                    "pct_oracle": pct_oracle, "pct_intact": pct_intact,
                    "oracle_fitness": oracle_best,
                })
            done = [s for s in summary if s["scenario"] == scenario["name"] and s["method"] == method]
            print(f"[adapt {run_dir.name}] {scenario['name']:18s} {method:11s} "
                  f"trials {np.mean([s['trials_used'] for s in done]):5.1f}  "
                  f"% oracle {np.mean([s['pct_oracle'] for s in done]):5.1f}  ({time.time() - t0:.0f} s)", flush=True)

    out_dir = Path(out_root) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "trials.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "adaptation.json").write_text(json.dumps({
        "run_dir": str(run_dir), "code_commit": git_commit(), "ite": ite_cfg, "f0": f0,
        "oracle_intact": oracle_intact, "time_s": time.time() - t0,
    }, indent=2))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--oracle-root", type=Path, default=REPO_ROOT / "results" / "oracle")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "adaptation")
    args = parser.parse_args()
    for run_dir in args.runs:
        print(run_adaptation(run_dir, args.oracle_root, args.out, scenarios=args.scenarios, n_repeats=args.repeats))


if __name__ == "__main__":
    main()
