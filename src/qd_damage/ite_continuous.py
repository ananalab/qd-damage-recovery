"""Continuous ITE: search the descriptor space with the descriptor-conditioned actor of DCRL-ME.

Usage:
    python -m qd_damage.ite_continuous --runs results/bonus/repertoires/dcrlme_seed0 ...

DCRL-ME learns an actor pi(a | observation, d): "walk with feet-contact profile d". Instead of searching the
~1000 cells of the repertoire, ITE searches N gaits produced by the actor for N target descriptors spread over
[0,1]^4, i.e. gaits "between the cells". Everything else is identical to ITE on the grid (re-evaluated prior,
Gaussian process, damages, number of repetitions), so both can be compared on the same run.

Outputs in results/bonus/continuous/<run>/: candidates.npz, trials.csv, summary.csv and continuous.json.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qd_damage.adaptation import write_csv
from qd_damage.config import load_run_config, load_yaml, performance_reference
from qd_damage.envs import damage_scale, load_damages, make_env
from qd_damage.ite import Normalizer, run_ite
from qd_damage.oracle import evaluate_all, make_evaluator
from qd_damage.repertoires import REPO_ROOT, git_commit, load_dc_actor

OUT = REPO_ROOT / "results" / "bonus" / "continuous"


class ConditionedPolicy:
    """The actor seen as a policy whose "parameters" are the target descriptor d.

    The existing machinery (QDax scoring, oracle, ITE) then applies unchanged.
    """

    def __init__(self, actor, actor_params):
        self.actor, self.actor_params = actor, actor_params

    def apply(self, desc, obs):
        return self.actor.apply(self.actor_params, obs, desc)


def run_continuous(
    run_dir: Path, n_candidates: int = 2048, n_episodes: int = 4, scenarios=None, n_repeats=None, out_root: Path = OUT
):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    common = load_run_config(run["algo"], run["profile"])["common"]
    ite = load_yaml("adaptation")["ite"]
    damages = load_damages()
    n_repeats = n_repeats or ite["n_repeats"]
    f0, scale = performance_reference(common["env"]["episode_length"])
    normalizer = Normalizer(f0=f0, scale=scale)
    # The intact scenario is always evaluated: it gives the prior and the achieved descriptors.
    chosen = [s for s in damages["scenarios"] if scenarios is None or s["name"] in scenarios or s["name"] == "intact"]

    actor, actor_params = load_dc_actor(run_dir, common, make_env(common))
    env, _, evaluate = make_evaluator(common, ConditionedPolicy(actor, actor_params))
    rng = np.random.default_rng(run["seed"])
    targets = jnp.asarray(rng.random((n_candidates, env.descriptor_length)), jnp.float32)  # targets in [0,1]^4
    key = jax.random.key(damages["oracle"]["seed"] + 1)
    t0 = time.time()
    fitness = {}
    for s in chosen:
        key, subkey = jax.random.split(key)
        f, d = evaluate_all(evaluate, targets, jnp.asarray(damage_scale(s, damages)), subkey, n_episodes)
        fitness[s["name"]] = f
        if s["name"] == "intact":
            achieved = d  # descriptors actually reached on the intact robot: input of the Gaussian process
        print(
            f"[continuous {run_dir.name}] {s['name']:18s} best: {f.mean(1).max():7.1f} ({time.time() - t0:.0f} s)",
            flush=True,
        )

    out_dir = Path(out_root) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "candidates.npz",
        targets=np.asarray(targets),
        achieved=achieved,
        scenarios=np.array(list(fitness)),
        fitness=np.stack(list(fitness.values())),
    )

    prior = fitness["intact"].mean(1)
    oracle_intact = float(prior.max())
    rows, summary = [], []
    for s in chosen:
        if s["name"] == "intact":
            continue
        true_mean = fitness[s["name"]].mean(1)
        scale_vec = jnp.asarray(damage_scale(s, damages))
        for rep in range(n_repeats):
            key, method_key = jax.random.split(key)
            keys = iter(jax.random.split(method_key, ite["max_trials"] + 1))

            def trial_fn(i, _keys=keys, scale_vec=scale_vec):
                f, _ = evaluate(targets[i : i + 1], next(_keys), scale_vec)
                return float(f[0])

            hist = run_ite(
                achieved,
                prior,
                trial_fn,
                normalizer,
                kappa=ite["kappa"],
                alpha=ite["alpha"],
                lengthscale=ite["lengthscale"],
                noise_variance=ite["noise_variance"],
                signal_variance=ite["signal_variance"],
                max_trials=ite["max_trials"],
            )
            for t in range(len(hist.cells)):
                rec = hist.best_cell(t + 1)
                rows.append(
                    {
                        "run": run_dir.name,
                        "seed": run["seed"],
                        "scenario": s["name"],
                        "rep": rep,
                        "trial": t + 1,
                        "recommended_true": float(true_mean[rec]),
                        "perf_abs": float(normalizer(true_mean[rec])),
                        "pct_intact": 100 * (true_mean[rec] - f0) / (oracle_intact - f0),
                    }
                )
            stop = hist.stop_trial or len(hist.cells)
            rec = hist.best_cell(stop)
            # pct_intact is relative to the best actor gait, not to the best elite: compare with the grid
            # using perf_abs only.
            summary.append(
                {
                    "run": run_dir.name,
                    "seed": run["seed"],
                    "scenario": s["name"],
                    "rep": rep,
                    "method": "ite_continuous",
                    "trials_used": stop,
                    "recommended_true": float(true_mean[rec]),
                    "perf_abs": float(normalizer(true_mean[rec])),
                    "pct_intact": 100 * (true_mean[rec] - f0) / (oracle_intact - f0),
                    "oracle_continuous_abs": float(normalizer(true_mean.max())),
                }
            )
    write_csv(out_dir / "trials.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "continuous.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "code_commit": git_commit(),
                "n_candidates": n_candidates,
                "n_episodes": n_episodes,
                "ite": ite,
                "time_s": time.time() - t0,
            },
            indent=2,
        )
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    for r in args.runs:
        print(run_continuous(r, args.candidates, scenarios=args.scenarios, n_repeats=args.repeats, out_root=args.out))


if __name__ == "__main__":
    main()
