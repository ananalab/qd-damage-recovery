"""Oracle: evaluate every policy of a repertoire under every damage.

Usage:
    python -m qd_damage.oracle --runs results/repertoires/me_seed0 results/repertoires/pgame_seed0 ...

For each damage of configs/damages.yaml and each filled cell, `n_episodes` episodes (different random starts)
are played and averaged. This gives:
- the oracle performance: the best policy of the repertoire under this damage. It is the ceiling of what ITE
  can find, but the oracle tries everything (about 1000 x n_episodes episodes per damage);
- the performance of the best intact policy replayed on the damaged robot (no adaptation);
- the resilience of the repertoire: (damaged oracle - F0) / (intact oracle - F0), F0 = standing robot.

The intact scenario is also the re-evaluated prior used by ITE (see qd_damage.adaptation).
Outputs in results/oracle/<run>/: oracle.npz (everything), summary.csv (one row per damage), oracle.json.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs

from qd_damage.config import load_run_config, performance_reference
from qd_damage.envs import DamageWrapper, damage_scale, descriptor_extractor, load_damages, make_env
from qd_damage.repertoires import REPO_ROOT, git_commit, make_play_step_fn, make_policy_network

CHUNK = 512  # policies evaluated at once (memory)


def make_evaluator(common: dict, policy_network=None):
    """Return (env, network, evaluate) with evaluate(params_batch, key, scale) -> (fitnesses, descriptors).

    The damage vector `scale` is an argument of the compiled function, so all damages share one compilation.
    `policy_network` defaults to the repertoire MLP; any object with `apply(params, obs)` works.
    """
    base_env = make_env(common)
    policy_network = policy_network or make_policy_network(common, base_env.action_size)
    extractor = descriptor_extractor(common)
    episode_length = common["env"]["episode_length"]

    @jax.jit
    def evaluate(params_batch, key, scale):
        env = DamageWrapper(base_env, scale)
        play_step_fn = make_play_step_fn(env, policy_network, "me")
        fitnesses, descriptors, _ = scoring_function_brax_envs(
            params_batch, key, episode_length, env.reset, play_step_fn, extractor
        )
        return fitnesses, descriptors

    return base_env, policy_network, evaluate


def load_genotypes(run_dir: Path, policy_network, observation_size: int):
    """Parameters of all filled cells (stacked pytree), indices of these cells, and the repertoire data."""
    data = np.load(Path(run_dir) / "repertoire.npz")
    filled = np.flatnonzero(np.isfinite(data["fitnesses"]))
    template = policy_network.init(jax.random.key(0), jnp.zeros((observation_size,)))
    _, unravel = ravel_pytree(template)
    params = jax.vmap(unravel)(jnp.asarray(data["genotypes"][filled]))
    return params, filled, data


def evaluate_all(evaluate, params, scale, key, n_episodes: int):
    """Fitness (N, n_episodes) and mean descriptors (N, 4) of N policies (a stacked pytree) under one damage."""
    n = jax.tree.leaves(params)[0].shape[0]
    pad = (-n) % CHUNK
    padded = jax.tree.map(lambda x: jnp.concatenate([x, jnp.repeat(x[:1], pad, axis=0)]), params)
    fits, descs = [], []
    for _ in range(n_episodes):
        f_ep, d_ep = [], []
        for start in range(0, n + pad, CHUNK):
            chunk = jax.tree.map(lambda x: x[start:start + CHUNK], padded)
            key, subkey = jax.random.split(key)
            f, d = evaluate(chunk, subkey, scale)
            f_ep.append(np.asarray(f))
            d_ep.append(np.asarray(d))
        fits.append(np.concatenate(f_ep)[:n])
        descs.append(np.concatenate(d_ep)[:n])
    return np.stack(fits, axis=1), np.mean(descs, axis=0)


def run_oracle(run_dir: Path, out_root: Path = REPO_ROOT / "results" / "oracle", n_episodes=None, scenarios=None):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    common = load_run_config(run["algo"], run["profile"])["common"]
    damages = load_damages()
    n_episodes = n_episodes or damages["oracle"]["n_episodes"]
    chosen = [s for s in damages["scenarios"] if scenarios is None or s["name"] in scenarios]

    env, policy_network, evaluate = make_evaluator(common)
    params, filled, data = load_genotypes(run_dir, policy_network, env.observation_size)
    n_cells = data["fitnesses"].shape[0]
    key = jax.random.key(damages["oracle"]["seed"])

    t0 = time.time()
    fitness = np.full((len(chosen), n_cells, n_episodes), np.nan, dtype=np.float32)
    descriptors = np.full((len(chosen), n_cells, 4), np.nan, dtype=np.float32)
    for i, scenario in enumerate(chosen):
        key, subkey = jax.random.split(key)
        scale = jnp.asarray(damage_scale(scenario, damages, env.action_size))
        fitness[i, filled], descriptors[i, filled] = evaluate_all(evaluate, params, scale, subkey, n_episodes)
        print(f"[oracle {run_dir.name}] {scenario['name']:18s} best: {np.nanmax(fitness[i].mean(1)):7.1f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)

    out_dir = Path(out_root) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [s["name"] for s in chosen]
    np.savez_compressed(
        out_dir / "oracle.npz", scenarios=np.array(names), fitness=fitness, descriptors=descriptors,
        repertoire_fitness=data["fitnesses"], repertoire_descriptors=data["descriptors"],
    )

    write_summary(out_dir, run)
    (out_dir / "oracle.json").write_text(json.dumps({
        "run_dir": str(run_dir), "code_commit": git_commit(), "n_episodes": n_episodes,
        "seed": damages["oracle"]["seed"], "devices": [str(d) for d in jax.devices()],
        "time_s": time.time() - t0,
    }, indent=2))
    return out_dir


def write_summary(out_dir: Path, run: dict):
    """summary.csv, one row per damage, computed from oracle.npz (can be rerun without simulating).

    Resilience = (damaged oracle - F0) / (intact oracle - F0), F0 = return of a standing robot: the share of
    the progress (beyond standing still) that the repertoire can still offer.
    The best intact cell is chosen on the re-evaluated intact fitness, as in qd_damage.adaptation: the fitness
    stored in the repertoire comes from one lucky episode and over-estimates the elites.
    """
    data = np.load(Path(out_dir) / "oracle.npz")
    names = list(data["scenarios"])
    episode_length = load_run_config(run["algo"], run["profile"])["common"]["env"]["episode_length"]
    f0, _ = performance_reference(episode_length)
    mean_fit = np.nanmean(data["fitness"], axis=2)  # (scenarios, cells)
    rep_fit = data["repertoire_fitness"]
    intact = mean_fit[names.index("intact")]
    best_intact_cell = int(np.nanargmax(intact))
    oracle_intact = float(intact[best_intact_cell])
    rows = []
    for i, name in enumerate(names):
        best_cell = int(np.nanargmax(mean_fit[i]))
        rows.append({
            "run": Path(out_dir).name, "algo": run["algo"], "seed": run["seed"], "scenario": name,
            "oracle_cell": best_cell, "oracle_fitness": float(mean_fit[i, best_cell]),
            "best_intact_cell": best_intact_cell,
            "best_intact_fitness_damaged": float(mean_fit[i, best_intact_cell]),
            "resilience": (float(mean_fit[i, best_cell]) - f0) / (oracle_intact - f0),
            "best_intact_retained": (float(mean_fit[i, best_intact_cell]) - f0) / (oracle_intact - f0),
            "n_policies": int(np.isfinite(rep_fit).sum()), "n_episodes": int(data["fitness"].shape[2]),
        })
    with open(Path(out_dir) / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "oracle")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--summary-only", action="store_true", help="recompute summary.csv from oracle.npz")
    args = parser.parse_args()
    for run_dir in args.runs:
        if args.summary_only:
            run = json.loads((Path(run_dir) / "run.json").read_text())
            write_summary(Path(args.out) / Path(run_dir).name, run)
        else:
            print(run_oracle(run_dir, args.out, args.episodes, args.scenarios))


if __name__ == "__main__":
    main()
