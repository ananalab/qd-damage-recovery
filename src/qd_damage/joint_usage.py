"""Alternative behaviour descriptor for ITE: how much each joint is used.

Usage:
    python -m qd_damage.joint_usage --runs results/repertoires/dcrlme_seed0 ...

The repertoires are organised by feet contact, but a damage acts on joints: a gait that hardly uses the front-right
hip should suffer little when that hip is paralysed. For every filled cell, this script plays the policy on the
intact robot and records the mean absolute action of each of the 8 actuators (in [0, 1], tanh outputs). ITE can
then model the effect of the damage on this 8D descriptor instead of the 4D feet-contact one (descriptor
ablation). Outputs in results/descriptors/<run>/joint_usage.npz (NaN for empty cells).
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qd_damage.config import load_run_config
from qd_damage.oracle import evaluate_all, load_genotypes, make_evaluator
from qd_damage.repertoires import REPO_ROOT

OUT = REPO_ROOT / "results" / "descriptors"


def mean_abs_action(data, mask):
    """QDax descriptor extractor: mean |action| of each actuator over the valid steps of an episode."""
    mask = jnp.expand_dims(mask, axis=-1)
    return jnp.sum(jnp.abs(data.actions) * (1.0 - mask), axis=1) / jnp.sum(1.0 - mask, axis=1)


def compute_joint_usage(run_dir: Path, n_episodes: int = 2, seed: int = 99, out_root: Path = OUT) -> Path:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    common = load_run_config(run["algo"], run["profile"])["common"]
    env, policy_network, evaluate = make_evaluator(common, extractor=mean_abs_action)
    params, filled, data = load_genotypes(run_dir, policy_network, env.observation_size)
    _, usage = evaluate_all(evaluate, params, jnp.ones(env.action_size), jax.random.key(seed), n_episodes)
    descriptors = np.full((data["fitnesses"].shape[0], env.action_size), np.nan, dtype=np.float32)
    descriptors[filled] = usage
    out_dir = Path(out_root) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "joint_usage.npz", descriptors=descriptors, n_episodes=n_episodes)
    return out_dir / "joint_usage.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    for run_dir in args.runs:
        print(compute_joint_usage(run_dir, out_root=args.out))


if __name__ == "__main__":
    main()
