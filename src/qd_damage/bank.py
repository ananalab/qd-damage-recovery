"""Condition bank: every policy of a repertoire played under every condition of qd_damage.conditions.

Usage:
    python -m qd_damage.bank --runs results/repertoires/dcrlme_seed0 ...
    python -m qd_damage.bank --robust results/rl_robust/seed0          (one policy, 32 episodes per condition)

This extends the oracle of the first campaign. With it, any adaptation method can be replayed offline
(qd_damage.offline): a trial on the damaged robot is one of the simulated episodes of the chosen gait, and the
quality of the recommended gait is measured on other, independent episodes.

Outputs in results/bank/<run>/: bank.npz (fitness of shape (conditions, cells, episodes), NaN for empty cells and
missing episodes), conditions.json and bank.json. Also stored: the return of a standing robot (zero action,
f0) and the joint usage of each gait on the intact robot (mean |action| per actuator, the alternative ITE
descriptor).

With --signatures, every episode also records a proprioceptive signature of what the robot did, i.e. what a real
robot measures during a trial with its encoders, IMU and foot contacts (see `signature_extractor`), stored in
signatures.npz: per episode for the test conditions, averaged over episodes for the intact robot and the simulated
damages.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs

from qd_damage.conditions import build_conditions
from qd_damage.config import load_run_config, load_yaml, run_config
from qd_damage.envs import ConditionWrapper, descriptor_extractor, make_env
from qd_damage.joint_usage import mean_abs_action
from qd_damage.oracle import evaluate_all, load_genotypes
from qd_damage.repertoires import REPO_ROOT, git_commit, make_play_step_fn, make_policy_network

OUT = REPO_ROOT / "results" / "bank"


def make_condition_evaluator(common: dict, physics=None, extractor=None):
    """(env, network, evaluate) with evaluate(params_batch, key, scale, offset, obs_mask) -> (fitnesses, descs)."""
    base_env = make_env(common, physics)
    policy_network = make_policy_network(common, base_env.action_size)
    extractor = extractor or descriptor_extractor(common)
    episode_length = common["env"]["episode_length"]

    @jax.jit
    def evaluate(params_batch, key, scale, offset, obs_mask):
        env = ConditionWrapper(base_env, scale, offset, obs_mask)
        play_step_fn = make_play_step_fn(env, policy_network, "me")
        fitnesses, descriptors, _ = scoring_function_brax_envs(
            params_batch, key, episode_length, env.reset, play_step_fn, extractor
        )
        return fitnesses, descriptors

    return base_env, policy_network, evaluate


def signature_extractor(robot: str):
    """QDax descriptor extractor returning a proprioceptive signature of an episode.

    Mean and mean absolute value, over the steps the robot is alive, of the observation of the torso and the joints
    (positions and velocities, without the contact forces of Ant), followed by the fraction of time each foot touches
    the ground. With a sensor damage the robot sees, and records, the corrupted observation.
    """
    cfg = load_yaml("robots")[robot]
    n_state = cfg["joint_obs"]["velocity"] + len(sum(cfg["limbs"].values(), []))

    def extract(data, mask):
        valid = jnp.expand_dims(1.0 - mask, -1)
        count = jnp.sum(valid, axis=1)
        obs = data.obs[..., :n_state]
        return jnp.concatenate(
            [
                jnp.sum(obs * valid, 1) / count,
                jnp.sum(jnp.abs(obs) * valid, 1) / count,
                jnp.sum(data.state_desc * valid, 1) / count,
            ],
            axis=-1,
        )

    return extract


def _physics_key(physics) -> str:
    return json.dumps(physics, sort_keys=True)


def play_conditions(
    common: dict, robot: str, params, n_policies: int, key, episodes_override=None, signatures: bool = False
):
    """Fitness (conditions, n_policies, max episodes) of a stacked pytree of policies under every condition, and
    with `signatures` the per-episode signatures (conditions, n_policies, max episodes, d) as float16.

    Conditions sharing the same physics share one compiled function.
    """
    env = make_env(common)
    conds = build_conditions(robot, env.action_size, env.observation_size)
    n_eps = [episodes_override or c["n_episodes"] for c in conds]
    fitness = np.full((len(conds), n_policies, max(n_eps)), np.nan, dtype=np.float32)
    sigs = None
    extractor = signature_extractor(robot) if signatures else None
    t0 = time.time()
    for physics_key in dict.fromkeys(_physics_key(c["physics"]) for c in conds):
        physics = json.loads(physics_key)
        _, _, evaluate = make_condition_evaluator(common, physics, extractor)
        for i, c in enumerate(conds):
            if _physics_key(c["physics"]) != physics_key:
                continue
            key, subkey = jax.random.split(key)
            args = (jnp.asarray(c["scale"]), jnp.asarray(c["offset"]), jnp.asarray(c["obs_mask"]))
            fitness[i, :, : n_eps[i]], d = evaluate_all(evaluate, params, args, subkey, n_eps[i], per_episode=True)
            if signatures:
                if sigs is None:
                    sigs = np.full(fitness.shape + (d.shape[-1],), np.nan, dtype=np.float16)
                sigs[i, :, : n_eps[i]] = d
        print(f"[bank] physics {physics_key[:60]}: done ({time.time() - t0:.0f} s)", flush=True)
    return (conds, fitness, sigs) if signatures else (conds, fitness)


def _conditions_json(conds) -> list:
    return [
        {
            "name": c["name"],
            "group": c["group"],
            "scale": c["scale"].tolist(),
            "offset": c["offset"].tolist(),
            "masked_obs": np.flatnonzero(c["obs_mask"] == 0).tolist(),
            "physics": c["physics"],
            "n_episodes": c["n_episodes"],
        }
        for c in conds
    ]


def run_bank(run_dir: Path, out_root: Path = OUT, seed: int = 2027, signatures: bool = False) -> Path:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    robot = run.get("robot", "ant")
    common = run_config(run)
    t0 = time.time()
    key = jax.random.key(seed)

    env, policy_network, evaluate = make_condition_evaluator(common)
    params, filled, data = load_genotypes(run_dir, policy_network, env.observation_size)
    n_cells = data["fitnesses"].shape[0]
    played = play_conditions(common, robot, params, len(filled), key, signatures=signatures)
    conds, fit_filled = played[:2]
    fitness = np.full((len(conds), n_cells, fit_filled.shape[2]), np.nan, dtype=np.float32)
    fitness[:, filled] = fit_filled

    # Standing robot (zero action): a network with zero weights outputs tanh(0) = 0.
    ones, zeros = jnp.ones(env.action_size), jnp.zeros(env.action_size)
    obs_ones = jnp.ones(env.observation_size)
    zero_params = jax.tree.map(lambda x: jnp.zeros((16,) + x.shape[1:], x.dtype), params)
    f_still, _ = evaluate(zero_params, jax.random.key(seed + 1), ones, zeros, obs_ones)

    # Joint usage on the intact robot (2 episodes), the alternative ITE descriptor.
    _, _, eval_usage = make_condition_evaluator(common, extractor=mean_abs_action)
    _, usage = evaluate_all(eval_usage, params, (ones, zeros, obs_ones), jax.random.key(seed + 2), 2)
    joint_usage = np.full((n_cells, env.action_size), np.nan, dtype=np.float32)
    joint_usage[filled] = usage

    out_dir = Path(out_root) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "bank.npz",
        names=np.array([c["name"] for c in conds]),
        groups=np.array([c["group"] for c in conds]),
        fitness=fitness,
        f0=float(np.mean(f_still)),
        repertoire_fitness=data["fitnesses"],
        descriptors=data["descriptors"],
        joint_usage=joint_usage,
    )
    (out_dir / "conditions.json").write_text(json.dumps(_conditions_json(conds)))
    if signatures:
        sig = played[2]  # (conditions, filled cells, episodes, d)
        groups = np.array([c["group"] for c in conds])
        test = np.flatnonzero(~np.isin(groups, ["intact", "train"]))
        np.savez_compressed(
            out_dir / "signatures.npz",
            intact=sig[groups == "intact"][0],  # (cells, episodes, d): the noise model comes from the intact robot
            train=np.nanmean(sig[groups == "train"].astype(np.float32), axis=2),
            test=sig[test],
            test_index=test,
            filled=filled,
        )
    (out_dir / "bank.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "robot": robot,
                "algo": run["algo"],
                "seed": run["seed"],
                "profile": run["profile"],
                "code_commit": git_commit(),
                "devices": [str(d) for d in jax.devices()],
                "f0": float(np.mean(f_still)),
                "time_s": time.time() - t0,
            },
            indent=2,
        )
    )
    print(f"[bank {run_dir.name}] done in {time.time() - t0:.0f} s", flush=True)
    return out_dir


def run_bank_robust(robust_dir: Path, out_root: Path = OUT, n_episodes: int = 32, seed: int = 2028) -> Path:
    """Zero-shot performance of a damage-robust TD3 policy under every condition."""
    robust_dir = Path(robust_dir)
    info = json.loads((robust_dir / "rl.json").read_text())
    robot = info.get("robot", "ant")
    common = load_run_config("me", None, robot)["common"]
    env = make_env(common)
    policy_network = make_policy_network(common, env.action_size)
    _, unravel = ravel_pytree(policy_network.init(jax.random.key(0), jnp.zeros((env.observation_size,))))
    params = unravel(jnp.asarray(np.load(robust_dir / "policy.npz")["params"]))
    copies = jax.tree.map(lambda x: jnp.repeat(x[None], n_episodes, axis=0), params)
    t0 = time.time()
    # n_episodes copies played once each: one episode per copy, different random starts.
    conds, fitness = play_conditions(common, robot, copies, n_episodes, jax.random.key(seed), episodes_override=1)
    out_dir = Path(out_root) / f"robust_{robot}_{robust_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "bank.npz",
        names=np.array([c["name"] for c in conds]),
        groups=np.array([c["group"] for c in conds]),
        fitness=fitness[:, :, 0],  # (conditions, episodes)
    )
    (out_dir / "bank.json").write_text(
        json.dumps(
            {
                "robust_dir": str(robust_dir),
                "robot": robot,
                "seed": info["seed"],
                "n_episodes": n_episodes,
                "code_commit": git_commit(),
                "time_s": time.time() - t0,
            },
            indent=2,
        )
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--robust", nargs="*", default=[], help="folders of damage-robust TD3 policies")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--signatures", action="store_true", help="also record proprioceptive signatures")
    args = parser.parse_args()
    for run_dir in args.runs:
        print(run_bank(run_dir, args.out, signatures=args.signatures))
    for robust_dir in args.robust:
        print(run_bank_robust(robust_dir, args.out))


if __name__ == "__main__":
    main()
