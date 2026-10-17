"""How much of a gait's resilience comes from feedback? Closed-loop play against open-loop replay.

Usage:
    python -m qd_damage.feedback --runs results/repertoires/dcrlme_seed0 ...

The policies of the repertoires are closed-loop: they read the robot's state at every step, so after a damage they
react to it. Cully et al. (2015) used open-loop gaits instead. For every gait and every test condition, from the SAME
initial state, we play:
- the closed-loop policy on the damaged robot;
- the open-loop replay, on the damaged robot, of the actions the policy produced on the intact robot.
Their difference is what feedback contributes after the damage. On the intact robot the replay reproduces the policy
exactly (the simulation is deterministic), which is checked and stored.

Outputs in results/feedback/<run>/feedback.npz: closed and open returns of shape (conditions, cells, episodes).
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qd_damage.bank import _physics_key
from qd_damage.conditions import build_conditions
from qd_damage.config import run_config
from qd_damage.envs import make_env
from qd_damage.oracle import load_genotypes
from qd_damage.repertoires import REPO_ROOT, git_commit, make_policy_network

OUT = REPO_ROOT / "results" / "feedback"
GROUPS = ("intact", "id", "ood_jammed", "ood_inverted", "ood_sensor", "ood_slippery", "ood_heavy")
CHUNK = 256


def make_rollouts(common: dict, physics=None):
    """jit(params_batch, keys, scale, offset, obs_mask) -> (closed returns, open returns), one episode per policy."""
    nominal = make_env(common)
    env = make_env(common, physics) if physics else nominal
    policy = make_policy_network(common, env.action_size)
    length = common["env"]["episode_length"]

    def episode_return(reset_env, step_fn, key, act_fn):
        state = reset_env.reset(key)

        def body(carry, t):
            s, alive, ret = carry
            a = act_fn(s, t)
            s2 = step_fn(s, a)
            ret = ret + alive * s2.reward
            alive = alive * (1.0 - s2.done)
            return (s2, alive, ret), a

        (_, _, ret), actions = jax.lax.scan(body, (state, 1.0, 0.0), jnp.arange(length))
        return ret, actions

    def one(params, key, scale, offset, obs_mask):
        # Actions of the policy on the intact robot (nominal physics), from this initial state.
        _, intact_actions = episode_return(nominal, nominal.step, key, lambda s, t: policy.apply(params, s.obs))

        def damaged_step(s, a):
            s2 = env.step(s, a * scale + offset)
            return s2.replace(obs=s2.obs * obs_mask)

        closed, _ = episode_return(env, damaged_step, key, lambda s, t: policy.apply(params, s.obs * obs_mask))
        opened, _ = episode_return(env, damaged_step, key, lambda s, t: intact_actions[t])
        return closed, opened

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None, None, None)))


def run_feedback(run_dir: Path, out_root: Path = OUT, n_episodes: int = 4, seed: int = 2029) -> Path:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    robot = run.get("robot", "ant")
    common = run_config(run)
    env = make_env(common)
    policy = make_policy_network(common, env.action_size)
    params, filled, data = load_genotypes(run_dir, policy, env.observation_size)
    conds = [c for c in build_conditions(robot, env.action_size, env.observation_size) if c["group"] in GROUPS]
    n = len(filled)
    pad = (-n) % CHUNK
    padded = jax.tree.map(lambda x: jnp.concatenate([x, jnp.repeat(x[:1], pad, axis=0)]), params)
    closed = np.full((len(conds), n, n_episodes), np.nan, np.float32)
    opened = np.full_like(closed, np.nan)
    t0 = time.time()
    for physics_key in dict.fromkeys(_physics_key(c["physics"]) for c in conds):
        rollouts = make_rollouts(common, json.loads(physics_key))
        for i, c in enumerate(conds):
            if _physics_key(c["physics"]) != physics_key:
                continue
            args = (jnp.asarray(c["scale"]), jnp.asarray(c["offset"]), jnp.asarray(c["obs_mask"]))
            for e in range(n_episodes):
                # Same initial states for every condition: episode e of cell j always starts from the same state.
                keys = jax.random.split(jax.random.key(seed + e), n + pad)
                cl, op = [], []
                for start in range(0, n + pad, CHUNK):
                    sl = slice(start, start + CHUNK)
                    a, b = rollouts(jax.tree.map(lambda x, sl=sl: x[sl], padded), keys[sl], *args)
                    cl.append(np.asarray(a))
                    op.append(np.asarray(b))
                closed[i, :, e] = np.concatenate(cl)[:n]
                opened[i, :, e] = np.concatenate(op)[:n]
        print(f"[feedback {run_dir.name}] physics {physics_key[:50]} done ({time.time() - t0:.0f} s)", flush=True)

    out_dir = Path(out_root) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "feedback.npz", names=np.array([c["name"] for c in conds]),
                        groups=np.array([c["group"] for c in conds]), closed=closed, open=opened, filled=filled)
    (out_dir / "feedback.json").write_text(json.dumps({
        "run_dir": str(run_dir), "robot": robot, "algo": run["algo"], "seed": run["seed"], "n_episodes": n_episodes,
        "intact_replay_max_abs_diff": float(np.nanmax(np.abs(closed[0] - opened[0]))),
        "code_commit": git_commit(), "time_s": time.time() - t0}, indent=2))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()
    for run_dir in args.runs:
        print(run_feedback(run_dir, args.out, args.episodes))


if __name__ == "__main__":
    main()
