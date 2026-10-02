"""RL re-training (TD3) on the damaged robot, from scratch or from the best intact policy of a repertoire.

Usage:
    python -m qd_damage.rl_retrain --scenario leg4_paralysed --variant finetune --seed 0 \
        --repertoire results/repertoires/dcrlme_seed0

Performance (deterministic evaluation, same fitness as the repertoires) is recorded against the number of
EPISODES played on the damaged robot, which is what costs on a real robot. In parallel simulation RL consumes
thousands of episodes in minutes, so compute time is not the right measure.

`n_envs` robots play in parallel; one round = `episode_length` steps = `n_envs` episodes.
Outputs in results/rl/<scenario>_<variant>_seed<seed>[_smoke]/: curve.csv and rl.json.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from qdax.baselines.td3 import TD3, TD3Config
from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition

from qd_damage.config import load_run_config, load_yaml
from qd_damage.envs import DamageWrapper, damage_scale, get_scenario, load_damages, make_env
from qd_damage.oracle import make_evaluator
from qd_damage.repertoires import REPO_ROOT, git_commit, load_policy_params


class _BatchedEnv:
    """n robots in parallel (vmap). QDax's AutoResetWrapper restarts every finished episode."""

    def __init__(self, env):
        self.env = env
        self.reset = jax.vmap(env.reset)
        self.step = jax.vmap(env.step)


def load_rl_config(smoke: bool) -> dict:
    cfg = load_yaml("rl")
    smoke_cfg = cfg.pop("smoke")
    if smoke:
        cfg.update(smoke_cfg)
    return cfg


def best_intact_cell(repertoire: Path, oracle_root: Path) -> tuple:
    """(cell, source) of the best intact policy: on the re-evaluated intact fitness when the oracle exists,
    as for ITE and the best-intact baseline, otherwise on the fitness stored in the repertoire."""
    oracle_file = Path(oracle_root) / Path(repertoire).name / "oracle.npz"
    if oracle_file.exists():
        oracle = np.load(oracle_file)
        intact = oracle["fitness"][list(oracle["scenarios"]).index("intact")].mean(-1)
        return int(np.nanargmax(intact)), "oracle"
    return int(np.nanargmax(np.load(Path(repertoire) / "repertoire.npz")["fitnesses"])), "repertoire"


def retrain(scenario: str, variant: str, seed: int, repertoire=None, smoke: bool = False,
            out_root: Path = REPO_ROOT / "results" / "rl", oracle_root: Path = REPO_ROOT / "results" / "oracle"):
    if variant not in ("scratch", "finetune"):
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "finetune" and repertoire is None:
        raise ValueError("--repertoire is required for finetune")
    cfg = load_rl_config(smoke)
    common = load_run_config("me", "smoke" if smoke else None)["common"]
    episode_length = common["env"]["episode_length"]
    damages = load_damages()
    scale = damage_scale(get_scenario(scenario, damages), damages)
    env = DamageWrapper(make_env(common), scale)
    venv = _BatchedEnv(env)

    t = cfg["td3"]
    td3 = TD3(
        TD3Config(
            episode_length=episode_length, batch_size=t["batch_size"], policy_delay=t["policy_delay"],
            soft_tau_update=t["soft_tau_update"], expl_noise=t["expl_noise"],
            critic_hidden_layer_size=tuple(t["critic_hidden_layer_size"]),
            policy_hidden_layer_size=tuple(common["policy"]["hidden_layer_sizes"]),  # same network as the repertoires
            critic_learning_rate=t["critic_learning_rate"], policy_learning_rate=t["policy_learning_rate"],
            discount=t["discount"], noise_clip=t["noise_clip"], policy_noise=t["policy_noise"],
            reward_scaling=t["reward_scaling"],
        ),
        action_size=env.action_size,
    )
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    state = td3.init(subkey, env.action_size, env.observation_size)

    _, policy_network, evaluate = make_evaluator(common)
    init_cell = init_source = None
    if variant == "finetune":
        # Same architecture (MLP (128, 128), tanh output): the repertoire weights load as they are.
        init_cell, init_source = best_intact_cell(repertoire, oracle_root)
        params = load_policy_params(repertoire, init_cell, policy_network, env.observation_size)
        assert jax.tree.structure(params) == jax.tree.structure(state.policy_params)
        state = state.replace(policy_params=params, target_policy_params=params)

    buffer = ReplayBuffer.init(
        buffer_size=cfg["buffer_size"],
        transition=Transition.init_dummy(env.observation_size, env.action_size),
    )
    key, subkey = jax.random.split(key)
    env_state = venv.reset(jax.random.split(subkey, cfg["n_envs"]))

    @jax.jit
    def train_round(env_state, state, buffer):
        """One round: episode_length steps for n_envs robots, updates_per_step gradient steps per step."""

        def one_step(carry, _):
            env_state, state, buffer = carry
            env_state, state, transitions = td3.play_step_fn(env_state, state, venv)
            buffer = buffer.insert(transitions)

            def one_update(c, _):
                s, b = c
                s, b, _ = td3.update(s, b)
                return (s, b), None

            (state, buffer), _ = jax.lax.scan(one_update, (state, buffer), (), length=cfg["updates_per_step"])
            return (env_state, state, buffer), None

        (env_state, state, buffer), _ = jax.lax.scan(one_step, (env_state, state, buffer), (), length=episode_length)
        return env_state, state, buffer

    def evaluate_policy(policy_params, key):
        batch = jax.tree.map(lambda x: jnp.repeat(x[None], cfg["eval_episodes"], axis=0), policy_params)
        fitnesses, _ = evaluate(batch, key, jnp.asarray(scale))
        return float(fitnesses.mean())

    rows, t0, episodes = [], time.time(), 0
    key, subkey = jax.random.split(key)
    rows.append({"episodes": 0, "eval_fitness": evaluate_policy(state.policy_params, subkey), "time_s": 0.0})
    while episodes < cfg["max_episodes"]:
        env_state, state, buffer = train_round(env_state, state, buffer)
        episodes += cfg["n_envs"]
        key, subkey = jax.random.split(key)
        rows.append({"episodes": episodes, "eval_fitness": evaluate_policy(state.policy_params, subkey),
                     "time_s": time.time() - t0})
        if len(rows) % 10 == 0 or episodes >= cfg["max_episodes"]:
            print(f"[rl {scenario} {variant} s{seed}] {episodes} episodes: {rows[-1]['eval_fitness']:.1f} "
                  f"({rows[-1]['time_s']:.0f} s)", flush=True)

    name = f"{scenario}_{variant}_seed{seed}" + ("_smoke" if smoke else "")
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "rl.json").write_text(json.dumps({
        "scenario": scenario, "variant": variant, "seed": seed, "repertoire": str(repertoire),
        "init_cell": init_cell, "init_cell_source": init_source,
        "config": cfg, "code_commit": git_commit(), "devices": [f"{d} ({d.device_kind})" for d in jax.devices()],
        "time_s": time.time() - t0,
    }, indent=2))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--variant", choices=["scratch", "finetune"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repertoire", type=Path, default=None, help="required for finetune")
    parser.add_argument("--oracle-root", type=Path, default=REPO_ROOT / "results" / "oracle")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "rl")
    args = parser.parse_args()
    print(retrain(args.scenario, args.variant, args.seed, args.repertoire, args.smoke, args.out, args.oracle_root))


if __name__ == "__main__":
    main()
