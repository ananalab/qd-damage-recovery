"""RL re-training (TD3) on the damaged robot, and a damage-robust TD3 policy.

Usage:
    python -m qd_damage.rl_retrain --scenario leg4_paralysed --variant finetune --seed 0 \
        --repertoire results/repertoires/dcrlme_seed0
    python -m qd_damage.rl_retrain --robust --seed 0

Variants trained on the damaged robot:
- scratch: TD3 from a random policy;
- finetune: TD3 from the best intact policy of the repertoire (best re-evaluated cell), with a new critic;
- finetune_warmup: same, but the actor is frozen during the first `warmup_episodes` so that the new critic learns
  before its gradient is used (these episodes are counted);
- finetune_pretrained: same, but the critic is first trained in simulation on the INTACT robot with the actor
  frozen (`pretrain_episodes`, not counted: no damaged robot is involved).

Performance (deterministic evaluation, same fitness as the repertoires) is recorded against the number of EPISODES
played on the damaged robot, which is what costs on a real robot. `n_envs` robots play in parallel; one round =
`episode_length` steps = `n_envs` episodes. Outputs in results/rl/<scenario>_<variant>_seed<seed>[_smoke]/.

The robust policy is trained in simulation with a random damage drawn for every episode (each joint independently
paralysed or weakened with probability `robust.p_damaged`), then evaluated without any adaptation on every damage
scenario: zero trials on the damaged robot. Outputs in results/rl_robust/seed<seed>[_smoke]/.
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
from qdax.baselines.td3 import TD3, TD3Config
from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition

from qd_damage.config import load_run_config, load_yaml
from qd_damage.envs import DamageWrapper, damage_scale, get_scenario, load_damages, make_env
from qd_damage.oracle import make_evaluator
from qd_damage.repertoires import REPO_ROOT, git_commit, load_policy_params

VARIANTS = ("scratch", "finetune", "finetune_warmup", "finetune_pretrained")


class _BatchedEnv:
    """n robots in parallel (vmap). QDax's AutoResetWrapper restarts every finished episode."""

    def __init__(self, env):
        self.env = env
        self.reset = jax.vmap(env.reset)
        self.step = jax.vmap(env.step)


class RandomDamageWrapper:
    """A new random damage at every episode: each joint is independently scaled by 0 or 0.5 with probability p.

    The damage lives in the state info, so it survives QDax's automatic resets and is redrawn when an episode ends.
    """

    def __init__(self, env, p_damaged: float):
        self.env = env
        self.p_damaged = p_damaged

    def _draw(self, key):
        k1, k2 = jax.random.split(key)
        damaged = jax.random.bernoulli(k1, self.p_damaged, (self.env.action_size,))
        level = jnp.where(jax.random.bernoulli(k2, 0.5, (self.env.action_size,)), 0.0, 0.5)
        return jnp.where(damaged, level, 1.0)

    def reset(self, rng):
        state = self.env.reset(rng)
        key, subkey = jax.random.split(jax.random.fold_in(rng, 1))
        state.info["damage_scale"] = self._draw(subkey)
        state.info["damage_key"] = key
        return state

    def step(self, state, action):
        scale, key = state.info["damage_scale"], state.info["damage_key"]
        state = self.env.step(state, action * scale)
        key, subkey = jax.random.split(key)
        state.info["damage_scale"] = jnp.where(state.done, self._draw(subkey), scale)
        state.info["damage_key"] = key
        return state

    def __getattr__(self, name):
        return getattr(self.env, name)


def load_rl_config(smoke: bool) -> dict:
    cfg = load_yaml("rl")
    smoke_cfg = cfg.pop("smoke")
    if smoke:
        for key, value in smoke_cfg.items():
            if isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
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


def make_td3(cfg: dict, common: dict, action_size: int) -> TD3:
    t = cfg["td3"]
    return TD3(
        TD3Config(
            episode_length=common["env"]["episode_length"],
            batch_size=t["batch_size"],
            policy_delay=t["policy_delay"],
            soft_tau_update=t["soft_tau_update"],
            expl_noise=t["expl_noise"],
            critic_hidden_layer_size=tuple(t["critic_hidden_layer_size"]),
            policy_hidden_layer_size=tuple(common["policy"]["hidden_layer_sizes"]),  # same network as the repertoires
            critic_learning_rate=t["critic_learning_rate"],
            policy_learning_rate=t["policy_learning_rate"],
            discount=t["discount"],
            noise_clip=t["noise_clip"],
            policy_noise=t["policy_noise"],
            reward_scaling=t["reward_scaling"],
        ),
        action_size=action_size,
    )


def make_train_round(td3: TD3, venv: _BatchedEnv, cfg: dict, episode_length: int, freeze_actor: bool = False):
    """Compiled round: episode_length steps for n_envs robots, updates_per_step gradient steps per step.

    With freeze_actor, only the critic learns: the policy, its target and its optimiser state are kept.
    """

    def one_update(carry, _):
        state, buffer = carry
        new_state, buffer, _ = td3.update(state, buffer)
        if freeze_actor:
            new_state = new_state.replace(
                policy_params=state.policy_params,
                target_policy_params=state.target_policy_params,
                policy_optimizer_state=state.policy_optimizer_state,
            )
        return (new_state, buffer), None

    def one_step(carry, _):
        env_state, state, buffer = carry
        env_state, state, transitions = td3.play_step_fn(env_state, state, venv)
        buffer = buffer.insert(transitions)
        (state, buffer), _ = jax.lax.scan(one_update, (state, buffer), (), length=cfg["updates_per_step"])
        return (env_state, state, buffer), None

    @jax.jit
    def train_round(env_state, state, buffer):
        (env_state, state, buffer), _ = jax.lax.scan(one_step, (env_state, state, buffer), (), length=episode_length)
        return env_state, state, buffer

    return train_round


def new_buffer(cfg: dict, env) -> ReplayBuffer:
    return ReplayBuffer.init(
        buffer_size=cfg["buffer_size"], transition=Transition.init_dummy(env.observation_size, env.action_size)
    )


