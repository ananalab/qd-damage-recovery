"""Build a repertoire with MAP-Elites, PGA-ME or DCRL-ME, at equal budget.

Usage:
    python -m qd_damage.repertoires --algo me --seed 0 --profile smoke
    python -m qd_damage.repertoires --algo pgame --seed 0            # full run (GPU)

Everything that must be identical across algorithms (environment, network, repertoire, evaluations per
iteration) comes from common.yaml; only the emitter changes. The code follows the official QDax 0.5.0
notebooks (mapelites, pgame, dcrlme).

Outputs in <out>/<algo>_seed<seed>[_<profile>]/:
    repertoire.npz  flattened genotypes, fitnesses, descriptors, centroids
    metrics.csv     QD-score, coverage, max fitness and time, every log_period iterations
    run.json        config, seed, git commit, library versions, device
    actor.npz       DCRL-ME only: the descriptor-conditioned actor

A checkpoint is written every `budget.checkpoint_every` iterations and the run resumes from it automatically.
"""

import argparse
import csv
import functools
import json
import os
import pickle
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.dcrl_me_emitter import DCRLMEConfig, DCRLMEEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition, QDTransition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.tasks.brax.v1 import reward_offset
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics

from qd_damage.config import ALGOS, load_run_config
from qd_damage.envs import descriptor_extractor, make_env

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_policy_network(common: dict, action_size: int) -> MLP:
    """Same MLP for the three algorithms: hidden layers from common.yaml, tanh output."""
    return MLP(
        layer_sizes=tuple(common["policy"]["hidden_layer_sizes"]) + (action_size,),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )


def make_play_step_fn(env, policy_network, algo: str):
    """One simulation step. DCRL-ME also stores descriptor fields in its transitions."""

    def play_step_fn(env_state, policy_params, key):
        actions = policy_network.apply(policy_params, env_state.obs)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)
        common_fields = dict(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )
        if algo == "dcrlme":
            nan_desc = jnp.full((env.descriptor_length,), jnp.nan)
            transition = DCRLTransition(**common_fields, desc=nan_desc, desc_prime=nan_desc)
        else:
            transition = QDTransition(**common_fields)
        return next_state, policy_params, key, transition

    return play_step_fn


def make_emitter(algo: str, algo_cfg: dict, common: dict, env, policy_network):
    """The only part that differs between algorithms. Each emitter produces `batch_size` offspring."""
    batch_size = common["budget"]["batch_size"]
    em = algo_cfg["emitter"]
    variation_fn = functools.partial(isoline_variation, iso_sigma=em["iso_sigma"], line_sigma=em["line_sigma"])

    if algo == "me":
        return MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=batch_size,
        )

    if algo == "pgame":
        td3 = algo_cfg["td3"]
        config = PGAMEConfig(
            env_batch_size=batch_size,
            batch_size=td3["transitions_batch_size"],
            proportion_mutation_ga=em["proportion_mutation_ga"],
            critic_hidden_layer_size=tuple(td3["critic_hidden_layer_size"]),
            critic_learning_rate=td3["critic_learning_rate"],
            greedy_learning_rate=td3["greedy_learning_rate"],
            policy_learning_rate=td3["policy_learning_rate"],
            noise_clip=td3["noise_clip"],
            policy_noise=td3["policy_noise"],
            discount=td3["discount"],
            reward_scaling=td3["reward_scaling"],
            replay_buffer_size=td3["replay_buffer_size"],
            soft_tau_update=td3["soft_tau_update"],
            num_critic_training_steps=td3["num_critic_training_steps"],
            num_pg_training_steps=td3["num_pg_training_steps"],
            policy_delay=td3["policy_delay"],
        )
        return PGAMEEmitter(config=config, policy_network=policy_network, env=env, variation_fn=variation_fn)

    if algo == "dcrlme":
        dcrl = algo_cfg["dcrl"]
        ga_batch_size = int(round(em["ga_fraction"] * batch_size))
        dcrl_batch_size = int(round(em["dcrl_fraction"] * batch_size))
        config = DCRLMEConfig(
            ga_batch_size=ga_batch_size,
            dcrl_batch_size=dcrl_batch_size,
            ai_batch_size=batch_size - ga_batch_size - dcrl_batch_size,
            lengthscale=dcrl["lengthscale"],
            critic_hidden_layer_size=tuple(dcrl["critic_hidden_layer_size"]),
            num_critic_training_steps=dcrl["num_critic_training_steps"],
            num_pg_training_steps=dcrl["num_pg_training_steps"],
            batch_size=dcrl["transitions_batch_size"],
            replay_buffer_size=dcrl["replay_buffer_size"],
            discount=dcrl["discount"],
            reward_scaling=dcrl["reward_scaling"],
            critic_learning_rate=dcrl["critic_learning_rate"],
            actor_learning_rate=dcrl["actor_learning_rate"],
            policy_learning_rate=dcrl["policy_learning_rate"],
            noise_clip=dcrl["noise_clip"],
            policy_noise=dcrl["policy_noise"],
            soft_tau_update=dcrl["soft_tau_update"],
            policy_delay=dcrl["policy_delay"],
        )
        actor_dc_network = make_dc_actor_network(common, env.action_size)
        return DCRLMEEmitter(
            config=config,
            policy_network=policy_network,
            actor_network=actor_dc_network,
            env=env,
            variation_fn=variation_fn,
        )

    raise ValueError(f"unknown algo {algo!r}")


def setup(algo: str, profile=None):
    """Build environment, network and MAP-Elites for one algorithm; also return the config used."""
    cfg = load_run_config(algo, profile)
    common = cfg["common"]
    env = make_env(common)
    policy_network = make_policy_network(common, env.action_size)

    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=common["env"]["episode_length"],
        play_reset_fn=jax.jit(env.reset),
        play_step_fn=make_play_step_fn(env, policy_network, algo),
        descriptor_extractor=descriptor_extractor(common),
    )
    # With positive_rewards the fitness is already >= 0, so the QD-score needs no offset.
    qd_offset = (
        0.0
        if common["env"]["positive_rewards"]
        else reward_offset[common["env"]["name"]] * common["env"]["episode_length"]
    )
    emitter = make_emitter(algo, cfg["algo"], common, env, policy_network)
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=functools.partial(default_qd_metrics, qd_offset=qd_offset),
    )
    return cfg, env, policy_network, map_elites


def git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        return commit.strip() + ("-dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Outside a git checkout (e.g. on Kaggle) the launcher passes the commit.
        return os.environ.get("QD_DAMAGE_COMMIT", "unknown")


class _Key:
    """Picklable wrapper for a JAX random key (typed keys do not go through np.asarray)."""

    def __init__(self, data):
        self.data = data


