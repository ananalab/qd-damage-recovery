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

from qd_damage.config import load_run_config, load_yaml, performance_reference
from qd_damage.envs import DamageWrapper, damage_scale, descriptor_extractor, load_damages, make_env
from qd_damage.repertoires import REPO_ROOT, git_commit, make_play_step_fn, make_policy_network

CHUNK = 512  # policies evaluated at once (memory)


def make_evaluator(common: dict, policy_network=None, physics: dict | None = None, extractor=None):
    """Return (env, network, evaluate) with evaluate(params_batch, key, scale) -> (fitnesses, descriptors).

    The damage vector `scale` is an argument of the compiled function, so all damages share one compilation.
    `policy_network` defaults to the repertoire MLP; any object with `apply(params, obs)` works.
    `physics` perturbs the simulator (see envs.make_env); `extractor` replaces the feet-contact descriptor.
    """
    base_env = make_env(common, physics)
    policy_network = policy_network or make_policy_network(common, base_env.action_size)
    extractor = extractor or descriptor_extractor(common)
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


