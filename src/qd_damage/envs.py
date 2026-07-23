"""ant_uni environment, damage wrapper and damage scenarios.

Uses `qdax.tasks.brax.v1`, like the three official QDax 0.5.0 notebooks.
"""

import copy

import jax.numpy as jnp
import numpy as np
import qdax.tasks.brax.v1 as qdax_envs
from qdax.tasks.brax.v1.wrappers.locomotion_wrappers import QDSystem
from qdax.tasks.brax.v1.wrappers.reward_wrappers import ClipRewardWrapper, OffsetRewardWrapper

from qd_damage.config import load_yaml

def make_env(common: dict, physics: dict | None = None):
    """Environment shared by the three algorithms, built from common.yaml only.

    `physics` optionally changes the simulator (e.g. {"friction": 0.7, "torso_mass": 12.0}); it stands for the
    gap between the simulator used to build the repertoire and the robot it is deployed on.
    """
    name = common["env"]["name"]
    env = qdax_envs.create(name, episode_length=common["env"]["episode_length"])
    if physics:
        _set_physics(env, physics)
    if common["env"]["positive_rewards"]:
        # As in the DCRL-ME notebook: shift the reward, then clip it at 0, so that the descriptor-conditioned
        # critic sees positive rewards. Applied to all three algorithms so that fitness means the same thing.
        env = OffsetRewardWrapper(env, offset=qdax_envs.reward_offset[name])
        env = ClipRewardWrapper(env, clip_min=0.0)
    return env


