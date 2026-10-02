"""ant_uni environment, damage wrapper and damage scenarios.

Uses `qdax.tasks.brax.v1`, like the three official QDax 0.5.0 notebooks.
"""

import jax.numpy as jnp
import numpy as np
import qdax.tasks.brax.v1 as qdax_envs
from qdax.tasks.brax.v1.wrappers.reward_wrappers import ClipRewardWrapper, OffsetRewardWrapper

from qd_damage.config import load_yaml

# (hip, ankle) actions of each leg. Checked from the actuator order in the Brax config, from the motion
# produced by each action alone, and on video (media/checks/leg_mapping/).
# The robot walks towards +x; leg 1 front left, 2 hind left, 3 hind right, 4 front right.
LEG_ACTIONS = {
    "leg_1": (0, 1),
    "leg_2": (2, 3),
    "leg_3": (4, 5),
    "leg_4": (6, 7),  # the leg damaged by Chalumeau et al. (ICLR 2023)
}


def make_env(common: dict):
    """Environment shared by the three algorithms, built from common.yaml only."""
    name = common["env"]["name"]
    env = qdax_envs.create(name, episode_length=common["env"]["episode_length"])
    if common["env"]["positive_rewards"]:
        # As in the DCRL-ME notebook: shift the reward, then clip it at 0, so that the descriptor-conditioned
        # critic sees positive rewards. Applied to all three algorithms so that fitness means the same thing.
        env = OffsetRewardWrapper(env, offset=qdax_envs.reward_offset[name])
        env = ClipRewardWrapper(env, clip_min=0.0)
    return env


def descriptor_extractor(common: dict):
    return qdax_envs.descriptor_extractor[common["env"]["name"]]


class DamageWrapper:
    """Damage: each action is multiplied by a factor (0 = paralysed joint, 0.5 = weakened).

    For Ant, actuator torque is action x strength, so this is the same as scaling the actuator strength.
    Observations, reward and descriptor are unchanged.
    """

    def __init__(self, env, action_scale):
        self.env = env
        self.action_scale = jnp.asarray(action_scale, dtype=jnp.float32)

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state, action):
        return self.env.step(state, action * self.action_scale)

    def __getattr__(self, name):
        return getattr(self.env, name)


def load_damages() -> dict:
    return load_yaml("damages")


def get_scenario(name: str, damages: dict) -> dict:
    return next(s for s in damages["scenarios"] if s["name"] == name)


def damage_scale(scenario: dict, damages: dict, action_size: int = 8) -> np.ndarray:
    """Per-action factors for a scenario of configs/damages.yaml."""
    scale = np.ones(action_size, dtype=np.float32)
    for joint in scenario["joints"]:
        leg, part = joint.split(".")
        scale[damages["legs"][leg][part]] = scenario["factor"]
    return scale
