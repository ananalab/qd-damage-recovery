"""RL re-training helpers: random damages for the robust policy, and the frozen-actor warm-up."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qd_damage.config import load_run_config
from qd_damage.envs import make_env
from qd_damage.rl_retrain import (
    RandomDamageWrapper,
    _BatchedEnv,
    load_rl_config,
    make_td3,
    make_train_round,
    new_buffer,
)


