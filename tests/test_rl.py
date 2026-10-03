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


def test_random_damage_is_redrawn_only_when_an_episode_ends():
    common = load_run_config("me", "smoke")["common"]  # 20-step episodes
    env = RandomDamageWrapper(make_env(common), p_damaged=0.5)
    state = jax.jit(env.reset)(jax.random.key(0))
    step = jax.jit(env.step)
    scales = []
    for _ in range(45):
        state = step(state, jnp.zeros(env.action_size))
        scales.append(np.asarray(state.info["damage_scale"]))
    scales = np.array(scales)
    assert set(np.unique(scales)) <= {0.0, 0.5, 1.0}
    # Constant inside an episode, new draw after steps 20 and 40.
    assert (scales[:19] == scales[0]).all() and (scales[20:39] == scales[20]).all()
    assert not (scales[0] == scales[20]).all() or not (scales[20] == scales[40]).all()


@pytest.mark.slow
def test_frozen_round_trains_the_critic_only():
    cfg = load_rl_config(smoke=True)
    common = load_run_config("me", "smoke")["common"]
    env = make_env(common)
    venv = _BatchedEnv(env)
    td3 = make_td3(cfg, common, env.action_size)
    state = td3.init(jax.random.key(0), env.action_size, env.observation_size)
    env_state = venv.reset(jax.random.split(jax.random.key(1), cfg["n_envs"]))
    frozen = make_train_round(td3, venv, cfg, common["env"]["episode_length"], freeze_actor=True)
    _, new_state, _ = frozen(env_state, state, new_buffer(cfg, env))
    same = jax.tree.map(lambda a, b: bool(jnp.all(a == b)), state.policy_params, new_state.policy_params)
    changed = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), state.critic_params, new_state.critic_params)
    assert all(jax.tree.leaves(same)) and any(jax.tree.leaves(changed))
