"""Setup: pinned dependencies and consistent configs."""

from importlib.metadata import version

import pytest

from qd_damage.config import ALGOS, load_run_config, load_yaml


def test_qdax_version_is_pinned():
    assert version("qdax") == "0.5.0"


def test_jax_brax_qdax_import():
    import brax  # noqa: F401
    import jax
    import qdax  # noqa: F401

    # Without a GPU, JAX must at least see the CPU.
    assert len(jax.devices()) >= 1


def test_ant_uni_env_exists():
    # The official MAP-Elites, PGA-ME and DCRL-ME notebooks of QDax 0.5.0 all use the v1 module.
    from qdax.tasks.brax import v1 as qdax_envs

    assert "ant_uni" in qdax_envs.descriptor_extractor
    env = qdax_envs.create("ant_uni", episode_length=10)
    # Ant: 8 joints (4 legs x hip + ankle), hence 8 actions.
    assert env.action_size == 8


def test_algo_configs_do_not_override_common_keys():
    """Environment, network and budget come from common.yaml only."""
    common_keys = set(load_yaml("common"))
    for algo in ALGOS:
        algo_keys = set(load_yaml(f"algos/{algo}"))
        assert not (algo_keys & common_keys), f"{algo} redefines {algo_keys & common_keys}"


@pytest.mark.parametrize("algo", ALGOS)
def test_smoke_config_is_complete(algo):
    cfg = load_run_config(algo, profile="smoke")["common"]
    assert cfg["env"]["name"] == "ant_uni"
    assert cfg["env"]["episode_length"] > 0
    assert cfg["budget"]["num_iterations"] % cfg["budget"]["log_period"] == 0
    assert cfg["budget"]["batch_size"] > 0


def test_damage_scenarios_are_unique_and_valid():
    damages = load_yaml("damages")
    names = [s["name"] for s in damages["scenarios"]]
    assert len(names) == len(set(names))
    for s in damages["scenarios"]:
        assert 0.0 <= s["factor"] <= 1.0
        for joint in s["joints"]:
            leg, part = joint.split(".")
            assert part in damages["legs"][leg]
    assert set(damages["rl_subset"]) <= set(names)
