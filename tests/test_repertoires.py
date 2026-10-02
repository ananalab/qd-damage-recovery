"""The three algorithms share the environment, the network and the budget (docs/protocol.md)."""

import json

import numpy as np
import pytest

from qd_damage.config import ALGOS
from qd_damage.repertoires import build, setup


@pytest.fixture(scope="module")
def setups():
    return {algo: setup(algo, profile="smoke") for algo in ALGOS}


def test_same_env_and_network(setups):
    ref_cfg, ref_env, ref_net, _ = setups["me"]
    for algo, (cfg, env, net, _) in setups.items():
        assert cfg["common"] == ref_cfg["common"], algo
        assert (env.observation_size, env.action_size, env.descriptor_length) == (
            ref_env.observation_size, ref_env.action_size, ref_env.descriptor_length
        ), algo
        assert net.layer_sizes == ref_net.layer_sizes, algo


def test_same_evaluations_per_iteration(setups):
    for algo, (cfg, _, _, map_elites) in setups.items():
        assert map_elites._emitter.batch_size == cfg["common"]["budget"]["batch_size"], algo


@pytest.mark.slow
def test_smoke_build_writes_outputs(tmp_path):
    shapes = {}
    for algo in ALGOS:
        out = build(algo, seed=0, profile="smoke", out_root=tmp_path)
        data = np.load(out / "repertoire.npz")
        run = json.loads((out / "run.json").read_text())
        shapes[algo] = (data["genotypes"].shape, data["descriptors"].shape)
        assert run["versions"]["qdax"] == "0.5.0"
        assert np.isfinite(data["fitnesses"]).any()  # at least one filled cell
    # Same repertoire shape for the three algorithms.
    assert len(set(shapes.values())) == 1, shapes


@pytest.mark.slow
@pytest.mark.parametrize("algo", ALGOS)
def test_resume_from_checkpoint_gives_same_repertoire(tmp_path, algo):
    """A run interrupted then resumed from its checkpoint gives exactly the same repertoire as a straight run."""
    straight = build(algo, seed=1, profile="smoke", out_root=tmp_path / "straight")
    interrupted = build(algo, seed=1, profile="smoke", out_root=tmp_path / "resumed", stop_after=2)
    assert (interrupted / "checkpoint.pkl").exists()
    resumed = build(algo, seed=1, profile="smoke", out_root=tmp_path / "resumed")
    assert not (resumed / "checkpoint.pkl").exists()
    a, b = np.load(straight / "repertoire.npz"), np.load(resumed / "repertoire.npz")
    np.testing.assert_array_equal(a["fitnesses"], b["fitnesses"])
    np.testing.assert_array_equal(a["genotypes"], b["genotypes"])
    rows = (resumed / "metrics.csv").read_text().splitlines()
    assert len(rows) == 1 + 3  # header + iterations 0, 2, 4


@pytest.mark.slow
def test_dcrlme_saves_conditioned_actor(tmp_path):
    import jax.numpy as jnp

    from qd_damage.config import load_run_config
    from qd_damage.envs import make_env
    from qd_damage.repertoires import load_dc_actor

    out = build("dcrlme", seed=0, profile="smoke", out_root=tmp_path)
    common = load_run_config("dcrlme", "smoke")["common"]
    env = make_env(common)
    actor, params = load_dc_actor(out, common, env)
    a = actor.apply(params, jnp.zeros(env.observation_size), jnp.full(env.descriptor_length, 0.5))
    assert a.shape == (env.action_size,) and bool(jnp.all(jnp.abs(a) <= 1.0))
