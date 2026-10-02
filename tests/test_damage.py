"""Damages (DamageWrapper, scenarios), oracle and adaptation."""

import csv

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qd_damage.config import load_run_config
from qd_damage.envs import LEG_ACTIONS, DamageWrapper, damage_scale, load_damages, make_env


def test_leg_mapping_matches_config():
    damages = load_damages()
    for leg, (hip, ankle) in LEG_ACTIONS.items():
        assert (damages["legs"][leg]["hip"], damages["legs"][leg]["ankle"]) == (hip, ankle)


def test_damage_scale():
    damages = load_damages()
    by_name = {s["name"]: s for s in damages["scenarios"]}
    np.testing.assert_array_equal(damage_scale(by_name["intact"], damages), np.ones(8))
    np.testing.assert_array_equal(damage_scale(by_name["leg4_paralysed"], damages), [1, 1, 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(damage_scale(by_name["leg2_weak"], damages), [1, 1, .5, .5, 1, 1, 1, 1])
    np.testing.assert_array_equal(damage_scale(by_name["ankle1_paralysed"], damages), [1, 0, 1, 1, 1, 1, 1, 1])


def test_paralysed_joints_ignore_their_actions():
    """With leg 4 paralysed, changing actions 6-7 has no effect; changing action 0 does."""
    common = load_run_config("me", "smoke")["common"]
    env = DamageWrapper(make_env(common), [1, 1, 1, 1, 1, 1, 0, 0])
    state = jax.jit(env.reset)(jax.random.key(0))
    step = jax.jit(env.step)
    base = step(state, jnp.zeros(8)).qp.pos
    same = step(state, jnp.zeros(8).at[6].set(1.0).at[7].set(-1.0)).qp.pos
    moved = step(state, jnp.zeros(8).at[0].set(1.0)).qp.pos
    np.testing.assert_allclose(base, same)
    assert not np.allclose(base, moved)


@pytest.mark.slow
def test_oracle_outputs(tmp_path):
    from qd_damage.oracle import run_oracle
    from qd_damage.repertoires import build

    run_dir = build("me", seed=0, profile="smoke", out_root=tmp_path / "rep")
    out = run_oracle(run_dir, tmp_path / "oracle", n_episodes=2, scenarios=["intact", "leg4_paralysed"])
    data = np.load(out / "oracle.npz")
    n_cells = data["repertoire_fitness"].shape[0]
    assert data["fitness"].shape == (2, n_cells, 2)
    filled = np.isfinite(data["repertoire_fitness"])
    assert np.isfinite(data["fitness"][:, filled]).all() and np.isnan(data["fitness"][:, ~filled]).all()
    rows = list(csv.DictReader(open(out / "summary.csv")))
    assert [r["scenario"] for r in rows] == ["intact", "leg4_paralysed"]
    assert float(rows[0]["resilience"]) == pytest.approx(1.0)


@pytest.mark.slow
def test_adaptation_end_to_end(tmp_path):
    from qd_damage.adaptation import METHODS, run_adaptation
    from qd_damage.oracle import run_oracle
    from qd_damage.repertoires import build

    run_dir = build("me", seed=0, profile="smoke", out_root=tmp_path / "rep")
    run_oracle(run_dir, tmp_path / "oracle", n_episodes=1, scenarios=["intact", "leg4_paralysed"])
    out = run_adaptation(run_dir, tmp_path / "oracle", tmp_path / "adapt",
                         scenarios=["leg4_paralysed"], n_repeats=1)
    summary = list(csv.DictReader(open(out / "summary.csv")))
    assert sorted(r["method"] for r in summary) == sorted(METHODS)
    for r in summary:
        assert 1 <= int(r["trials_used"]) <= 20
        assert float(r["pct_oracle"]) <= 100.0 + 1e-6  # the recommendation cannot beat the oracle
    trials = list(csv.DictReader(open(out / "trials.csv")))
    assert max(int(t["trial"]) for t in trials if t["method"] == "best_intact") == 1


@pytest.mark.slow
def test_continuous_ite_end_to_end(tmp_path):
    from qd_damage.ite_continuous import run_continuous
    from qd_damage.repertoires import build

    run_dir = build("dcrlme", seed=0, profile="smoke", out_root=tmp_path / "rep")
    out = run_continuous(run_dir, n_candidates=64, n_episodes=1, scenarios=["leg4_paralysed"], n_repeats=1,
                         out_root=tmp_path / "cont")
    data = np.load(out / "candidates.npz")
    assert data["fitness"].shape == (2, 64, 1) and data["achieved"].shape == (64, 4)
    rows = list(csv.DictReader(open(out / "summary.csv")))
    assert len(rows) == 1 and 1 <= int(rows[0]["trials_used"]) <= 20
