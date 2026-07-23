"""Damages (DamageWrapper, scenarios), oracle and adaptation."""

import csv

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qd_damage.config import load_run_config
from qd_damage.envs import LEG_ACTIONS, DamageWrapper, damage_scale, load_damages, make_env


def test_damage_scale():
    damages = load_damages()
    by_name = {s["name"]: s for s in damages["scenarios"]}
    np.testing.assert_array_equal(damage_scale(by_name["intact"], damages), np.ones(8))
    np.testing.assert_array_equal(damage_scale(by_name["leg4_paralysed"], damages), [1, 1, 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(damage_scale(by_name["leg2_weak"], damages), [1, 1, 0.5, 0.5, 1, 1, 1, 1])
    np.testing.assert_array_equal(damage_scale(by_name["ankle1_paralysed"], damages), [1, 0, 1, 1, 1, 1, 1, 1])


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


