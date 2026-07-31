"""Setup: pinned dependencies and consistent configs."""

from importlib.metadata import version

import pytest

from qd_damage.config import ALGOS, load_run_config, load_yaml


def test_qdax_version_is_pinned():
    assert version("qdax") == "0.5.0"


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
