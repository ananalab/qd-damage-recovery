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
            ref_env.observation_size,
            ref_env.action_size,
            ref_env.descriptor_length,
        ), algo
        assert net.layer_sizes == ref_net.layer_sizes, algo


def test_same_evaluations_per_iteration(setups):
    for algo, (cfg, _, _, map_elites) in setups.items():
        assert map_elites._emitter.batch_size == cfg["common"]["budget"]["batch_size"], algo


