"""YAML configuration loading.

A run config is configs/common.yaml (shared by the three algorithms) plus configs/algos/<algo>.yaml.
They are kept in separate sections so that an algorithm cannot silently change the environment,
the network or the budget (see docs/protocol.md).
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
ALGOS = ("me", "pgame", "dcrlme")

