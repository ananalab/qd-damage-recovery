"""Shared constants and loading of the campaign results (results/ only)."""

import glob
import json

import pandas as pd

from qd_damage.repertoires import REPO_ROOT

RES = REPO_ROOT / "results"
ALGOS = ["me", "pgame", "dcrlme"]
ALGO_LABELS = {"me": "MAP-Elites", "pgame": "PGA-ME", "dcrlme": "DCRL-ME"}
TAG = {"me": "ME", "pgame": "PGA", "dcrlme": "DCRL"}
WORD = {1: "One", 3: "Three", 5: "Five", 10: "Ten", 20: "Twenty"}
METHOD_EN = {"ite": "ITE", "top_k": "top-$k$ (no GP)", "random": "random", "best_intact": "best-intact"}
UNIT = ["algo", "run", "scenario"]  # statistical unit: one (repertoire, damage) pair

