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

# Damage names. Legs: 1 front left, 2 hind left, 3 hind right, 4 front right.
SCENARIO_EN = {
    "leg1_paralysed": "FL leg paralysed",
    "leg2_paralysed": "HL leg paralysed",
    "leg3_paralysed": "HR leg paralysed",
    "leg4_paralysed": "FR leg paralysed",
    "leg1_weak": "FL leg weakened",
    "leg2_weak": "HL leg weakened",
    "leg3_weak": "HR leg weakened",
    "leg4_weak": "FR leg weakened",
    "legs12_paralysed": "FL + HL legs paralysed",
    "legs14_paralysed": "FL + FR legs paralysed",
    "ankle1_paralysed": "FL ankle paralysed",
    "ankle3_paralysed": "HR ankle paralysed",
    "intact": "intact",
}
SCENARIO_FR = {
    "leg1_paralysed": "patte 1 (AvG) paralysée",
    "leg2_paralysed": "patte 2 (ArG) paralysée",
    "leg3_paralysed": "patte 3 (ArD) paralysée",
    "leg4_paralysed": "patte 4 (AvD) paralysée",
    "leg1_weak": "patte 1 affaiblie",
    "leg2_weak": "patte 2 affaiblie",
    "leg3_weak": "patte 3 affaiblie",
    "leg4_weak": "patte 4 affaiblie",
    "legs12_paralysed": "pattes 1+2 paralysées",
    "legs14_paralysed": "pattes 1+4 paralysées",
    "ankle1_paralysed": "cheville 1 paralysée",
    "ankle3_paralysed": "cheville 3 paralysée",
    "intact": "intact",
}

RL_SCENARIOS = ["leg4_paralysed", "leg2_weak", "legs14_paralysed", "ankle1_paralysed"]


