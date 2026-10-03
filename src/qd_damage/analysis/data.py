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


def units(df: pd.DataFrame, value: str) -> pd.Series:
    """Mean of `value` over the repetitions of each (repertoire, damage) pair."""
    return df.groupby(UNIT)[value].mean()


def load(runs_glob: str):
    runs = sorted(p.parent for p in (RES / "repertoires").glob(f"{runs_glob}/run.json"))
    if not runs:
        raise SystemExit(f"no run matches results/repertoires/{runs_glob}")
    meta = {r.name: json.loads((r / "run.json").read_text()) for r in runs}
    quality = pd.DataFrame(
        [
            {
                "run": r.name,
                "algo": meta[r.name]["algo"],
                "seed": meta[r.name]["seed"],
                **pd.read_csv(r / "metrics.csv")
                .iloc[-1][["evaluations", "qd_score", "coverage", "max_fitness"]]
                .to_dict(),
            }
            for r in runs
        ]
    )
    curves = pd.concat([pd.read_csv(r / "metrics.csv").assign(run=r.name, algo=meta[r.name]["algo"]) for r in runs])

    def gather(kind, name):
        return pd.concat(
            [pd.read_csv(RES / kind / r.name / name) for r in runs if (RES / kind / r.name / name).exists()]
        )

    oracle, adapt, trials = (
        gather("oracle", "summary.csv"),
        gather("adaptation", "summary.csv"),
        gather("adaptation", "trials.csv"),
    )
    rl = []
    for p in (RES / "rl").glob("*/curve.csv"):
        info = json.loads((p.parent / "rl.json").read_text())
        if info["config"]["max_episodes"] >= 100:  # skip smoke runs
            rl.append(pd.read_csv(p).assign(scenario=info["scenario"], variant=info["variant"], seed=info["seed"]))
    rl = pd.concat(rl) if rl else pd.DataFrame()
    return meta, quality, curves, oracle, adapt, trials, rl


def load_bonus(f0: float, scale: float):
    """Grid ITE, continuous ITE and oracle summaries of the dedicated DCRL-ME runs (None if absent)."""

    def read(kind):
        files = glob.glob(str(RES / "bonus" / kind / "*" / "summary.csv"))
        return pd.concat(map(pd.read_csv, files)) if files else None

    grid, cont, oracle = read("adaptation"), read("continuous"), read("oracle")
    if grid is None or cont is None or oracle is None:
        return None
    grid = grid[grid.method == "ite"].assign(perf_abs=lambda d: (d.recommended_true - f0) / scale)
    oracle = oracle[oracle.scenario != "intact"].assign(oracle_abs=lambda d: (d.oracle_fitness - f0) / scale)
    # Unit = (run, damage): repetitions are averaged, then grid and continuous are paired on the same unit.
    key = ["run", "scenario"]
    pairs = (
        grid.groupby(key)
        .perf_abs.mean()
        .rename("grid")
        .to_frame()
        .join(cont.groupby(key).perf_abs.mean().rename("cont"))
    )
    oracles = cont.groupby(key).oracle_continuous_abs.first().to_frame().join(oracle.set_index(key).oracle_abs)
    return grid, cont, pairs, oracles
