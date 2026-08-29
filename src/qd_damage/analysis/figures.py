"""Figures of the analysis, in French (portfolio) and English (report)."""

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qd_damage.analysis.data import ALGO_LABELS, ALGOS, RL_SCENARIOS, SCENARIO_EN, SCENARIO_FR  # noqa: E402

ALGO_COLORS = {"me": "#2a78d6", "pgame": "#e07b39", "dcrlme": "#3a9d5d"}
RL_STYLE = {  # line style and colour of each TD3 variant
    "scratch": (":", "#7a5cc4"),
    "finetune": ("-", "#e07b39"),
    "finetune_warmup": ("--", "#c8413b"),
    "finetune_pretrained": ("-.", "#2a78d6"),
}
METHOD_COLORS = {"ite": "#3a9d5d", "top_k": "#7a5cc4", "random": "#8a8a85", "best_intact": "#c8413b"}
# Figure texts: French figures for the portfolio and the French report, English ones for the English reports.
TEXT = {
    "fr": {
        "qd": "QD-score",
        "cov": "Couverture (%)",
        "maxfit": "Meilleure fitness",
        "evals": "Évaluations (épisodes)",
        "res_y": "Résilience (% du progrès intact\nencore atteignable après la blessure)",
        "res_med": "Résilience médiane (%)",
        "trials_log": "Essais (épisodes) sur le robot blessé — échelle log",
        "perf_y": "Performance retrouvée\n(% de la meilleure démarche intacte)",
        "rl_scratch": "RL (TD3) de zéro",
        "rl_finetune": "RL (TD3) depuis la politique intacte",
        "trials": "Essais sur le robot blessé",
        "ite_on": "ITE sur",
        "methods": {"ite": "ITE", "top_k": "top-k (sans GP)", "random": "aléatoire", "best_intact": "meilleur-intact"},
        "scenarios": SCENARIO_FR,
        "episodes": "Épisodes sur le robot blessé",
        "perf": "Performance $v$\n(1 ≈ 15 m en 12,5 s)",
        "scratch": "TD3 de zéro",
        "finetune": "TD3 depuis la politique intacte",
        "finetune_warmup": "idem, échauffement du critique",
        "finetune_pretrained": "idem, critique pré-entraîné",
        "ite": "ITE (DCRL-ME, même seed)",
        "grid": "ITE sur la grille\n(cases du répertoire)",
        "cont": "ITE sur l'acteur\n(descripteurs continus)",
        "oracle": "meilleure démarche disponible (oracle)",
    },
    "en": {
        "qd": "QD-score",
        "cov": "Coverage (%)",
        "maxfit": "Max fitness",
        "evals": "Evaluations (episodes)",
        "res_y": "Resilience (% of intact progress\nstill reachable after damage)",
        "res_med": "Median resilience (%)",
        "trials_log": "Trials (episodes) on the damaged robot, log scale",
        "perf_y": "Recovered performance\n(% of best intact gait)",
        "rl_scratch": "RL (TD3) from scratch",
        "rl_finetune": "RL (TD3) from intact policy",
        "trials": "Trials on the damaged robot",
        "ite_on": "ITE on",
        "methods": {"ite": "ITE", "top_k": "top-k (no GP)", "random": "random", "best_intact": "best-intact"},
        "scenarios": SCENARIO_EN,
        "episodes": "Episodes on the damaged robot",
        "perf": "Performance $v$\n(1 ≈ 15 m in 12.5 s)",
        "scratch": "TD3 from scratch",
        "finetune": "TD3 from intact policy",
        "finetune_warmup": "same, critic warm-up",
        "finetune_pretrained": "same, pre-trained critic",
        "ite": "ITE (DCRL-ME, same seed)",
        "grid": "ITE on the grid\n(repertoire cells)",
        "cont": "ITE on the actor\n(continuous descriptors)",
        "oracle": "best available gait (oracle)",
    },
}


