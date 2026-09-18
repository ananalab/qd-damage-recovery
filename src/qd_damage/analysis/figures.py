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


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})  # reproducible files
    plt.close(fig)


def _style(ax, grid_axis="both"):
    ax.grid(alpha=0.3, axis=grid_axis)
    ax.spines[["top", "right"]].set_visible(False)


def figures(curves, oracle, trials, rl, adapt, bonus, fig_dir: Path, f0: float, scale: float, lang: str):
    T = TEXT[lang]
    # 1) Repertoire progress (median and quartiles over seeds)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for algo in ALGOS:
        c = curves[curves.algo == algo]
        for ax, key in zip(axes, ["qd_score", "coverage", "max_fitness"], strict=False):
            g = c.groupby("evaluations")[key]
            ax.plot(g.median().index, g.median(), color=ALGO_COLORS[algo], lw=2, label=ALGO_LABELS[algo])
            ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color=ALGO_COLORS[algo], alpha=0.2)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    for ax, title in zip(axes, [T["qd"], T["cov"], T["maxfit"]], strict=False):
        ax.set_title(title)
        ax.set_xlabel(T["evals"])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
        _style(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, fig_dir / "repertoire_progress")

    # 2) Resilience by algorithm (one value per (seed, damage))
    res = oracle[oracle.scenario != "intact"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([res[res.algo == a].resilience * 100 for a in ALGOS], patch_artist=True, widths=0.55)
    for patch, a in zip(bp["boxes"], ALGOS, strict=False):
        patch.set_facecolor(ALGO_COLORS[a])
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, 4), [ALGO_LABELS[a] for a in ALGOS])
    ax.set_ylabel(T["res_y"])
    ax.axhline(100, color="#999", lw=1, ls="--")
    _style(ax, "y")
    _save(fig, fig_dir / "resilience_by_algo")

    # 3) Resilience by damage and algorithm (median over seeds)
    pivot = res.groupby(["scenario", "algo"]).resilience.median().unstack()[ALGOS] * 100
    pivot.index = [T["scenarios"].get(i, i) for i in pivot.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    w, x = 0.27, np.arange(len(pivot))
    for k, a in enumerate(ALGOS):
        ax.bar(x + (k - 1) * w, pivot[a], w, color=ALGO_COLORS[a], label=ALGO_LABELS[a])
    ax.set_xticks(x, pivot.index, rotation=35, ha="right")
    ax.set_ylabel(T["res_med"])
    ax.axhline(100, color="#999", lw=1, ls="--")
    ax.legend(frameon=False, ncol=3)
    _style(ax, "y")
    _save(fig, fig_dir / "resilience_by_damage")

    # 4) Performance against trials: adaptation methods (all repertoires) and RL (episodes), log axis
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in ["ite", "top_k", "random"]:
        g = trials[trials.method == m].groupby("trial").pct_intact
        ax.plot(g.median().index, g.median(), color=METHOD_COLORS[m], lw=2.2, label=T["methods"][m])
        ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color=METHOD_COLORS[m], alpha=0.15)
    bi = trials[trials.method == "best_intact"].pct_intact
    ax.scatter(
        [1], [bi.median()], color=METHOD_COLORS["best_intact"], zorder=5, s=40, label=T["methods"]["best_intact"]
    )
    if not rl.empty:
        ref = oracle[(oracle.scenario == "intact") & (oracle.algo == "dcrlme")].set_index("seed").oracle_fitness
        rl_pct = rl.assign(pct_intact=lambda d: 100 * (d.eval_fitness - f0) / (d.seed.map(ref) - f0))
        for v, ls in [("scratch", ":"), ("finetune", "--")]:
            g = rl_pct[(rl_pct.variant == v) & (rl_pct.episodes > 0)].groupby("episodes").pct_intact.median()
            ax.plot(g.index, g, color="#222", ls=ls, lw=1.8, label=T[f"rl_{v}"])
    ax.set_xscale("log")
    ax.set_xlabel(T["trials_log"])
    ax.set_ylabel(T["perf_y"])
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, fig_dir / "performance_vs_trials")

    # 5) ITE by repertoire algorithm
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in ALGOS:
        g = trials[(trials.method == "ite") & (trials.algo == a)].groupby("trial").pct_intact
        ax.plot(g.median().index, g.median(), color=ALGO_COLORS[a], lw=2.2, label=f"{T['ite_on']} {ALGO_LABELS[a]}")
        ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color=ALGO_COLORS[a], alpha=0.15)
    ax.set_xlabel(T["trials"])
    ax.set_ylabel(T["perf_y"])
    _style(ax)
    ax.legend(frameon=False)
    _save(fig, fig_dir / "ite_by_algo")

    # 6) RL curves, one panel per damage, against the level reached by ITE
    if not rl.empty:
        fig, axes = plt.subplots(1, len(RL_SCENARIOS), figsize=(4.0 * len(RL_SCENARIOS), 3.3))
        for ax, sc in zip(axes, RL_SCENARIOS, strict=False):
            for (v, seed), c in rl[rl.scenario == sc].groupby(["variant", "seed"]):
                ls, color = RL_STYLE[v]
                ax.plot(
                    c.episodes,
                    (c.eval_fitness - f0) / scale,
                    ls=ls,
                    color=color,
                    lw=1.4,
                    alpha=0.9,
                    label=T[v] if seed == 0 else None,
                )
            for seed, color in [(0, "#3a9d5d"), (1, "#2a6f3f")]:
                ite = adapt[
                    (adapt.method == "ite") & (adapt.algo == "dcrlme") & (adapt.seed == seed) & (adapt.scenario == sc)
                ]
                if len(ite):
                    ax.axhline(
                        (ite.recommended_true.mean() - f0) / scale,
                        color=color,
                        lw=1.4,
                        ls="--",
                        label=T["ite"] if seed == 0 else None,
                    )
            ax.set_title(T["scenarios"][sc], fontsize=10)
            ax.set_xlabel(T["episodes"])
            _style(ax)
        axes[0].set_ylabel(T["perf"])
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        _save(fig, fig_dir / "rl_curves")

    # 7) Bonus: grid ITE against continuous ITE (one point per (run, damage))
    if bonus is not None:
        _, _, pairs, oracles = bonus
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        bp = ax.boxplot([pairs.grid, pairs.cont], patch_artist=True, widths=0.5)
        for patch, col in zip(bp["boxes"], ["#3a9d5d", "#7a5cc4"], strict=False):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        ax.scatter([1] * len(oracles), oracles.oracle_abs, marker="_", s=300, color="#222", label=T["oracle"])
        ax.scatter([2] * len(oracles), oracles.oracle_continuous_abs, marker="_", s=300, color="#222")
        ax.set_xticks([1, 2], [T["grid"], T["cont"]])
        ax.set_ylabel(T["perf"])
        _style(ax, "y")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        _save(fig, fig_dir / "bonus_grid_vs_continuous")
