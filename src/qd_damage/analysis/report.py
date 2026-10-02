"""Campaign analysis: tables, statistics, figures and LaTeX numbers, computed from results/ only.

Usage:
    python -m qd_damage.analysis.report          # final campaign: results/analysis, media/final, report/generated
    python -m qd_damage.analysis.report --dev    # development runs (*_demo): results/analysis_dev, media/dev

Reads results/repertoires/<run>/{run.json,metrics.csv}, results/oracle/<run>/summary.csv,
results/adaptation/<run>/{summary.csv,trials.csv}, results/rl/*/{curve.csv,rl.json} and results/bonus/.

Statistics. The five repetitions of an adaptation only differ by episode noise, so they are averaged first:
the statistical unit is a (repertoire, damage) pair, i.e. 3 seeds x 12 damages = 36 units per algorithm.
Algorithms are compared with Mann-Whitney tests on these units; ITE and top-k (or grid and continuous ITE)
are compared with paired Wilcoxon signed-rank tests. Units of the same seed share a repertoire, so they are
not fully independent and p-values are indicative.
"""

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

from qd_damage.config import load_run_config, performance_reference  # noqa: E402
from qd_damage.repertoires import REPO_ROOT  # noqa: E402

RES = REPO_ROOT / "results"
ALGOS = ["me", "pgame", "dcrlme"]
ALGO_LABELS = {"me": "MAP-Elites", "pgame": "PGA-ME", "dcrlme": "DCRL-ME"}
ALGO_COLORS = {"me": "#2a78d6", "pgame": "#e07b39", "dcrlme": "#3a9d5d"}
TAG = {"me": "ME", "pgame": "PGA", "dcrlme": "DCRL"}
WORD = {1: "One", 3: "Three", 5: "Five", 10: "Ten", 20: "Twenty"}
METHOD_COLORS = {"ite": "#3a9d5d", "top_k": "#7a5cc4", "random": "#8a8a85", "best_intact": "#c8413b"}
METHOD_EN = {"ite": "ITE", "top_k": "top-$k$ (no GP)", "random": "random", "best_intact": "best-intact"}
UNIT = ["algo", "run", "scenario"]  # statistical unit: one (repertoire, damage) pair

# Damage names. Legs: 1 front left, 2 hind left, 3 hind right, 4 front right.
SCENARIO_EN = {
    "leg1_paralysed": "FL leg paralysed", "leg2_paralysed": "HL leg paralysed",
    "leg3_paralysed": "HR leg paralysed", "leg4_paralysed": "FR leg paralysed",
    "leg1_weak": "FL leg weakened", "leg2_weak": "HL leg weakened",
    "leg3_weak": "HR leg weakened", "leg4_weak": "FR leg weakened",
    "legs12_paralysed": "FL + HL legs paralysed", "legs14_paralysed": "FL + FR legs paralysed",
    "ankle1_paralysed": "FL ankle paralysed", "ankle3_paralysed": "HR ankle paralysed", "intact": "intact",
}
SCENARIO_FR = {
    "leg1_paralysed": "patte 1 (AvG) paralysée", "leg2_paralysed": "patte 2 (ArG) paralysée",
    "leg3_paralysed": "patte 3 (ArD) paralysée", "leg4_paralysed": "patte 4 (AvD) paralysée",
    "leg1_weak": "patte 1 affaiblie", "leg2_weak": "patte 2 affaiblie",
    "leg3_weak": "patte 3 affaiblie", "leg4_weak": "patte 4 affaiblie",
    "legs12_paralysed": "pattes 1+2 paralysées", "legs14_paralysed": "pattes 1+4 paralysées",
    "ankle1_paralysed": "cheville 1 paralysée", "ankle3_paralysed": "cheville 3 paralysée", "intact": "intact",
}

# Figure texts: French figures for the portfolio and the French report, English ones for the English reports.
TEXT = {
    "fr": {
        "qd": "QD-score", "cov": "Couverture (%)", "maxfit": "Meilleure fitness", "evals": "Évaluations (épisodes)",
        "res_y": "Résilience (% du progrès intact\nencore atteignable après la blessure)",
        "res_med": "Résilience médiane (%)",
        "trials_log": "Essais (épisodes) sur le robot blessé — échelle log",
        "perf_y": "Performance retrouvée\n(% de la meilleure démarche intacte)",
        "rl_scratch": "RL (TD3) de zéro", "rl_finetune": "RL (TD3) depuis la politique intacte",
        "trials": "Essais sur le robot blessé", "ite_on": "ITE sur",
        "methods": {"ite": "ITE", "top_k": "top-k (sans GP)", "random": "aléatoire", "best_intact": "meilleur-intact"},
        "scenarios": SCENARIO_FR,
        "episodes": "Épisodes sur le robot blessé", "perf": "Performance $p$\n(1 ≈ 15 m en 12,5 s)",
        "scratch": "TD3 de zéro", "finetune": "TD3 depuis la politique intacte", "ite": "ITE (DCRL-ME, même seed)",
        "grid": "ITE sur la grille\n(cases du répertoire)", "cont": "ITE sur l'acteur\n(descripteurs continus)",
        "oracle": "meilleure démarche disponible (oracle)",
    },
    "en": {
        "qd": "QD-score", "cov": "Coverage (%)", "maxfit": "Max fitness", "evals": "Evaluations (episodes)",
        "res_y": "Resilience (% of intact progress\nstill reachable after damage)",
        "res_med": "Median resilience (%)",
        "trials_log": "Trials (episodes) on the damaged robot, log scale",
        "perf_y": "Recovered performance\n(% of best intact gait)",
        "rl_scratch": "RL (TD3) from scratch", "rl_finetune": "RL (TD3) from intact policy",
        "trials": "Trials on the damaged robot", "ite_on": "ITE on",
        "methods": {"ite": "ITE", "top_k": "top-k (no GP)", "random": "random", "best_intact": "best-intact"},
        "scenarios": SCENARIO_EN,
        "episodes": "Episodes on the damaged robot", "perf": "Performance $p$\n(1 ≈ 15 m in 12.5 s)",
        "scratch": "TD3 from scratch", "finetune": "TD3 from intact policy", "ite": "ITE (DCRL-ME, same seed)",
        "grid": "ITE on the grid\n(repertoire cells)", "cont": "ITE on the actor\n(continuous descriptors)",
        "oracle": "best available gait (oracle)",
    },
}
RL_SCENARIOS = ["leg4_paralysed", "leg2_weak", "legs14_paralysed", "ankle1_paralysed"]


# --------------------------------------------------------------------------- helpers


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _style(ax, grid_axis="both"):
    ax.grid(alpha=0.3, axis=grid_axis)
    ax.spines[["top", "right"]].set_visible(False)


def iqr(x):
    return f"{np.median(x):.1f} [{np.percentile(x, 25):.1f} ; {np.percentile(x, 75):.1f}]"


def fmt_p(p: float) -> str:
    """p-value with its relation sign, to be written as $p\\Macro$ in LaTeX: "=0.042" or "<0.001"."""
    return "<0.001" if p < 1e-3 else f"={p:.3f}"


def tex_table(df: pd.DataFrame, path: Path, fmt="{:.2f}", index_name: str = ""):
    """Minimal booktabs table (no jinja2 / Styler dependency)."""
    cols = list(df.columns)
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             " & ".join([index_name] + [str(c) for c in cols]) + " \\\\", "\\midrule"]
    fmts = fmt if isinstance(fmt, (list, tuple)) else [fmt] * len(cols)
    for idx, row in df.iterrows():
        cells = [f.format(v) if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v)
                 else ("--" if pd.isna(v) else str(v)) for f, v in zip(fmts, row.to_numpy())]
        lines.append(" & ".join([str(idx)] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def units(df: pd.DataFrame, value: str) -> pd.Series:
    """Mean of `value` over the repetitions of each (repertoire, damage) pair."""
    return df.groupby(UNIT)[value].mean()


# --------------------------------------------------------------------------- loading


def load(runs_glob: str):
    runs = sorted(p.parent for p in (RES / "repertoires").glob(f"{runs_glob}/run.json"))
    if not runs:
        raise SystemExit(f"no run matches results/repertoires/{runs_glob}")
    meta = {r.name: json.loads((r / "run.json").read_text()) for r in runs}
    quality = pd.DataFrame([
        {"run": r.name, "algo": meta[r.name]["algo"], "seed": meta[r.name]["seed"],
         **pd.read_csv(r / "metrics.csv").iloc[-1][["evaluations", "qd_score", "coverage", "max_fitness"]].to_dict()}
        for r in runs
    ])
    curves = pd.concat([pd.read_csv(r / "metrics.csv").assign(run=r.name, algo=meta[r.name]["algo"]) for r in runs])

    def gather(kind, name):
        return pd.concat([pd.read_csv(RES / kind / r.name / name) for r in runs if (RES / kind / r.name / name).exists()])

    oracle, adapt, trials = gather("oracle", "summary.csv"), gather("adaptation", "summary.csv"), gather("adaptation", "trials.csv")
    rl = []
    for p in (RES / "rl").glob("*/curve.csv"):
        info = json.loads((p.parent / "rl.json").read_text())
        if info["config"]["max_episodes"] >= 100:  # skip smoke runs
            rl.append(pd.read_csv(p).assign(scenario=info["scenario"], variant=info["variant"], seed=info["seed"]))
    rl = pd.concat(rl) if rl else pd.DataFrame()
    return meta, quality, curves, oracle, adapt, trials, rl


# --------------------------------------------------------------------------- statistics


def mann_whitney_table(values: pd.Series) -> pd.DataFrame:
    """Pairwise two-sided Mann-Whitney tests between algorithms; `values` is indexed by UNIT."""
    rows = []
    for a, b in combinations(ALGOS, 2):
        x, y = values.xs(a, level="algo").dropna(), values.xs(b, level="algo").dropna()
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        rows.append({"a": a, "b": b, "n_a": len(x), "n_b": len(y),
                     "median_a": np.median(x), "median_b": np.median(y), "U": stat, "p": p})
    return pd.DataFrame(rows)


def ite_vs_topk(trials: pd.DataFrame, at=(3, 5, 10)) -> pd.DataFrame:
    """Paired comparison of ITE and top-k at equal numbers of trials, per repertoire type.

    For each unit, the absolute performance of the recommended gait is averaged over repetitions; the
    ITE - top-k differences are tested with a two-sided Wilcoxon signed-rank test (zero differences dropped).
    """
    rows = []
    for t in at:
        sel = trials[(trials.trial == t) & trials.method.isin(["ite", "top_k"])]
        u = sel.groupby(UNIT + ["method"]).perf_abs.mean().unstack()
        for a in ALGOS + ["all"]:
            ua = u if a == "all" else u.xs(a, level="algo", drop_level=False)
            d = ua.ite - ua.top_k
            rows.append({"algo": a, "trial": t, "n": len(d), "ite_median": ua.ite.median(),
                         "top_k_median": ua.top_k.median(), "diff_mean": d.mean(), "diff_median": d.median(),
                         "ite_better": int((d > 0).sum()), "equal": int((d == 0).sum()), "ite_worse": int((d < 0).sum()),
                         "p": wilcoxon(ua.ite, ua.top_k).pvalue if (d != 0).any() else 1.0})
    return pd.DataFrame(rows)


def rl_episodes_to_match(rl: pd.DataFrame, adapt: pd.DataFrame) -> pd.DataFrame:
    """RL episodes needed to match the gait found by ITE (DCRL-ME repertoire of the same seed)."""
    rows = []
    for (scenario, variant, seed), c in rl.groupby(["scenario", "variant", "seed"]):
        ite = adapt[(adapt.method == "ite") & (adapt.algo == "dcrlme") & (adapt.seed == seed) & (adapt.scenario == scenario)]
        if ite.empty:
            continue
        target = ite.recommended_true.mean()
        reached = c[c.eval_fitness >= target]
        rows.append({
            "scenario": scenario, "variant": variant, "seed": seed, "ite_trials": ite.trials_used.mean(),
            "ite_fitness": target, "rl_episodes_to_match": int(reached.episodes.iloc[0]) if len(reached) else np.nan,
            "rl_best_fitness": c.eval_fitness.max(), "rl_max_episodes": int(c.episodes.max()),
        })
    return pd.DataFrame(rows)


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
    pairs = (grid.groupby(key).perf_abs.mean().rename("grid").to_frame()
             .join(cont.groupby(key).perf_abs.mean().rename("cont")))
    oracles = (cont.groupby(key).oracle_continuous_abs.first().to_frame()
               .join(oracle.set_index(key).oracle_abs))
    return grid, cont, pairs, oracles


# --------------------------------------------------------------------------- figures


def figures(curves, oracle, trials, rl, adapt, bonus, fig_dir: Path, f0: float, scale: float, lang: str):
    T = TEXT[lang]
    # 1) Repertoire progress (median and quartiles over seeds)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for algo in ALGOS:
        c = curves[curves.algo == algo]
        for ax, key in zip(axes, ["qd_score", "coverage", "max_fitness"]):
            g = c.groupby("evaluations")[key]
            ax.plot(g.median().index, g.median(), color=ALGO_COLORS[algo], lw=2, label=ALGO_LABELS[algo])
            ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color=ALGO_COLORS[algo], alpha=0.2)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    for ax, title in zip(axes, [T["qd"], T["cov"], T["maxfit"]]):
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
    for patch, a in zip(bp["boxes"], ALGOS):
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
    ax.scatter([1], [bi.median()], color=METHOD_COLORS["best_intact"], zorder=5, s=40, label=T["methods"]["best_intact"])
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
        for ax, sc in zip(axes, RL_SCENARIOS):
            for (v, seed), c in rl[rl.scenario == sc].groupby(["variant", "seed"]):
                ax.plot(c.episodes, (c.eval_fitness - f0) / scale, ls=":" if v == "scratch" else "-",
                        color="#7a5cc4" if v == "scratch" else "#e07b39", lw=1.6, alpha=0.9,
                        label=T[v] if seed == 0 else None)
            for seed, color in [(0, "#3a9d5d"), (1, "#2a6f3f")]:
                ite = adapt[(adapt.method == "ite") & (adapt.algo == "dcrlme") & (adapt.seed == seed) & (adapt.scenario == sc)]
                if len(ite):
                    ax.axhline((ite.recommended_true.mean() - f0) / scale, color=color, lw=1.4, ls="--",
                               label=T["ite"] if seed == 0 else None)
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
        for patch, col in zip(bp["boxes"], ["#3a9d5d", "#7a5cc4"]):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        ax.scatter([1] * len(oracles), oracles.oracle_abs, marker="_", s=300, color="#222", label=T["oracle"])
        ax.scatter([2] * len(oracles), oracles.oracle_continuous_abs, marker="_", s=300, color="#222")
        ax.set_xticks([1, 2], [T["grid"], T["cont"]])
        ax.set_ylabel(T["perf"])
        _style(ax, "y")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        _save(fig, fig_dir / "bonus_grid_vs_continuous")


# --------------------------------------------------------------------------- LaTeX output


def write_latex(out_dir: Path, meta, quality, oracle, adapt, trials, tables, bonus):
    """Tables of the reports (English) and every number of their text, as LaTeX macros.

    Generated files: never edit them by hand.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    m = {}

    # Repertoires
    q = quality.groupby("algo")[["coverage", "max_fitness", "qd_score"]].median().reindex(ALGOS)
    q.index = [ALGO_LABELS[a] for a in q.index]
    q = q.assign(qd_score=q.qd_score / 1e3)
    q.columns = ["Coverage (\\%)", "Max fitness", "QD-score ($\\times 10^3$)"]
    tex_table(q, out_dir / "table_quality.tex", ["{:.1f}", "{:.0f}", "{:.0f}"], "Algorithm")
    budget = next(iter(meta.values()))["config"]["common"]["budget"]
    m["NIterations"] = f"{budget['num_iterations']}"
    m["NEvaluations"] = f"{int(quality.evaluations.max()):,}".replace(",", "{,}")
    m["NSeeds"] = f"{quality.seed.nunique()}"

    # Resilience
    res = oracle[oracle.scenario != "intact"]
    m["NScenarios"] = f"{res.scenario.nunique()}"
    m["NUnits"] = f"{len(res) // res.algo.nunique()}"
    m["OraclePolicies"] = f"{int(res.n_policies.min())}--{int(res.n_policies.max())}"
    pct = res.pivot_table(index="scenario", columns="algo", values="resilience", aggfunc="median")[ALGOS] * 100
    ab = res.pivot_table(index="scenario", columns="algo", values="oracle_abs", aggfunc="median")[ALGOS]
    both = pd.concat([pct.round(0), ab.reindex(pct.index)], axis=1)
    both.index = [SCENARIO_EN.get(i, i) for i in both.index]
    both.columns = [f"{ALGO_LABELS[a]} (\\%)" for a in ALGOS] + [f"{ALGO_LABELS[a]} ($p$)" for a in ALGOS]
    tex_table(both, out_dir / "table_resilience.tex", ["{:.0f}"] * 3 + ["{:.2f}"] * 3, "Damage")

    # Adaptation methods
    meth = adapt.groupby("method")[["trials_used", "pct_intact", "pct_oracle", "perf_abs"]].median()
    meth = meth.reindex(["ite", "top_k", "random", "best_intact"])
    meth.index = [METHOD_EN[i] for i in meth.index]
    meth.columns = ["Trials", "\\% intact", "\\% oracle", "Absolute $p$"]
    tex_table(meth, out_dir / "table_methods.tex", ["{:.0f}", "{:.0f}", "{:.0f}", "{:.2f}"], "Method")

    at = tables["pct_intact_at_equal_trials"].reindex(["ite", "top_k", "random"])
    for meth_name, name in [("ite", "ITE"), ("top_k", "TopK"), ("random", "Random")]:
        for t in (3, 5, 20):
            m[f"{name}At{WORD[t]}"] = f"{at.loc[meth_name, t]:.0f}"
    at.index = [METHOD_EN[i] for i in at.index]
    at.columns = [f"{c} trial" + ("s" if c > 1 else "") for c in at.columns]
    tex_table(at, out_dir / "table_equal_trials.tex", "{:.0f}", "Method")

    # Equal trials, by repertoire type (medians over all rows)
    sel = trials[trials.trial.isin([1, 3, 5, 10, 20]) & trials.method.isin(["ite", "top_k", "random"])]
    rows = []
    for a in ALGOS:
        p = sel[sel.algo == a].pivot_table(index="method", columns="trial", values="perf_abs", aggfunc="median")
        for meth_name, name in [("ite", "ITE"), ("top_k", "TopK"), ("random", "Random")]:
            rows.append(pd.Series(p.loc[meth_name].to_numpy(), index=p.columns,
                                  name=f"{ALGO_LABELS[a]} -- {METHOD_EN[meth_name].split(' ')[0]}"))
            for t in p.columns:
                m[f"{name}Abs{WORD[t]}{TAG[a]}"] = f"{p.loc[meth_name, t]:.2f}"
    table = pd.DataFrame(rows)
    table.columns = [f"{c} trial" + ("s" if c != 1 else "") for c in table.columns]
    tex_table(table, out_dir / "table_equal_trials_by_algo.tex", "{:.2f}", "Repertoire -- method")

    # ITE against top-k: paired tests on (repertoire, damage) units
    vs = tables["ite_vs_topk"]
    t_vs = vs[vs.algo != "all"].copy()
    t_vs["cell"] = [f"${r.diff_mean:+.2f}$ ({r.ite_better}/{r.equal}/{r.ite_worse}; $p{fmt_p(r.p)}$)"
                    for r in t_vs.itertuples()]
    t_vs = t_vs.pivot(index="algo", columns="trial", values="cell").reindex(ALGOS)
    t_vs.index = [ALGO_LABELS[a] for a in t_vs.index]
    t_vs.columns = [f"{c} trials" for c in t_vs.columns]
    tex_table(t_vs, out_dir / "table_ite_vs_topk.tex", "{}", "Repertoire")
    for r in vs.itertuples():
        tag = "" if r.algo == "all" else TAG[r.algo]
        m[f"PTopK{WORD[r.trial]}{tag}"] = fmt_p(r.p)
        m[f"DiffTopK{WORD[r.trial]}{tag}"] = f"{r.diff_mean:+.2f}"
    five = vs[(vs.algo == "all") & (vs.trial == 5)].iloc[0]
    m["NPairsTopK"] = f"{five.n}"
    m["ITEWinsTopK"], m["ITETiesTopK"], m["ITELosesTopK"] = (
        f"{100 * five.ite_better / five.n:.0f}", f"{100 * five.equal / five.n:.0f}", f"{100 * five.ite_worse / five.n:.0f}")

    # ITE on each repertoire type
    ite = adapt[adapt.method == "ite"]
    ia = ite.groupby("algo")[["trials_used", "pct_intact", "perf_abs"]].median().reindex(ALGOS)
    ia.index = [ALGO_LABELS[a] for a in ia.index]
    ia.columns = ["Trials", "\\% intact", "Absolute $p$"]
    tex_table(ia, out_dir / "table_ite_by_algo.tex", ["{:.0f}", "{:.0f}", "{:.2f}"], "Repertoire")
    for a in ALGOS:
        q, ia_a = quality[quality.algo == a], ite[ite.algo == a]
        m[f"Cov{TAG[a]}"] = f"{q.coverage.median():.1f}"
        m[f"MaxFit{TAG[a]}"] = f"{q.max_fitness.median():.0f}"
        m[f"QD{TAG[a]}"] = f"{q.qd_score.median() / 1e3:.0f}k"
        m[f"Res{TAG[a]}"] = f"{100 * res[res.algo == a].resilience.median():.0f}"
        m[f"OracleAbs{TAG[a]}"] = f"{res[res.algo == a].oracle_abs.median():.2f}"
        m[f"ITETrials{TAG[a]}"] = f"{ia_a.trials_used.median():.0f}"
        m[f"ITEPctIntact{TAG[a]}"] = f"{ia_a.pct_intact.median():.0f}"
        m[f"ITEAbs{TAG[a]}"] = f"{ia_a.perf_abs.median():.2f}"
    m["ITETrials"] = f"{ite.trials_used.median():.0f}"
    m["ITETrialsQone"] = f"{ite.trials_used.quantile(0.25):.0f}"
    m["ITETrialsQthree"] = f"{ite.trials_used.quantile(0.75):.0f}"
    m["ITEPctIntact"] = f"{ite.pct_intact.median():.0f}"
    m["ITEPctOracle"] = f"{ite.pct_oracle.median():.0f}"
    m["ITEStopRate"] = f"{100 * ite.stopped_by_criterion.mean():.0f}"
    m["BestIntactPct"] = f"{adapt[adapt.method == 'best_intact'].pct_intact.median():.0f}"
    for label, df in [("Res", tables["stats_resilience"]), ("ITEAbs", tables["stats_ite_perf_abs"])]:
        for r in df.itertuples():
            m[f"P{label}{TAG[r.a]}{TAG[r.b]}"] = fmt_p(r.p)
        m[f"P{label}Min"] = fmt_p(df.p.min())

    # RL re-training
    if "rl_episodes_to_match_ite" in tables:
        rl = tables["rl_episodes_to_match_ite"].copy()
        for v, name in [("scratch", "Scratch"), ("finetune", "Finetune")]:
            r = rl[rl.variant == v]
            m[f"RL{name}Reached"] = f"{r.rl_episodes_to_match.notna().sum()}/{len(r)}"
            m[f"RL{name}Episodes"] = (f"{r.rl_episodes_to_match.median():.0f}"
                                      if r.rl_episodes_to_match.notna().any() else "n/a")
        m["RLMaxEpisodes"] = f"{int(rl.rl_max_episodes.max())}"
        rl["scenario"] = rl.scenario.map(lambda s: SCENARIO_EN.get(s, s))
        rl["variant"] = rl.variant.map({"scratch": "from scratch", "finetune": "from intact policy"})
        rl = rl.set_index(["scenario"])[["variant", "seed", "ite_trials", "rl_episodes_to_match", "rl_max_episodes"]]
        rl.columns = ["RL", "Seed", "ITE trials", "RL episodes to match", "RL budget"]
        tex_table(rl, out_dir / "table_rl.tex", "{:.0f}", "Damage")

    # Bonus: continuous ITE
    if bonus is not None:
        grid, cont, pairs, oracles = bonus
        m.update({
            "BonusGridTrials": f"{grid.trials_used.median():.0f}", "BonusGridAbs": f"{pairs.grid.median():.2f}",
            "BonusContTrials": f"{cont.trials_used.median():.0f}", "BonusContAbs": f"{pairs.cont.median():.2f}",
            "BonusContWins": f"{(pairs.cont > pairs.grid).sum()}", "BonusN": f"{len(pairs)}",
            "BonusP": fmt_p(wilcoxon(pairs.cont, pairs.grid).pvalue),
            "BonusOracleGrid": f"{oracles.oracle_abs.median():.2f}",
            "BonusOracleCont": f"{oracles.oracle_continuous_abs.median():.2f}",
            "BonusOracleContBetter": f"{(oracles.oracle_continuous_abs > oracles.oracle_abs).sum()}",
            "BonusNRuns": f"{grid.run.nunique()}",
        })
        t = pd.DataFrame({
            "Trials": [grid.trials_used.median(), cont.trials_used.median()],
            "Found $p$": [pairs.grid.median(), pairs.cont.median()],
            "Oracle $p$": [oracles.oracle_abs.median(), oracles.oracle_continuous_abs.median()],
        }, index=["Grid (repertoire cells)", "Continuous (conditioned actor)"])
        tex_table(t, out_dir / "table_bonus.tex", ["{:.0f}", "{:.2f}", "{:.2f}"], "ITE search space")

    lines = ["% Generated by qd_damage.analysis.report: do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (out_dir / "numbers.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dev", action="store_true",
                        help="analyse the development runs (*_demo) instead: no RL, no bonus, no LaTeX output")
    args = parser.parse_args()
    if args.dev:
        runs_glob, out, fig_dir, tex_dir = "*_demo", RES / "analysis_dev", REPO_ROOT / "media" / "dev" / "figures", None
    else:
        runs_glob, out, fig_dir = "*_seed[0-9]", RES / "analysis", REPO_ROOT / "media" / "final" / "figures"
        tex_dir = REPO_ROOT / "report" / "generated"

    meta, quality, curves, oracle, adapt, trials, rl = load(runs_glob)
    if args.dev:
        rl = pd.DataFrame()
    any_run = next(iter(meta.values()))
    f0, scale = performance_reference(load_run_config(any_run["algo"], any_run["profile"])["common"]["env"]["episode_length"])
    # Absolute performance, comparable across repertoires: p = (F - F0) / scale (1 is about 15 m in 12.5 s).
    adapt = adapt.assign(perf_abs=(adapt.recommended_true - f0) / scale)
    trials = trials.assign(perf_abs=(trials.recommended_true - f0) / scale)
    oracle = oracle.assign(oracle_abs=(oracle.oracle_fitness - f0) / scale)
    damaged = oracle[oracle.scenario != "intact"]
    ite = adapt[adapt.method == "ite"]

    tables = {
        "repertoire_quality": quality.groupby("algo")[["qd_score", "coverage", "max_fitness"]].agg(["median", "min", "max"]),
        "resilience_by_damage": damaged.pivot_table(index="scenario", columns="algo", values="resilience",
                                                    aggfunc="median")[ALGOS],
        "stats_resilience": mann_whitney_table(damaged.set_index(UNIT).resilience),
        "oracle_abs_by_damage": damaged.pivot_table(index="scenario", columns="algo", values="oracle_abs",
                                                    aggfunc="median")[ALGOS],
        "adaptation_by_method": adapt.groupby("method")[["trials_used", "pct_oracle", "pct_intact", "perf_abs"]].median(),
        "pct_intact_at_equal_trials": trials[trials.trial.isin([1, 3, 5, 10, 20])].pivot_table(
            index="method", columns="trial", values="pct_intact", aggfunc="median"),
        "ite_by_algo": ite.groupby("algo")[["trials_used", "pct_oracle", "pct_intact", "perf_abs"]].median(),
        "stats_ite_perf_abs": mann_whitney_table(units(ite, "perf_abs")),
        "stats_ite_pct_intact": mann_whitney_table(units(ite, "pct_intact")),
        "ite_vs_topk": ite_vs_topk(trials),
    }
    if not rl.empty:
        tables["rl_episodes_to_match_ite"] = rl_episodes_to_match(rl, adapt)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv")
    bonus = None if args.dev else load_bonus(f0, scale)

    lines = ["# Analysis summary (generated by qd_damage.analysis.report)", "", f"Runs: {', '.join(sorted(meta))}", ""]
    for name, t in tables.items():
        lines += [f"## {name}", "", t.round(3).to_markdown(), ""]
    lines += ["## Key numbers", "",
              f"- ITE: {iqr(ite.trials_used)} trials (median [Q1 ; Q3]), {iqr(ite.pct_intact)} % of the best intact "
              f"gait, {iqr(ite.pct_oracle)} % of the oracle.",
              f"- Best-intact (no adaptation): {iqr(adapt[adapt.method == 'best_intact'].pct_intact)} % of the best "
              "intact gait.", ""]
    (out / "summary.md").write_text("\n".join(lines))

    if tex_dir:
        write_latex(tex_dir, meta, quality, oracle, adapt, trials, tables, bonus)
    for lang, d in (("fr", fig_dir), ("en", fig_dir.with_name("figures_en"))):
        figures(curves, oracle, trials, rl, adapt, bonus, d, f0, scale, lang)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
