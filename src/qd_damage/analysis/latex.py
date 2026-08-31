"""LaTeX output: generated tables and every number of the report as macros (never edit the outputs by hand)."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from qd_damage.analysis.data import ALGO_LABELS, ALGOS, METHOD_EN, SCENARIO_EN, TAG, WORD
from qd_damage.analysis.stats import fmt_p


def tex_table(df: pd.DataFrame, path: Path, fmt="{:.2f}", index_name: str = ""):
    """Minimal booktabs table (no jinja2 / Styler dependency)."""
    cols = list(df.columns)
    lines = [
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        " & ".join([index_name] + [str(c) for c in cols]) + " \\\\",
        "\\midrule",
    ]
    fmts = fmt if isinstance(fmt, (list, tuple)) else [fmt] * len(cols)
    for idx, row in df.iterrows():
        cells = [
            f.format(v)
            if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v)
            else ("--" if pd.isna(v) else str(v))
            for f, v in zip(fmts, row.to_numpy(), strict=False)
        ]
        lines.append(" & ".join([str(idx)] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_latex(out_dir: Path, meta, quality, oracle, adapt, trials, tables, bonus, extra_macros=None):
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
    both.columns = [f"{ALGO_LABELS[a]} (\\%)" for a in ALGOS] + [f"{ALGO_LABELS[a]} ($v$)" for a in ALGOS]
    tex_table(both, out_dir / "table_resilience.tex", ["{:.0f}"] * 3 + ["{:.2f}"] * 3, "Damage")

    # Adaptation methods
    meth = adapt.groupby("method")[["trials_used", "pct_intact", "pct_oracle", "perf_abs"]].median()
    meth = meth.reindex(["ite", "top_k", "random", "best_intact"])
    meth.index = [METHOD_EN[i] for i in meth.index]
    meth.columns = ["Trials", "\\% intact", "\\% oracle", "Absolute $v$"]
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
            rows.append(
                pd.Series(
                    p.loc[meth_name].to_numpy(),
                    index=p.columns,
                    name=f"{ALGO_LABELS[a]} -- {METHOD_EN[meth_name].split(' ')[0]}",
                )
            )
            for t in p.columns:
                m[f"{name}Abs{WORD[t]}{TAG[a]}"] = f"{p.loc[meth_name, t]:.2f}"
    table = pd.DataFrame(rows)
    table.columns = [f"{c} trial" + ("s" if c != 1 else "") for c in table.columns]
    tex_table(table, out_dir / "table_equal_trials_by_algo.tex", "{:.2f}", "Repertoire -- method")

    # ITE against top-k: paired tests on (repertoire, damage) units
    vs = tables["ite_vs_topk"]
    t_vs = vs[vs.algo != "all"].copy()
    t_vs["cell"] = [
        f"${r.diff_mean:+.2f}$ ({r.ite_better}/{r.equal}/{r.ite_worse}; $p{fmt_p(r.p)}$)" for r in t_vs.itertuples()
    ]
    t_vs = t_vs.pivot(index="algo", columns="trial", values="cell").reindex(ALGOS)
    t_vs.index = [ALGO_LABELS[a] for a in t_vs.index]
    t_vs.columns = [f"{c} trials" for c in t_vs.columns]
    tex_table(t_vs, out_dir / "table_ite_vs_topk.tex", "{}", "Repertoire")
    for r in vs.itertuples():
        tag = "" if r.algo == "all" else TAG[r.algo]
        m[f"PTopK{WORD[r.trial]}{tag}"] = fmt_p(r.p)
        m[f"DiffTopK{WORD[r.trial]}{tag}"] = f"\\ensuremath{{{r.diff_mean:+.2f}}}"
    five = vs[(vs.algo == "all") & (vs.trial == 5)].iloc[0]
    m["NPairsTopK"] = f"{five.n}"
    m["ITEWinsTopK"], m["ITETiesTopK"], m["ITELosesTopK"] = (
        f"{100 * five.ite_better / five.n:.0f}",
        f"{100 * five.equal / five.n:.0f}",
        f"{100 * five.ite_worse / five.n:.0f}",
    )

    # ITE on each repertoire type
    ite = adapt[adapt.method == "ite"]
    ia = ite.groupby("algo")[["trials_used", "pct_intact", "perf_abs"]].median().reindex(ALGOS)
    ia.index = [ALGO_LABELS[a] for a in ia.index]
    ia.columns = ["Trials", "\\% intact", "Absolute $v$"]
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
            m[f"RL{name}Episodes"] = (
                f"{r.rl_episodes_to_match.median():.0f}" if r.rl_episodes_to_match.notna().any() else "n/a"
            )
        m["RLMaxEpisodes"] = f"{int(rl.rl_max_episodes.max())}"
        # One row per (damage, seed), one column per variant: episodes needed to match ITE ("--": never).
        names = {
            "scratch": "Scratch",
            "finetune": "Fine-tune",
            "finetune_warmup": "Warm-up",
            "finetune_pretrained": "Pre-trained critic",
        }
        ite_trials = rl.groupby(["scenario", "seed"]).ite_trials.first()
        t = rl.pivot_table(index=["scenario", "seed"], columns="variant", values="rl_episodes_to_match", dropna=False)
        t = t.reindex(columns=[v for v in names if v in t.columns])
        t.insert(0, "ITE trials", ite_trials)
        t.columns = [names.get(c, c) for c in t.columns]
        t.index = [f"{SCENARIO_EN.get(sc, sc)}, seed {seed}" for sc, seed in t.index]
        tex_table(t, out_dir / "table_rl.tex", ["{:.1f}"] + ["{:.0f}"] * (len(t.columns) - 1), "Damage")

    # Bonus: continuous ITE
    if bonus is not None:
        grid, cont, pairs, oracles = bonus
        m.update(
            {
                "BonusGridTrials": f"{grid.trials_used.median():.0f}",
                "BonusGridAbs": f"{pairs.grid.median():.2f}",
                "BonusContTrials": f"{cont.trials_used.median():.0f}",
                "BonusContAbs": f"{pairs.cont.median():.2f}",
                "BonusContWins": f"{(pairs.cont > pairs.grid).sum()}",
                "BonusN": f"{len(pairs)}",
                "BonusP": fmt_p(wilcoxon(pairs.cont, pairs.grid).pvalue),
                "BonusOracleGrid": f"{oracles.oracle_abs.median():.2f}",
                "BonusOracleCont": f"{oracles.oracle_continuous_abs.median():.2f}",
                "BonusOracleContBetter": f"{(oracles.oracle_continuous_abs > oracles.oracle_abs).sum()}",
                "BonusNRuns": f"{grid.run.nunique()}",
            }
        )
        t = pd.DataFrame(
            {
                "Trials": [grid.trials_used.median(), cont.trials_used.median()],
                "Found $v$": [pairs.grid.median(), pairs.cont.median()],
                "Oracle $v$": [oracles.oracle_abs.median(), oracles.oracle_continuous_abs.median()],
            },
            index=["Grid (repertoire cells)", "Continuous (conditioned actor)"],
        )
        tex_table(t, out_dir / "table_bonus.tex", ["{:.0f}", "{:.2f}", "{:.2f}"], "ITE search space")

    m.update(extra_macros or {})
    lines = ["% Generated by qd_damage.analysis.report: do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (out_dir / "numbers.tex").write_text("\n".join(lines) + "\n")
