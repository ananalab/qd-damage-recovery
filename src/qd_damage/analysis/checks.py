"""Robustness checks of the main results.

- unbiased oracle: the best cell of each scenario re-evaluated on fresh episodes (oracle.check_best_cells);
- mixed models: algorithm effects with a random intercept per repertoire, as a check of the rank tests;
- reality gap: ITE and baselines when the trials run on a robot whose physics differ from the simulator;
- descriptor ablation: ITE with a Gaussian process on joint usage instead of feet contact;
- stronger RL baselines: fine-tuning with a critic warm-up or a pre-trained critic, and a damage-robust policy.

Each part is skipped when its results are missing. Every function returns (tables, macros); the LaTeX tables are
written in `tex_dir` when given.
"""

import glob
import re
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from qd_damage.analysis.data import ALGO_LABELS, ALGOS, RES, SCENARIO_EN, TAG, UNIT, WORD, units
from qd_damage.analysis.latex import tex_table
from qd_damage.analysis.stats import fmt_p, mixed_contrasts, mixed_mean_by_algo


def _signed(x: float) -> str:
    """Signed number typeset in math mode, so that LaTeX prints a real minus sign."""
    return f"\\ensuremath{{{x:+.2f}}}"


def _ci(estimate: float, low: float, high: float) -> str:
    return f"\\ensuremath{{{estimate:+.2f}\\ [{low:+.2f}, {high:+.2f}]}}"


def _read_runs(kind: str, name: str, runs) -> pd.DataFrame | None:
    files = [RES / kind / r / name for r in runs if (RES / kind / r / name).exists()]
    return pd.concat(map(pd.read_csv, files)) if files else None


# --------------------------------------------------------------------------- unbiased oracle


def oracle_check(oracle: pd.DataFrame, f0: float, scale: float, tex_dir: Path | None):
    checked = _read_runs("oracle", "best_checked.csv", sorted(oracle.run.unique()))
    if checked is None:
        return {}, {}
    algo = oracle.drop_duplicates("run").set_index("run").algo
    checked = checked.assign(
        algo=checked.run.map(algo), bias=(checked.oracle_fitness - checked.checked_fitness) / scale
    )
    intact = checked[checked.scenario == "intact"].set_index("run").checked_fitness
    damaged = checked[checked.scenario != "intact"].copy()
    damaged["resilience"] = (damaged.checked_fitness - f0) / (damaged.run.map(intact) - f0)
    damaged["checked_abs"] = (damaged.checked_fitness - f0) / scale
    res_units = damaged.set_index(UNIT).resilience
    contrasts = mixed_contrasts(res_units)
    m = {}
    for a in ALGOS:
        m[f"OracleBias{TAG[a]}"] = f"{checked[checked.algo == a].bias.mean():.2f}"
        m[f"ResChecked{TAG[a]}"] = f"{100 * damaged[damaged.algo == a].resilience.median():.0f}"
        m[f"OracleChecked{TAG[a]}"] = f"{damaged[damaged.algo == a].checked_abs.median():.2f}"
    m["PResCheckedMin"] = fmt_p(contrasts.p.min())
    for r in contrasts.itertuples():
        m[f"MixResChecked{TAG[r.b]}vs{TAG[r.a]}"] = _ci(r.estimate, r.ci_low, r.ci_high)
        m[f"PMixResChecked{TAG[r.b]}vs{TAG[r.a]}"] = fmt_p(r.p)
    m["OracleCheckEpisodes"] = f"{int(checked.n_episodes.iloc[0])}"
    if tex_dir:
        t = pd.DataFrame(
            {
                "Bias of the oracle ($v$)": [checked[checked.algo == a].bias.mean() for a in ALGOS],
                "Resilience, checked (\\%)": [100 * damaged[damaged.algo == a].resilience.median() for a in ALGOS],
                "Best gait after damage, checked ($v$)": [
                    damaged[damaged.algo == a].checked_abs.median() for a in ALGOS
                ],
            },
            index=[ALGO_LABELS[a] for a in ALGOS],
        )
        tex_table(t, tex_dir / "table_oracle_check.tex", ["{:.2f}", "{:.0f}", "{:.2f}"], "Repertoire")
    return {"oracle_check": checked, "resilience_checked_contrasts": contrasts}, m


# --------------------------------------------------------------------------- mixed models


def mixed_checks(oracle: pd.DataFrame, adapt: pd.DataFrame, trials: pd.DataFrame, tex_dir: Path | None):
    damaged = oracle[oracle.scenario != "intact"]
    ite = adapt[adapt.method == "ite"]
    res = mixed_contrasts(damaged.set_index(UNIT).resilience)
    ab = mixed_contrasts(damaged.set_index(UNIT).oracle_abs)
    it = mixed_contrasts(units(ite, "perf_abs"))
    m, rows = {}, []
    for label, df in (("Res", res), ("OracleAbs", ab), ("ITEAbs", it)):
        for r in df.itertuples():
            key = f"Mix{label}{TAG[r.b]}vs{TAG[r.a]}"
            m[key] = _ci(r.estimate, r.ci_low, r.ci_high)
            m[f"P{key}"] = fmt_p(r.p)
    for label, df in (("Resilience", res), ("Best gait after damage ($v$)", ab), ("Gait found by ITE ($v$)", it)):
        for r in df.itertuples():
            rows.append(
                {
                    "Quantity": label,
                    "Comparison": f"{ALGO_LABELS[r.b]} $-$ {ALGO_LABELS[r.a]}",
                    "Difference [95\\% CI]": _ci(r.estimate, r.ci_low, r.ci_high),
                    "$p$-value": fmt_p(r.p).lstrip("="),
                }
            )
    diffs = []
    for t in (3, 5, 10):
        u = trials[(trials.trial == t) & trials.method.isin(["ite", "top_k"])]
        u = u.groupby([*UNIT, "method"]).perf_abs.mean().unstack()
        mm = mixed_mean_by_algo(u.ite - u.top_k).assign(trial=t)
        diffs.append(mm)
        for r in mm.itertuples():
            m[f"MixTopK{WORD[t]}{TAG[r.algo]}"] = _ci(r.estimate, r.ci_low, r.ci_high)
            m[f"PMixTopK{WORD[t]}{TAG[r.algo]}"] = fmt_p(r.p)
    diffs = pd.concat(diffs)
    if tex_dir:
        tex_table(pd.DataFrame(rows).set_index("Quantity"), tex_dir / "table_mixed.tex", "{}", "Quantity")
        t = diffs.assign(cell=[_ci(r.estimate, r.ci_low, r.ci_high) for r in diffs.itertuples()])
        t = t.pivot(index="algo", columns="trial", values="cell").reindex(ALGOS)
        t.index = [ALGO_LABELS[a] for a in t.index]
        t.columns = [f"{c} trials" for c in t.columns]
        tex_table(t, tex_dir / "table_mixed_topk.tex", "{}", "Repertoire")
    return {"mixed_contrasts": pd.DataFrame(rows), "mixed_ite_minus_topk": diffs}, m


def run_checks(oracle, adapt, trials, rl, f0: float, scale: float, tex_dir: Path | None):
    tables, macros = {}, {}
    for t, m in (
        oracle_check(oracle, f0, scale, tex_dir),
        mixed_checks(oracle, adapt, trials, tex_dir),
        reality_gap(adapt, f0, scale, tex_dir),
        descriptor_ablation(trials, f0, scale, tex_dir),
        robust_policy(adapt, f0, scale, tex_dir),
    ):
        tables.update(t)
        macros.update(m)
    macros.update(gap_intact_cost(oracle, f0, scale))
    macros.update(rl_variants(rl, adapt))
    return tables, macros
