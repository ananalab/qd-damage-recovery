"""Statistics on (repertoire, damage) units: medians, Mann-Whitney and Wilcoxon tests, mixed models."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from qd_damage.analysis.data import ALGOS, UNIT


def iqr(x):
    return f"{np.median(x):.1f} [{np.percentile(x, 25):.1f} ; {np.percentile(x, 75):.1f}]"


def fmt_p(p: float) -> str:
    """p-value with its relation sign, to be written as $p\\Macro$ in LaTeX: "=0.042" or "<0.001"."""
    return "<0.001" if p < 1e-3 else f"={p:.3f}"


def mann_whitney_table(values: pd.Series) -> pd.DataFrame:
    """Pairwise two-sided Mann-Whitney tests between algorithms; `values` is indexed by UNIT."""
    rows = []
    for a, b in combinations(ALGOS, 2):
        x, y = values.xs(a, level="algo").dropna(), values.xs(b, level="algo").dropna()
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        rows.append(
            {
                "a": a,
                "b": b,
                "n_a": len(x),
                "n_b": len(y),
                "median_a": np.median(x),
                "median_b": np.median(y),
                "U": stat,
                "p": p,
            }
        )
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
            rows.append(
                {
                    "algo": a,
                    "trial": t,
                    "n": len(d),
                    "ite_median": ua.ite.median(),
                    "top_k_median": ua.top_k.median(),
                    "diff_mean": d.mean(),
                    "diff_median": d.median(),
                    "ite_better": int((d > 0).sum()),
                    "equal": int((d == 0).sum()),
                    "ite_worse": int((d < 0).sum()),
                    "p": wilcoxon(ua.ite, ua.top_k).pvalue if (d != 0).any() else 1.0,
                }
            )
    return pd.DataFrame(rows)


def mixed_contrasts(values: pd.Series) -> pd.DataFrame:
    """Differences between algorithms (b - a) from `value ~ algo + damage + (1 | run)`, with 95 % intervals.

    `values` is indexed by UNIT. The damage enters as a fixed effect, so the comparison is made within damages;
    the random intercept absorbs what the units of one repertoire have in common.
    """
    df = values.rename("y").reset_index()
    rows = []
    for ref in ALGOS:
        fit = _fit_mixed(f"y ~ C(algo, Treatment('{ref}')) + C(scenario)", df)
        ci = fit.conf_int()
        for other in ALGOS:
            name = f"C(algo, Treatment('{ref}'))[T.{other}]"
            if name in fit.params and ALGOS.index(other) > ALGOS.index(ref):
                rows.append(
                    {
                        "a": ref,
                        "b": other,
                        "estimate": fit.params[name],
                        "ci_low": ci.loc[name, 0],
                        "ci_high": ci.loc[name, 1],
                        "p": fit.pvalues[name],
                    }
                )
    return pd.DataFrame(rows)


