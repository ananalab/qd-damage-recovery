"""Statistics on (repertoire, damage) units: medians, Mann-Whitney and Wilcoxon tests, mixed models."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from qd_damage.analysis.data import ALGOS, UNIT


def iqr(x):
    return f"{np.median(x):.1f} [{np.percentile(x, 25):.1f} ; {np.percentile(x, 75):.1f}]"


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


