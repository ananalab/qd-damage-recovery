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


