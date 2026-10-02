"""Statistics of the analysis: repetitions are averaged before tests, ITE and top-k are paired per unit."""

import pandas as pd

from qd_damage.analysis.report import ite_vs_topk, mann_whitney_table, units


def _trials(diff_dcrl: float) -> pd.DataFrame:
    rows = []
    for algo, diff in (("me", 0.0), ("pgame", 0.0), ("dcrlme", diff_dcrl)):
        for scenario in range(12):
            for rep in range(5):
                base = 0.1 * scenario
                rows += [{"algo": algo, "run": f"{algo}_seed0", "scenario": scenario, "rep": rep, "trial": 5,
                          "method": "top_k", "perf_abs": base},
                         {"algo": algo, "run": f"{algo}_seed0", "scenario": scenario, "rep": rep, "trial": 5,
                          "method": "ite", "perf_abs": base + diff + 0.001 * (scenario % 3 - 1)}]
    return pd.DataFrame(rows)


def test_ite_vs_topk_pairs_units_not_repetitions():
    res = ite_vs_topk(_trials(0.2), at=(5,)).set_index("algo")
    assert res.loc["dcrlme", "n"] == 12  # 12 units, not 60 rows
    assert res.loc["dcrlme", "ite_better"] == 12 and res.loc["dcrlme", "p"] < 0.01
    assert res.loc["me", "p"] > 0.05


def test_mann_whitney_on_units():
    df = _trials(0.2)
    table = mann_whitney_table(units(df[df.method == "ite"], "perf_abs"))
    assert set(table.n_a) == {12}
