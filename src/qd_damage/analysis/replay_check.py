"""Check that replaying trials from the condition bank gives the same results as real simulated episodes.

Usage:
    python -m qd_damage.analysis.replay_check --banks results/bank --adaptation results/adaptation

On the 9 Ant repertoires of the first campaign, ITE, top-k, random and best-intact were run with real episodes
(qd_damage.adaptation, 5 repetitions). The same methods, with the same settings, are replayed here from the bank
(20 repetitions), and the performance of the recommended gait after k trials is compared, per (repertoire, damage),
in the normalised performance v of the first campaign. Writes results/analysis_gecco/replay_check.csv.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from qd_damage.analysis.gecco import REPO_ROOT
from qd_damage.config import load_yaml, performance_reference
from qd_damage.offline import adapt_bank, load_bank, offline_config

METHODS = ("best_intact", "random", "top_k", "ite")
RUNS = [f"{a}_seed{s}" for a in ("me", "pgame", "dcrlme") for s in range(3)]


def replay(bank_dir: Path, n_repeats: int) -> pd.DataFrame:
    """Offline replay with the ITE settings of the first campaign (given in v, converted to u for this bank)."""
    bank = load_bank(bank_dir)
    f0 = float(np.load(Path(bank_dir) / "bank.npz")["f0"])
    intact = np.nanmean(np.load(Path(bank_dir) / "bank.npz")["fitness"][0], -1)
    progress = float(np.nanmax(intact) - f0)
    _, scale = performance_reference(250)
    ite = load_yaml("adaptation")["ite"]
    cfg = copy.deepcopy(offline_config())
    k = (scale / progress) ** 2  # variance in v -> variance in u
    cfg["gp"]["ant"]["feet"] = {"lengthscale": ite["lengthscale"], "variance": ite["signal_variance"] * k,
                                "noise": ite["noise_variance"] * k}
    cfg["kappa"]["ite"] = ite["kappa"]  # kappa multiplies a standard deviation: same in both units
    rows = adapt_bank(bank_dir, METHODS, n_repeats, cfg, groups=("id",))
    df = pd.DataFrame(rows, columns=["run", "robot", "algo", "seed", "condition", "group", "method", "rep", "trial",
                                     "u_true", "u_observed", "stop_trial", "distinct_cells", "u_oracle"])
    df["v"] = df.u_true * progress / scale
    del bank
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--banks", type=Path, default=REPO_ROOT / "results" / "bank")
    parser.add_argument("--adaptation", type=Path, default=REPO_ROOT / "results" / "adaptation")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "analysis_gecco")
    args = parser.parse_args()
    f0, scale = performance_reference(250)
    rows = []
    for run in RUNS:
        off = replay(args.banks / run, args.repeats)
        real = pd.read_csv(args.adaptation / run / "trials.csv")
        real["v"] = (real.recommended_true - f0) / scale
        for t in (1, 3, 5, 10, 20):
            a = off[off.trial == t].groupby(["condition", "method"]).v.mean()
            b = real[real.trial == t].groupby(["scenario", "method"]).v.mean()
            b.index = b.index.set_names(["condition", "method"])
            both = pd.concat({"replay": a, "real": b}, axis=1).dropna().reset_index()
            both["run"], both["trial"] = run, t
            rows.append(both)
    df = pd.concat(rows, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "replay_check.csv", index=False)
    summary = df.groupby(["method", "trial"]).apply(
        lambda g: pd.Series({"replay": g.replay.mean(), "real": g.real.mean(),
                             "corr": np.corrcoef(g.replay, g.real)[0, 1] if g.real.std() > 0 else np.nan,
                             "mean_abs_diff": (g.replay - g.real).abs().mean()}),
        include_groups=False,
    )
    print(summary.round(3))


if __name__ == "__main__":
    main()
