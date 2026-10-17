"""Analysis of the second campaign: adaptation replayed on the condition banks (qd_damage.offline).

Usage:
    python -m qd_damage.analysis.gecco --offline results/offline/final --banks results/bank

Statistical unit: one (repertoire, condition) pair; the repetitions of a method on a unit are averaged first.
Aggregates are interquartile means (IQM) with 95 % stratified bootstrap intervals, resampling repertoires within each
robot (Agarwal et al. 2021). Paired comparisons between methods use two-sided Wilcoxon signed-rank tests on the units
and linear mixed models with a random intercept per repertoire; p-values of the pre-registered family (H1 to H3) are
Holm-corrected.

Outputs: results/analysis_gecco/*.csv, figures in media/gecco/, LaTeX tables and macros in report/gecco/generated/.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, trim_mean, wilcoxon

from qd_damage.config import CONFIG_DIR

REPO_ROOT = CONFIG_DIR.parent
TRIALS = (1, 3, 5, 10, 20)
GROUP_ORDER = ["id", "ood_jammed", "ood_inverted", "ood_sensor", "ood_slippery", "ood_heavy", "gap"]
GROUP_LABELS = {
    "id": "In-family damages",
    "ood_jammed": "Jammed motor",
    "ood_inverted": "Inverted motor",
    "ood_sensor": "Lost joint sensors",
    "ood_slippery": "Slippery foot",
    "ood_heavy": "Heavy foot",
    "gap": "Reality gap (+ damage)",
    "ood": "Out-of-family damages",
}
METHOD_LABELS = {
    "best_intact": "Best intact",
    "random": "Random",
    "top_k": "Top-$k$",
    "ite": "ITE",
    "ite_norep": "ITE, no repeat",
    "ite_joint": "ITE, joint usage",
    "robust_top_k": "Top-$k$, sim.\\ prior",
    "aprol": "APROL-like",
    "itesd": "ITE-SD (ours)",
    "robust_td3": "Robust TD3",
}
ROBOT_LABELS = {"ant": "Ant", "walker2d": "Walker2d", "halfcheetah": "HalfCheetah"}
ALGO_LABELS = {"me": "MAP-Elites", "pgame": "PGA-ME", "dcrlme": "DCRL-ME"}


def iqm(x) -> float:
    return float(trim_mean(np.asarray(x, float), 0.25))


def family(group: str) -> str:
    """id, ood (all out-of-family damage types) or gap."""
    return "ood" if group.startswith("ood") else group


# ----------------------------------------------------------------------------------------------- loading


def load_offline(folder: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(Path(folder).glob("*.csv.gz"))]
    df = pd.concat(frames, ignore_index=True)
    df["family"] = df.group.map(family)
    return df


def units(df: pd.DataFrame, trials=TRIALS) -> pd.DataFrame:
    """Mean over repetitions, one row per (repertoire, condition, method, trial)."""
    keys = ["run", "robot", "algo", "seed", "condition", "group", "family", "method", "trial"]
    sel = df[df.trial.isin(trials)]
    return sel.groupby(keys, as_index=False).agg(
        u=("u_true", "mean"),
        oracle=("u_oracle", "mean"),
        distinct=("distinct_cells", "mean"),
        stop=("stop_trial", "median"),
    )


# ----------------------------------------------------------------------------------------------- statistics


def bootstrap_iqm(values: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple:
    """IQM of `values.u` and its 95 % interval, resampling repertoires (runs) within each robot."""
    rng = np.random.default_rng(seed)
    by_run = {r: g.u.to_numpy() for r, g in values.groupby("run")}
    strata = values.groupby("robot").run.unique()
    stats = []
    for _ in range(n_boot):
        sample = np.concatenate([by_run[r] for runs in strata for r in rng.choice(runs, size=len(runs), replace=True)])
        stats.append(iqm(sample))
    return iqm(values.u), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def aggregate(u: pd.DataFrame, by=("family", "method", "trial"), n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for key, g in u.groupby(list(by)):
        est, lo, hi = bootstrap_iqm(g, n_boot)
        rows.append(
            {
                **dict(zip(by, key, strict=True)),
                "iqm": est,
                "ci_low": lo,
                "ci_high": hi,
                "mean": g.u.mean(),
                "n_units": len(g),
            }
        )
    return pd.DataFrame(rows)


def _mixed_mean(diff: pd.DataFrame):
    """Mean paired difference with a random intercept per repertoire: (estimate, ci_low, ci_high, p)."""
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        fit = smf.mixedlm("d ~ 1", diff, groups=diff["run"]).fit(reml=True)
    ci = fit.conf_int().loc["Intercept"]
    return float(fit.params["Intercept"]), float(ci[0]), float(ci[1]), float(fit.pvalues["Intercept"])


def paired(u: pd.DataFrame, a: str, b: str, trial: int, subset: pd.Series | None = None) -> dict:
    """a - b on the units of `subset`, at a given number of trials."""
    sel = u[(u.trial == trial) & u.method.isin([a, b])]
    if subset is not None:
        sel = sel[subset.reindex(sel.index, fill_value=False)]
    wide = sel.pivot_table(index=["run", "robot", "condition"], columns="method", values="u").dropna()
    d = (wide[a] - wide[b]).rename("d").reset_index()
    est, lo, hi, p_mixed = _mixed_mean(d) if d.run.nunique() > 2 else (d.d.mean(), np.nan, np.nan, np.nan)
    nonzero = d.d[d.d != 0]
    return {
        "a": a,
        "b": b,
        "trial": trial,
        "n": len(d),
        "mean_diff": d.d.mean(),
        "median_diff": d.d.median(),
        "a_better": int((d.d > 0).sum()),
        "a_worse": int((d.d < 0).sum()),
        "p_wilcoxon": float(wilcoxon(nonzero).pvalue) if len(nonzero) > 0 else 1.0,
        "mixed_est": est,
        "mixed_low": lo,
        "mixed_high": hi,
        "p_mixed": p_mixed,
    }


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m, running, out = len(p), 0.0, pd.Series(index=p.index, dtype=float)
    for k, idx in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[idx]))
        out[idx] = running
    return out


def hypothesis_tests(u: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered family (docs/protocol.md): H1, H2, H3, Holm-corrected together."""
    rows = []
    for t in (10, 20):
        rows.append({"hypothesis": "H1", "family": "id", **paired(u, "ite_norep", "ite", t, u.family == "id")})
    for t in (3, 5, 10):
        for b in ("ite", "top_k"):
            rows.append({"hypothesis": "H2", "family": "id", **paired(u, "itesd", b, t, u.family == "id")})
    for fam in ("ood", "gap"):
        for t in (3, 5, 10):
            rows.append({"hypothesis": "H3", "family": fam, **paired(u, "itesd", "ite", t, u.family == fam)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_wilcoxon)
    return out


# ----------------------------------------------------------------------------------------------- banks


def bank_features(bank_root: Path, runs) -> tuple:
    """Per (run, test condition): prior fidelity and how much of the damage response each model explains.

    - rho: Spearman correlation between intact and damaged performance over the cells;
    - best_rank: rank, in the intact order, of the best cell after damage (0 = the best intact gait);
    - r2_sd: share of the variance of the response (damaged - intact) explained by its projection on the principal
      components of the simulated responses (the subspace of the ITE-SD kernel);
    Also returns the spectrum of the simulated responses of each repertoire (explained variance ratios).
    """
    from qd_damage.offline import load_bank

    rows, spectra = [], {}
    for run in runs:
        bank = load_bank(Path(bank_root) / run)
        prior = bank["prior"]
        resid_train = bank["train"] - prior
        centred = resid_train - resid_train.mean(0)
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        spectra[run] = (s**2) / (s**2).sum()
        mean_resp = resid_train.mean(0)
        for c, (name, group) in enumerate(zip(bank["names"], bank["groups"], strict=True)):
            if group in ("intact", "train"):
                continue
            u_cond = bank["u"][c]
            n_eps = int(np.isfinite(u_cond[0]).sum())
            truth = u_cond[:, n_eps // 2 : n_eps].mean(-1)
            resp = truth - prior - mean_resp
            proj = vt.T @ (vt @ resp)
            order = np.argsort(-prior)
            rank = int(np.flatnonzero(order == np.argmax(truth))[0])
            rows.append(
                {
                    "run": run,
                    "condition": name,
                    "group": group,
                    "family": family(group),
                    "rho": float(spearmanr(prior, truth).statistic),
                    "best_rank": rank,
                    "r2_sd": float(1 - ((resp - proj) ** 2).sum() / (resp**2).sum()),
                    "oracle_gain": float(truth.max() - truth[order[0]]),
                }
            )
    return pd.DataFrame(rows), spectra


# ----------------------------------------------------------------------------------------------- outputs


def macro_name(*parts) -> str:
    """LaTeX macro name from parts (letters only)."""
    words = {
        "0": "Zero",
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
    }
    out = ""
    for p in parts:
        for chunk in str(p).replace("-", "_").split("_"):
            if chunk.isdigit():
                out += "".join(words[ch] for ch in chunk)
            else:
                out += chunk[:1].upper() + chunk[1:]
    return "".join(ch for ch in out if ch.isalpha())


def write_macros(path: Path, macros: dict):
    lines = ["% Generated by qd_damage.analysis.gecco; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    path.write_text("\n".join(lines) + "\n")


def fmt_p(p: float) -> str:
    return "<0.001" if p < 1e-3 else f"={p:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offline", type=Path, default=REPO_ROOT / "results" / "offline" / "final")
    parser.add_argument("--banks", type=Path, default=REPO_ROOT / "results" / "bank")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "analysis_gecco")
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_offline(args.offline)
    u = units(df)
    u.to_csv(args.out / "units.csv.gz", index=False)
    agg = aggregate(u, ("family", "method", "trial"), args.boot)
    agg.to_csv(args.out / "iqm_family.csv", index=False)
    aggregate(u, ("robot", "family", "method", "trial"), args.boot).to_csv(args.out / "iqm_robot.csv", index=False)
    aggregate(u, ("group", "method", "trial"), args.boot).to_csv(args.out / "iqm_group.csv", index=False)
    tests = hypothesis_tests(u)
    tests.to_csv(args.out / "hypotheses.csv", index=False)
    feats, spectra = bank_features(args.banks, sorted(u.run.unique()))
    feats.to_csv(args.out / "bank_features.csv", index=False)
    with open(args.out / "spectra.json", "w") as f:
        json.dump({k: v[:20].tolist() for k, v in spectra.items()}, f)
    print(agg[agg.trial.isin([1, 5, 20])].pivot_table(index=["family", "method"], columns="trial", values="iqm"))
    print(tests[["hypothesis", "family", "a", "b", "trial", "mean_diff", "a_better", "a_worse", "p_holm"]])


if __name__ == "__main__":
    main()
