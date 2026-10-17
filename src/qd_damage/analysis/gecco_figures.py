"""Figures of the second campaign (paper), from the tables written by qd_damage.analysis.gecco."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qd_damage.analysis.gecco import GROUP_LABELS, METHOD_LABELS, ROBOT_LABELS  # noqa: E402

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Fixed colour per method (never re-assigned when a figure shows fewer methods).
COLORS = {
    "itesd": "#2a78d6",
    "ite": "#eb6834",
    "ite_norep": "#1baf7a",
    "top_k": "#eda100",
    "robust_top_k": "#e87ba4",
    "aprol": "#008300",
    "robust_td3": "#6250d6",
    "itesd_hybrid": "#2a78d6",
    "ite_joint": "#e34948",
    "random": "#8a8985",
    "best_intact": "#8a8985",
}
STYLES = {"itesd_hybrid": (0, (4, 2)), "best_intact": (0, (1, 2)), "random": (0, (1, 2))}
MAIN = ["top_k", "ite", "ite_norep", "robust_top_k", "aprol", "itesd"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6, "pdf.fonttype": 42,
    "svg.hashsalt": "qd", "savefig.dpi": 200,
})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def anytime(curves: pd.DataFrame, robust: pd.DataFrame | None, out: Path, families=("id", "ood", "gap"),
            methods=MAIN, name="anytime"):
    """IQM of u after k trials, one panel per condition family; shaded 95 % bootstrap interval."""
    fig, axes = plt.subplots(1, len(families), figsize=(7.0, 2.3), sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), families, strict=True):
        sub = curves[curves.family == fam]
        for m in methods:
            c = sub[sub.method == m].sort_values("trial")
            if c.empty:
                continue
            ax.plot(c.trial, c.iqm, color=COLORS[m], ls=STYLES.get(m, "-"), label=METHOD_LABELS[m])
            ax.fill_between(c.trial, c.ci_low, c.ci_high, color=COLORS[m], alpha=0.12, lw=0)
        if robust is not None and fam in set(robust.family):
            r = robust[robust.family == fam].iloc[0]
            ax.axhline(r.iqm, color=COLORS["robust_td3"], ls=(0, (5, 2)), lw=1.2, label=METHOD_LABELS["robust_td3"])
        ax.set_title(GROUP_LABELS[fam])
        ax.set_xlabel("Trials on the damaged robot")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    np.atleast_1d(axes)[0].set_ylabel("Recommended gait, IQM of $u$\n(share of best intact progress)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.12))
    _save(fig, out, name)


def spectrum(spectra: dict, out: Path, name="spectrum"):
    """Cumulative variance of the simulated damage responses explained by the first principal components."""
    fig, ax = plt.subplots(figsize=(2.6, 2.1))
    robots = {}
    for run, ev in spectra.items():
        robot = next((r for r in ("walker2d", "halfcheetah") if run.startswith(r)), "ant")
        robots.setdefault(robot, []).append(np.cumsum(ev))
    colors = {"ant": "#2a78d6", "walker2d": "#eb6834", "halfcheetah": "#1baf7a"}
    for robot, curves in robots.items():
        k = min(len(c) for c in curves)
        arr = np.array([c[:k] for c in curves])
        x = np.arange(1, k + 1)
        ax.plot(x, arr.mean(0), color=colors[robot], label=ROBOT_LABELS[robot], marker="o", ms=3)
        ax.fill_between(x, arr.min(0), arr.max(0), color=colors[robot], alpha=0.12, lw=0)
    ax.set_xlabel("Principal components")
    ax.set_ylabel("Variance of simulated\nresponses explained")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    _save(fig, out, name)


def gain_vs_fidelity(points: pd.DataFrame, out: Path, x="rho", name="gain_vs_rho", xlabel=None):
    """Gain of ITE-SD over top-k after 5 trials against a property of the condition, one dot per unit."""
    fig, ax = plt.subplots(figsize=(2.8, 2.2))
    fams = {"id": ("#2a78d6", "In-family"), "ood": ("#eb6834", "Out-of-family"), "gap": ("#1baf7a", "Reality gap")}
    for fam, (color, label) in fams.items():
        p = points[points.family == fam]
        ax.scatter(p[x], p.gain, s=8, color=color, alpha=0.55, lw=0, label=label)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel("ITE-SD $-$ top-$k$, $u$ after 5 trials")
    ax.legend(frameon=False, markerscale=1.5)
    _save(fig, out, name)
