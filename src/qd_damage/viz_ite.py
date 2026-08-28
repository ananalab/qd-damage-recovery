"""Adaptation visuals: the robot intact, damaged, then damaged after ITE (the "three-phase" video).

Usage:
    python -m qd_damage.viz_ite --run results/repertoires/dcrlme_seed0 --scenario leg4_paralysed \
        --out media/final/ite_main --rep 2

Needs the oracle and the adaptation results of this repertoire (results/oracle/<run>, results/adaptation/<run>).
Writes in --out: videos/ (MP4 + GIF), interactive/ (self-contained 3D), figures/ and figures_en/ (PNG + PDF),
data/ and index.html. Figures exist in French (portfolio) and English (report).
"""

import argparse
import html as html_lib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qd_damage.config import load_run_config, performance_reference  # noqa: E402
from qd_damage.envs import damage_scale, get_scenario, load_damages, make_env  # noqa: E402
from qd_damage.repertoires import REPO_ROOT, git_commit, load_policy_params, make_policy_network  # noqa: E402
from qd_damage.viz import (  # noqa: E402
    _fixed_camera,
    _save,
    image,
    plot_gait,
    plot_trajectories,
    render,
    render_frames,
    rollout,
    side_by_side,
    write_gif,
    write_interactive,
    write_mp4,
)

RES = REPO_ROOT / "results"
TEXT = {
    "fr": {
        "feet": ["patte 1\navant G", "patte 2\narrière G", "patte 3\narrière D", "patte 4\navant D"],
        "trials_title": "Démarches essayées : fraction du temps où chaque pied touche le sol (noir = souvent)",
        "observed": "performance observée",
        "best_so_far": "meilleure jusqu'ici",
        "stop": "arrêt d'ITE",
        "trial": "essai sur le robot blessé",
        "perf": "performance\n(1 ≈ 15 m en 12,5 s)",
        "intact": "Robot intact",
        "damage": "Blessure",
        "front": "pattes avant au sol (fraction du temps)",
        "hind": "pattes arrière au sol (fraction du temps)",
        "ite_trials": "essais d'ITE",
        "perf_bar": "performance (1 ≈ 15 m en 12,5 s)",
        "clips": ["1. Intact", "2. Blessé, même démarche", "3. Blessé, après ITE"],
        "time": "t = {:.1f} s",
    },
    "en": {
        "feet": ["leg 1\nfront L", "leg 2\nhind L", "leg 3\nhind R", "leg 4\nfront R"],
        "trials_title": "Gaits tried by ITE: fraction of time each foot touches the ground (black = often)",
        "observed": "observed performance",
        "best_so_far": "best so far",
        "stop": "ITE stops",
        "trial": "trial on the damaged robot",
        "perf": "performance\n(1 ≈ 15 m in 12.5 s)",
        "intact": "Intact robot",
        "damage": "Damage",
        "front": "front feet on ground (fraction of time)",
        "hind": "hind feet on ground (fraction of time)",
        "ite_trials": "ITE trials",
        "perf_bar": "performance (1 ≈ 15 m in 12.5 s)",
        "clips": ["1. Intact", "2. Damaged, same gait", "3. Damaged, after ITE"],
        "time": "t = {:.1f} s",
    },
}


def damaged_legs(scenario: dict):
    """Indices (0-3) of the legs affected by the damage."""
    return sorted({int(j.split(".")[0].split("_")[1]) - 1 for j in scenario["joints"]})


def plot_ite_trials(
    trials: pd.DataFrame, descriptors: np.ndarray, hurt, stem: Path, f0: float, scale: float, lang: str = "fr"
):
    """For each ITE trial: the feet-contact profile of the gait tried, and its performance."""
    T = TEXT[lang]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    cells = trials.cell.to_numpy()
    im = ax1.imshow(
        descriptors[cells].T,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=[0.5, len(cells) + 0.5, 3.5, -0.5],
    )
    ax1.set_yticks(range(4), T["feet"], fontsize=8)
    for k in hurt:
        ax1.get_yticklabels()[k].set_color("#c8413b")
        ax1.get_yticklabels()[k].set_fontweight("bold")
    ax1.set_title(T["trials_title"], fontsize=10, loc="left")
    fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.01)
    perf = (trials.observed.to_numpy() - f0) / scale
    ax2.plot(np.arange(1, len(cells) + 1), perf, "o-", color="#3a9d5d", label=T["observed"])
    ax2.plot(
        np.arange(1, len(cells) + 1), np.maximum.accumulate(perf), color="#222", lw=1.2, ls="--", label=T["best_so_far"]
    )
    stop = trials.stop_trial.dropna()
    if len(stop):
        ax2.axvline(float(stop.iloc[0]), color="#c8413b", lw=1.2, ls=":", label=T["stop"])
    ax2.set_xlabel(T["trial"])
    ax2.set_xticks(range(1, len(cells) + 1))
    ax2.set_ylabel(T["perf"])
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, stem)


def plot_map_before_after(
    oracle: np.lib.npyio.NpzFile, scenario: str, stem: Path, f0: float, scale: float, tested=None, lang: str = "fr"
):
    """Repertoire map: front vs hind feet ground contact, colour = performance."""
    T = TEXT[lang]
    names = list(oracle["scenarios"])
    filled = np.isfinite(oracle["repertoire_fitness"])
    d = oracle["repertoire_descriptors"][filled]
    x, y = d[:, [0, 3]].mean(1), d[:, [1, 2]].mean(1)
    before = (oracle["fitness"][names.index("intact")].mean(-1)[filled] - f0) / scale
    after = (oracle["fitness"][names.index(scenario)].mean(-1)[filled] - f0) / scale
    vmin, vmax = np.percentile(np.concatenate([before, after]), [2, 98])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    for ax, v, title in zip(axes, [before, after], [T["intact"], f"{T['damage']}: {scenario}"], strict=False):
        order = np.argsort(v)
        sc = ax.scatter(x[order], y[order], c=v[order], s=10, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(T["front"])
        ax.set_ylabel(T["hind"])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
    if tested is not None:
        all_d = oracle["repertoire_descriptors"]
        axes[1].plot(
            all_d[tested][:, [0, 3]].mean(1),
            all_d[tested][:, [1, 2]].mean(1),
            "o-",
            ms=5,
            mfc="none",
            color="#c8413b",
            lw=1,
            label=T["ite_trials"],
        )
        axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=axes, shrink=0.8, label=T["perf_bar"])
    _save(fig, stem)


def representative_rollout(env, net, params, action_scale=None, n: int = 8):
    """Play n episodes (different random starts) and return the median one in fitness.

    A single episode is very noisy and could mislead either way; the median episode is neither the best nor the
    worst. The mean and standard deviation over the n episodes are kept as well.
    """
    runs = [rollout(env, net, params, seed=s, action_scale=action_scale) for s in range(n)]
    fits = np.array([r["fitness"] for r in runs])
    median = runs[int(np.argsort(fits)[n // 2])]
    dists = np.array([r["distance"] for r in runs])
    return {
        **median,
        "fitness_mean": float(fits.mean()),
        "fitness_std": float(fits.std()),
        "distance_mean": float(dists.mean()),
        "distance_std": float(dists.std()),
        "n_episodes": n,
    }


def write_follow_gifs(env, clips: dict, stop: int, out_dir: Path):
    """Three-phase GIF with following cameras (the robot stays large), in French and English."""
    dt = float(env.sys.config.dt)
    for lang, name in (("fr", "trois_temps_camera_suiveuse.gif"), ("en", "three_phases_follow.gif")):
        titles = TEXT[lang]["clips"]
        unit = "essais" if lang == "fr" else "trials"
        titles = [titles[0], titles[1], f"{titles[2]} ({stop} {unit})"]
        labels = [f"{t}: {c['distance']:.1f} m" for t, c in zip(titles, clips.values(), strict=False)]
        frames = side_by_side(
            [
                render_frames(env, c["qps"], 320, 240, lab, None, every=2)
                for c, lab in zip(clips.values(), labels, strict=False)
            ]
        )
        write_gif(frames, out_dir / "videos" / name, 2 * dt)


