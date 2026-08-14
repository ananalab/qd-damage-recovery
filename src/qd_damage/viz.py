"""Repertoire visuals: figures, videos and interactive 3D pages, laid out for reuse (portfolio, report).

Usage:
    python -m qd_damage.viz --runs results/repertoires/*_seed? --out media/final/repertoires --title "..."

Layout written in --out:
    index.html      overview page (light: it links to the files below)
    README.md       file catalogue, provenance and the command that regenerates everything
    figures/        each figure as PNG (web, slides) and vector PDF (LaTeX)
    videos/         H.264 MP4 and GIF
    interactive/    self-contained 3D pages (Brax viewer + three.js inlined), work offline
    data/           displayed numbers (summary.json), trajectories, copies of the metrics.csv / run.json used

Page and figure texts are in French (portfolio); the report figures in English come from qd_damage.analysis.
"""

import argparse
import csv
import datetime
import html as html_lib
import json
import shutil
from pathlib import Path
from string import Template

import imageio
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from brax.v1.io import image  # noqa: E402  (software rendering, needs pytinyrenderer)
from brax.v1.io.json import dumps as brax_dumps  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from qd_damage.config import load_run_config  # noqa: E402
from qd_damage.envs import LEG_ACTIONS, make_env  # noqa: E402
from qd_damage.repertoires import REPO_ROOT, git_commit, load_policy_params, make_policy_network  # noqa: E402

VIEWER_JS = Path(__file__).parent / "assets" / "brax_viewer_v0.1.0_three_r135.js"
TEMPLATES = Path(__file__).parent / "templates"
# Font shipped with matplotlib: handles accents, unlike the default PIL font.
_FONT_PATH = font_manager.findfont("DejaVu Sans")

ALGO_LABELS = {"me": "MAP-Elites", "pgame": "PGA-ME", "dcrlme": "DCRL-ME"}
ALGO_COLORS = {"me": "#2a78d6", "pgame": "#e07b39", "dcrlme": "#3a9d5d"}
CONDITION_COLORS = {"intact": "#2a78d6", "damaged": "#c8413b"}


# --------------------------------------------------------------------------- figures


def _save(fig, stem: Path):
    """Save as PNG (200 dpi) and vector PDF."""
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})  # reproducible files
    plt.close(fig)


def read_metrics(run_dir: Path) -> dict:
    with open(Path(run_dir) / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _run_info(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text())


def plot_progress(runs: dict, stem: Path):
    """QD-score, coverage and max fitness against the number of evaluations (one line per run)."""
    panels = [("qd_score", "QD-score"), ("coverage", "Couverture (%)"), ("max_fitness", "Meilleure fitness")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    labelled = set()
    for run_dir in runs.values():
        algo = _run_info(run_dir)["algo"]
        m = read_metrics(run_dir)
        for ax, (key, _) in zip(axes, panels, strict=False):
            ax.plot(
                m["evaluations"],
                m[key],
                color=ALGO_COLORS[algo],
                lw=2,
                alpha=0.85,
                label=None if algo in labelled else ALGO_LABELS[algo],
            )
        labelled.add(algo)
    for ax, (_, title) in zip(axes, panels, strict=False):
        ax.set_title(title)
        ax.set_xlabel("Évaluations (épisodes)")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, stem)


def rollout(env, policy_network, params, seed: int = 0, paralysed_leg=None, action_scale=None):
    """Play one episode. Damage: `action_scale` (per-action factor, see envs.damage_scale) or, as a shortcut,
    `paralysed_leg` (e.g. "leg_4"), which zeroes the actions of that leg."""
    scale = jnp.ones(env.action_size) if action_scale is None else jnp.asarray(action_scale, jnp.float32)
    if paralysed_leg is not None:
        scale = scale.at[jnp.array(LEG_ACTIONS[paralysed_leg])].set(0.0)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    act = jax.jit(lambda s: policy_network.apply(params, s.obs) * scale)
    state = reset(jax.random.key(seed))
    qps, contacts, total_reward = [], [], 0.0
    while True:
        qps.append(state.qp)
        state = step(state, act(state))
        total_reward += float(state.reward)
        if bool(state.done):
            # QDax envs have an AutoResetWrapper: at the end `state` is already back at the start,
            # so we stop at the last state before the reset.
            break
        contacts.append(np.asarray(state.info["state_descriptor"]))
    path_xy = np.array([np.asarray(qp.pos[0, :2]) for qp in qps])  # torso path seen from above
    return {
        "qps": qps,
        "contacts": np.array(contacts),  # (T, 4): 1 when the foot touches the ground
        "fitness": total_reward,
        "path_xy": path_xy,
        "distance": float(path_xy[-1, 0] - path_xy[0, 0]),  # forward displacement (x)
    }


def side_by_side(columns):
    n = max(len(c) for c in columns)
    columns = [c + [c[-1]] * (n - len(c)) for c in columns]  # when an episode ends earlier
    return [np.concatenate([c[t] for c in columns], axis=1) for t in range(n)]


