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


