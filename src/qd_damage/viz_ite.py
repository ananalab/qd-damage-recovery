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


def plot_strip(env, clips: dict, stem: Path, lang: str, stop: int, times=(0.0, 4.0, 8.0, 12.4)):
    """Still frames for print: one row per situation, a few time points."""
    T = TEXT[lang]
    dt = float(env.sys.config.dt)
    fig, axes = plt.subplots(len(clips), len(times), figsize=(2.6 * len(times), 2.1 * len(clips)))
    titles = [T["clips"][0], T["clips"][1], f"{T['clips'][2]} ({stop} {'essais' if lang == 'fr' else 'trials'})"]
    for r, ((_, c), title) in enumerate(zip(clips.items(), titles, strict=True)):
        for j, t in enumerate(times):
            idx = min(int(round(t / dt)), len(c["qps"]) - 1)
            # Camera following the robot (close-up), readable in print; the distance is given on the left.
            axes[r, j].imshow(image.render_array(env.sys, c["qps"][idx], 360, 270))
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            if r == 0:
                axes[r, j].set_title(T["time"].format(idx * dt), fontsize=9)
        axes[r, 0].set_ylabel(f"{title}\n{c['distance']:.1f} m", fontsize=9)
    fig.tight_layout()
    _save(fig, stem)


def load_episode(run_dir: Path, scenario_name: str, rep: int):
    """Everything needed to show one ITE run: the trials, where ITE stopped, and the three median clips
    (best intact gait intact, same gait damaged, gait recommended by ITE damaged). Seeds are fixed, so the
    clips are the same at every call."""
    run = json.loads((run_dir / "run.json").read_text())
    common = load_run_config(run["algo"], run["profile"])["common"]
    damages = load_damages()
    scenario = get_scenario(scenario_name, damages)
    dmg_scale = damage_scale(scenario, damages)

    oracle = np.load(RES / "oracle" / run_dir.name / "oracle.npz")
    names = list(oracle["scenarios"])
    best_intact_cell = int(np.nanargmax(oracle["fitness"][names.index("intact")].mean(-1)))
    trials = pd.read_csv(RES / "adaptation" / run_dir.name / "trials.csv")
    trials = trials[(trials.scenario == scenario_name) & (trials.method == "ite") & (trials.rep == rep)]
    stop = int(trials.stop_trial.dropna().iloc[0]) if trials.stop_trial.notna().any() else int(trials.trial.max())
    ite_cell = int(trials[trials.trial == stop].recommended_cell.iloc[0])

    env = make_env(common)
    net = make_policy_network(common, env.action_size)
    p_intact = load_policy_params(run_dir, best_intact_cell, net, env.observation_size)
    p_ite = load_policy_params(run_dir, ite_cell, net, env.observation_size)
    clips = {
        "intact": {**representative_rollout(env, net, p_intact), "label": "1. Intact"},
        "blesse": {**representative_rollout(env, net, p_intact, dmg_scale), "label": "2. Blessé, même démarche"},
        "apres_ite": {
            **representative_rollout(env, net, p_ite, dmg_scale),
            "label": f"3. Blessé, après ITE ({stop} essais)",
        },
    }
    return dict(
        run=run,
        common=common,
        scenario=scenario,
        oracle=oracle,
        trials=trials,
        stop=stop,
        ite_cell=ite_cell,
        best_intact_cell=best_intact_cell,
        env=env,
        clips=clips,
    )


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


def make_ite_media(run_dir: Path, scenario_name: str, out_dir: Path, rep: int = 0):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    for sub in ("figures", "figures_en", "videos", "interactive", "data"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    e = load_episode(run_dir, scenario_name, rep)
    run, env, clips, trials, stop, oracle = e["run"], e["env"], e["clips"], e["trials"], e["stop"], e["oracle"]
    f0, scale = performance_reference(e["common"]["env"]["episode_length"])
    dt = float(env.sys.config.dt)
    labels = {k: f"{c['label']} : {c['distance']:.1f} m" for k, c in clips.items()}

    # Videos
    paths = [c["path_xy"] for c in clips.values()]
    cam = _fixed_camera(paths, 480, 360)
    hd = side_by_side([render_frames(env, c["qps"], 480, 360, labels[k], cam) for k, c in clips.items()])
    write_mp4(hd, out_dir / "videos" / "trois_temps_camera_fixe.mp4", fps=1 / dt)
    cam_s = _fixed_camera(paths, 320, 240)
    small = side_by_side([render_frames(env, c["qps"], 320, 240, labels[k], cam_s, every=2) for k, c in clips.items()])
    write_gif(small, out_dir / "videos" / "trois_temps_camera_fixe.gif", 2 * dt)
    for k, c in clips.items():
        write_mp4(
            render_frames(env, c["qps"], 768, 576, labels[k]), out_dir / "videos" / f"{k}_camera_suiveuse.mp4", 1 / dt
        )
        write_interactive(
            env,
            c["qps"],
            out_dir / "interactive" / f"{k}.html",
            title=f"Ant — {c['label']}",
            caption=f"Blessure {scenario_name} · avance de {c['distance']:.1f} m en {len(c['qps']) * dt:.1f} s",
        )

    write_follow_gifs(env, clips, stop, out_dir)

    # Figures (French for the portfolio, English for the report)
    all_desc = oracle["repertoire_descriptors"]
    hurt = damaged_legs(e["scenario"])
    for lang, sub in (("fr", "figures"), ("en", "figures_en")):
        plot_ite_trials(
            trials, all_desc, hurt, out_dir / sub / ("essais_ite" if lang == "fr" else "ite_trials"), f0, scale, lang
        )
        plot_map_before_after(
            oracle,
            scenario_name,
            out_dir / sub / ("carte_avant_apres" if lang == "fr" else "map_before_after"),
            f0,
            scale,
            tested=trials.cell.to_numpy()[:stop],
            lang=lang,
        )
        plot_strip(
            env, clips, out_dir / sub / ("planche_trois_temps" if lang == "fr" else "three_phase_strip"), lang, stop
        )
    plot_gait(clips, out_dir / "figures" / "contacts_pieds")
    plot_trajectories(
        {"intact": clips["intact"], "damaged": clips["blesse"]}, out_dir / "figures" / "trajets_intact_blesse"
    )
    plot_trajectories(clips, out_dir / "figures" / "trajets_trois_temps", colors=["#2a78d6", "#c8413b", "#3a9d5d"])

    summary = {
        "run": run_dir.name,
        "algo": run["algo"],
        "seed": run["seed"],
        "scenario": scenario_name,
        "rep": rep,
        "code_commit": git_commit(),
        "ite_trials_to_stop": stop,
        "ite_cell": e["ite_cell"],
        "best_intact_cell": e["best_intact_cell"],
        "clips": {
            k: {
                "label": c["label"],
                "shown_episode": "médian sur 8",
                "distance_x_m": c["distance"],
                "fitness": c["fitness"],
                "distance_mean_m": c["distance_mean"],
                "distance_std_m": c["distance_std"],
                "fitness_mean": c["fitness_mean"],
                "fitness_std": c["fitness_std"],
                "perf_norm_mean": (c["fitness_mean"] - f0) / scale,
            }
            for k, c in clips.items()
        },
        "command": (
            f"python -m qd_damage.viz_ite --run {run_dir} --scenario {scenario_name} --out {out_dir} --rep {rep}"
        ),
    }
    (out_dir / "data" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    trials.to_csv(out_dir / "data" / "essais_ite.csv", index=False)
    _write_index(out_dir, summary)
    return out_dir


def _write_index(out_dir: Path, s: dict):
    c = s["clips"]
    title = f"Adaptation à la blessure « {s['scenario']} » avec ITE"
    (out_dir / "index.html").write_text(
        render(
            "ite_index.html",
            title=html_lib.escape(title),
            algo=s["algo"],
            seed=s["seed"],
            trials=s["ite_trials_to_stop"],
            code=s["code_commit"],
            command=html_lib.escape(s["command"]),
            **{
                f"{name}_{stat}": f"{c[key][f'distance_{stat}_m']:.1f}"
                for name, key in (("intact", "intact"), ("damaged", "blesse"), ("ite", "apres_ite"))
                for stat in ("mean", "std")
            },
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rep", type=int, default=0)
    args = parser.parse_args()
    print(make_ite_media(args.run, args.scenario, args.out, args.rep))


if __name__ == "__main__":
    main()
