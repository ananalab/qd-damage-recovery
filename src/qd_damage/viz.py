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
# Font shipped with matplotlib: handles accents, unlike the default PIL font.
_FONT_PATH = font_manager.findfont("DejaVu Sans")

ALGO_LABELS = {"me": "MAP-Elites", "pgame": "PGA-ME", "dcrlme": "DCRL-ME"}
ALGO_COLORS = {"me": "#2a78d6", "pgame": "#e07b39", "dcrlme": "#3a9d5d"}
CONDITION_COLORS = {"intact": "#2a78d6", "damaged": "#c8413b"}


# --------------------------------------------------------------------------- figures


def _save(fig, stem: Path):
    """Save as PNG (200 dpi) and vector PDF."""
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
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
        for ax, (key, _) in zip(axes, panels):
            ax.plot(m["evaluations"], m[key], color=ALGO_COLORS[algo], lw=2, alpha=0.85,
                    label=None if algo in labelled else ALGO_LABELS[algo])
        labelled.add(algo)
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xlabel("Évaluations (épisodes)")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, stem)


def plot_repertoire(run_dir: Path, stem: Path, title: str):
    """The descriptor has 4 dimensions (one per foot): show the 6 two-dimensional projections.

    Each point is a gait of the repertoire, placed by the ground-contact time of two feet and coloured by fitness.
    """
    data = np.load(Path(run_dir) / "repertoire.npz")
    filled = np.isfinite(data["fitnesses"])
    desc, fit = data["descriptors"][filled], data["fitnesses"][filled]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), layout="constrained")
    order = np.argsort(fit)  # best gaits drawn last (on top)
    for ax, (i, j) in zip(axes.flat, pairs):
        sc = ax.scatter(desc[order, i], desc[order, j], c=fit[order], s=9, cmap="viridis")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(f"pied {i + 1} au sol")
        ax.set_ylabel(f"pied {j + 1} au sol")
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes, shrink=0.6, label="fitness")
    fig.suptitle(f"{title} — {filled.sum()} démarches dans le répertoire")
    _save(fig, stem)


def plot_trajectories(rollouts: dict, stem: Path, colors=None):
    """Torso path seen from above: the robot is rewarded for moving right (+x)."""
    colors = colors or [CONDITION_COLORS[key] for key in rollouts]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for r, color in zip(rollouts.values(), colors):
        xy = r["path_xy"]
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.2, label=f"{r['label']} ({r['distance']:.1f} m)")
        ax.plot(*xy[-1], "o", color=color, ms=7)
    ax.plot(0, 0, "ks", ms=7, label="départ")
    ax.set_xlabel("x (m) — sens de la marche récompensé →")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    _save(fig, stem)


def plot_gait(rollouts: dict, stem: Path):
    """Feet-contact profile: one row per foot, black = the foot touches the ground."""
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(10, 1.5 * len(rollouts)), sharex=True)
    for ax, r in zip(np.atleast_1d(axes), rollouts.values()):
        ax.imshow(r["contacts"].T, aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
        ax.set_yticks(range(4), [f"pied {k + 1}" for k in range(4)])
        ax.set_title(f"{r['label']} — avance de {r['distance']:.1f} m", loc="left", fontsize=10)
    np.atleast_1d(axes)[-1].set_xlabel("pas de temps")
    fig.tight_layout()
    _save(fig, stem)


# --------------------------------------------------------------------------- simulation


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


# --------------------------------------------------------------------------- videos


def _fixed_camera(paths, width: int, height: int, ssaa: int = 2) -> image.Camera:
    """Fixed camera framing all paths, so that the robot is seen moving and not only its legs."""
    pts = np.concatenate(paths)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    d = max(3.5, 0.8 * extent + 2.5)
    target = [float(center[0]), float(center[1]), 0.3]
    hfov = 58.0
    return image.Camera(
        viewWidth=width * ssaa,
        viewHeight=height * ssaa,
        position=[target[0] + 0.3 * d, target[1] - d, 0.8 * d],
        target=target,
        up=[0, 0, 1],
        hfov=hfov,
        vfov=hfov * height / width,
    )


def render_frames(env, qps, width: int, height: int, label: str, camera=None, every: int = 1):
    """Frames of a rollout with a label on top. camera=None: Brax camera that follows the robot."""
    font = ImageFont.truetype(_FONT_PATH, max(14, height // 24))
    band = int(font.size * 1.8)
    frames = []
    for qp in qps[::every]:
        img = Image.fromarray(image.render_array(env.sys, qp, width, height, camera=camera))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width, band], fill=(255, 255, 255))
        draw.text((10, (band - font.size) // 2), label, fill=(20, 20, 20), font=font)
        frames.append(np.asarray(img))
    return frames


def side_by_side(columns):
    n = max(len(c) for c in columns)
    columns = [c + [c[-1]] * (n - len(c)) for c in columns]  # when an episode ends earlier
    return [np.concatenate([c[t] for c in columns], axis=1) for t in range(n)]


def write_mp4(frames, path: Path, fps: float):
    # H.264 + yuv420p: plays everywhere (browsers, slides).
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")


def write_gif(frames, path: Path, frame_duration_s: float):
    imageio.mimsave(path, frames, duration=round(1000 * frame_duration_s), loop=0)  # in ms


# --------------------------------------------------------------------------- 3D interactive


def write_interactive(env, qps, path: Path, title: str, caption: str):
    """Self-contained HTML page: trajectory and Brax 3D viewer inlined (no external resource)."""
    viewer_js = VIEWER_JS.read_text().replace("</script", "<\\/script")
    system_json = brax_dumps(env.sys, qps).replace("</script", "<\\/script")
    path.write_text(f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)}</title>
<style>
  html, body {{ margin:0; height:100%; background:#fff; font:14px/1.4 system-ui, sans-serif; }}
  #brax-viewer {{ position:relative; height:100vh; }}
  #caption {{ position:absolute; left:10px; bottom:10px; z-index:10; background:rgba(255,255,255,.85);
             padding:6px 10px; border-radius:6px; color:#222; max-width:70%; pointer-events:none; }}
</style></head><body>
<div id="brax-viewer"><div id="caption"><b>{html_lib.escape(title)}</b><br>{html_lib.escape(caption)}<br>
<small>Souris : tourner · molette : zoomer · clic droit : déplacer</small></div></div>
<script>var system = {system_json};</script>
<script>{viewer_js}</script>
<script>new BraxViewer(document.getElementById("brax-viewer"), system);</script>
</body></html>""")


# --------------------------------------------------------------------------- page and catalogue


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def make_media(run_dirs, out_dir: Path, title: str) -> Path:
    """Write figures, videos, 3D pages, data and the overview page. Return the path of index.html."""
    out_dir = Path(out_dir)
    dirs = {name: out_dir / name for name in ("figures", "videos", "interactive", "data")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    # One run per name (e.g. dcrlme_seed0), fixed order: ME, PGA-ME, DCRL-ME, then seed.
    order = list(ALGO_LABELS)
    run_dirs = sorted((Path(r) for r in run_dirs),
                      key=lambda r: (order.index(_run_info(r)["algo"]), _run_info(r)["seed"]))
    runs = {r.name: r for r in run_dirs}

    # Repertoire figures
    plot_progress(runs, dirs["figures"] / "progression_qd")
    for name, r in runs.items():
        info = _run_info(r)
        plot_repertoire(r, dirs["figures"] / f"repertoire_{name}", f"{ALGO_LABELS[info['algo']]} (seed {info['seed']})")

    # Best gait over all runs: intact, then with leg 4 paralysed.
    best_name, best_idx, best_fit = None, None, -np.inf
    for name, r in runs.items():
        fit = np.load(r / "repertoire.npz")["fitnesses"]
        if np.nanmax(fit) > best_fit:
            best_name, best_idx, best_fit = name, int(np.nanargmax(fit)), float(np.nanmax(fit))
    best_run = _run_info(runs[best_name])
    best_algo = best_run["algo"]
    common = load_run_config(best_algo, best_run["profile"])["common"]
    env = make_env(common)
    net = make_policy_network(common, env.action_size)
    params = load_policy_params(runs[best_name], best_idx, net, env.observation_size)
    rollouts = {
        "intact": {**rollout(env, net, params), "label": "Intact"},
        "damaged": {**rollout(env, net, params, paralysed_leg="leg_4"), "label": "Patte 4 paralysée"},
    }
    plot_gait(rollouts, dirs["figures"] / "contacts_pieds")
    plot_trajectories(rollouts, dirs["figures"] / "trajets_vus_de_dessus")

    # Videos. dt = duration of a simulation step (0.05 s): videos play in real time.
    dt = float(env.sys.config.dt)
    labels = {k: f"{r['label']} : {r['distance']:.1f} m" for k, r in rollouts.items()}
    paths = [r["path_xy"] for r in rollouts.values()]
    hd_cam = _fixed_camera(paths, 640, 480)
    hd = side_by_side([render_frames(env, r["qps"], 640, 480, labels[k], hd_cam) for k, r in rollouts.items()])
    write_mp4(hd, dirs["videos"] / "intact_vs_patte4_camera_fixe.mp4", fps=1 / dt)
    small_cam = _fixed_camera(paths, 360, 270)
    small = side_by_side(
        [render_frames(env, r["qps"], 360, 270, labels[k], small_cam, every=2) for k, r in rollouts.items()]
    )
    write_gif(small, dirs["videos"] / "intact_vs_patte4_camera_fixe.gif", 2 * dt)
    for key, r in rollouts.items():
        name = "intact" if key == "intact" else "patte4_paralysee"
        write_mp4(render_frames(env, r["qps"], 768, 576, labels[key]), dirs["videos"] / f"{name}_camera_suiveuse.mp4", 1 / dt)

    # Interactive 3D
    for key, r in rollouts.items():
        name = "intact" if key == "intact" else "patte4_paralysee"
        write_interactive(
            env, r["qps"], dirs["interactive"] / f"{name}.html",
            title=f"Ant — {r['label'].lower()}",
            caption=f"Meilleure démarche {ALGO_LABELS[best_algo]} · avance de {r['distance']:.1f} m en {len(r['qps']) * dt:.1f} s",
        )

    # Data and provenance
    summary = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "command": (
            f"python -m qd_damage.viz --runs {' '.join(_rel(r) for r in runs.values())} "
            f"--out {_rel(out_dir)} --title {json.dumps(title, ensure_ascii=False)}"
        ),
        "best_policy": {"algo": best_algo, "run": best_name, "cell_index": best_idx, "repertoire_fitness": best_fit},
        "rollouts": {
            k: {"label": r["label"], "distance_x_m": r["distance"], "fitness": r["fitness"],
                "duration_s": len(r["qps"]) * dt, "paralysed_actions": list(LEG_ACTIONS["leg_4"]) if k == "damaged" else []}
            for k, r in rollouts.items()
        },
        "runs": {},
    }
    for name, r in runs.items():
        run_info = _run_info(r)
        m = read_metrics(r)
        summary["runs"][name] = {
            "algo": run_info["algo"], "run_dir": _rel(r), "profile": run_info["profile"], "seed": run_info["seed"],
            "run_commit": run_info["git_commit"], "total_evaluations": int(m["evaluations"][-1]), "iterations": int(m["iteration"][-1]),
            "coverage_pct": float(m["coverage"][-1]), "max_fitness": float(m["max_fitness"][-1]),
            "qd_score": float(m["qd_score"][-1]), "time_min": run_info["total_time_s"] / 60,
            "device": run_info["devices"][0],
        }
        shutil.copy(r / "metrics.csv", dirs["data"] / f"metrics_{name}.csv")
        shutil.copy(r / "run.json", dirs["data"] / f"run_{name}.json")
    (dirs["data"] / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez_compressed(
        dirs["data"] / "rollouts.npz",
        **{f"{k}_path_xy": r["path_xy"] for k, r in rollouts.items()},
        **{f"{k}_contacts": r["contacts"] for k, r in rollouts.items()},
    )

    _write_index(out_dir, title, summary, runs)
    _write_readme(out_dir, title, summary)
    return out_dir / "index.html"


def _status(run: dict, machine: str = "") -> str:
    """One sentence saying whether the page shows demo runs or final runs."""
    where = f" sur {machine}" if machine else ""
    if run["profile"] is None:
        return f"Runs complets{where} ({run['iterations']} itérations par run) : résultats finaux, avant blessure."
    return f"Mini-entraînement{where} (profil « {run['profile']} », {run['iterations']} itérations) : démo, pas des résultats finaux."


def _write_index(out_dir: Path, title: str, summary: dict, runs: dict):
    ro = summary["rollouts"]
    best = summary["best_policy"]
    first = next(iter(summary["runs"].values()))
    on_cpu = "cpu" in first["device"].lower()
    machine = "le CPU du Mac" if on_cpu else f"un GPU {first['device'].split('(')[-1].rstrip(')')} sur Kaggle"
    dcrl_note = (
        " En profil local, DCRL-ME entraîne son critique 10× moins que prévu, pour tenir sur CPU."
        if first["profile"] == "local" and any(s.get("algo", a) == "dcrlme" for a, s in summary["runs"].items()) else ""
    )
    rows = "".join(
        f"<tr><td>{ALGO_LABELS[s.get('algo', a)]} (seed {s['seed']})</td><td>{s['total_evaluations']:,}</td><td>{s['coverage_pct']:.1f} %</td>"
        f"<td>{s['max_fitness']:.0f}</td><td>{s['time_min']:.1f} min</td></tr>".replace(",", " ")
        for a, s in summary["runs"].items()
    )
    repertoires = "".join(
        f'<div class="card"><img src="figures/repertoire_{a}.png" alt="Répertoire {a}"></div>' for a in runs
    )
    (out_dir / "index.html").write_text(f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)}</title>
<style>
  :root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#e4e2dc; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#161614; --fg:#ecebe6; --muted:#a3a29b; --card:#1f1f1c; --line:#33332f; }} }}
  body {{ background:var(--bg); color:var(--fg); font:16px/1.55 system-ui, sans-serif; margin:0; padding:24px 16px 64px; }}
  main {{ max-width:1100px; margin:0 auto; }}
  h1 {{ font-size:28px; margin:0 0 4px; }} h2 {{ font-size:20px; margin:40px 0 8px; }}
  a {{ color:inherit; }}
  .lead, .note {{ color:var(--muted); }} .note {{ font-size:14px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:12px 0; overflow-x:auto; }}
  img, video {{ max-width:100%; height:auto; background:#fff; border-radius:6px; display:block; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; }}
  iframe {{ width:100%; height:420px; border:0; border-radius:8px; background:#fff; }}
  table {{ border-collapse:collapse; width:100%; }} td, th {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; }}
</style></head><body><main>
<h1>{html_lib.escape(title)}</h1>
<p class="lead">{_status(first, machine)} Tous les fichiers sont listés dans <a href="README.md">README.md</a>.</p>

<h2>1. Le robot, avec sa meilleure démarche</h2>
<p>Meilleure démarche trouvée ({ALGO_LABELS[best['algo']]}, fitness {best['repertoire_fitness']:.0f}). À gauche intact ; à droite <b>la même démarche avec la patte 4 paralysée</b>. C'est le problème que le projet va résoudre.</p>
<div class="card"><video src="videos/intact_vs_patte4_camera_fixe.mp4" autoplay muted loop playsinline controls></video></div>
<div class="grid">
  <div class="card"><video src="videos/intact_camera_suiveuse.mp4" autoplay muted loop playsinline controls></video></div>
  <div class="card"><video src="videos/patte4_paralysee_camera_suiveuse.mp4" autoplay muted loop playsinline controls></video></div>
</div>
<div class="card"><img src="figures/trajets_vus_de_dessus.png" alt="Trajets vus de dessus">
<p class="note">Trajet vu de dessus pendant les {ro['intact']['duration_s']:.1f} s de l'épisode. La fitness récompense l'avancée selon x.</p></div>

<h2>2. En 3D interactive</h2>
<p>Tourne la caméra à la souris, zoome à la molette. Chaque vue est aussi une page autonome dans <code>interactive/</code>, qui marche hors ligne.</p>
<div class="grid">
  <div class="card"><b>Intact</b> — {ro['intact']['distance_x_m']:.1f} m <a href="interactive/intact.html">plein écran ↗</a><iframe src="interactive/intact.html" loading="lazy"></iframe></div>
  <div class="card"><b>Patte 4 paralysée</b> — {ro['damaged']['distance_x_m']:.1f} m <a href="interactive/patte4_paralysee.html">plein écran ↗</a><iframe src="interactive/patte4_paralysee.html" loading="lazy"></iframe></div>
</div>
<div class="card"><img src="figures/contacts_pieds.png" alt="Profil de contact des pieds">
<p class="note">Chaque ligne est un pied, noir = le pied touche le sol. C'est ce que mesure le descripteur (fraction du temps en noir, pour chaque pied).</p></div>

<h2>3. L'album se remplit</h2>
<div class="card"><img src="figures/progression_qd.png" alt="Progression QD"></div>
<div class="card"><table><tr><th>Algo</th><th>Évaluations</th><th>Couverture</th><th>Meilleure fitness</th><th>Temps de calcul</th></tr>{rows}</table>
<p class="note">Même robot, même réseau, même nombre d'évaluations pour les trois (vérifié par les tests).{dcrl_note}</p></div>

<h2>4. À quoi ressemblent les albums</h2>
{repertoires}
<p class="note">Chaque point est une démarche. Les points près de 0 sur un axe n'utilisent presque pas ce pied : ce sont elles qui serviront après une blessure.</p>
<p class="note">Généré le {summary['generated']} · code {summary['code_commit']}</p>
</main></body></html>""")


def _write_readme(out_dir: Path, title: str, summary: dict):
    def listing(sub):
        return "\n".join(f"- `{sub}/{p.name}`" for p in sorted((out_dir / sub).iterdir()))

    ro = summary["rollouts"]
    (out_dir / "README.md").write_text(f"""# {title}

Généré le {summary['generated']} (code `{summary['code_commit']}`). {_status(next(iter(summary['runs'].values())))}

Régénérer : `{summary['command']}`

Meilleure démarche : {ALGO_LABELS[summary['best_policy']['algo']]}, case {summary['best_policy']['cell_index']}.
Intact : {ro['intact']['distance_x_m']:.2f} m vers l'avant en {ro['intact']['duration_s']:.1f} s.
Patte 4 paralysée (actions {ro['damaged']['paralysed_actions']} à 0) : {ro['damaged']['distance_x_m']:.2f} m.

## Pour quoi faire

| Usage | Fichier |
|---|---|
| README GitHub | `videos/intact_vs_patte4_camera_fixe.gif` |
| LinkedIn, slides | `videos/*.mp4` (H.264, temps réel, 20 i/s) |
| Portfolio web (3D à la souris) | `interactive/*.html` : pages autonomes, à mettre en `<iframe>` ou en lien |
| Rapport LaTeX | `figures/*.pdf` (vectoriel) |
| Web, slides | `figures/*.png` (200 dpi) |
| Chiffres et provenance | `data/summary.json`, `data/run_*.json`, `data/metrics_*.csv` |

## Fichiers

### figures/
{listing('figures')}

### videos/
{listing('videos')}

### interactive/
{listing('interactive')}

### data/
{listing('data')}
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True, help="run folders (results/repertoires/...)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Premiers pas du robot")
    args = parser.parse_args()
    print(make_media(args.runs, args.out, args.title))


if __name__ == "__main__":
    main()
