"""YAML configuration loading.

A run config is configs/common.yaml (shared by the three algorithms) plus configs/algos/<algo>.yaml.
They are kept in separate sections so that an algorithm cannot silently change the environment,
the network or the budget (see docs/protocol.md).
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
ALGOS = ("me", "pgame", "dcrlme")

# Section of common.yaml that each profile key overrides.
_PROFILE_KEYS = {
    "episode_length": "env",
    "batch_size": "budget",
    "num_iterations": "budget",
    "log_period": "budget",
    "checkpoint_every": "budget",
    "num_centroids": "repertoire",
    "num_init_cvt_samples": "repertoire",
}
# Section of the algorithm config that its `profile_overrides` modify.
_ALGO_RL_SECTION = {"pgame": "td3", "dcrlme": "dcrl"}


def load_yaml(name: str) -> dict:
    """Load configs/<name>.yaml, e.g. load_yaml("damages") or load_yaml("algos/me")."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def load_run_config(algo: str, profile: str | None = None, robot: str = "ant") -> dict:
    """Full config of a run: {"common": ..., "algo": ..., "profile": ..., "robot": ...}.

    profile=None is a full run; "smoke" is tiny (CPU tests); "local" and "demo" are 200-iteration demos.
    `robot` (configs/robots.yaml) sets the environment and the episode length; everything else is shared.
    """
    if algo not in ALGOS:
        raise ValueError(f"unknown algo {algo!r}, expected one of {ALGOS}")
    common = load_yaml("common")
    profiles = common.pop("profiles")
    robots = load_yaml("robots")
    if robot not in robots:
        raise ValueError(f"unknown robot {robot!r}, expected one of {tuple(robots)}")
    common["env"]["name"] = robots[robot]["env"]
    common["env"]["episode_length"] = robots[robot]["episode_length"]
    algo_cfg = load_yaml(f"algos/{algo}")
    algo_overrides = algo_cfg.pop("profile_overrides", {})

    if profile is not None:
        if profile not in profiles:
            raise ValueError(f"unknown profile {profile!r}, expected one of {tuple(profiles)}")
        for key, value in profiles[profile].items():
            common[_PROFILE_KEYS[key]][key] = value
        for key, value in algo_overrides.get(profile, {}).items():
            algo_cfg[_ALGO_RL_SECTION[algo]][key] = value

    return {"common": common, "algo": algo_cfg, "profile": profile, "robot": robot}


def run_name(algo: str, seed: int, profile: str | None = None, robot: str = "ant") -> str:
    """Folder name of a run. Ant runs keep the names of the first campaign (no robot prefix)."""
    prefix = "" if robot == "ant" else f"{robot}_"
    return f"{prefix}{algo}_seed{seed}" + (f"_{profile}" if profile else "")


def run_config(run: dict) -> dict:
    """Common config of a finished run (from its run.json); runs of the first campaign are Ant runs."""
    return load_run_config(run["algo"], run["profile"], run.get("robot", "ant"))["common"]


def performance_reference(episode_length: int) -> tuple:
    """(f0, scale) for this episode length: return of a standing robot, and the fixed performance scale.

    Normalised performance is p = (F - f0) / scale (configs/adaptation.yaml).
    """
    perf = load_yaml("adaptation")["performance"]
    return perf["f0_still_per_step"] * episode_length, perf["scale_per_step"] * episode_length
