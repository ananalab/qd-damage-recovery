"""Conditions of the bank: intact robot, simulated training damages, and the test damages.

A condition is a dict with a name, a group (intact, train, id, ood_*, gap), the action scale and offset, the
observation mask, the physics changes (None = nominal simulator) and a number of episodes. See
configs/conditions.yaml for the meaning of each group.
"""

import numpy as np

from qd_damage.config import load_yaml


def _condition(name, group, robot_cfg, action_size, obs_size, n_episodes, physics=None):
    return {
        "name": name,
        "group": group,
        "scale": np.ones(action_size, np.float32),
        "offset": np.zeros(action_size, np.float32),
        "obs_mask": np.ones(obs_size, np.float32),
        "physics": physics,
        "n_episodes": n_episodes,
    }


def named_damages(robot: str, robot_cfg: dict) -> list:
    """(name, {joint: factor}) of the in-family test damages.

    Ant keeps the 12 scenarios of the first campaign (configs/damages.yaml). The two-legged robots get every limb
    paralysed, every limb weakened, and every single joint paralysed.
    """
    if robot == "ant":
        damages = load_yaml("damages")
        out = []
        for s in damages["scenarios"]:
            if s["name"] == "intact":
                continue
            joints = [damages["legs"][j.split(".")[0]][j.split(".")[1]] for j in s["joints"]]
            out.append((s["name"], {j: s["factor"] for j in joints}))
        return out
    out = []
    for limb, joints in robot_cfg["limbs"].items():
        out.append((f"{limb}_paralysed", {j: 0.0 for j in joints}))
        out.append((f"{limb}_weak", {j: 0.5 for j in joints}))
    for limb, joints in robot_cfg["limbs"].items():
        for k, j in enumerate(joints):
            out.append((f"{limb}_joint{k}_paralysed", {j: 0.0}))
    return out


def train_damages(n: int, p_damaged: float, levels, action_size: int, rng: np.random.Generator, exclude=()) -> list:
    """n random damages of the robust-policy family. Draws without any damaged joint, and draws equal to one of
    the `exclude` scale vectors (the test damages), are redrawn, so that no test damage is seen in training."""
    out = []
    while len(out) < n:
        damaged = rng.random(action_size) < p_damaged
        scale = np.where(damaged, rng.choice(levels, size=action_size), 1.0).astype(np.float32)
        if damaged.any() and not any(np.array_equal(scale, e) for e in exclude):
            out.append(scale)
    return out


def build_conditions(robot: str, action_size: int, obs_size: int) -> list:
    cfg, robot_cfg = load_yaml("conditions"), load_yaml("robots")[robot]
    ep = cfg["episodes"]
    rng = np.random.default_rng(cfg["seed"])
    limbs = robot_cfg["limbs"]
    conds = [_condition("intact", "intact", robot_cfg, action_size, obs_size, ep["intact"])]

    named, id_conds = named_damages(robot, robot_cfg), []
    for name, factors in named:
        c = _condition(name, "id", robot_cfg, action_size, obs_size, ep["test"])
        for j, f in factors.items():
            c["scale"][j] = f
        id_conds.append(c)
    train = train_damages(cfg["train"]["n"], cfg["train"]["p_damaged"], cfg["train"]["levels"], action_size, rng,
                          exclude=[c["scale"] for c in id_conds])
    for i, scale in enumerate(train):
        c = _condition(f"train_{i:03d}", "train", robot_cfg, action_size, obs_size, ep["train"])
        c["scale"] = scale
        conds.append(c)
    conds += id_conds

    ood = cfg["ood"]
    for limb, joints in limbs.items():
        for off in ood["jammed_offsets"]:
            c = _condition(f"{limb}_jammed{off:+.1f}", "ood_jammed", robot_cfg, action_size, obs_size, ep["test"])
            c["scale"][joints] = 0.0
            c["offset"][joints] = off
            conds.append(c)
    for limb, joints in limbs.items():
        c = _condition(f"{limb}_inverted", "ood_inverted", robot_cfg, action_size, obs_size, ep["test"])
        c["scale"][joints[0]] = ood["inverted_scale"]
        conds.append(c)
    angle, velocity = robot_cfg["joint_obs"]["angle"], robot_cfg["joint_obs"]["velocity"]
    for limb, joints in limbs.items():
        c = _condition(f"{limb}_sensors", "ood_sensor", robot_cfg, action_size, obs_size, ep["test"])
        for j in joints:
            c["obs_mask"][angle + j] = 0.0
            c["obs_mask"][velocity + j] = 0.0
        conds.append(c)
    for limb, foot in zip(limbs, robot_cfg["feet"], strict=True):
        conds.append(_condition(f"{limb}_slippery", "ood_slippery", robot_cfg, action_size, obs_size, ep["test"],
                                {"body_friction": {foot: ood["slippery_friction"]}}))
        conds.append(_condition(f"{limb}_heavy", "ood_heavy", robot_cfg, action_size, obs_size, ep["test"],
                                {"body_mass_scale": {foot: ood["heavy_foot_scale"]}}))

    for gap_name, physics in cfg["gap"].items():
        conds.append(_condition(f"{gap_name}_intact", "gap", robot_cfg, action_size, obs_size, ep["test"], physics))
        for name, factors in named:
            c = _condition(f"{gap_name}_{name}", "gap", robot_cfg, action_size, obs_size, ep["test"], physics)
            for j, f in factors.items():
                c["scale"][j] = f
            conds.append(c)
    return conds
