"""Kaggle (2 x T4): stronger RL baselines.

- damage-robust TD3 policy (random damage at every episode), 2 seeds, evaluated zero-shot on every damage;
- TD3 fine-tuning seed 0 re-run from the best RE-EVALUATED intact DCRL-ME policy (the first campaign used the elite
  with the highest stored fitness because the oracle ran in parallel);
- fine-tuning with a critic warm-up (actor frozen at first) and with a critic pre-trained on the intact robot,
  4 damages x 2 seeds;
- reality gap: oracle of every repertoire on perturbed physics, then ITE, top-k and best-intact with the prior from
  the nominal simulator and the trials on the perturbed robot.
Inputs: the repertoires (qd-damage-recovery-full) and their oracle (qd-damage-recovery-campaign).
Outputs in /kaggle/working/outputs/{rl,rl_robust,oracle_gap,adaptation_gap}/.
"""

import glob
import os
import shutil
import sys
import tarfile

CODE = "/kaggle/working/code"
# The dataset holds the code archive, which Kaggle sometimes extracts by itself: handle both cases.
archives = glob.glob("/kaggle/input/**/qd_damage_code.tar.gz", recursive=True)
if archives:
    with tarfile.open(archives[0]) as tar:
        tar.extractall(CODE)
else:
    src = os.path.dirname(glob.glob("/kaggle/input/**/CODE_COMMIT", recursive=True)[0])
    shutil.copytree(src, CODE)  # copy: /kaggle/input is read-only and pip -e needs to write
sys.path.insert(0, f"{CODE}/kaggle")
import kernel_utils as ku  # noqa: E402

env = ku.install(CODE)
OUT, py = "/kaggle/working/outputs", ku.PYTHON


def find(pattern):
    return glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)[0]


all_runs = sorted(
    os.path.dirname(p) for p in glob.glob("/kaggle/input/**/outputs/*_seed[0-9]/repertoire.npz", recursive=True)
)
repertoires = {s: os.path.dirname(find(f"outputs/dcrlme_seed{s}/repertoire.npz")) for s in (0, 1)}
oracle_root = os.path.dirname(os.path.dirname(find("outputs/oracle/dcrlme_seed0/oracle.npz")))
print("repertoires:", repertoires, "oracle:", oracle_root, flush=True)
SCENARIOS = ["leg4_paralysed", "leg2_weak", "legs14_paralysed", "ankle1_paralysed"]


def finetune(scenario, variant, seed):
    return (
        f"rl_{scenario}_{variant}_s{seed}",
        [
            py,
            "-m",
            "qd_damage.rl_retrain",
            "--scenario",
            scenario,
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--repertoire",
            repertoires[seed],
            "--oracle-root",
            oracle_root,
            "--out",
            f"{OUT}/rl",
        ],
    )


jobs = [
    (f"robust_s{s}", [py, "-m", "qd_damage.rl_retrain", "--robust", "--seed", str(s), "--out", f"{OUT}/rl_robust"])
    for s in (0, 1)
]  # longest first
jobs += [
    (
        f"gap_{os.path.basename(r)}",
        [
            "bash",
            "-c",
            f"{py} -m qd_damage.oracle --reality-gap --runs {r} --out {OUT}/oracle_gap && "
            f"{py} -m qd_damage.adaptation --reality-gap --runs {r} --oracle-root {OUT}/oracle_gap "
            f"--prior-oracle-root {oracle_root} --methods ite top_k best_intact --out {OUT}/adaptation_gap",
        ],
    )
    for r in all_runs
]
jobs += [finetune(sc, "finetune", 0) for sc in SCENARIOS]
jobs += [finetune(sc, v, s) for v in ("finetune_warmup", "finetune_pretrained") for s in (0, 1) for sc in SCENARIOS]
ku.run_jobs(jobs, CODE, env, log_dir="/kaggle/working/logs")
