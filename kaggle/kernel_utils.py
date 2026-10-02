"""Helpers shared by the Kaggle scripts: project installation and runs spread over several GPUs.

Each kernel script extracts the code archive into /kaggle/working/code, then imports this module.
"""

import os
import queue
import subprocess
import sys
import threading
import time

PIP = [sys.executable, "-m", "pip", "install", "--progress-bar", "off"]


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def install(code_dir: str) -> dict:
    """Install the project and return the environment to pass to subprocesses.

    brax 0.10.4 requires pytinyrenderer (image rendering), whose setup.py uses distutils, removed in
    Python 3.12: it cannot be installed here and is not needed for computing. So the exact local versions
    (requirements-kaggle.txt = requirements-lock.txt without pytinyrenderer) are installed without dependency
    resolution, then the project, then CUDA support for jax.
    """
    print(sys.version, flush=True)
    subprocess.run(["nvidia-smi", "-L"])
    sh(PIP + ["--no-deps", "-r", f"{code_dir}/requirements-kaggle.txt"])
    sh(PIP + ["--no-deps", "-e", code_dir])
    sh(PIP + ["jax[cuda12]==0.4.28"])
    env = dict(os.environ)
    commit_file = os.path.join(code_dir, "CODE_COMMIT")
    if os.path.exists(commit_file):
        env["QD_DAMAGE_COMMIT"] = open(commit_file).read().strip()
    return env


def num_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    except FileNotFoundError:  # no NVIDIA GPU (e.g. local test)
        return 1
    return max(1, sum(line.startswith("GPU") for line in out.splitlines()))


def run_jobs(jobs, code_dir: str, env: dict, log_dir: str):
    """Run the `jobs` commands in order (longest first), one per GPU at a time.

    Each GPU has a thread that takes the next job from the queue. A failure does not stop the other jobs;
    an error is raised at the end if any job failed.
    """
    os.makedirs(log_dir, exist_ok=True)
    todo = queue.Queue()
    for name, cmd in jobs:
        todo.put((name, cmd))
    failures, lock = [], threading.Lock()

    def worker(gpu: int):
        while True:
            try:
                name, cmd = todo.get_nowait()
            except queue.Empty:
                return
            start = time.time()
            print(f"[GPU {gpu}] start {name}", flush=True)
            with open(os.path.join(log_dir, f"{name}.log"), "w") as log:
                proc = subprocess.run(
                    cmd, cwd=code_dir, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log, stderr=subprocess.STDOUT,
                )
            status = "ok" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
            print(f"[GPU {gpu}] end {name}: {status} in {(time.time() - start) / 60:.1f} min", flush=True)
            if proc.returncode:
                with lock:
                    failures.append(name)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(num_gpus())]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise RuntimeError(f"failed jobs: {failures}")
    print("ALL_DONE", flush=True)
