"""Helpers shared by the Kaggle scripts: project installation and runs spread over several GPUs.

Each kernel script extracts the code archive into /kaggle/working/code, then imports this module.
"""

import os
import queue
import subprocess
import sys
import threading
import time

# Python used to run the project; replaced by a Python 3.12 environment when the image is newer (see install).
PYTHON = sys.executable


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def install(code_dir: str) -> dict:
    """Install the project and return the environment to pass to subprocesses.

    The pinned stack (jax 0.4.28) only has wheels up to Python 3.12. When the Kaggle image runs a newer Python, a
    Python 3.12 virtual environment is created with uv and used for every job (`PYTHON`).
    brax 0.10.4 requires pytinyrenderer (image rendering), which does not install on Python 3.12 and is not needed
    for computing. So the exact local versions (requirements-kaggle.txt = requirements-lock.txt without
    pytinyrenderer) are installed without dependency resolution, then the project, then CUDA support for jax.
    """
    global PYTHON
    print(sys.version, flush=True)
    subprocess.run(["nvidia-smi", "-L"])
    if sys.version_info >= (3, 13):
        venv = "/tmp/py312"  # outside /kaggle/working, which Kaggle saves as output
        sh([sys.executable, "-m", "pip", "install", "--progress-bar", "off", "uv"])
        sh([sys.executable, "-m", "uv", "venv", "--python", "3.12", "--seed", venv])
        PYTHON = f"{venv}/bin/python"
    pip = [PYTHON, "-m", "pip", "install", "--progress-bar", "off"]
    sh(pip + ["--no-deps", "-r", f"{code_dir}/requirements-kaggle.txt"])
    sh(pip + ["--no-deps", "-e", code_dir])
    sh(pip + ["jax[cuda12]==0.4.28"])
    env = dict(os.environ)
    commit_file = os.path.join(code_dir, "CODE_COMMIT")
    if os.path.exists(commit_file):
        env["QD_DAMAGE_COMMIT"] = open(commit_file).read().strip()
    return env


