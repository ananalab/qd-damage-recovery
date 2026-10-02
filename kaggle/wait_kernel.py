"""Wait for a Kaggle kernel: poll its status every minute, with a timeout on each call.

    python kaggle/wait_kernel.py anaisazouaoui/qd-damage-recovery-full [max_minutes]
Prints one line per minute; stops on complete / error / cancel, or after max_minutes.
"""

import subprocess
import sys
import time
from pathlib import Path

KAGGLE = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "kaggle")
kernel, max_minutes = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 105
for _ in range(max_minutes):
    try:
        out = subprocess.run([KAGGLE, "kernels", "status", kernel], capture_output=True, text=True, timeout=45).stdout
        status = out.strip().splitlines()[-1] if out.strip() else "no answer"
    except subprocess.TimeoutExpired:
        status = "Kaggle call timed out (network?), retrying"
    print(time.strftime("%H:%M"), status, flush=True)
    if any(w in status.lower() for w in ("complete", "error", "cancel")):
        break
    time.sleep(60)
