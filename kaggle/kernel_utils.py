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


