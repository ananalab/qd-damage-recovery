"""Build the example notebook proposed to QDax: ite_damage_recovery.ipynb (in English, like the QDax examples).

    python contrib/qdax/build_notebook.py && jupyter nbconvert --to notebook --execute contrib/qdax/ite_damage_recovery.ipynb \
        --output ite_damage_recovery_executed.ipynb
"""

import json
from pathlib import Path

CELLS = [
    (
        "markdown",
        """
# Damage recovery with Intelligent Trial & Error (ITE)

