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

This notebook shows how to use a MAP-Elites repertoire built with QDax to let a robot **recover from damage
in a few trials**, with *Intelligent Trial & Error* (Cully et al., *Robots that can adapt like animals*,
Nature 2015).

1. Build a repertoire of Ant gaits with MAP-Elites (descriptor: feet contact).
2. Damage the robot: one leg is paralysed.
3. Use M-BOA (the Bayesian optimisation at the heart of ITE) to find a gait that still works, trying only a
   handful of policies on the damaged robot.

