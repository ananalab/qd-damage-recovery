"""Kaggle (2 x T4): continuous ITE with the descriptor-conditioned actor of DCRL-ME.

For each seed: a full DCRL-ME run that saves its actor (actor.npz), then the oracle and the grid adaptation
(ITE and baselines), then continuous ITE, all on the SAME run.
Outputs in /kaggle/working/outputs/{repertoires,oracle,adaptation,continuous}/.
"""

