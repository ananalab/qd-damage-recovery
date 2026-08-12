# Question and protocol

## Question

A four-legged robot (the Brax Ant) loses the use of a leg. What lets it walk again with the fewest trials?

| Approach | Idea |
|---|---|
| MAP-Elites (evolution only) | Build a repertoire of gaits, then try the most promising ones |
| PGA-ME / DCRL-ME (evolution + RL) | Same, but the repertoire is built with the help of a TD3 critic |
| RL re-training | Train the gait again, directly on the damaged robot |

We measure the number of trials on the damaged robot (what is expensive on a real robot) and the performance
recovered. Adaptation uses Intelligent Trial & Error (ITE, Cully et al., Nature 2015): Bayesian optimisation
with a Gaussian process over the repertoire.

## Prior work and what this project adds

Chalumeau et al. (ICLR 2023) tested Ant-uni with a weakened or paralysed leg (MAP-Elites, PGA-ME and
skill-discovery RL). Their post-damage selection is an oracle: every skill is evaluated 100 times on the damaged
robot and the best one is kept. They found that PGA-ME adapts better than MAP-Elites, and that QD methods are
more robust to large perturbations.

## Protocol

## Rules followed

