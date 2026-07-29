# Question and protocol

## Question

A four-legged robot (the Brax Ant) loses the use of a leg. What lets it walk again with the fewest trials?

| Approach | Idea |
|---|---|
| MAP-Elites (evolution only) | Build a repertoire of gaits, then try the most promising ones |
| PGA-ME / DCRL-ME (evolution + RL) | Same, but the repertoire is built with the help of a TD3 critic |
| RL re-training | Train the gait again, directly on the damaged robot |

## Rules followed

