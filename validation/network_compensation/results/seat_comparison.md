# Player 0 against player 1, between trials

100 social sessions, paired by session. The logged rows are measured on 48 Hz frames with the per-frame cutoffs (0.5 deg, 0.05 units), the recovered rows on the 30 Hz ticks with the tick cutoffs (0.8 deg, 0.08 units). Levels therefore do not carry between the two halves of the table. The gap between the players does, as both players are measured the same way within a half, and that gap is the whole question.
Regenerate with `validation/network_compensation/seat_comparison.py`.

| source | metric | player 0 | player 1 | difference | 95% CI |
|---|---|---|---|---|---|
| as logged | `no_rotational` | 0.558 | 0.458 | +0.100 | [+0.061, +0.139] |
| as logged | `resting` | 0.309 | 0.250 | +0.059 | [+0.014, +0.103] |
| as logged | `distance` | 2216.986 | 2460.331 | -243.345 | [-623.562, +136.873] |
| as logged | `rotation` | 47389.238 | 43422.702 | +3966.535 | [-2700.124, +10633.194] |
| recovered | `no_rotational` | 0.549 | 0.545 | +0.004 | [-0.031, +0.039] |
| recovered | `resting` | 0.303 | 0.293 | +0.010 | [-0.034, +0.054] |
| recovered | `distance` | 2202.752 | 2500.821 | -298.068 | [-685.963, +89.826] |
| recovered | `rotation` | 46932.316 | 49657.234 | -2724.916 | [-11468.599, +6018.768] |

## What the totals cover, and how big the gap is

Each total is one session's worth of between-trial movement for one player, summed over every ITI in it. That is a median of 601 s (10 min) per session, about half the session. One value per player per session, averaged over the 100 sessions, so every one of the 200 people appears once.

| | player 0 | player 1 | gap | as % of player 0 | between-session SD | gap in SD |
|---|---|---|---|---|---|---|
| ITI distance (units) | 2203 | 2501 | -298 | -13.5% | 1723 | 0.17 |
| ITI rotation (deg) | 46932 | 49657 | -2725 | -5.8% | 27854 | 0.10 |

The spread between players is what makes these gaps hard to read. The standard deviation of ITI distance across sessions is 1723 units against a mean of 2203, so players differ from each other far more than the two positions differ on average. That is why the confidence intervals are wide, and it is a fact about the players rather than about the recovery.

The recovery's own error on a total is much smaller. The round trips in `lambda_sweep.md` put the distance ratio at 1.000 to 1.001 and the rotation ratio at 1.002 to 1.011, so the method distorts a total by at most about 1%. The gap between the players is several times that, and is not something the recovery introduced.

## Is player 1 a more peripheral set of players?

A tempting reading of the distance gap is that the people who happened to play as player 1 simply move around more. Onset centrality per session, with the solo sessions as a control that never touched the network:

| session | player 0 | player 1 | gap | paired p | sessions with player 1 more peripheral |
|---|---|---|---|---|---|
| social | 0.604 | 0.527 | +0.077 | 0.00046 | 62 / 100 |
| solo1 | 0.584 | 0.543 | +0.040 | 0.16 | 55 / 100 |
| solo2 | 0.559 | 0.478 | +0.081 | 0.0034 | 61 / 100 |

Player 1 is the more peripheral of the pair in 62 of 100 social sessions, and that gap is still there in the second solo, where nothing crossed the network. It is not there in the first solo. So it is about the players and not the logging, and it appears to arrive with the social session rather than to have been there all along.

It does not explain the distance gap either. Centrality and ITI distance are only weakly related (r = 0.194, p = 0.006 over the 200 players), and the relation runs the wrong way: more central players move slightly more, so the centrality gap predicts player 0 covering +116 units more than player 1, against the -298 observed. Whatever the distance gap is, it is not centrality.
