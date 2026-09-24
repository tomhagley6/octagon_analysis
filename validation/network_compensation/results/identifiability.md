# What is recoverable, and what is assumed

Regenerate with `validation/network_compensation/identifiability.py`.

## The filter attenuates, it does not erase

A pure oscillation of known size is put on the tick grid, run through the network path and recovered. `surviving` is how much of it is still in the logged trace. The two recovered columns are how much the solve gets back, without a penalty and with the one in use.

| Hz | surviving in the log | recovered, lambda 0 | recovered, lambda 0.003 |
|---|---|---|---|
| 0.25 | 0.990 | 1.000 | 1.000 |
| 0.5 | 0.962 | 1.000 | 1.000 |
| 1 | 0.870 | 1.001 | 1.000 |
| 2 | 0.658 | 1.000 | 1.000 |
| 4 | 0.388 | 1.000 | 0.995 |
| 8 | 0.181 | 1.000 | 0.779 |
| 12 | 0.108 | 1.000 | 0.292 |
| 14 | 0.090 | 1.000 | 0.197 |

Two things to read here. Even at 14 Hz, near the fastest the tick grid can carry, a tenth of the movement is still in the logged trace. The path multiplies movement by a small number, it does not throw it away, and with no penalty the solve returns every frequency at full size. So the information is in the log and the question is whether it can be pulled out without pulling out error with it, which is conditioning rather than identifiability.

The second thing is what the penalty costs. Below about 4 Hz it takes almost nothing. By 8 Hz it is holding back a fifth of the movement and by 12 Hz most of it. **The recovery is therefore conservative at the fast end, not inflationary.** Whatever it reports for fast turning is a floor.

This test flatters the solve, as a clean oscillation gives error nothing to hide behind. It shows the ceiling of what is identifiable, not what happens on a real trace.

## Does the penalty invent the extra turning?

Player 1's between-trial rotation total, recovered at a range of penalties. The penalty prefers calmer answers, so it can only hold a total down. If the uplift were the penalty inventing movement, relaxing the penalty would shrink it.

| lambda | ITI rotation (deg) | against logged |
|---|---|---|
| logged, on frames | 17439 | |
| 0.0 | 21592 | +23.8% |
| 0.001 | 19809 | +13.6% |
| 0.003 | 19477 | +11.7% |
| 0.01 | 19302 | +10.7% |
| 0.03 | 19218 | +10.2% |
| 0.1 | 19227 | +10.2% |

It goes the other way. The total falls as the penalty rises and then settles: at lambda 0.03 and 0.1, penalties ten and thirty times the one in use, the uplift is still +10%. **An uplift that survives any penalty you care to apply is in the data, not in the penalty.** No penalty at all gives +24%, which is the unregularised inverse amplifying error, and is the failure the penalty exists to stop.

At the chosen 0.003 the uplift is +11.7%, about 1.4 points above where it settles. That gap is the part of the uplift that is amplified error rather than recovered movement. If rotation totals are the main interest, 0.01 to 0.03 sits closer to the settled value at no measurable cost on the held-out test.
