# Which total to believe

Sessions [0, 1, 2, 3, 4]. Regenerate with `validation/network_compensation/totals_fidelity.py`.

## Against a known truth

A trace is run through the network path, then measured both ways and compared with what it really was on the 30 Hz grid. The logged column is how a networked player is measured without any recovery, on frames, as the log holds it.

| truth used | logged, distance | logged, rotation | recovered, distance | recovered, rotation |
|---|---|---|---|---|
| player 0's own trace | 0.979 | 0.936 | 1.000 | 1.004 |
| player 1's recovered trace | 0.978 | 0.928 | 1.000 | 1.000 |

The logged rotation total comes out 6 to 7% low, and recovery puts it back to within half a percent. The direction is not an accident. The smoothing is a low-pass filter, so it can only take angular change out of a trace, never add it, and a logged rotation total is therefore an underestimate of the real one. Distance survives either way, within about 2%.

How much is removed depends on the player, which is why the two rows differ. A low-pass filter takes more from sharp turning than from smooth turning, and player 1 turns more sharply: 0.637 against 0.563 on the sharpness measure in `turn_sharpness`. Player 1's figure comes from the recovered trace, which the penalty smooths, so the gap is if anything understated.

## Without a truth at all

The test above still leans on a stand-in for the truth. This one does not. The recovered trace is pushed back through the network path and measured the way the log is measured. If the recovery explains the log, it lands on the logged total.

| | ITI rotation (deg) | against logged |
|---|---|---|
| logged, on frames | 43611 | |
| recovered, on ticks | 48945 | +12.2% |
| recovered, pushed back through the path | 43578 | -0.1% |

The replay lands on the logged total, so the recovered trace does explain the log and the uplift is rotation the smoothing had removed.

One limit on this check. The solve fits the log, so agreement here is partly built in, and it cannot see movement the filter would erase completely. That is exactly what the second difference penalty is for, and it is why `LAMBDA` is tuned rather than set to zero.

The real uplift is a little larger than the round trip predicts, 12% against 8%. The likely reason is the part of the network path that is not modelled, which the validation section covers: the real log loses slightly more than the deterministic filter alone would take. The recovered total may therefore sit a fraction high, but it is far closer to the truth than the logged one.
