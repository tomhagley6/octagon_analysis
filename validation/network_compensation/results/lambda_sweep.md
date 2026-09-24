# Choosing LAMBDA

Sessions [0, 1, 2], first 300 s of each. Regenerate with
`validation/network_compensation/lambda_sweep.py`.

## Round trip, exact frame times

No time-axis error, so nothing for the penalty to absorb. This is the test the original tuning table used, and it cannot do anything but favour the smallest penalty.

| lambda | distance ratio | rotation ratio | no-rotation error | resting error | position error |
|---|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | -0.000 | +0.000 | 0.0000 |
| 0.003 | 1.000 | 1.005 | -0.002 | -0.002 | 0.0004 |
| 0.01 | 1.000 | 1.008 | -0.003 | -0.002 | 0.0006 |
| 0.03 | 1.000 | 1.011 | -0.004 | -0.003 | 0.0008 |
| 0.1 | 1.001 | 1.016 | -0.009 | -0.005 | 0.0011 |

## Round trip, millisecond timestamps

Every time column in the log is written to the millisecond, so the engine smoothed with times the solver never sees. The path runs on times inside that millisecond and the solver is handed the rounded ones.

| lambda | distance ratio | rotation ratio | no-rotation error | resting error | position error |
|---|---|---|---|---|---|
| 0.0 | 1.002 | 1.011 | -0.001 | -0.000 | 0.0023 |
| 0.003 | 1.001 | 1.006 | -0.002 | -0.001 | 0.0011 |
| 0.01 | 1.000 | 1.008 | -0.003 | -0.002 | 0.0010 |
| 0.03 | 1.001 | 1.011 | -0.005 | -0.004 | 0.0010 |
| 0.1 | 1.001 | 1.017 | -0.010 | -0.006 | 0.0013 |

## Round trip, interpolator stepped on a 59 Hz render clock

The logger samples in FixedUpdate and the interpolator steps in Update, so one logged row does not correspond to one smoothing step.

| lambda | distance ratio | rotation ratio | no-rotation error | resting error | position error |
|---|---|---|---|---|---|
| 0.0 | 1.001 | 1.006 | +0.001 | +0.001 | 0.0085 |
| 0.003 | 1.000 | 1.002 | -0.003 | -0.003 | 0.0096 |
| 0.01 | 0.999 | 1.004 | -0.004 | -0.003 | 0.0100 |
| 0.03 | 0.999 | 1.006 | -0.006 | -0.004 | 0.0101 |
| 0.1 | 0.999 | 1.011 | -0.011 | -0.007 | 0.0104 |

## Held out frames, real player 1

Every 5th frame is left out of the fit and scored on what the fit predicts there, as a share of the trace's own spread. No forward model stands between the answer and the data, so this is the only test that sees every model error at once.

| lambda | held-out error |
|---|---|
| 0.0 | 0.00388 |
| 0.003 | 0.00138 |
| 0.01 | 0.00137 |
| 0.03 | 0.00140 |
| 0.1 | 0.00151 |
