# What the shaky assumptions cost

Regenerate with `validation/network_compensation/assumption_errors.py`.

## One smoothing step per logged row

The model applies one smoothing step per logged row, with the weight read from the logged timestamps. The interpolator really steps once per render frame, and `targetFrameRate` is 144 against a log near 48 Hz, so a logged row spans about three steps. This runs the path on a render clock and gives the solve only the log times.

| render rate | distance | rotation | no-rotation error | position error |
|---|---|---|---|---|
| 48 Hz | 1.001 | 1.005 | -0.002 | 0.0017 |
| 60 Hz | 0.999 | 1.002 | -0.003 | 0.0105 |
| 90 Hz | 0.998 | 0.998 | -0.002 | 0.0282 |
| 144 Hz | 0.997 | 0.997 | -0.002 | 0.0410 |
| 240 Hz | 0.997 | 0.996 | -0.002 | 0.0495 |
| one step per row (what the model assumes) | 1.000 | 1.005 | -0.002 | 0.0004 |

At the real 144 Hz the totals come back at 0.997 and 0.997, so this costs well under half a percent, and in the direction of reporting slightly too little rather than too much. The stillness share is unaffected, -0.002 against -0.002 when the assumption holds exactly.

What it does cost is position: 0.0410 units against 0.0004, a hundredfold increase, though still a quarter of a percent of the arena radius.

The reason is that several small smoothing steps compose into very nearly one larger one, so the shape of the trace survives while its effective lag shifts a little. A timing error moves a trace without changing its path length or how often it holds still, which is exactly the pattern in the table. **The assumption that is known to be wrong is the one that matters least for what this data is used for.** It lands on the same weak axis as the whole-tick lag, which is timing, and leaves the strong axis, which is shape, alone.

## The tick rate

`TICK_S` comes from `TickRate: 30` in the scene file. Held-out frames cannot choose it, scoring flat from 26 to 38 Hz, because the movement carries little near any of those Nyquist limits. So the question is not whether the data confirms it but whether being wrong would matter.

| tick rate | rotation total (deg) | per 100 ticks | stillness share |
|---|---|---|---|
| 27 Hz | 19472 | 225.6 | 0.517 |
| 28.5 Hz | 19477 | 225.6 | 0.510 |
| 30 Hz (in use) | 19477 | 225.6 | 0.503 |
| 31.5 Hz | 19484 | 225.7 | 0.496 |
| 33 Hz | 19482 | 225.6 | 0.491 |

Barely. A ten percent error in the tick rate moves the rotation total by 0.05%. The stillness share drifts more, about 2.6 points across the same range, as the number of ticks changes and rescaling the cutoff only approximately preserves its meaning. Since the rate is read from the scene file rather than guessed, neither is a live concern.

## What is still untested

Two assumptions have no number here. Sends are threshold gated at 0.001 units and 0.01 degrees and packets can be lost, so not every tick necessarily carries a sample. And the buffer can consume several items at once or extrapolate up to three ticks, neither of which the interpolation matrix can represent. Both are network side, and they are the most likely home for the unexplained residual in the validation section.
