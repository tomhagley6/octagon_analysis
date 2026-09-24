# Recovering the networked player's movement

How `network_compensation.py` undoes what the network did to player 1 in social sessions, why it is needed,
how it was tested, and which analyses should use its output.

Every number below is produced by a script in
[`validation/network_compensation/`](../validation/network_compensation/), which writes its tables to
`results/` so they can be regenerated and checked.

## Ticks, frames and rows

Three clocks run at once.

| | what it is | rate |
|---|---|---|
| **network tick** | Netcode's own clock, fixed and independent of the frame rate. The owner samples and sends its transform once per tick, and the host places remote players against it | 30 a second (`TickRate: 30` in `OctagonStage.unity`) |
| **render frame** | one pass of Unity's `Update`, where the interpolator advances what it draws | up to 144 a second (`Application.targetFrameRate`, `GameManager.cs:249`) |
| **logged row** | one line in the session log, written from `FixedUpdate` against a 0.02 s accumulator | 50 a second by that accumulator, about 48 as the timestamps land |

Player 1's movement only ever travelled on the tick clock, so that is the grid the recovery solves for.
Movement between two ticks was never sent and cannot be recovered. The other two clocks do not line up
with it or with each other. The smoothing steps on render frames, and the log is written on its own
schedule.

**"Frame" below means a logged row.** Where a render frame is meant it says so.

## The problem

A social session is logged on one machine. Player 0 is that machine's own player, so its position and
yaw are written down as they happen, about 48 times a second. Player 1 plays on the other machine, and
its transform reaches the log over the network.

The game moves it with Netcode's `ClientNetworkTransform` (NGO 1.7.1, `Interpolate: 1` on
`Assets/Prefabs/Player/FirstPersonPlayer.prefab`, `TickRate: 30` in `OctagonStage.unity`). Four things
happen to player 1's movement before it is logged:

1. The owner sends its transform on each network tick, 30 times a second.
2. The host renders it one tick behind, as it is the server (`NetworkTransform.cs:2799`).
3. Render frames between two ticks are filled by linear interpolation.
4. The result is smoothed again each render frame with `lerp(current, target, deltaTime / 0.1)`
   (`BufferedLinearInterpolator.cs:220`).

Step 4 is what does the damage. When a player stops turning, the logged angle keeps easing toward the
last value for several frames, so player 1 almost never logs an exact hold.

Measured over ITI frames of the 100 social sessions, as logged:

| | player 0 | player 1 | paired p |
|---|---|---|---|
| exact-zero yaw steps | 0.40 | 0.06 | |
| `prop_no_rotational` | 0.55 | 0.41 | 3e-6 |
| `proportion_resting` | 0.29 | 0.21 | .004 |

The same players show no such difference in their solo sessions, where both are logged locally, so this
is the logging path and not the players. Measuring over 0.2 s windows instead of frames does not remove
it (p = .002).

## The idea, in short

The four steps above are all in the engine source, so the exact process is known. That turns the problem
around. Rather than asking what the smoothed line came from, ask which 30 Hz movement, run through the
game's own process, reproduces the logged line. Search for that movement, and prefer the calmer of two
answers that fit equally well. The answer is player 1's real movement at 30 Hz.

### Why anything is left to recover

A low-pass filter is usually taken to destroy what it removes. This one does not. The smoothing is lower
triangular with a positive diagonal, so it is invertible, and the interpolation has full column rank.
Neither has a null space, so no movement is mapped to nothing.

Measured, an oscillation at 14 Hz, near the fastest the tick grid can carry, still has a tenth of its size
in the logged trace, and with no penalty the solve returns every frequency at full size. The path
multiplies movement by a small number rather than throwing it away. The difficulty is therefore
conditioning and not identifiability. The question is whether the attenuated signal can be pulled out
without pulling out the error sitting beside it, which is what the penalty is for. Table in
[`results/identifiability.md`](../validation/network_compensation/results/identifiability.md).

What is genuinely gone is movement between ticks, which was never sent, and the fast end of what remains,
which the penalty holds back. Both make the answer conservative rather than inflated.

## Why not invert it step by step

Every stage has an exact algebraic reverse. Undoing the smoothing is
`target_k = y_(k-1) + (y_k - y_(k-1)) / a_k`, with `a_k = dt_k / 0.1` about 0.2 per frame.

That division multiplies every error by about five. The process depends on Unity's own frame delta, and
only the log's timestamps are available, which are written to the millisecond and come from a different
clock (see Values chosen). Applied to real sessions, this recovered stillness but wrecked the totals: ITI
distance came out 53% high and rotation 48% high against player 0, because the amplified error accumulates
over tens of thousands of steps.

## The method

Both stages are linear, so the whole path is one linear operator:

```
logged = S A s
```

with `s` the owner's values on the tick grid, `A` the interpolation (each frame a weighted pair of
neighbouring ticks) and `S` the smoothing. `solve_block` finds

```
argmin_s  || S A s - logged ||^2  +  lambda || D2 s ||^2
```

where `D2` is the second difference of `s` (each triple of neighbouring ticks giving
`s[k] - 2 s[k+1] + s[k+2]`, the discrete counterpart of a second derivative rather than a derivative
itself). It is left unnormalised, with no division by the tick spacing, so `lambda` is tied to a 30 Hz
grid.

The first term asks the answer to explain the log. The second penalises jitter, which stops the solution
using the error in the time axis to fit something that is not movement. A straight line through three
ticks scores zero, so the penalty charges for a change in speed and not for speed. A player moving
steadily pays nothing. This is the same information the algebraic inverse uses, minus its error
amplification.

Practical points:

- **The solve** runs `scipy.sparse.linalg.lsqr` on a `LinearOperator`, so `S` never forms as a matrix.
  `smooth_forward` and `smooth_adjoint` apply it and its transpose, each as a first-order recursion
  solved in chunks (`_first_order`) rather than frame by frame.
- **The tick phase is not logged.** Each block tries `N_PHASE` offsets across one tick and keeps the
  best fitting one. x, z and yaw are fitted together on one phase, as they travel in the same transform
  update, with each trace's residual scaled by its own spread so degrees cannot outvote units.
- **Blocks.** The two machines' clocks drift over a 20 minute session, so the fit runs in `BLOCK_S`
  blocks, each with its own phase. Blocks overlap by `EDGE_S` at both ends and only the interior is
  kept, as the solution is least constrained at an edge.
- **Block joins.** A new phase can place the first tick of a block less than a tick after the last of
  the one before. Those samples are dropped, or the pair would read as a smaller step than it was.
- **Player 0** is not recovered. `sample_player` reads its local trace at the same tick instants, so
  both players end up on one grid: sampled at 30 Hz, never delayed or smoothed.

## Values chosen

| constant | value | why |
|---|---|---|
| `TICK_S` | 1/30 | `TickRate: 30` in `OctagonStage.unity` |
| `LAG_TICKS` | 1 | the host is the server (`NetworkTransform.cs:2799`) |
| `SMOOTH_S` | 0.1 | `MaximumInterpolationTime` in `BufferedLinearInterpolator.cs` |
| `LAMBDA` | 0.003 | tuned, see below |
| `BLOCK_S` | 60 | short enough that clock drift does not build up, long enough to fit a phase |
| `EDGE_S` | 2 | 60 ticks of overlap, well past the 0.1 s smoothing |
| `N_PHASE` | 8 | 4 ms steps across a tick |

`LAMBDA` was chosen against the errors the log's time axis really carries. Full tables in
[`validation/network_compensation/results/lambda_sweep.md`](../validation/network_compensation/results/lambda_sweep.md).

Two things make the frame times the solver reads differ from the ones the engine smoothed with:

- **Every timestamp is written to the millisecond.** `timeLocal`, `timeApplication` and `timeReferenced`
  are all whole milliseconds, so a frame interval carries half a millisecond of rounding and the
  smoothing weight `a_k = dt / 0.1` inherits about 2.5% of error.
- **The logger and the interpolator run on different clocks.** Rows are written from `FixedUpdate`
  against a 0.02 s accumulator (`ArenaLogger.cs:231`), while the smoothing steps once per render frame in
  `Update`. One logged row is not one smoothing step. Player 1's x repeats exactly in 1.1% of consecutive
  rows and its yaw in 3.2%, which is the logger sampling one render frame twice.

An earlier version of this file tuned `LAMBDA` on a round trip that used the same frame times forwards and
backwards. That case has no time-axis error, so there is nothing for the penalty to absorb and the sweep
can only favour the smallest penalty offered. It is kept as the first table in the sweep to show this.

Scoring instead against each error above, and against held-out real frames:

| lambda | ms timestamps, position | render clock, rotation | held out, real player 1 |
|---|---|---|---|
| 0 | 0.0023 | 1.006 | 0.00388 |
| **0.003** | **0.0011** | **1.002** | **0.00138** |
| 0.01 | 0.0010 | 1.004 | 0.00137 |
| 0.03 | 0.0010 | 1.006 | 0.00140 |
| 0.1 | 0.0013 | 1.011 | 0.00151 |

The held-out test is the one to weigh most: every fifth real player 1 frame is left out of the fit and
scored on what the fit predicts there, so no forward model stands between the answer and the data. It
rules out `LAMBDA = 0` firmly, at 2.8 times the error of the rest, and then flattens. 0.003, 0.01 and 0.03
are within 2% of each other and cannot be separated.

`LAMBDA` is a preference rather than a measurement. It says that among the traces explaining the log
equally well, the calmer one is the answer. It is the only part of the method not read off the engine
source, so it is worth knowing what it can do to a result.

It cannot be the source of extra movement, as a penalty can only hold a total down. Player 1's rotation
total falls as the penalty rises and then settles: 23.8% above the logged total at zero, 11.7% at 0.003,
and still 10.2% at 0.1, thirty times the value in use. An uplift that survives any penalty is in the data.
The zero case is the unregularised inverse amplifying error, which is the failure the penalty exists to
stop.

0.003 is kept because the recovery is used mainly for stillness, where it is the best of the candidates,
and the held-out test cannot separate it from the others. It is not the best choice for rotation totals,
where it sits about 1.4 points above where the uplift settles. **For a rotation total, quote the 10%
plateau rather than the value at 0.003.** That needs no rebuild.

## Validation

Every table here is regenerated by the scripts in
[`validation/network_compensation/`](../validation/network_compensation/), which write their output to
`results/` so the numbers can be traced back to the code that made them.

### Does the modelled path describe the real log?

Running a known trace through `network_forward` and recovering it tests the solve, but takes the forward
path as given: a wrong path would pass that test just as well. What the path cannot explain shows up
instead as the fit residual on real player 1 frames, against the same fit on cases where the path is exact
by construction.

Yaw residual per frame, in degrees:

| case | median | max |
|---|---|---|
| real player 1 | 0.107 | 0.272 |
| player 0 through the path, exact times | 0.009 | 0.021 |
| player 0, millisecond timestamps | 0.018 | 0.033 |
| player 0, 59 Hz render clock | 0.014 | 0.028 |

Real player 1 leaves about six times more unexplained than either known time-axis error accounts for. The
deterministic filter is right, and was read off the engine source, but something beyond it is present.
The likely sources are all on the network side and none of them are modelled: packet jitter moving when
the buffer is consumed, loss stretching an interpolation across more than one tick, the extrapolation the
interpolator allows (`MaxInterpolationBound = 3`), and clock drift inside a 60 s block.

Two things follow. The penalty is doing more work than a clean derivation suggests, which is why
`LAMBDA = 0` fails the held-out test so clearly. And any accuracy figure taken from a round trip is a
floor rather than an estimate, because a round trip never contains this component.

In absolute terms it is small: 0.107 degrees per frame against a trace whose spread is about 100 degrees,
and the fit predicts held-out real frames to 0.14% of that spread.

### How much of the gap between the players does it remove?

Both players do the same task and show no difference in their solo sessions, where both are logged
locally, so a gap in social is the logging path. Paired over the 100 sessions, between trials:

| metric | source | player 0 | player 1 | difference | 95% CI |
|---|---|---|---|---|---|
| `no_rotational` | as logged | 0.558 | 0.458 | +0.100 | [+0.061, +0.139] |
| `no_rotational` | recovered | 0.549 | 0.545 | +0.004 | [-0.031, +0.039] |
| `resting` | as logged | 0.309 | 0.250 | +0.059 | [+0.014, +0.103] |
| `resting` | recovered | 0.303 | 0.293 | +0.010 | [-0.034, +0.054] |
| distance | as logged | 2217 | 2460 | -243 | [-624, +137] |
| distance | recovered | 2203 | 2501 | -298 | [-686, +90] |
| rotation | as logged | 47389 | 43423 | +3967 | [-2700, +10633] |
| rotation | recovered | 46932 | 49657 | -2725 | [-11469, +6019] |

Levels do not carry between the two halves, as the logged rows are 48 Hz frames with the per-frame cutoffs
and the recovered rows are 30 Hz ticks with the tick cutoffs. The gap between the players does carry, as
both players are measured the same way within a half.

**The stillness result is solid.** As logged, player 0 holds still 10 points more often than player 1, and
the interval is nowhere near zero. After recovery the gap is 0.004, and its interval excludes everything
the logged interval allowed. The recovery removes that bias, and this is a demonstration rather than a
failure to detect one.

**The totals are a different story, in both directions.** Their intervals already contain zero as logged,
so there was never a demonstrated bias in them to fix, and they are wide enough after recovery to allow
player 1 moving up to 31% further. Nothing here shows the recovery helps the totals, and nothing shows it
harms them. An earlier version of this file read the same result as the two players sitting on top of each
other; they do not, the measure is simply too loose to say.

These intervals are about comparing the two players, not about whether either player's own total is
right. On that second question the answer is clear, and it is in
[`results/totals_fidelity.md`](../validation/network_compensation/results/totals_fidelity.md): running a
known trace through the network path leaves the logged rotation total 6 to 7% low and the recovered one
within half a percent, so **for rotation the recovered total is the one to use and the logged total is
wrong**. The direction is forced, as a low-pass filter can only take angular change out of a trace and
never add it. Distance comes through either way, within about 2%.

That is also why recovery lifts player 1's rotation total by around 11%. It is not the method disagreeing
with the data, it is the method putting back what the smoothing removed, and pushing the recovered trace
back through the network path reproduces the logged total to 0.1%.

Each total is one player's between-trial movement across a whole session, a median of 601 s of ITI, which
is about half of it. The gap of -298 units is 13.5% of player 0's, and the rotation gap of -2725 degrees
is 5.8%. Those sound large, but the standard deviation across sessions is 1723 units against a mean of
2203, so the gap is 0.17 of a standard deviation and players differ from one another far more than the
two positions differ on average. **This does not mean player 1's totals are inaccurate.** The round trips
put the recovery's own error on a total at about half a percent, well under the gap, so the width here is
a fact about how varied the players are and not about the method.

Player 1 is also the more peripheral of the pair in 62 of 100 sessions (p = .0005), which invites the
reading that more peripheral players happened to land in that position. The solo sessions say otherwise
about the cause: the same gap is in the second solo, where nothing crossed the network (61 of 100,
p = .003), but not in the first (55 of 100, p = .16). So it is the players rather than the logging, and it
seems to arrive with the social session rather than to have been there from the start. It does not explain
the distance gap in any case, as centrality and ITI distance are barely related (r = 0.19) and the
relation runs the wrong way.

Reporting these as intervals rather than p values is deliberate. A p value above .05 only says a gap was
not detected, which a noisy measure achieves by doing nothing at all, and an earlier fix that degraded
player 0 to match player 1 passed exactly that test while moving both to the wrong numbers.

One caveat on the whole approach. Player 1 is genuinely more peripheral than player 0 in social sessions,
so seat-linked differences in behaviour do exist. A leftover gap is not automatically leftover method
bias, and matching player 0 is not automatically success. The solo sessions are the control that carries
the argument, not the social comparison on its own.

### Unit tests

`tests/test_network_compensation.py` (15 tests) covers:

- `smooth_forward` against a plain frame loop.
- `smooth_adjoint` against the explicit matrix, and by inner product. The solve is only correct if the
  adjoint matches the forward operator, so this is the load-bearing test.
- Round trips on synthetic traces: a constant, a moving trace, and a trace that holds still half the
  time. With frame times known exactly the path is invertible, so these run at `lambda = 0` and demand
  near-exact recovery.
- The penalty earning its place when frame times carry a millisecond of noise.
- Phase recovery when the tick grid is not aligned to the first frame.
- Grid and column structure of the saved table, and that player 0's values are read from its own trace.
- The stillness cutoffs matching the frame-level thresholds, and sitting above the safe floor.

`tests/test_log_data_properties.py` pins the facts about the logs that the method rests on: the
millisecond timestamps, the logging rate, that solo sessions have no player 1 columns, and the head pitch
problem below.

## Limits

**30 Hz is the ceiling.** Movement between ticks was never transmitted, so no method recovers it. This
is why player 0 is sampled onto the same grid for any comparison.

**Fast turning comes back only in part.** Within the tick grid the path attenuates rather than erases, and
with no penalty every frequency returns at full size, so the information is in the log. The penalty is
what limits it. Below about 4 Hz it costs almost nothing, by 8 Hz it holds back a fifth of the movement
and by 12 Hz most of it, so the recovery understates fast turning and whatever it reports there is a
floor. Table in
[`results/identifiability.md`](../validation/network_compensation/results/identifiability.md).

**What the answer rests on.** The recovery does not return player 1's movement. It returns the smoothest
30 Hz movement that, under the model below, would have produced the logged trace. Each of these is an
assumption, and they are not equally safe:

| assumption | standing |
|---|---|
| the owner sampled on a uniform 30 Hz grid | verified, `SentTime` is the tick number over 30 (`NetworkTransform.cs:2372`). The rate itself the data cannot choose, but a 10% error moves a total by 0.06% |
| every tick produced a sample that arrived | not verified. Sends are threshold gated at 0.001 units and 0.01 degrees, so a perfectly still player sends nothing, and packets can be lost |
| the host rendered exactly one tick behind | **assumed and untestable from the data**, see below |
| each frame reads a linear interpolation of two neighbouring ticks | breaks when the buffer consumes more than one item at once, or extrapolates, which it allows up to three ticks |
| one smoothing step per logged row, with `a = dt / 0.1` from the logged timestamps | **known to be wrong**. Rows come from `FixedUpdate` at 50 Hz and smoothing steps from `Update` at up to 144, so a row spans about three steps. Measured cost below |
| the tick phase is constant within a 60 s block | reasonable, the fit finds a clear best phase |
| among traces that fit equally well, the calmer one is right | a preference, not a fact. It is what `LAMBDA` encodes |

Costs are measured in
[`results/assumption_errors.md`](../validation/network_compensation/results/assumption_errors.md). The one
worth knowing is that **the assumption known to be wrong is the one that matters least here**. Stepping
the path at the real 144 Hz and solving with the log times leaves the rotation total at 0.997 and the
distance total at 0.997, and the stillness share is unchanged. Several small smoothing steps compose into
very nearly one larger one, so the shape of the trace survives and only its effective lag shifts. That
shows up as position error, 0.041 units against 0.0004 when the assumption holds exactly, which is a
quarter of a percent of the arena radius.

The pattern across all of these is the same. **Timing is the weak axis and shape is the strong one.** The
whole-tick lag, the render clock and the millisecond timestamps all move the trace in time without
changing its path length or how often it holds still. Totals and stillness are what this data is used
for, and they are the robust part; anything read at a single instant is the exposed part.

The residual test measures what is left over. Real player 1 leaves about six times more unexplained than
the two time-axis errors account for, so the model is a good approximation and not an exact account. The two untested assumptions above, threshold gated sends and buffer skipping, are the
most likely home for that.

**Exact holds do not come back.** The penalty that keeps the totals honest also stops the solution
sitting at exactly zero. Cutoffs finer than about 0.4 degrees per tick therefore still separate the two
players, and must not be used.

**Player 1 has no head pitch, and never did.** Only yaw is networked (`SyncRotAngleY: 1`, with
`SyncRotAngleX: 0` on `FirstPersonPlayer.prefab`). Pitch and roll come from the camera, and
`ArenaLogger.BuildPlayerPosition` reads them with `GetComponentInChildren<Camera>()`, which finds the
host's own camera for both players. In all 100 social sessions `data.playerPosition.1.rotation.x` and
`.z` are an exact copy of player 0's, while yaw never is. This is not something the recovery can address,
as the values were never transmitted. Nothing in the repo reads these columns today; the constants in
`globals.py` carry a warning and `tests/test_log_data_properties.py` pins it. Player 0's pitch is its own
and is fine. Solo logs have no player 1 columns at all.

**The whole-tick part of the delay is assumed, not fitted.** `LAG_TICKS` only shifts which times the
recovered knots are labelled with; it does not change the shape of the fit. Setting it to 0, 1, 2, 3 or 5
gives a residual identical to every decimal place, so nothing in the data chooses it, and the round trip
cannot test it either as it uses the same constant both ways. The value rests entirely on reading
`ticksAgo` off `NetworkTransform.cs:2799`, and that reading is incomplete. The owner stamps `SentTime`
from its own network tick (`NetworkTransform.cs:2372`), which leads the server's, and `TryConsumeFromBuffer`
holds an item until server time reaches it. The effective delay therefore depends on the round trip time
and need not be exactly one tick.

Step sizes are unaffected, so nothing in the movement or stillness results depends on this. What does
depend on it is everything read at one instant: slice-onset position, wall visibility, head angle. If the
true delay is two ticks rather than one, recovered heading is still about 33 ms out, which is roughly half
the error the recovery is there to remove.

This is testable and has not been tested. On trials player 1 wins, the moment the recovered trace crosses
the wall collider can be compared against the logged trigger time. For player 0 that gap is near zero by
construction, so the difference between the two players measures the delay directly.

### Stillness cutoffs to use

Rotation, from the round trip (6 sessions) and from recovered player 1 against logged player 0 (100
sessions). This table predates `validation/`, so unlike the rest of this file it has no script behind it
and cannot be regenerated. The conclusion it supports is unchanged by anything since, but treat the exact
figures as provisional until it is rebuilt:

| cutoff (deg/tick) | round-trip error | player 0 - player 1 | p |
|---|---|---|---|
| 0.05 | -0.037 | +0.161 | 5e-10 |
| 0.1 | -0.025 | +0.118 | 5e-7 |
| 0.2 | -0.015 | +0.079 | .0007 |
| 0.4 | -0.007 | +0.030 | .087 |
| **0.8** | **-0.003** | **+0.009** | **.75** |
| 1.2 | -0.002 | +0.005 | .88 |
| 1.6 | -0.002 | -0.004 | .63 |

Translation recovers well at every cutoff tested (round-trip error 0.003 or less from 0.005 to 0.24
units per tick), but the small-cutoff shares still differ between players, as a locally logged player
repeats its position exactly while a recovered one does not.

**Use `STILL_ROT_PER_TICK = 0.8` degrees and `STILL_SPEED_PER_TICK = 0.08` units per tick.** These are
the frame-level thresholds of the agent pipeline (0.5 degrees and 0.05 units per frame at about 48 Hz)
rescaled to a 33 ms step, so they keep the same physical meaning: about 24 degrees per second and about
2.4 units per second. They also sit where recovery is faithful. Do not tighten them for recovered data.

## The saved data

`build_recovered_social` writes `social_recovered_ticks.pkl` to `data_strings.DATAFRAME_PATH`
(`/home/tom/repos/data/`), beside the session pickles. It takes about 20 minutes to rebuild and is not
rebuilt automatically.

- 3,622,081 rows, one per network tick per session, 149 MB.
- `experiment`, `t` (seconds from the session's first frame), `p1_x`, `p1_z`, `p1_yaw` (recovered),
  `p0_x`, `p0_z`, `p0_yaw` (read locally at the same instants), `trial_num`, `is_iti`.
- Yaw is unwrapped, so rotation can be differenced without unwrapping again. Floats are float32, which
  is far finer than the recovery's accuracy.
- `trial_num` and `is_iti` come from the frame each tick falls in, so movement can be cut by trial and
  by epoch without loading anything else.

### What it does not have

Everything in the table is positions and rotations. Compared with `social_trials_list.pkl` it is missing:

- **Events.** No `slice onset`, `trial end` or trigger rows, so no response times, no winner, no scores.
- **Trial detail.** No wall identities, trial type or wall separation.
- **Epochs** beyond the `is_iti` flag. No `slices active` or `post choice`.
- **Head pitch and roll**, and every other logged column. For player 1 the pickles do not really have
  these either, as noted in Limits.
- **Frame resolution.** 30 Hz, not the logged 48 Hz.
- **Solo sessions.** Social only. Solo players are logged locally and need no recovery, so they are
  sampled onto the same grid instead and saved as `solo_local_ticks.pkl` (6.9M rows, 283 MB, about 15
  seconds to rebuild, one row per player per tick with the same columns plus `player`, `session_type`
  and `source`). `human_characterisation.load_tick_traces` returns both as one long table.

Use it alongside the session pickles, not instead of them.

### The opponent view table

`collect_opponent_views` writes `social_opponent_views.pkl` beside the others, 26,480 rows, one per trial
per viewer. It takes a few minutes and is not rebuilt automatically.

- `experiment`, `trial_num`, `viewer`.
- `true_angle`, `true_visible` from where both players really were.
- `seen_angle`, `seen_visible` from what that viewer's screen showed.

Read it with `human_characterisation.load_opponent_views`.

## Which data to use for what

| analysis | use |
|---|---|
| Per-frame stillness, resting, scanning in **social** | recovered ticks, with the cutoffs above |
| ITI **rotation** totals in **social** | recovered. The logged total is 6 to 7% low by construction, as the smoothing can only take angular change out of a trace |
| ITI **distance** totals in **social** | either, they agree within about 2%. Recovered is still the better of the two |
| **Solo** sessions, anything | the session pickles as logged. Solo players are local, so there is nothing to recover. Sample onto a 30 Hz grid only when comparing against social |
| Position at one instant in social (slice-onset centrality, occupancy) | recovered for player 1 if precision matters. The logged trace lags 33 ms plus smoothing, which is 0 error at the median but 1.2 units at the 95th percentile when a player is moving. Subject to the whole-tick caveat in Limits |
| Opponent visibility used as a condition on choice or outcome | each player's own screen, not the true geometry. A player can only act on what was drawn. Pass `screen_views=hc.screen_view_trial_lists(...)` to `conditioned_player_choice`. Player 0's screen is the session as logged, so only player 1's is built |
| Wall visibility and head direction in social | recovered for player 1. Logged heading errs 6.7 degrees at the median, 38 at the 95th percentile, enough to change whether a wall is in view. Subject to the whole-tick caveat in Limits |
| Winner, scores, wall identities, trial structure | the session pickles |
| Response times, and any margin between the two players | the session pickles, but read the note below first |
| Agent (ML-Agents) sessions | as logged. There is no netcode in the inference builds |

## What each player's data goes through

Player 0 hosts, so its own values are never touched. Player 1's are the only ones the network altered,
and the only ones reconstructed. Solo players are all logged locally, as player 0 of their own session.

| measurement | social player 0 | social player 1 | solo player |
|---|---|---|---|
| Between-trial movement, rotation, stillness | logged, **read at the 30 Hz ticks** | **recovered** at 30 Hz | logged, **read at the 30 Hz ticks** |
| Wall visibility, head angle | logged frames, 48 Hz, untouched | **recovered**, interpolated back onto frames | logged frames, untouched |
| Route to the wall, path ratio | logged frames, untouched | **recovered**, interpolated onto frames | logged frames, untouched |
| Position at slice onset, centrality, occupancy | logged frames, untouched | **recovered**, interpolated onto frames | logged frames, untouched |
| Group labels, `onset_centrality` in the player table | logged frames, untouched | **logged frames, untouched** (shared with `strategy_characterisation.ipynb`) | logged frames, untouched |
| Winner, scores, trial times, wall identities | log events, untouched | log events, untouched | log events, untouched |
| Response time | log events, untouched | log events, **late by about a round trip** | log events, untouched |
| Head pitch and roll | logged camera, its own | **not logged at all**, the column holds player 0's | logged camera, its own |
| Opponent visibility, **where they really were** | logged frames, untouched | **recovered**, interpolated onto frames | not applicable |
| Opponent visibility, **what player 0 saw** | logged frames, untouched | **logged, untouched**, already the host's rendered copy | not applicable |
| Opponent visibility, **what player 1 saw** | **simulated with a 2-tick lag** | **recovered**, interpolated onto frames | not applicable |

The three opponent visibility rows differ in which trace is filtered, and each answers a different
question. Where they really were takes both players true, and is nobody's screen. What player 0 saw keeps
player 0 true and takes player 1 as the host drew it, which is the log left alone. What player 1 saw keeps
player 1 true, from the recovery, and puts player 0 through the network path. A player sees itself directly
and the other player through the network, so each screen holds one true trace and one filtered one.

Use the first row to ask whether an opponent was really in view. Use a player's own screen to ask whether
seeing them changed what that player did, since a player can only act on what was drawn.

The two screen rows are the one place where simulating is right rather than a compromise. Reconstructing
what was on a screen means applying the network path, not undoing it. What player 0 saw needs no work, as
the logged player 1 values already are the copy the host rendered. What player 1 saw is not in the log at
all, and the client renders remote players **two** ticks behind rather than one
(`ticksAgo = !IsServerAuthoritative() && !IsServer ? 2 : 1`, `NetworkTransform.cs:2799`).

`screen_view_trial_list` builds either screen. For player 0 it returns the session as logged, since its
own values were never filtered and the logged player 1 values already are the host's rendered copy. For
player 1 it puts player 0's logged trace through `network_view(..., lag_ticks=NET_CLIENT_LAG_TICKS)` and
takes player 1's own values from the recovery. `true_view_trial_list` gives the matching case where both
players are where they really were.

`opponent_view_rows` measures the angle to the opponent and whether they were visible at slice onset under
both, and `collect_opponent_views` runs it over every session. Full table in
[`validation/network_compensation/results/opponent_view_impact.md`](../validation/network_compensation/results/opponent_view_impact.md).

| viewer | visible, true | visible, on screen | angle error, median | p90 | screen disagrees |
|---|---|---|---|---|---|
| player 0, host, 1 tick behind | 0.406 | 0.406 | 0.00 deg | 7.47 deg | 1.53% |
| player 1, client, 2 ticks behind | 0.469 | 0.469 | 0.00 deg | 7.16 deg | 1.50% |

**The marginal rate is preserved, and that is not the same as the error being harmless.** The visible share
matches to three decimal places for both players, because the disagreements fall both ways in near equal
numbers: 96 trials where player 0's screen showed an opponent who was not really in view against 107 the
other way, 101 against 98 for player 1. But the flag is used as a condition, and a condition that is wrong
on some trials pulls whatever is conditioned on it toward no effect. The two directions add to that rather
than cancelling. Sensitivity and specificity are about 0.98 and 0.99, which shrinks a conditional effect by
roughly 3%.

**The errors also land somewhere specific.** A disagreement needs the opponent to have moved between being
sent and being drawn, so every one of them sits on a trial where the opponent was already running at slice
onset: median speed 9.2 and 9.9 units per second on disagreeing trials against 0.00 on the rest. Those
trials carry a different outcome with them, an own-win rate of 0.394 and 0.417 against about 0.50
elsewhere. The misclassification is therefore tied to the outcome, not independent of it, so the 3% figure
is a guide to the size and not a guarantee of the direction.

**Slice onset is the most forgiving frame in the trial.** 60% of players are within half a unit per second
of standing still at onset, and a filter applied to someone who is not moving changes nothing, which is why
the median angle error is zero and only 1.5% of trials disagree. At trial end only 27% are that still and
the median speed is 7.8 units per second. A single-frame visibility measure taken later in the trial would
be wrong far more often, so this result does not transfer to other epochs.

The extra tick the client carries does not make its screen measurably worse than the host's: paired over
sessions, the disagreement rates are 0.0151 and 0.0149 (p = .92) and the median angle errors 1.47 and 1.06
degrees (p = .16).

One asymmetry to keep in mind. For player 0 the seen side is the log itself and the true side is the
recovery, while for player 1 it is the other way round. The modelled side is not the same one for the two
players, so their error bars are not on the same footing even though the numbers match.

Reading player 0 at the tick instants is a change of sampling only. Nothing is delayed, smoothed or
interpolated, and it is done so that both players' movement is measured over the same 33 ms step. Every
other row leaves player 0 exactly as logged.

Recovered values answer where the players really were, not what appeared on either screen.

### What the trial events do and do not pass through

The table above sends every outcome to the session pickles. That is right, but the reason is not that
outcomes avoid the network, and an earlier version of this file said they did.

Each wall trigger is evaluated on its own player's machine, gated on `IsLocalPlayer`
(`WallTrigger.cs:152`), so the geometry uses that player's true position rather than the host's smoothed
copy. **The winner is not corrupted by the smoothing**, which is the thing worth knowing and does not
follow from anything else here.

What does cross the network is the report. `UpdateTriggerActivation` reaches the server through
`UpdateTriggerActivationServerRPC` (`GameManager.cs:787`). Player 0's trigger is a local call on the
server; player 1's is an RPC, so it registers about one one-way trip later, and an RPC waits for the next
tick. Slice onset travels the other way, as a server-side network variable, so player 1 sees it late too.
Player 1's logged response time therefore carries roughly a full round trip that player 0's does not.

Measured, the effect is not visible in who wins: player 0 takes 49.2% of trials across the 100 sessions
(t = -0.52, p = .6), consistent with the two machines being on a LAN. Use response times across players
with the caveat in mind, and do not read a small player 0 advantage as behaviour.

### Where this is used

`player_fingerprint.ipynb` and `analysis/human_characterisation.py` use it as follows:

- `load_iti_table` measures every between-trial movement metric on the tick grid, for social and solo
  alike, with the cutoffs above. An ITI is a run of ITI ticks, named by the trial it leads into. Pairs
  spanning a block join are dropped, and the wait after the last trial is not an ITI.
- `load_fingerprint_tables(recovered=True)` hands each social session player 1's recovered trace, which
  `frames_from_ticks` interpolates back onto frames. Wall visibility, head angle, routes and slice-onset
  position are then measured on frames that are player 1's own, not the host's delayed copy of them.
- Trial outcomes come from log events, and are untouched. They did cross the network, but only in the
  way set out above, which does not change any of them.
- `screen_view_trial_list` and `collect_opponent_views` build the two screens and the true case, for the
  opponent visibility question. The fingerprint uses the true case.
- `conditioned_player_choice` takes `screen_views` from `screen_view_trial_lists`, and then measures
  opponent visibility on each player's own screen. Choices, walls and outcomes stay on the logged trials,
  as they come from log events. Without the argument it behaves as it did before.

### The forward simulation

An earlier fix went the other way (`analysis/human_characterisation.py`, `simulate_network_view` and `network_view`).
It puts locally logged players through the same network path, so everyone is measured through the same
blurred view. It makes players comparable, but by degrading good data to match the bad, and it cannot
undo player 1's 33 ms lag.

It has three uses left. Reconstructing what a player's screen showed is the one where it is the right
tool rather than a compromise, and `screen_view_trial_list` is built on it. It is also worth using when
frame-level (48 Hz) values are needed rather than the 30 Hz tick grid, and as a cross-check: a result that
holds under both the simulation and the recovery does not rest on either.

It is not needed for solo sessions on their own, for totals, for anything built on events (winner,
scores, response times), or for agent sessions.
