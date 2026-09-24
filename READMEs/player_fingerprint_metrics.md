# Player fingerprint metrics

How every metric in `demos/strategy_characterisation/humans/player_fingerprint.ipynb` is derived from
the raw session logs. One metric per entry, from the logged rows to the number in the table.

This covers the human pipeline. The agent version, and the model shortlist ranking built on top of it,
are in [`demos/strategy_characterisation/fingerprint_metrics.md`](../demos/strategy_characterisation/fingerprint_metrics.md).
The strategy axis and the session-level results are in
[`human_strategy_characterisation.md`](human_strategy_characterisation.md).

## The path from log to metric

A raw session log is one row per logged frame, with event rows interleaved. `trial start`,
`slice onset`, `selected trigger activation` and `trial end` mark the trial, and `trial_epoch` marks
the wait between trials. The pickles hold these split into trials, one list per session.

Three passes turn that into one row per player per session.

1. **The onset table** (`hc.load_tables`) reduces each trial to the player's position at slice onset.
   It sets the strategy axis and the groups.
2. **The fingerprint tables** (`hc.load_fingerprint_tables`) reduce the same trials four more ways.
3. **`hc.build_player_fingerprint`** groups each table by player and session and merges them into one row.

| Table | A row is | Rows | Built by |
|---|---|---|---|
| `onset` | one trial x player, at the slice-onset frame | 47,689 | `hc.collect_onset_table` |
| `iti` | one ITI x player | 47,281 | `hc.iti_metrics_ticks` via `hc.load_iti_table` |
| `vis` | one trial x player, that player's own view | 47,690 | `mc.visibility_metrics` |
| `win` | one won trial, the winner's route | 34,450 | `hc._win_path` |
| `outcome` | one trial x player, result and trial time | 47,690 | `hc._trial_outcome` |

The fingerprint holds 600 rows, 200 players x 3 sessions, over 100 experiments.

## Shared ground

**Trials.** The first solo drops its first 5 trials. Social and second solo keep every trial. The
pre-trial block and any final trial with no trial end are already dropped at save time. Every table
covers the same trials, and rows are keyed by `trialNum` rather than by position.

**Groups.** `hc.extreme_groups` ranks the players of one session by mean onset centrality. The top 20%
are central, the bottom 20% peripheral, and the middle 60% are left out of the comparisons. That is 40
players per group per session. Sessions are ranked separately and never pooled, so the social groups
are not the solo groups.

**Two clocks.** Between-trial metrics are measured on the 30 Hz network tick grid, and everything else
on the logged frames. Social player 1's transform reaches the host through Netcode's interpolation, so
their per-frame movement is not comparable with player 0's. Their real movement is recoverable only at
the tick rate, so every player is put on that grid for the ITI metrics. Frame-level metrics take social
player 1's frames from the recovered trace instead. The derivation is in
[`network_compensation.md`](network_compensation.md).

**Time in seconds.** A frame index is converted with the trial's own `timeApplication`, by
`mc.window_times`. The human logs run at about 48.25 rows per second and some sessions at 30, so
dividing by a nominal 50 reported latencies 3.6% short on clean files and 65% short on the slowest
session. Trial time in the `outcome` table always came from the logged times and was never affected.

**Score.** 1.0 for reaching High, 0.4 for Low, 0 for losing the race, which is the agent scale. On a
forced trial both walls pay the same, so the trial scores from its type. High/Low trials are 79.8% of
the data, forced trials the rest. The human builds log 50 and 20 in the cumulative score, and
`score_logged / 50` matches the outcome score on all but 4 trials, where the logged value is missing.

**Windows.** Visibility runs from the slice-onset row to the trigger row inclusive. Routes run from the
slice-onset row up to, but not including, the trigger row.

## The axis

### `onset_centrality`

Take the player's x, z on the first `slice onset` row (`get_indices.get_player_slice_onset_loc`), then
`1 - hypot(x, z) / ARENA_RADIUS`, where `ARENA_RADIUS` is the alcove back. 1 is the arena centre and 0
the alcove back. `mc.slice_onset_centrality` gives one value per trial per player, and `hc.player_table`
means them over the session. Slice onset is the frame the walls appear, so the position there is the
waiting position the player chose.

`sd_centrality` is the spread over the same trials, and `frac_peripheral` is the share of trials past
the fitted trial boundary.

## Between trials

An ITI is a run of consecutive ITI ticks, labelled with the trial it leads into. The run before a
session's first trial follows no trial end, so it is not an ITI and goes. A run whose ticks all carry
one trial is the wait at the end of the session, so it goes too. A run needs at least 2 tick pairs to
give a row.

Each tick pair gives two quantities. `step` is `hypot(dx, dz)`, the translation over 33 ms. `rot` is
the shortest turn in yaw, in degrees, so a wrap past 360 is not read as a full spin. The per-player
value is the unweighted mean over that player's ITIs.

| Metric | Derivation |
|---|---|
| `prop_no_translational` | share of tick pairs with `step < 0.08` units |
| `prop_no_rotational` | share with `rot < 0.8` degrees |
| `proportion_resting` | share meeting both |
| `iti_path_len` | sum of `step` over the ITI |
| `rot_total` | sum of `rot` over the ITI, in degrees |
| `iti_centrality` | `1 - mean(hypot(x, z)) / ARENA_RADIUS` over the ITI's ticks |

The cutoffs are the agent per-frame thresholds rescaled to a 33 ms step. Finer cutoffs do not survive
the recovery and must not be used. They have not been checked against human input.

## Wall visibility

`trajectory_headangle.get_wall_visible` returns an 8 x timepoints boolean array for one player. Per
frame it takes the smoothed head-angle vector and calls a wall visible when any part of its coloured
face falls within half the FoV of it with a clear line of sight. The human FoV is 110.36 degrees.
Walls sit at their physical positions, alcove walls block sight from inside an alcove, and there is no
distance limit.

The rows of the two trial walls are OR'd, so visible means either trial wall is in view. Column 0 is
slice onset. Per trial that gives `visible_at_onset` (column 0), `ever_visible` (any column) and
`latency_to_wall_visible` (the first visible column's logged time, referenced to slice onset, NaN when
never visible). Every trial counts, won or lost, as each player is measured from their own view.

| Metric | Derivation |
|---|---|
| `frac_visible_at_onset` | mean of `visible_at_onset` over the player's trials |
| `frac_never_visible` | `1 - mean(ever_visible)` |
| `late_latency_to_wall_visible` | mean latency over trials with no wall at onset but one seen later |
| `all_latency_to_wall_visible` | mean latency over every trial a wall was seen on, scoring visible-at-onset as 0 |

`n_late_latency_to_wall_visible` counts the trials behind the late latency, which runs to single figures
for some peripheral players.

## Route to the wall

Measured on the trigger activator only, as a loser never completes a route. In solo the player is
always the winner. The window runs from slice onset to the frame before the trigger, and each metric is
averaged over the player's won trials.

| Metric | Derivation |
|---|---|
| `winner_path_len` | sum of the euclidean distance between consecutive frames |
| `winner_direct_len` | straight-line distance from the first frame to the last |
| `path_ratio` | `winner_path_len / winner_direct_len`, per trial, then averaged |

## Timing, choice and score

Per trial, the winner is the trigger activator, `chose_high` is whether the wall they triggered is the
trial's wall 1, which is High by repo convention, and `rt_s` is the trigger's `timeApplication` minus
slice onset's. A losing player's row carries the same trial time with a score of 0. Players win 72.2% of
the trials they are in, as solo trials are always won.

| Metric | Trials used | Derivation |
|---|---|---|
| `latency_to_reach_wall` | own wins | median `rt_s`, as a few trials run on for tens of seconds |
| `frac_trials_over_5s` | own wins | share of `rt_s` over 5 s |
| `p_high` | own wins, High/Low only | mean of `chose_high` |
| `mean_score` | every trial | mean score, a loss scoring 0 |
| `score_rate` | every timed trial | total score over total trial time, summed over the session then divided once |
| `win_rate` | every trial | share won, social only and NaN elsewhere |

Latency, the slow-trial share, the route and P(High) use the player's own wins, so they describe that
player rather than their opponent. In social a lost trial ends on the opponent's trigger, so its time
and route are not the player's.

`score_rate` sums score and time over the session before dividing. Averaging a per-trial rate lets a
win in under one logged frame rate at hundreds of score per second and rank players backwards. Time is
the decision window only, as the ITI is a fixed wait the player cannot shorten.

`n_wins`, `n_high_low_wins` and `n_outcome_trials` carry the trial counts behind these, so a metric
resting on a handful of trials is visible.

## What the numbers do not settle

- The groups come from onset centrality, so every metric that tracks centrality separates by
  construction. Read the eta² ordering, not the p values.
- Partners can fall in the same group, so the players in a comparison are not independent.
- `path_ratio` is averaged per trial, and a trial starting almost at the wall can reach a ratio of 100.
- `frac_never_visible` partly measures trial length, as a trial the opponent ends at once never gives a
  wall time to come into view.

The notebook's known-problems cell holds the full list, with the size of each effect.

## Code and tests

| Path | What it holds |
|---|---|
| `analysis/human_characterisation.py` | The human tables and the fingerprint |
| `analysis/model_characterisation.py` | The visibility, ITI and route metrics shared with the agents |
| `data_extraction/network_compensation/` | The tick grid and the recovery of social player 1 |
| `tests/test_human_characterisation.py` | Unit tests for the tables and the fingerprint |
| `tests/test_visibility_timing.py` | Unit tests for converting a frame index to seconds |
| `tests/test_network_compensation.py` | Unit tests for the recovery |

Run them from the repo root with the conda env python:

```
/home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest \
    tests.test_human_characterisation tests.test_visibility_timing tests.test_network_compensation -v
```
