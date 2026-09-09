# Fingerprint metrics

How each metric in the model characterisation notebooks is calculated. All the numbers come
from [`analysis/model_characterisation.py`](../../analysis/model_characterisation.py); the
notebooks hold only the per-batch config, the plots and the written-up results.

## Shape of the pipeline

Three passes:

1. `collect_model_data` loads one model's self-play session at a time and reduces it to six
   long tables. `slice` (row per trial x agent), `iti` (row per ITI x agent), `vis` (row per
   trial x agent), `win` (row per trial, winner only), `score` (row per trial), `rt` (row per
   completed trial).
2. `onset_centrality_by_model` then `split_strategies` collapse `slice` to one mean per model
   and label central/peripheral.
3. `build_fingerprint` groups every other table by model and merges them into one row per model.

Both agents are pooled wherever they can be (`PLAYERS = (0, 1)`), as self-play runs the same
model in both.

Every table covers the same trials. Two are always dropped. The session's **first trial** goes by
convention, and the **last trial** has no `trial end` event, as logging stops on the final
trigger, so it has a time but no score. One `keep_trials` set in `collect_model_data` drives all
six tables, and events are paired on `trialNum` rather than by row position, so a trial missing
an event cannot shift the rest.

Note that `trialNum` 0 is not a trial. It is the wait before trial 1, carrying only a
`logging start` event, and `split_session_by_trial` already drops it along with an unfinished
final trial. Real trials run from 1.

## The strategy axis

`onset_centrality` takes the agent's x, z at the slice-onset frame and computes
`1 - hypot(x, z) / ARENA_RADIUS`. One value per trial per agent, and the per-model number is
the plain mean over all of them. `onset_sd` is the spread.

`split_strategies` labels the per-model means three ways. A fixed cut at 0.5, a 2-component
GMM that finds its own boundary, and a 3-component GMM whose middle component tags
`intermediate` models, applied only if 3 components beat 2 on BIC. The working label is the
GMM one.

## ITI metrics

An ITI is every contiguous run of frames where `trial_epoch` is `ITI` or `trial started`, so
it spans trial end to the next slice onset. Runs of fewer than 3 valid frames are dropped.
Yaw is unwrapped first so a spin past 360 degrees does not fold back. Two per-frame
quantities are then taken:

- `step = hypot(diff(x), diff(z))`, translation per frame
- `rot = |diff(unwrapped yaw)|`, rotation per frame in degrees

| metric | logic |
|---|---|
| `prop_no_translational` | fraction of frames with `step < 0.05` units/frame |
| `prop_no_rotational` | fraction of frames with `rot < 0.5` deg/frame |
| `proportion_resting` | fraction with both, the all-zero action the penalty-break exempts |
| `iti_path_len` | sum of `step` over the ITI |
| `rot_total` | sum of `rot` over the ITI |
| `iti_centrality` | `1 - mean(hypot(x, z)) / R`, averaged over frames in the ITI |

The per-model value is the unweighted mean over ITI rows.

## Route to the wall

`winner_path_metrics` measures only the trigger activator, as the loser never completes a
route. Over the window from slice onset to trigger:

- `winner_path_len` is the sum of consecutive euclidean steps
- `winner_direct_len` is the straight-line distance from first to last point
- `path_ratio` divides actual by direct

## Wall visibility

`get_wall_visible` returns an 8 x timepoints boolean array. Per frame it takes the smoothed
head-angle vector and the vector to the closest point of each wall, and calls the wall
visible when that angle is under FoV/2 (55 degrees at FoV 110). The test is purely angular,
with no occlusion and no distance limit. Rows are indexed `wall - 1` for the two trial walls
and OR'd together, so "visible" means either trial wall is in view.

Frame 0 is slice onset and the window ends at the trigger, giving per trial:

- `visible_at_onset` is `any_vis[0]`
- `ever_visible` is `any_vis.any()`
- `latency_to_wall_visible` is `argmax(any_vis) / 50` seconds, NaN if never visible

Aggregated per model:

- `frac_visible_at_onset`, the mean of the flag
- `frac_never_visible`, `1 - mean(ever_visible)`
- `late_latency_to_wall_visible`, mean latency over trials not visible at onset but eventually
  visible
- `all_latency_to_wall_visible`, mean over every trial ever seen, scoring visible-at-onset as 0

`full_trial_window=True` computes over trial start to trial end and crops back, which keeps
trials shorter than 25 frames that would otherwise return NaN. Agents parked on a trigger
produce those on about a quarter of their trials.

## Timing and scoring

- `latency_to_reach_wall` is the median of (trigger `timeApplication` - slice onset
  `timeApplication`). Median rather than mean, as a few trials run on for tens of seconds
- `frac_trials_over_5s` is the fraction of those over 5 s
- `p_high` is the fraction of trials where `winner_score = max(score_p0, score_p1)` is 1.0,
  using 1.0 = High, 0.4 = Low, 0 = lost the race
- `mean_score` is the mean of agent 0's terminal score
- `score_rate` is `sum(winner_score) / sum(rt_s)` over the session, so score per second of
  trial time

These use terminal score, not agent reward. Reward folds in step/time penalties and the
no-movement penalty-break, and is not even applied at inference.

`score_rate` uses `winner_score` because both agents are the same model, so the loser scores 0
and the winner's score is all the score the model produced on that trial. Time is the decision
window only (slice onset to trigger), as the ITI is a fixed wait set by the environment and not
something the model can shorten.

The score and the time are summed over the whole session and divided once, giving total score
over total time. Averaging a per-trial `score_i / rt_i` does not work here. An agent parked on a
trigger wins in 0.007 s, under one logged frame, and that trial alone rates at 143 score per
second. On the 260723 batch 23 of the 24 peripheral models win 21-39% of their trials in under
0.1 s, so trimming outliers cannot remove them. Averaging per-trial rates puts those models at
11-23 score per second against 0.23-0.40 for the session sum, and ranks the 55 models backwards
(Spearman -0.58 against the session sum). Summing first keeps every trial's contribution finite.
An instant win adds its full score to the numerator and almost nothing to the denominator, which
is the largest contribution a single trial can make.

The two sums are taken over the same trials. Scores come from trial-end rows and times come from
slice-onset and trigger rows, and those cover different trials, so both tables carry `trial_num`
and are merged on it before summing.

## Picking the best models of a strategy

Every worked example below comes from the **260723** batch, 55 models, 25 central and 24
peripheral.

`rank_models` ranks the models of one strategy, best first. It looks for the **best** models of a
group rather than the most typical ones, so it ranks on performance and does not measure distance
from a group centre. That is mainly aimed at the peripheral models. The central ones are
near-identical, so any of them stands in for the others.

| metric | better when |
|---|---|
| `score_rate` | higher, more score per second of trial time |
| `latency_to_reach_wall` | lower, gets to the wall sooner |
| `frac_visible_at_onset` | higher, already has a wall in view when the trial starts |
| `late_latency_to_wall_visible` | lower, finds a wall faster when it has to turn for it |
| `winner_path_len` | lower, takes a shorter route to the wall it triggers |

### The seven steps

1. Take the models of one strategy. The ranking is always within a group, never across groups.
2. Blank a `late_latency_to_wall_visible` measured on fewer than `MIN_LATE_TRIALS` (20) trials,
   then drop any metric more than `MAX_MISSING_FRAC` (25%) of the group is missing. That keeps
   every model on the same metric set. On the 260723 peripheral group late latency goes, missing
   for 42%, and the ranking runs on the other four. It survives for the central group.
3. Find each metric's centre and spread across the group. **This is the only step where the two
   ranking methods differ.**
4. Standardise every value against its own metric's centre and spread.
5. Flip the sign on the lower-is-better metrics, so a positive score always means better.
6. Average the standardised scores with **equal weight** on every metric. That is `quality`.
7. Sort best first. `margin` is the gap down to the next model, which says how firm a cut is.

### The two ranking methods

**Mean of signed z-scores** (`method='classic'`) uses the arithmetic **mean** as the centre and
the **standard deviation** as the spread. Both are computed from every model, so an extreme model
pulls the centre and inflates the spread.

**Robust z** (`method='robust'`, the default) uses the **median** as the centre and
**MAD x 1.4826** as the spread, where MAD is the median of each model's distance from the median.
Both are fixed by the middle of the group. The 1.4826 rescales a MAD to read like an SD, so a
quality score means roughly the same thing under either method.

Strengths and weaknesses:

| | classic | robust |
|---|---|---|
| centre and spread | mean, SD | median, MAD x 1.4826 |
| one extreme model | pulls the centre and stretches the spread | barely moves either |
| equal weight | exactly true, every metric has unit SD by construction | only nominal, metrics with a tight middle sway the average more |
| reads | familiar z-scores | same scale, set by the typical model |

The trade is a real one. Under classic standardising each metric contributes exactly one unit of
spread, so equal weight is literally equal. Under robust standardising each metric contributes
one MAD, and the standard deviations of the four columns come out unequal. On the 260723
peripheral group that gives the most influential metric twice the sway of the least, because
three of the four have a tight middle and one far tail. `score_rate` runs 0.233 for model 41
against 0.265 for the next lowest and a median of 0.356; `latency_to_reach_wall` runs 3.04 s for
model 22 against 2.73 s for the next slowest and a median of 2.30 s. Classic keeps the weights
honest but lets those single models set the scale for everybody; robust describes the typical
model faithfully but lets the metrics drift apart in influence.

**The two usually agree.** On the 260723 peripheral group they rank the 24 models at Spearman
0.991 and pick the same five models, differing only in the order within that five. Expect the
choice of method to decide the edge of a shortlist, not its core.

### Comparing the two

`report_method_comparison` runs both and prints the two shortlists, which models make both and
which make only one, the rank agreement, the biggest movers, and how far outliers are pulling
each metric. `compare_ranking_methods` returns the same as a table.

Outlier pull is read from `sd_over_mad`. Above 1 the standard deviation is stretched by the
tails, so classic standardising divides by a spread the typical model never sees and squashes
everyone toward zero. `influence` is how much each standardised metric actually varies, so how
much sway it has in the equal-weight average. It is 1 for every metric under classic by
construction, and is not under robust.

### Looking at chosen models

`inspect_models` takes any set of models and gives the ranking metrics for each, with the raw
value, its rank in the group and its standardised score under **both** methods. `hidden_metrics`
gives the same models the fingerprint metrics the ranking never sees, which is everything left
over once the ranking metrics are removed, plus any ranking metric dropped at step 2. No
direction is implied there, as those metrics are descriptive, so `rank_high_to_low` is simply the
group ordering by value.

That pairing is the point of the whole comparison. Where the two methods disagree, the reason is
visible in the metric values, and the unseen fingerprint often decides it. A model can rank well
on the four ranking metrics while resting most of the ITI, winning the High wall rarely, or
carrying a slow-trial tail, and only the second table shows that.

### Other checks

`rank_stability` re-ranks with each metric dropped in turn. A model that stays near the top
however the metrics are cut is a firm pick, and one that moves a long way is riding on a single
metric. `metric_overlap` gives the within-group rank correlations between the metrics, so a
signal counted twice is visible. On the 260723 peripheral group the strongest pairs are
`score_rate` with `winner_path_len` at 0.64 and `latency_to_reach_wall` with `winner_path_len` at
0.61, as a shorter route takes less time and earns more score per second.

The scores are within-group, so they do not compare across strategies. A quality of +1.2 means
the best of that group, however tight the group is. Across the 260723 batch the central models
span 11.8% on `winner_path_len` and 12.7% on `score_rate`, against 27.7% and 47.3% for the
peripheral ones, so ranking the central group mostly stretches noise to unit variance.

## Strategy comparison

`between_strategy_variance` reports eta^2 and a one-way ANOVA per metric, but the strategy
labels come from a GMM fit to `onset_centrality` on the same data. The comparison is
circular for every metric that tracks centrality, so read the eta^2 ordering and not the
p-values. The function's docstring lists the specific metrics this misleads on.
