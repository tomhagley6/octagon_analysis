# Human strategy characterisation

Where human players wait when a trial starts, and what else those players do differently.

Both notebooks here describe players on one axis, **centrality at slice onset** (`1 - r / R`, 1 = arena
centre, 0 = alcove back). Slice onset is the frame the walls appear, so a player's position there is
the waiting position they chose. The RL agents are characterised on the same axis in
`../models/competitive/`, so humans and agents can be read against each other.

## Files

| File | What it holds |
|---|---|
| `strategy_characterisation.ipynb` | The high-level strategy. Who waits centrally, who waits at a wall, how steady that is, and how it changes across sessions. |
| `player_fingerprint.ipynb` | The top and bottom 20% of players on the same axis, compared on the behavioural metrics used for the agents. |
| `model_characterisation_demo260723.ipynb` | The agent version of the fingerprint, kept here for comparison. |
| `../fingerprint_metrics.md` | Definitions of every fingerprint metric. |
| `analysis/human_characterisation.py` (repo root) | Everything that computes a number. The notebooks only plot and describe. |
| `tests/test_human_characterisation.py` (repo root) | 106 unit tests for that module. |

## Data and conventions

- **Sessions.** Each player plays a first solo, a social session with a partner, and a second solo.
  An **experiment** is one pair on one day, holding its social session and four solo sessions.
  100 experiments, 200 players, 2024-09 to 2026-03.
- **Sessions are never pooled.** The social context differs, so every fit is done per session and the
  sessions are compared only afterwards.
- **Trials.** The first solo drops its first 5 trials (first exposure to the task). Social and second
  solo keep every trial. The repo-wide rules are in the parsing code: the pre-trial block is dropped at
  save time, as is a final trial with no trial end.
- **One exception to fitting per session.** The boundary between central and peripheral *trials* is a
  single value for all three sessions, fitted on social trials (0.49). Fitting it per session made
  "peripheral" mean a different position in each, so cross-session label changes were partly an
  artefact of the cut moving.
- **Protocol changes.** 2024 solos ran about half as many trials, and a few 2024 experiments used a
  different mix of wall separations. 2026 experiments all ran the same protocol, so they serve as a
  clean subset. Year is not used as a general covariate.
- **Caching.** Tables and any resampling run through `data_extraction.data_saving.cached`. Set
  `RECOMPUTE = True` in the first cell to rebuild. Rebuilding the session file list needs the raw data
  mounted.

## `strategy_characterisation.ipynb`

1. **Protocol across the timeline.** Session lengths and wall separation shares per experiment, so the
   2024 differences are visible before anything is pooled.
2. **Trial-level split.** A 2-component Gaussian mixture on social trial centralities sets one boundary
   (0.49) for all sessions. Onset positions are clearly bimodal, but the middle is flat rather than
   empty, so the boundary is sensitive.
3. **Social strategy types.** A fixed cut on the fraction of peripheral trials (≤20% central, ≥80%
   peripheral, mixed between) is the working label, because it needs no fitting and its edges do not
   move with the data. Mixtures with k = 1-4 on player means are shown beside it, with a bootstrap over
   players for how stable the best k is. BIC separates k = 2, 3 and 4 by under 2 points, so the number
   of groups is weakly supported and k = 3 is exploratory.
4. **Consistency.** A raster of every player's trials, a shared-rate null, split-half reliability, and a
   run test.
   - **4a. Win-stay, lose-shift.** Onset centrality against the *previous onset*, split by the previous
     trial's outcome, as signed change (direction) and absolute change (size).
   - **4b. Drift within a session.** Half split, largest step and OLS slope per player, against a null
     that reorders each player's own 10-trial blocks, so runs of one strategy are not counted as drift.
   - **4c. Alternating or switching once.** A switch score, the number of label changes over the number
     expected if the same trials came in random order.
5. **Solo strategy types.** The same labels and mixtures refitted on each solo, then all three sessions
   on one axis.
6. **Change across sessions.** Player paths, label transitions, a timeline, partner influence (6d), and
   first solo to second solo (6f), which is the cleanest comparison as both are the same task played
   alone.

### Main results

- **Players separate into two clear groups**, waiting near the centre (~0.8) or near a wall (~0.1), with
  a spread of mixed players between. Players differ far more than a shared habit allows.
- **Individual players are consistent within a session** (odd-even split-half 0.98 to 0.996) but only
  moderately so between the two solos (r = 0.59). They genuinely change.
- **Everyone drifts toward the wall a little**, in every session (mean half difference −0.03). Drift is
  largest in the first solo and smallest in the second.
- **Partners' positions predict each other.** A partner who started 0.1 more central goes with a player
  sitting 0.024 more central in social, beyond the player's own start, and 0.021 still in the second
  solo. Partners are already slightly alike in the first solo, before they meet.
- **Outcome does not change where a player waits next (4a).** What looked like win-stay, lose-shift
  tracks where the previous trial left the player, and disappears when that is held fixed.

## `player_fingerprint.ipynb`

Players are ranked within each session by mean onset centrality, and the top and bottom 20% (40 players
each) are compared on the agent metrics: route length, latencies, wall visibility, movement between
trials, rest, score and win rate. Groups are ranked again in each session, so they are not the same
players throughout.

Two things to know before reading it:

- **Player 1's positions arrive over the network** and are interpolated and smoothed by Netcode, which
  makes them look less still than player 0. The notebook reconstructs player 1's real movement by
  inverting the engine's own interpolation. The write-up is
  `READMEs/network_compensation.md`.
- **Its "known problems" cell lists what is not fixed**, including an inflated path ratio and second-solo
  first trials starting at the spawn point.

## Statistics, and why each one

- **Gaussian mixtures** (trial split, player means). Used because the distributions are bimodal.
  Reported with BIC and a bootstrap over players, as one BIC cannot say whether the best k is stable.
- **Reordering nulls rather than parametric tests** wherever trials are involved. Trials come in runs, so
  a test that assumes independent trials would call ordinary runs a real effect. Blocks of 10 trials are
  reordered, which keeps runs and removes only the slow trend.
- **Group-level rather than per-player claims for drift (4b).** The share of players passing a per-player
  test depends on session length, because a longer session gives the test more power. The group test
  (`p_group`) and the size of real drift (`drift_rms`) do not. Per-player p values are uncorrected and
  are only ever shown against the ~5% expected by chance.
- **Standard errors clustered by experiment** in every model, as the two players of an experiment share
  a social session and are not independent.
- **Covariates instead of differences.** When a later value is compared with a starting value, the
  starting value goes into the model as a predictor. Subtracting it from both sides builds a slope into
  the plot whatever players do. This is why 6d fits `later ~ own_solo + partner_solo` and plots it as an
  added-variable plot, whose slope is exactly the partner coefficient.
- **Effect sizes, not just p values.** With 200 players and ~26,000 trial pairs, tiny effects reach
  small p values. Results are reported with the median, Cohen's d, the share of players on one side of
  zero, and the size against a typical move.
- **Positions are logged exactly.** No interval here is a measurement error. A bootstrap interval asks
  whether the same player would show the same result over a different set of trials.

## Tests

```
/home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_human_characterisation -v
```

106 tests, all on synthetic data with a known answer (pytest is not installed in the conda env). They
cover:

- **Table building.** Solo files line up with their experiment and player, the first-solo cut, previous
  trial outcomes, and wall separations.
- **Labels.** The boundary falls between the two modes, one reference session sets it for all sessions,
  and the fixed cut behaves at its edges.
- **Mixtures.** Components are named by centrality whatever order sklearn returns, and known clusters
  are recovered.
- **Consistency and drift.** Block reordering keeps order inside a block, a planted trend is found, runs
  without a trend are not called drift, a planted step is found, and the group drift size is recovered.
- **Outcome effects.** Known win and loss changes, a planted lose-shift, gaps between trials breaking a
  pair, and a spurious win effect vanishing when the previous trial end is controlled.
- **Across sessions.** Partner effect recovery, a null that centres on zero, the added-variable slope
  equalling the coefficient, clustered standard errors widening for linked partners, and the shrinkage
  slope.
- **Fingerprints.** Metric collection, the network reconstruction, ITI metrics on the tick grid, and the
  group cuts.
- **Real data.** One class checks the cached session file list against known low-frame-rate sessions, and
  is skipped when that list is missing.

## Known limitations

- The trial boundary sits in a flat part of the distribution, so every label-based result depends on it.
  There is no sensitivity table yet.
- The number of strategy groups is not settled. Labels come from a fixed rule, not from evidence of
  distinct types.
- 6c and 6f step 4 cannot separate a social effect from more time on the task, as every player took the
  sessions in the same order.
- Experiment 260116_1 is absent from the pickles (its second solo has long logging blackouts), so 100 of
  101 experiments are analysed.
- Some sessions are short (one first solo has 3 trials after the cut) and 11 sessions were logged below
  48 Hz.
- Social logs put some frames past the reachable arena, mostly after the trigger, so trial-end positions
  are the least reliable ones.
