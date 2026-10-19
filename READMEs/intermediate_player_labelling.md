# Labelling intermediate human players

How human players who are neither clearly central nor clearly peripheral are labelled by the way their
waiting position changes over a session. Covers the simulated players used to test the labels, the
behavioural measures, the labelling rules, how they were checked, and the main results.

The analysis lives in section 5 of
[`demos/strategy_characterisation/humans/mixed_strategy_characterisation.ipynb`](../demos/strategy_characterisation/humans/mixed_strategy_characterisation.ipynb).
Every number below comes from that notebook's tables unless marked as a scratch check. The wider human
strategy work (fixed-cut labels, drift, partner influence) is described in
[`human_strategy_characterisation.md`](human_strategy_characterisation.md).

## Files

| file | what it holds |
|---|---|
| [`analysis/mixed_labels.py`](../analysis/mixed_labels.py) | every measure, the shuffle tests, the simulated players and the labelling rules |
| [`analysis/human_characterisation.py`](../analysis/human_characterisation.py) | the onset table, the trial boundary, the fixed-cut labels, and the gap-to-spread ratio with its intervals |
| [`tests/test_mixed_labels.py`](../tests/test_mixed_labels.py) | tests for each measure and rule against synthetic sessions with a known answer |
| `mixed_strategy_characterisation.ipynb` | sections 1 to 4 are the earlier routes (spread, own positions, soft shapes). Section 5 is the labelling. A sandbox at the end plots every social one-switch, long-runs, weak-trend, short-runs and unclear player trial by trial, for checking labels by eye |

Run the tests with
`/home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_mixed_labels -v`.

## The question

The fixed cut labels a player from the share of their trials below one boundary. At most 20% peripheral
trials is **central**, at least 80% is **peripheral**, and anything between is **intermediate**. That
label says how often a player waits at the wall. It does not say how they get there. An intermediate
player might switch once from the centre to the wall, move between the two in long runs, slide steadily
across the session, wait in a narrow band near the middle, or move back and forth on most trials. Section
5 labels each intermediate player by which of these they do.

**Social is the main focus.** It has the most trials (median 133 after the spawn trial is dropped), so
every measure is most reliable there. The first and second solo are labelled the same way for comparison.

## Data and conventions

- **One trial per row**, measured at slice onset (the frame the walls appear), as onset centrality
  (1 = arena centre, 0 = alcove back).
- **The spawn trial is dropped.** The first trial of every social and second-solo session starts at the
  arena centre, where players spawn, so it shows the spawn position and not a choice. The first solo
  already starts after a 5-trial cut, so it has no spawn trial. No other trials are dropped. Dropping it
  moved 3 social players out of one switch (their apparent switch was the spawn trial).
- **One boundary, 0.484**, fitted on social trials, marks trials as peripheral or central. It differs
  from the 0.485 in `strategy_characterisation`, which keeps the spawn trial.
- **Fixed-cut labels** come from all of a player's kept trials. Social has 95 central, 69 intermediate and
  36 peripheral players. Only intermediate players get the labels below (65 first solo, 69 social, 54
  second solo, 188 player sessions).
- **Sessions are never combined.** Each session is measured and labelled on its own.
- **Where a player starts each trial is their choice.** Nothing moves players but themselves, and the time
  between trials is long enough to reach any point. So consecutive trials sitting close together is
  expected behaviour, not a flaw in the data.

## Behavioural measures

All use the player's onset centrality in play order, with no boundary, unless stated. A session needs at
least 20 trials.

| measure | column | what it is | how it reads | role |
|---|---|---|---|---|
| trend strength | `trend_r2` | share of variation a straight line over the session explains | 0 = no steady rise or fall, 1 = every trial on the line | drifter |
| leftover spread | `resid_sd` | SD of trials around that line | small = a narrow band | hoverer |
| runs | `runs_r1` | correlation between consecutive trials' distances from the line | about 0 = no link between trials, positive = stays put for several trials | long runs, short runs |
| local jump | `local_jump` | largest change between the mean of the 10 trials after any point and the 10 before | about the size of a switch, small for a slow drift | description only |
| shifts beyond noise | `shift_share` | share of trial-to-trial differences bigger than the shift cut | high = moves back and forth often | mixer |
| two-section fit | `two_section_r2` | share of variation explained by two steady sections joined by a short transition | high = the session splits in two | one switch |
| section gap | `section_gap` | second section's mean minus the first's | negative = toward the wall | one switch |
| section purity | `purity_min` | share of the less pure section's trials nearer its own level | 1 = every trial at its own level | one switch |
| longest excursion | `longest_excursion` | longest unbroken run of trials, inside a section, nearer the other level | 0 = no trips across | one switch |
| runs left after two sections | `resid_runs_r1` | the runs measure on each trial's distance from the two-section fit | positive = runs inside the sections | short-runs flag for one switch |
| two positions | `gap_vs_sd` | gap between a player's own wall-side and centre-side means, over the spread within each side (section 3) | above 3.46 (one even spread cut in two) suggests two positions | description only |

**A trial-to-trial difference** is the size of the change in a player's onset centrality from one trial to
the next. Every pair of consecutive trials gives one, so a session of n trials has n − 1. Most are small.
**A shift** is a difference bigger than the shift cut, a move well beyond a one-position player's
trial-to-trial variation.

**The two-section fit** describes a session as section A (a steady level from the first trial), a
transition (a straight move from A's level to B's, 0 trials up to 20% of the session), and section B (a
steady level to the last trial). Each section has at least 5 trials, so the switch can fall anywhere that
leaves 5 trials either side. The split and transition length are chosen to fit best.

**Section purity** asks whether each section is mostly at its own level. A trial is nearer its own
section's level if it sits on that level's side of the midpoint between the two levels. Trips to the other
level lower purity. Consecutive trials sitting close together do not.

### Shuffle tests

Trend, runs, local jump, the two-section fit and the runs left after it each have a shuffle test. It asks
whether the order of a player's trials matters.

1. Compute the measure on the real order.
2. Put the player's own trials in random orders (1000, or 200 for the two-section tests), keeping every
   value, and compute the measure on each.
3. p is the share of shuffled orders at least as large as the real one. Any search over split points is
   repeated in every shuffle, as searching always finds something.

**A player passes when p < 0.05.** Their real order then gives a larger value than at least 95% of random
orderings of the same trials. Failing does not show there is no structure, only that it cannot be told
apart from random order. The tests are per player and uncorrected, so about 5% of players pass any one by
chance.

### Data anchors

Thresholds come from central players, who wait in one place, wherever possible.

| anchor | how it is set | social |
|---|---|---|
| narrow cut | 75th percentile of central players' leftover spread | 0.145 |
| middle range | between the typical (median) mean centrality of peripheral and central players | 0.151 to 0.760 |
| shift cut | 95th percentile of central players' trial-to-trial differences, counting only pairs where both trials sit on the centre side | 0.272 |
| shift reference | 95th percentile of central players' own shift shares | 0.21 |

## Simulated players

**Why.** A labelling rule is only trusted after it recovers players whose behaviour is known. Each kind
below is simulated with 200 players at the median trial count of each session (54 first solo, 133 social,
60 second solo). The rules are then run on them, and the share given the right label is reported.

**Settings.** All sessions use the social anchors. Only the number of trials changes, which widens the
clouds of the shorter solos without moving their centres.

| setting | value | source |
|---|---|---|
| two positions | 0.29 and 0.69 | intermediate players' own wall-side and centre-side means (section 3) |
| spread around a position | SD 0.11 | intermediate players' within-side spread |
| hoverer and drifter spread | SD 0.124 | central players' median leftover spread |
| hoverer position | 0.46 | midway between the typical peripheral and central means |
| run length | 10 trials on average | a choice, as the data gives no direct value |
| noisy one switch trips | 5% chance per trial of a 1 to 3 trial trip; consecutive scatter correlated 0.2 | a choice |

**Kinds.** Direction and starting position are random.

| kind | how it is built | label it should get |
|---|---|---|
| mixer | picks one of the two positions at random on every trial | mixer |
| one switch | holds one position, then the other. The move takes 1 trial up to 20% of the session, starting anywhere that leaves 5 trials either side | one switch |
| noisy one switch | one switch, plus brief trips to the other position and consecutive trials sitting close together, as real players show | one switch |
| long runs | moves between the two positions in runs averaging 10 trials | long runs |
| drifter | moves steadily from one position to the other, in a narrow band | drifter |
| hoverer | waits at 0.46 in a narrow band | hoverer |

**Limits.** The simulated players draw from fixed settings, several of which are choices. No kind is built
for short runs, so that label's recovery is not checked. Real players can combine behaviours (for example
drift plus runs), which no kind covers.

## Labelling rules

Only intermediate players are labelled. Each gets the first label they fit, in this order.

| label | rule |
|---|---|
| **one switch** | the two-section fit passes its shuffle test, the section gap is at least the shift cut, each section is stable (purity at least 0.80 and no excursion over 5 trials), and two sections explain more than a straight line (`two_section_r2` > `trend_r2`) |
| **long runs** | runs pass the shuffle test and are at least 0.46 (runs of about 5 or more trials, once spread within a position is allowed for) |
| **drifter** | the trend passes its shuffle test, and trend strength at least 0.10 |
| **hoverer** | no significant trend, leftover spread at or below the narrow cut, and mean inside the middle range |
| **mixer** | no significant trend, wider than the narrow cut, and a shift share above the shift reference |
| **short runs** | runs pass the shuffle test, but the player fits none of the labels above |
| **unclear** | anything else (weak or mixed signals) |

**Where the cut points come from.**
- Purity 0.80 sits just above where simulated noisy one switch (10th percentile 0.77 in social) and long
  runs (90th percentile 0.75) meet.
- Trend 0.10 sits above simulated mixers and hoverers (90th percentile 0.02 to 0.06), so players with no
  steady trend stay out. It is low enough to include players whose line clearly moves but whose trials
  scatter widely around it (examples in the notebook's sandbox).
- The two-section-beats-line condition keeps drifters out of one switch. Simulated one switch fits two
  sections better (0.68 to 0.71 against 0.45 to 0.47), and the drifter fits the line better (0.48 to 0.49
  against 0.42 to 0.46).

**Short runs beside other labels.** Labels are descriptive, so runs that pass the test do not block drifter,
hoverer or mixer. A yes/no column, `has_short_runs`, marks short runs beside any label. For a one-switch player it
means runs left inside the two sections, as the switch itself leaves runs around a straight line. For
anyone else it means runs that pass the test without being long runs.

**Every label a player passes.** The priority label is the first rule a player fits. Each rule is also checked
on its own, ignoring the order. `fits_<label>` columns record whether a player passes each rule, and
`all_labels` lists every label passed. Hoverer, mixer and drifter never occur together, as their rules need
a narrow or wide spread and a trend or no trend. A single switch tilts the straight line, so one-switch
players also tend to pass the drifter rule. The one-switch rule separates them by needing two sections to
fit better than the line.

**Two positions are not part of any label.** A gap-to-spread ratio above 3.46 is only what one even spread
gives on average, so it cannot show two positions on its own. Players are unlikely to choose their
centrality precisely enough to hold two exact positions.

## How the rules were checked

Share of each simulated kind given its own label.

| simulated kind | social | first and second solo |
|---|---|---|
| one switch | 1.00 | 0.99 to 1.00 |
| noisy one switch | 0.76 | 0.70 to 0.78 |
| long runs | 0.93 | 0.53 to 0.55 |
| drifter | 1.00 | 0.91 to 0.94 |
| hoverer | 0.95 | 0.95 to 0.96 |
| mixer | 0.96 | 0.93 to 0.97 |

- **Social is clean.** Most errors are small (no simulated mixer is called one switch, 0.01 of long runs
  are, and 0.00 to 0.02 of mixers and hoverers are called drifters).
- **Noisy one switch** loses most of the rest (0.16 to 0.20) to drifter, where a section falls just below
  the purity cut but the line still moves.
- **The solos are weaker for long runs.** With 54 to 60 trials, 0.17 to 0.24 of simulated long runs are
  called one switch, as fewer trials hold fewer runs.

## Results

| label | first solo | social | second solo | all |
|---|---|---|---|---|
| one switch | 20 | 12 | 13 | 45 |
| long runs | 6 | 15 | 4 | 25 |
| drifter | 10 | 11 | 10 | 31 |
| hoverer | 1 | 0 | 7 | 8 |
| mixer | 17 | 19 | 18 | 54 |
| short runs | 8 | 7 | 1 | 16 |
| unclear | 3 | 5 | 1 | 9 |
| intermediate players | 65 | 69 | 54 | 188 |

- **Social intermediate players mostly mix, wait in runs, switch once or drift.** Of 69, 19 are mixers,
  15 long runs, 12 one switch and 11 drifters. 7 are short runs and 5 unclear.
- **Social drifters drift weakly toward the wall.** Their median trend strength is 0.16, and their median
  line moves 0.21 toward the wall over the session.
- **One-switch players mostly move toward the wall.** The median second section sits 0.31 below the first
  in social (0.54 and 0.42 in the solos). The median social switch falls 39% of the way through the
  session.
- **Social mixers shift back and forth beyond noise.** For the median social mixer, 0.30 of their
  trial-to-trial differences are shifts, above the 0.21 reference, with no steady trend.
- **Most social intermediate players pass more than one label.** 40 of 69 pass two, 2 pass three and 7
  pass four. All 12 one-switch and 12 of 15 long-runs players also pass the drifter rule.
- **Short runs is mostly a description.** 10 of 11 social drifters, 14 of 19 mixers and 8 of 12 one-switch
  players also have short runs, against 7 players labelled short runs.
- **Few switchers can be shown to use two separate positions.** 4 of 12 social one-switch and 7 of 15
  long-runs players have a gap interval entirely above one even spread, and none above the two-position
  reference.
- **The weaker drifts do not follow the partner** (scratch check, not yet in the notebook). Of the 13
  social players with a trend below the earlier 0.30 cut who were short runs at the time, 10 drift toward
  the wall (median line change −0.22, against −0.11 for all intermediate players). Their drift heads toward
  their partner's mean position in 7 of 13, about half. It looks like a stronger version of the general
  drift toward the wall found in every session.

## How the rules got here

Each change was made after checking on simulated players or real examples.

1. **First rules.** Switchers from runs alone, and wide players with no order split into mixer and broad
   by the two-position ratio. This missed most single switches, and a ratio above 3.46 cannot show two
   positions.
2. **Mixers redefined** by shifts beyond noise (back and forth, no runs needed for two exact positions).
   The shift reference comes from central players' own shift shares, as a fixed 5% rate was too low.
3. **One switch from a large local jump, split by trend strength.** Dropped, as it missed single switches
   near the start or end of a session. The local jump now only describes.
4. **One switch as two stable sections.** The first stability check required no runs left after the fit.
   Real players' consecutive trials often sit close together within a section, so clear single switches
   with brief trips back went to long runs. Section purity replaced it.
5. **Short runs moved to lowest priority.** Runs no longer block drifter, hoverer or mixer, as labels are
   descriptive.
6. **Drifter cut lowered from 0.30 to 0.10.** Plots of individual players showed clear moves toward the
   wall at trend strengths of 0.11 to 0.25, which had been labelled short runs.
7. **Every label passed is recorded** beside the priority label, as players often show several patterns.

## Open decisions

- **Short runs as a label or only a description.** It appears beside most drifters, mixers and one-switch
  players, and only 7 social players are labelled short runs. It could be dropped as a label, keeping
  `has_short_runs`.
- **The solos.** Long runs and one switch are hard to separate with 54 to 60 trials.
- **Label stability.** Labels have no measure of how firm they are. Redrawing each session in blocks would
  show which players keep their label over a different set of their own trials.
