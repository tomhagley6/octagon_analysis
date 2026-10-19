# Analytical modelling: assumptions and decisions

Running notes for the geometric race model (`analysis/analytical_race.py`, tests in
`tests/test_analytical_race.py`, notebook `demos/analytical_modelling/nearer_wall_choice.ipynb`,
built by `build_nearer_wall_choice.py` beside it). To be checked, then shaped into a README.

Each entry is tagged by who made it:
- **[user]**: agreed with Tom.
- **[data]**: measured from logs or read from the Unity source.
- **[claude]**: my call, still to be checked.

## Question (stage 1)

Against a central opponent, would a peripheral agent score more per second if it chose the nearer
wall more often? Perception, inference and alcove occlusion are set aside for now **[user]**.

## Model structure

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 1 | Time to a wall = start-up delay + shortest path / speed (+ timing residual) | [user] | Agreed method |
| 2 | Both agents share speed, delay and residual spread | [user] | "put these assumptions to one side". The extra-delay sensitivity (#14) relaxes it |
| 3 | List all 40 High/Low layouts with exact probabilities, rather than sampling trials | [claude] | Gives exact expectations, and each result traces to a layout |
| 4 | High uniform over 8 walls. Low at 1, 2 or 4 walls (0.5/0.25/0.25), random side | [data] | `OctagonArenaSettings.cs` L200-250, `GeneralGlobals.cs` L33-36 |
| 5 | Peripheral alcove uniform relative to High (walls independent of positions) | [user] + [data] | Anchor wall is `Random.Range(0, 8)` |
| 6 | Central opponent always goes High | [claude] | 260929 central p_high 0.998. Section 6 uses each entrant's own p_high |
| 7 | An agent overlapping a lit trigger at onset wins that wall at once, whatever its intended wall | [data] | Logged 0.992 instant-win rate (260929 notebook, Stage 6) |
| 8 | Neither agent blocks the other. Neither changes behaviour for a different opponent | [claude] | Same as `race_estimate.py`. Policy replay shows the opponent channel is mostly ignored |
| 9 | Ties split evenly | [claude] | Only matters with no noise |

## Geometry

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 10 | Reachable space for the agent centre = octagon (face 16.747) + alcoves (half-width 1.90, back 19.747), from `globals.py` | [data] | Settled 2026-09-14 (measured reachable space) |
| 11 | Corners of the reachable space are sharp, not rounded by the 0.61 body | [claude] | Rounding shortens paths around mouth corners by a few hundredths of a unit |
| 12 | Target = trigger line at p = 18.247 (box front 18.857 − body 0.61), any point across the alcove width | [data] | Lowest logged trigger depth 18.20-18.25 (within one 50 Hz frame of movement) |
| 13 | Shortest path bends only at the 16 alcove mouth corners (visibility graph) | [claude] | Exact for this polygon (the other corners are convex). Tested by hand for adjacent and opposite walls |
| 13a | Logged start points up to ~0.08 outside the polygon are moved to the nearest point 0.02 inside | [claude] | Found 5 of 3000 entrant positions past alcove sides. Changes paths by ~0.01 s |
| 13b | Peripheral agent waits at the centre of the alcove back (idealised). Section 6 uses entrants' real onset positions | [user] | Model 32 median onset radius 19.76 |
| 13c | Central agent placed at the exact centre (idealised), then at 400 sampled 260929 central onset positions (empirical) | [user] | Both reported |

## Timing (from 260929 logs)

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 14 | Speed fixed at 10.0 units/s | [data] | Median and 99th percentile of moving speed both 10.0. Diagonal moves are not faster |
| 15 | Delay = median(rt − path / speed) on central won trials (0.256 s). Residuals = remainder (sd 0.425 s) | [claude] | A loser's time is not logged. Won trials are faster, so the delay is biased low (shared, mostly cancels) |
| 16 | The free regression slope (0.170 s/unit) is not used, only shown | [claude] | Central paths span only 14.6-20.2 units. The fixed speed is cautious for this question, as a steeper slope would favour adjacent-wall wins |
| 17 | Residuals drawn independently for the two agents. Win probability and race time computed exactly over all residual pairs (1500 sampled residuals) | [claude] | Checked against 200k-race simulation (difference 0.0005) |
| 18 | Overhead per trial = whole trial − onset-to-trigger time, mean 7.09 s (sd 0.91), from 5 central models | [data] | Trial start → onset, trigger → end, and the ITI. These agent delays were set to match the human task [user], so the overhead holds for humans too |
| 19 | Same overhead for every trial, whatever the outcome | [claude] | Overhead sd 0.91 s. Not checked whether it depends on who won |

## Scoring and policies

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 20 | Primary measure = score per second over whole trials (overhead included). Secondary = score per trial | [user] | Players are told to maximise score over time |
| 21 | Also report score per second of race time only | [claude] | It reverses one comparison: race-only, a nearer-wall peripheral agent (0.274) beats central-vs-central (0.244) |
| 22 | Policies: always High, nearer wall, q sweep (P(Low \| Low nearer)), best per trial, best for score per second (Dinkelbach) | [user] + [claude] | Agreed set. Dinkelbach finds the true rate-maximising per-layout choice |
| 23 | "Nearer" = shorter path. Equal paths count as High | [claude] | Mirror-image ties would otherwise flip on rounding |

## Validation and handicap

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 24 | Known-answer checks: tie at a shared point, hand counts 7/16, 0.3625 and 0.425, and Monte Carlo agreement | [claude] | All in the notebook and unit tests |
| 25 | Validate against the 13 peripheral-vs-central tournament pairs (260715 + 260914_vs_260723) | [claude] | Only tournaments with such pairs. No 260929 tournament exists yet |
| 26 | Extra peripheral delay as a single lump handicap. Pooled fit 0.60 s on win share (grid 0-2 s, step 0.05) | [claude] | Stands in for perception, unseen walls, slow starts and path inefficiency. Per-pair fits 0.40-1.30 s |
| 27 | The extra delay is not applied to instant wins | [claude] | An agent on a lit trigger wins at onset |
| 28 | 260929 timing applied to 260715/260723/260914 entrants | [claude] | Same prefab and speed. Delay and spread may differ by batch |

## Open points to discuss

- Should the handicap be split into parts with their own mechanism (e.g. a first-sight delay for walls not seen from the alcove) rather than one lump? This is the natural stage 2.
- Overhead: is the ITI for agents fixed by design or variable? If it depends on outcome, #19 needs revisiting.
- Human version: rerun with human speed, start-up delay and timing spread (overhead is shared with agents by design), to ask why human peripheral players keep pace.
- A 260929 peripheral-vs-central tournament would check the fitted handicap on the current batch.
