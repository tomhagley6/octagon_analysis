# Geometric race model: assumptions and decisions

Covers the current model in `analysis/analytical_race.py` (tests in `tests/test_analytical_race.py`).
`demos/analytical_modelling/geometric_model_assembly.ipynb` (built by `build_geometric_model_assembly.py`)
walks through the model step by step for understanding; `nearer_wall_choice.ipynb` is the working record.
The notebook `demos/analytical_modelling/nearer_wall_choice.ipynb` (built by
`build_nearer_wall_choice.py` beside it) shows the first model (both walls known at onset) in
sections 3-6. Section 7 shows the logged central turning behaviour that sets the ray-sensing
inputs. Sections 8-9 show the ray-sensing model: the peripheral agent's design, and the three
peripheral rules compared with tables and figures.

Each entry is tagged by who made it:
- **[user]**: decided or agreed by Tom.
- **[data]**: measured from logs or read from the Unity source.
- **[claude]**: my call, still to be checked.

## Intention

The model is about arena geometry, not learned behaviour **[user]**. It takes the peripheral agents'
slice-onset positions and a stereotyped central agent. It asks what score rate both agents get
when the peripheral agent follows each of three rules:
1. always move to the High wall;
2. always move to the closest lit wall;
3. move to High, unless High takes more than Δ seconds longer than Low, then move to the closest wall.

The aim is to find how well a peripheral strategy **can** be played within the limits of the game, not to
copy how the trained agents play **[user]**. Where a more efficient behaviour is allowed by the game,
the model uses it, and differences from logged agents are recorded rather than copied.

**Key assumption: the central agent always goes High [user].** This copies the stereotyped play of
trained central agents (High on 0.998 of 260929 central won trials). It is not necessarily the best
central play against a peripheral opponent, for example when a peripheral agent is near High and
Low is close to the centre's path. All central scores in this model depend on it.

**Information rule [user]:** a model agent's behaviour may depend only on what it could know: what its
rays have sensed, its own position and heading, and the fixed arena layout. It must never depend on
where an unseen wall is.

Gameplay components are added one at a time. Wall visibility is now in for the central agent only.
Peripheral wall visibility and opponent visibility come next.

## Control scheme

Applies to humans, agents and this model **[user]**.

- Movement combines four directions relative to the facing direction: Up (always the facing
  direction), Down, Left and Right. Pressing two together (e.g. Up and Right) gives a diagonal. So
  movement can only point in 8 directions relative to the heading, 45 deg apart.
- Turning is separate from movement. Agents choose a move action (none / forward / back), a strafe
  action (none / right / left) and a rotate action (none / clockwise / anticlockwise) every step
  (`OctagonAgent.cs` L486-514). Turning is capped at 360 deg/s (`turnSpeed`).
- Speed is 10 units/s in every direction, diagonals included **[data]** (median and 99th
  percentile of logged moving speed are both 10.0, and diagonals make up most moving frames, see
  observation 7).
- In the model, travel is a straight line at 10 units/s. Once High is centred it lies along Up, so
  the straight line is allowed by the control scheme.
- While turning with High in view (#22a), the model's agent zig-zags **[user]**. It mixes the two
  allowed directions either side of High, in the share that cancels sideways drift, so on average
  it moves along the straight line to High and stays on it. Its speed along the line is the
  empirical closing speed (7.57 units/s), so that it stays relevant to the agents **[user]**.
- The best speed along the line that a zig-zag allows is 10 · sin 45° / (sin θ + sin(45° − θ)), where
  θ is High's angle past the nearest allowed direction (`zigzag_closing_speed`). That is 10 units/s
  when High lies on an allowed direction and 9.24 units/s midway between two. Averaged over a 77.5 deg
  in-view turn it is 9.46 units/s.
- The peripheral agent's travel is not yet constrained by the control scheme. That comes with its
  field of view and facing **[user]**.

## Audit of the model (2026-10-22)

Every step of `analysis/analytical_race.py` was re-read for logic errors, especially decisions that use
information the agent does not have, and inference or rules applied to one wall but not the other.
Numeric checks were run on the cached results.

### Problems found and fixed

1. **First-sense times used a position the agent was not at (information leak).** In the build, the time
   at which each wall is first sensed (`high_tsense`, `low_tsense`) was set to 0 ("sensed at onset") when
   the wall was in view at the onset heading from the *stepped-out* position left over from the earlier
   step-then-search version, rather than from the agent's start position. It affected 450 of 190,080
   races (0.24%), through the inferred search direction of "High, always", "nearest" and the High-only Δ
   rule, the "Low, if seen first" rule, and reported search times. Effect at most 0.0005 per second (e.g.
   "Low, if seen first" margin -0.0559 -> -0.0564; mean search for a hidden High 0.252 -> 0.254 s). The
   two-wall Δ rule (`delta2`, the headline result) does its own sensing and was not affected. Fixed in
   `_position_arrays` (uses onset sensing `S0`), test `TestAuditFixes`, cache patched (stamp `v8`).
   After the fix, `delta2` at large Δ matches "High, always" with inferred direction in every race (489
   races differed before).
2. **Exact ties at Δ = 0 were decided by rounding.** When High and Low are equally far (e.g. both
   adjacent to an alcove back), the gap is about 1e-13 s and its sign set the choice. A mirror-image check
   (layout reflected, search direction reversed) failed in 8 of 160 pairs, all at Δ = 0. Fixed with a
   tolerance (`GAP_TIE` = 1e-6 s; ties go High); the mirror check now passes (test). Δ > 0 was unaffected.

### Checks that passed

- Probabilities sum to 1 per model and per position-layout.
- No infinite, negative or physically impossible times (no peripheral time below its shortest path at
  10 units/s; no central time below 1.8247 s, straight travel from the centre).
- Central times are symmetric in Low's side (Low at ±1 and ±2 walls give equal means).
- `delta_outcome` and `delta2_outcome` (stored decision tables) match direct simulations (tests).
- Choosing Δ in-sample costs nothing measurable: chosen on five models and scored on the sixth, the
  held-out loss is at most 0.0003 per second (model 20) and zero for the other five.

### Remaining soundness concerns (not bugs; to keep in mind when reading results)

3. **The two agents are not optimised equally.** The peripheral agent's Δ is tuned, and it re-decides at
   every new wall. The central agent always goes High (key assumption), chooses its search direction once
   at onset, and stands still while High is unseen. The near-equal score rates compare a tuned peripheral
   player with a stereotyped central one, not two equally good players.
4. **The choice uses expected distance, leaving out search time** (user's deliberate choice). Searching
   for an unseen wall takes about 0.15-0.27 s, so adding it would be like shifting Δ by that much for
   some decisions, over a range of Δ (0.25-1.15 s with D11-D12 on; 0.30-1.15 s before) where the score barely changes. Not a concern.
5. **Decisions are revisited only when a new wall is sensed**, not continuously. While searching for
   an unseen wall the agent keeps its search direction even if new unlit walls make the other way more
   likely (no reversing). **Decision (user, 2026-10-23): keep no reversing.** Humans may reverse when the
   likely side turns out empty, but the effect is likely small and it adds complexity. The agent does
   remember every wall sensed in the trial; the user confirms this is correct.
6. **The search moves straight out of the alcove.** Moving out is there so that walls not yet seen can
   come into view; it goes along the alcove's axis. Moving out at an angle toward the side where the
   wanted wall probably is might reveal it slightly sooner. A possible refinement, not an error.
7. **Start positions.** The model is one rule set. The peripheral agent's start positions are logged
   self-play positions from six 260929 peripheral models (results "by model" mean by these position
   sets). Positions from model 20 sit nearer the alcove mouth (median depth 18.56 vs about 19.67, adjacent
   wall seen at onset 0.52 vs 0.03-0.34) and do best; why positions from models 3 and 15 do worse than the
   similar positions from model 23 is not established. The central agent is at the exact centre.
8. **Reporting:** `summarise`'s `p_go_low` counts instant Low wins only for `delta2`, so "share heading
   Low" is not comparable across rules. The new notebook reports "ends at Low" consistently.
9. **Documentation brought up to date (2026-10-23):** the module docstring, and sections 11-13 and the
   interpretation of `nearer_wall_choice.ipynb` (now show the Δ rule with inference about both walls, and
   the numbers after fix 1). Entries D1-D5 below describe the superseded High-only Δ rule and are kept as
   a record.

## Arena and layouts

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 1 | Reachable space for the agent centre = octagon (face 16.747) + alcoves (half-width 1.90, back 19.747), from `globals.py` | [data] | Settled 2026-09-14 (measured reachable space) |
| 2 | Corners of the reachable space are sharp, not rounded by the 0.61 body | [claude] | Rounding shortens paths around mouth corners by a few hundredths of a unit |
| 3 | Target = trigger line at p = 18.247 (box front 18.857 − body 0.61), any point across the alcove width | [data] | Lowest logged trigger depth 18.20-18.25 (within one 50 Hz frame of movement) |
| 4 | Shortest path bends only at the 16 alcove mouth corners (visibility graph) | [claude] | Exact for this polygon (the other corners are convex). Hand-checked in tests |
| 5 | Logged start points up to ~0.08 outside the polygon are moved to the nearest point 0.02 inside | [claude] | 5 of 3,000 sampled positions sit past alcove sides. Changes paths by ~0.01 s. The instant-win check uses the unmoved point |
| 6 | High uniform over 8 walls. Low at 1, 2 or 4 walls (0.5/0.25/0.25), random side | [data] | `OctagonArenaSettings.cs` L200-250, `GeneralGlobals.cs` L33-36 |
| 7 | Every peripheral position meets all 40 layouts, weighted by exact layout probability | [claude] | Walls are drawn independently of positions, so this gives the exact expectation |

## Agents

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 8 | Peripheral positions = logged slice-onset positions of 260929 peripheral models 3, 15, 20, 23, 32, 34 (2,376 agent-trials, both agents) | [user] | Peripheral = mean onset centrality < 0.5. Pooled results weigh positions equally. Per-model results weigh each model's positions equally |
| 9 | Central agent waits at the exact arena centre and always goes High | [user] + [data] | 260929 central models: High on 0.998 of 5,344 won trials, 0.975 when Low was > 6 units nearer (40 trials). Won trials only |
| 10 | Both agents move at 10 units/s along the shortest path | [data] | `moveSpeed` = 10. Median moving speed 10.0 in all 990 measured trials |
| 11 | No start-up delay and no timing noise | [user] | Not justified for a geometric model |
| 12 | Neither agent blocks the other | [claude] | |
| 13 | An agent overlapping a lit trigger at onset wins that wall at once, whatever its rule | [data] | Game rule. Logged 0.992 instant-win rate (260929 notebook, Stage 6) |
| 14 | Equal arrival times split the win | [claude] | |

## Central ray sensing

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 15 | The agent senses walls with its 7 rays, as in `analysis/agent_raycast.py` (`cast_rays`, the replica of Unity's `SelectivePassThroughRaycast`): rays at 0, ±18.3, ±36.7 and ±55 deg from the heading, first hit only. Any part of a lit wall (faces, alcove sides, alcove back) or its trigger box tells the agent that wall's identity. A ray starting inside a trigger box does not detect that box. Used through `high_sensed` | [user] + [data] | Replaces the earlier 110 deg cone. The rays span ±55 deg (110 deg) but leave 18.3 deg gaps; walls are wide enough that from the centre no wall falls in a gap |
| 16 | From the centre, High is sensed when the angle between heading and High's centre is below 77.5 deg (the ±55 deg ray reaching High's vertex). This is computed from the rays (`central_visible_half_angle`), not assumed | [data] | Logged share of central agent-trials with High sensed at onset 0.406, against 0.431 at the exact centre |
| 17 | Rays pass through the opponent in the model, so wall sensing is never blocked | [user] | True of builds from 261014. In the 260929 logs (earlier builds) player 1's rays are blocked by player 0's body; the log measurements keep that block, as it is what those agents sensed. It changes High sensing at onset in 11 of 990 player-1 agent-trials (1.1%) and in 0.5% of frames |
| 18 | Central onset heading is uniform at random | [user] + [data] | Logged angle-to-High deciles step by ~18 deg (19, 37, 57 ... 163) |
| 19 | The agent registers High as soon as a ray hits it. While High is out of view it only turns. Once High is in view (at onset, or for the last 77.5 deg of the turn, when the outer ray first reaches High), it keeps turning to centre High and closes on High at the same time (#22a). Once High is centred it travels the rest of the way straight at 10 units/s | [user] | It cannot move toward a wall it has not seen. Logged agents also move while High is out of view (see observations) |
| 20 | If High is in view at onset, the agent turns the short way to centre it (up to 77.5 deg) | [user] | Logged agents turned the short way on 0.92 of agent-trials with High sensed at onset (679, leaving out those already facing High) |
| 21 | If High is out of view, the agent does not know which way to turn. It turns either way with equal chance and keeps turning until High is centred (φ or 360 − φ degrees) | [user] | See the observation below |
| 22 | Rotation speed = 331 deg/s, the central agents' own turning rate. Kept as the model's turning rate because it reflects how fast agents turn while turning [user]. For each agent-trial, take the total heading change from slice onset until it faces High (within ±10 deg), over that time. Use the median over the 1,799 agent-trials that faced High after onset and before the trigger (central models 1, 2, 4, 5, 6) | [user] (turning rate, independent of direction) + [claude] (±10 deg tolerance, median) | Consistent: in view 347, out of view 328, by model 309-349, by angle to High 325-360 deg/s (IQR overall 279-360). Code maximum is 360 deg/s. A pooled total over total time gives 272 deg/s, pulled down by a few trials with long pauses (model 5: 207 pooled, 309 median) |
| 22a | Closing speed while turning with High in view = 7.57 units/s, moving along the straight line to High by zig-zagging (Control scheme). For each agent-trial, take the reduction in straight-line distance to High's trigger line over frames with High in view before facing it, over the time in those frames. Use the median over the 1,799 agent-trials with such frames | [user] (rule: same period and behaviour as the turning rate) + [claude] (median, straight-line distance) | Consistent: IQR 6.16-8.70, by model 6.36-8.39, in view at onset 8.53, out of view at onset 7.17, pooled 7.62 units/s. Full speed is 10 |
| 22b | **Assumption:** the empirical closing speed (7.57 units/s) is below the best the control scheme allows (9.24-10 units/s, mean 9.46 over a 77.5 deg in-view turn; 7.57 is 0.80 of that). The model keeps the empirical value, so both agents close on a sensed wall as slowly as logged agents do | [user] (2026-10-21: keep agent-derived turning and movement unless clearly too inefficient; humans are likely less efficient still, and human play is the endpoint the model will be applied to) | Logged agents lose speed by moving at an angle to High (median 23 deg off the line, observation 7): at full speed their angle alone predicts 7.74 units/s. The measured 7.57 is a per-agent-trial median of distance closed over time, which also counts frames below full speed |
| 22c | **Check (2026-10-23): which wall the measured agent-trials were going for.** `collect_central_alignment` measures turning to face High on every agent-trial, without checking the chosen wall. Matched to the wall each agent reached (winners) or was heading for (losers, `agent_loser_choice.infer_choice`): 990 winners all reached High; of 990 losers, 649 were heading for High, 341 were unclear, 0 for Low. So no Low-going trials are in the measure. Using only agent-trials heading for High gives 333.7 deg/s (vs 331.0) and 7.71 units/s (vs 7.57); winners heading for High alone give 331.0 and 8.13 | [claude] (check) | Unclear losers close more slowly (median 6.78 units/s). Whether to restrict the measure to agent-trials known to head for High is open for the user. Script: scratchpad `align_choice.py`; cache `agent_trajectories_260929_central5` |
| 23 | Central time is a distribution over 3,600 headings × 2 turn directions, each equally likely. Each race is scored exactly against all of it | [claude] | Checked against a direct loop over headings in the tests |
| 24 | Peripheral agent has no field of view yet (knows both lit walls at onset) | [user] | Its own visibility method (facing at onset, alcove walls blocking view) is the next step |

### What logged central agents do instead (not in the model)

From the same 1,799 agent-trials (notebook section 7; observation 7 from a separate check, not yet in the notebook). The model keeps the cleaner turning above;
these are the differences, kept on record.

**1. They turn toward High more often than chance.** With High out of view at onset, they turned
the short way on 0.62 of 1,120 agent-trials, against 0.5 for a random direction. The bias is
strongest when High is just outside the outer rays, and close to chance when High is behind:

| Angle to High at onset | 77.5-100° | 100-120° | 120-140° | 140-160° | 160-180° |
|---|---|---|---|---|---|
| Short-way share | 0.77 | 0.67 | 0.59 | 0.51 | 0.54 |
| Agent-trials | 231 | 225 | 221 | 210 | 217 |

Likely reason **[user]**: the agent can use the Low wall's position to infer where High is. Low is
always 1, 2 or 4 walls from High, so a Low wall in view at onset narrows down which way High lies.
This is not part of the model yet (#21 keeps the direction random) and will be added later. Not
tested yet. Other possible contributors: a learned turning habit.

**2. They turn back and forth.** Reversals of turning direction occur in 0.31 (sensed at onset) and 0.65
(not sensed) of agent-trials. Total turning is 71 deg for a net 37 deg (sensed at onset) and 229 deg for a
net 157 deg (out of view). The model turns 39 deg and 180 deg with no reversals. Their net turn is
smaller than the model's random-direction turn (157 vs 180 deg), but the back-and-forth more than
cancels this.

**3. They pause.** Frames with no rotation add 0.17 s (in view) and 0.15 s (out of view) before
facing High. While actually rotating they turn at 365 deg/s, the code maximum (360 deg/s). They do
not wait before starting to turn (mean 0.002 s).

**4. So they take longer to face High.** Mean time to face High is 0.36 s (in view) and 0.78 s (out of
view), against 0.12 s and 0.54 s in the model.

**5. They move while turning, including while High is out of view.** Before facing High they move
3.5 units and get 3.0 closer to High (sensed at onset), or 7.1 units and 3.4 closer (out of view).
The model moves only while High is in view (#19).

**6. Net effect on race time.** Mean central-vs-central race (slice onset to trigger, faster agent):
logs 2.10 s (990 trials); model with turn-then-move 2.04 s; model with moving while turning 1.93 s;
straight travel only 1.82 s. The model's central agent is 0.17 s faster than logged agents, because
it turns without back-and-forth or pauses.

**7. While closing on High, they move along the allowed direction nearest to High, not straight at
it and not along the view centre.** Checked on 31,261 moving frames with High in view but not yet
centred. Movement relative to heading: straight ahead 0.26, forward-diagonal +45 deg 0.39 and -45 deg
0.16, sideways 0.13, backwards or back-diagonal 0.06. The median movement is 45 deg off the heading on
High's side at every angle of High within view, and closer to the direction of High than to the
heading in 0.68 of frames. The median angle between movement and the direct line to High is 23 deg.
The mean component of movement toward High is 0.774, which at 10 units/s gives 7.74 units/s, close
to the measured closing speed of 7.57. So closing is slower than full speed because movement is at
an angle to High, not because agents move more slowly. Forward-right diagonals are more common than
forward-left (0.39 vs 0.16).

## Peripheral ray sensing (in progress)

Implemented: the shared search and travel, the rule "High, always" (`heading_sensing`,
`first_leg_bearing`, `step_to_sense`, `target_time`, `build_races_rays`) and the rule "Low, if Low
seen first, else High" (`sweep_angle`, `low_sensed_first`, policy `'low_first'`) and the rule
"Nearest, once both are sensed" (`high_trace`, `trace_first_sensed`, `time_from_sensed`, policy
`'nearest'`).

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| P1 | Start from each peripheral agent-trial's logged slice-onset position and logged heading | [user] | Logged peripheral agents face out of the alcove: median 14 deg from facing the arena centre, 90% within 26 deg |
| P2 | Sensing uses the same 7 rays (`agent_raycast.cast_rays`), recast at the agent's current position and heading. Alcove walls block the rays. Rays pass through the opponent | [user] | From the alcove back no heading reaches the adjacent walls |
| P3 | Same turning rate (331 deg/s), closing speed while turning with the target sensed (7.57 units/s, zig-zag) and control scheme as the central agent | [user] | |
| P4 | Search: if the target wall is not sensed, turn in place in a random direction (one direction per trial, either with equal chance) until a ray hits it | [user] | Logged peripheral agents mostly search by turning in place (see observations below) |
| P5 | If the target wall is not sensed at onset, an agent starting in an alcove moves toward the arena centre at 10 units/s **while** turning in its search direction at 331 deg/s, until a ray hits the target (`step_search_trace`, `move_out=True`). It then turns the short way to centre the first leg of its path. An agent not starting in an alcove turns in place | [user] (2026-10-20: turn and move at once; search the most efficient way; use only information the agent has) + [claude] (toward the centre; turn in place outside an alcove, where no direction helps before a wall is seen) | Replaces two earlier versions that decided whether to step out from whether some heading could sense the wall, i.e. from where the unseen wall is (`move_out=False`; with `merge_step` False or True). Moving out is the wrong way only for a target in the agent's own alcove that it does not sense at onset (0.017 of trials, still won) |
| P6 | Once the target is sensed, keep turning to centre the first straight leg of the shortest path (e.g. toward the alcove mouth corner), not the direct bearing to the wall. If the search turned past that leg before sensing, turn back the short way | [user] (first leg) + [claude] (turning back) | The direct line from an alcove often crosses walls |
| P7 | After centring, follow the shortest path at 10 units/s with no extra time at bends | [user] | The control scheme allows moving in 8 directions without turning |
| P8 | Headings are searched on a 0.5 deg grid | [claude] | 0.5 deg at 331 deg/s is 1.5 ms |
| P9 | An agent overlapping a lit trigger at onset still wins it at once | [data] | As #13 |
| P10 | "Low, if Low seen first": both lit walls sensed at onset is a tie and goes High. Otherwise the first wall a ray hits during the search decides. A wall hit at the same heading as the other is a tie and goes High | [user] | |
| P11 | Under "Low, if Low seen first", if neither lit wall is sensed at onset the agent moves out while turning (as P5), and the lit wall its rays hit first is chosen (ties go High). A wall sensed at onset counts as first | [user] | Consequence: from the alcove back an adjacent High is hidden, while a Low further away often is not, so the agent often takes a far Low |
| P12 | When High is the first wall sensed, the time to High is identical to "High, always" | [claude] | Checked in tests |
| P12b | Earlier search versions are kept for comparison as cache copies `periph_ray_races_260929_step_then_search` and `periph_ray_races_260929_step_with_search` (notebook section 10, third table) | [claude] | |
| P12a | `build_races_rays` splits positions across worker processes (`n_jobs`, default `N_JOBS` = 12 of 16 cores, leaving 4 for training). Results are identical to a serial build (`n_jobs=1`) | [user] (default on) + [claude] (12 workers, spawned processes) | Checked in tests. Speeds up the ~35 min single-core build |
| P13 | "Nearest": the agent's route under "High, always" is traced every 0.005 s (step out, search turn, turn to the first leg while closing, then the path). The rays are cast at every step to find when Low is first sensed | [claude] (0.005 s step: at most 1.7 deg of turn or 0.05 units of travel per step) | The route does not depend on Low, so one trace per position, High wall and search direction serves every Low |
| P14 | Turning and moving are uncoupled (Control scheme). At each bend of the path the agent keeps moving at 10 units/s and turns the short way at 331 deg/s to face the new stretch. The rays follow that heading | [user] | Bends still cost no time; only what the agent senses changes |
| P15 | Both walls are known at the later of High's and Low's first sensing. At that moment the agent compares the remaining shortest paths from where it is and goes to the nearer (equal paths go High). It decides once | [user] (nearest by path length) | Turning is fast, so path length rather than time with turning |
| P16 | If it switches to Low, it knows where Low is, so it turns the short way to centre the first leg toward Low (closing at 7.57 units/s), then follows that path at 10 units/s | [claude] | Same as the time to a wall already in view |
| P17 | If Low is not sensed before the agent reaches High, the time is the "High, always" time | [user] | |

**Rules for the peripheral agent** (replacing always High / closest / threshold, which need both walls known) **[user]**:
1. **High, always.** Search for High, then go to High.
2. **Low, if Low seen first, else High.** Go to Low if Low is sensed strictly before High. If High is sensed first, or both at the same moment (e.g. both at onset), go to High.
3. **Nearest, once both are sensed.** Behave as "High, always" until both walls have been sensed. Then go to whichever wall has the shorter remaining path from where the agent is at that moment. If Low is never sensed, it simply reaches High.
4. **Inference on nearest**
5. **Inference on nearest, flexible to High if below the average trial duration (against central opponent, against peripheral opponent)**
6. **The above, with a corner agent instead of alcove agent**

### What logged peripheral agents do at onset (260929)

Models 3, 15, 20, 23, 32, 34; 2,376 agent-trials. 539 (0.227) start overlapping a lit trigger and win at
once. Of the other 1,586 (trials lasting at least 0.1 s), by what the rays sense at onset:

| | Both seen | Only High | Only Low | Neither |
|---|---|---|---|---|
| Agent-trials (share) | 572 (0.36) | 398 (0.25) | 361 (0.23) | 255 (0.16) |
| Destination High / Low / no call | 0.82 / 0.01 / 0.17 | 0.76 / 0.01 / 0.23 | 0.70 / 0.02 / 0.28 | 0.49 / 0.02 / 0.49 |
| Turning in first 0.5 s (median) | 86 deg | 86 deg | 108 deg | 86 deg |
| Moved in first 0.5 s (median) | 4.7 units | 4.5 units | 4.6 units | 2.7 units |

Destination is the wall reached (winners) or the `agent_loser_choice` call (losers).
- Only Low seen: they search for High rather than take Low. They sense High after a median 0.16 s,
  22 deg of turning and 1.3 units of movement. They find it by moving within the alcove (0.50),
  turning in place (0.39) or leaving the alcove first (0.11).
- Neither seen: they turn in place first (82% move less than 1 unit before sensing a lit wall; 1.2%
  leave the alcove first). They find High after a median 0.20 s and 22 deg, by turning in place (0.56),
  moving within the alcove (0.38) or leaving the alcove first (0.07).
- Caveats: self-play, so the opponent is also peripheral. Many losers have no destination call.

## Inference of High's position (implemented 2026-10-21, notebook section 11)

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| I1 | At onset the agent knows every wall its rays hit and whether it is High, Low or unlit (an unlit wall returns a distance with no tag) | [data] | `agent_raycast` notes |
| I2 | It works out where High could be: the layout prior conditioned on what was sensed (`high_posterior_given`, matching `agent_raycast.high_posterior`). Walls hit and unlit cannot be High or Low; a sensed Low fixes Low | [user] | |
| I3 | If High is not sensed at onset, it turns in the direction with the lower expected time until a ray first hits High, weighting each candidate wall by its probability. A tie is 50/50. The times come from the known arena map for each candidate, so no hidden information is used | [user] (decided once, at onset; all rules search toward the likely High) | Peripheral: reweights the two search directions of each race row (`infer_direction`). Central: one time distribution per Low offset (`central_times_inferred`), which `evaluate` and `central_reference` accept |
| I4 | The opponent's position is not used | [user] | To add later, with opponent sensing |

## Peripheral Δ rule with inference (implemented 2026-10-21, notebook section 12)

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| D1 | The agent follows "High, always" and decides at onset and whenever a ray first hits a new wall. Once Low is known, it goes to Low if (path to High - path to Low) / 10 units/s > Δ, otherwise it keeps on to High | [user] (version b) | Δ = 0 is close to "nearest"; large Δ is "High, always" |
| D2 | While High is still unseen, it uses its expected path to High: the paths to the candidate walls weighted by where High could be, given every wall sensed so far (`high_posterior_given`) | [user] | Lets it give up on a probably distant High without finding it first |
| D3 | The decision is rechecked at every new wall. After switching to Low, if High comes into view, it rechecks with the actual paths and switches back to High if the gap is at most Δ | [user] | |
| D4 | While heading to Low with High unseen, new unlit walls do not make it turn back to search | [claude] | Searching mid-arena would mean turning in place there |
| D5 | Each race stores up to 9 decision points (`ev_*` columns: time, gap, arrival at Low, switch-back time, gap and arrival at High), so any Δ is evaluated instantly (`delta_outcome`); checked against a direct simulation (`delta_direct`) | [claude] | Cache v6, 409 s on 12 workers |

## Peripheral Δ rule with inference about both walls ('delta2', implemented 2026-10-22)

Replaces the Δ rule above (D1-D5), which inferred only where High was and always steered by High. That
narrowing was mine, not the user's, and it lost the races where High is far and in view and an adjacent
Low is hidden by the alcove.

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| D6 | At onset and whenever a ray first hits a new wall, the agent works out where both walls could be (`joint_posterior`: the layout prior, conditioned on every wall sensed; unlit walls rule out both; a sensed wall is fixed) | [user] | Inference applies to Low as well as High |
| D7 | It goes for Low if (distance to High - distance to Low) / 10 units/s > Δ, otherwise High. A sensed wall's distance is its actual path; an unsensed wall's is its expected path, averaged over where it could be (`_gap_both`) | [user] | Δ swept to maximise score per second, so the agent picks walls it is more likely to win |
| D8 | If the chosen wall is sensed it goes to it (turn onto the path while closing, then the path). If not, it searches for it: moves out of an alcove while turning (turns in place elsewhere), in the direction that should sense that wall sooner given where it could be; a tie uses the race's random direction (`_motion_to`) | [user] (search for the chosen wall) + [claude] (direction by expected first-sense time) | |
| D9 | The choice is rechecked at every new wall and can change either way. A motion is kept while its target and the agent's knowledge of it are unchanged | [user] | |
| D10 | Because the route depends on Δ, each race stores its outcome as pieces over Δ (up to 12 ranges, each with a time and a wall; `d2_*` columns, `delta2_outcome`). Checked against a direct simulation at fixed Δ (`delta2_direct`) | [claude] | |
| D11 | **ON from 2026-10-27 (`BACKPEDAL = True`, cache stamp v10; paused 2026-10-26). Backpedal into the own trigger.** If the agent is in its own alcove but not on its trigger, and the probability (from `joint_posterior`, given every wall sensed) that its current target (High or Low, as chosen by the Δ rule) is its own wall is at least **0.8**, it moves straight back into the trigger at 10 units/s without turning (`_motion_kind`, `blind_trace`). It need not see the wall. If the own wall holds the other lit wall, entering the trigger claims that wall. If the own wall is unlit (the 20% gamble lost), the agent learns this on reaching the trigger and the rule decides again from there. Nothing else moves without turning | [user] (backpedal when the wall behind can easily be inferred; **0.8 is an arbitrary value picked for now**, to be revisited) + [claude] (probability that the *target* is the own wall; strictly inside the own alcove, as the user specified; the motion is re-chosen whenever its kind changes) | `BACKPEDAL_P` in analytical_race.py. Logged agents nearly all start on their trigger, so it rarely applies to them; it matters for designed start positions further out in the alcove. Cache stamp v10 |
| D12 | **ON with D11 from 2026-10-27 (`OWN_TRIGGER_KNOWN = True`).** The agent knows walls only from experience (ray hits) or inference. Standing on its own trigger at onset without the trial ending is also experience: it shows the own wall is unlit. So that wall is treated as sensed, whether or not a ray hits it (`onset_known`) | [claude] (information the agent has; found while adding D11, 2026-10-26) | Before, it was known only if a ray hit it (0.934 of logged starts). Without this, an agent moving out of its trigger could infer its own wall was probably lit and backpedal into the trigger it had just left |
| D11 (withdrawn) | A broader version, implemented on 2026-10-26 by mistake: any target located with certainty, from anywhere, was driven to at full speed without turning. It went beyond the user's request (a hidden adjacent wall placed by inference from the alcove back was driven to without a search) and moved the logged-position margin from -0.0004 to +0.0027. Withdrawn the same day | [claude] (error) | Cache kept as `periph_ray_races_260929_v9_broad` for the record. Most of its gain came from skipping the search, not from the full speed (turning like a seen wall changed the margin by 0.0001 on 300 sampled starts) |

## Scoring

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 25 | Score 1.0 for a High win, 0.4 for a Low win, 0 for a loss | [data] | Task rules |
| 26 | Time outside the race per trial = **human social mean, 6.545 s** (`ar.human_overhead`; 13,140 trials; start delay + end delay + ITI from `ar.trial_delays`). Switched 2026-10-23 from the agent mean (7.09 s). The delays are not treated as fixed; only the mean enters score per second (total score / total time), provided it does not depend on the race | [user] (2026-10-23: human play is the endpoint) + [data] | Game settings (`GeneralGlobals.cs`): start delay uniform 0.5-1.5 s, end delay fixed 2 s, ITI uniform 2-5 s, so mean 6.5 s, sd 0.91. Humans match: uniform start delay and ITI, social 6.545 s, solo 6.484 s. Agent logs give 7.09 s because each wait ends on the next frame, 1/3 s of game time apart (`Maximum Allowed Timestep` 0.333 s in TimeManager.asset; timed waits resume only on frames, so a sped-up simulation hits the cap every frame); start delay only 0.67/1.00/1.33/1.67 s, end delay 2.00-2.33 s. Physics runs at 0.02 s, so races are unaffected. Effect of the switch: every score per second rises about 7%; best Δ unchanged (1.05 s); delta2 margin -0.0003 -> -0.0004; final rule 0.0527 vs 0.0530, central vs central 0.0591. Not checked whether the time outside the race depends on who won |
| 27 | Primary measure = score per second = total expected score / total expected (race + overhead). Secondary = score per trial | [user] | Players are told to maximise score over time |
| 28 | Relative score = peripheral minus central score per second. A central player against the same opponent ties (margin 0) | [user] | Human players may try to beat the other player as well as maximise their own score |
| 29 | Central-vs-central reference = both agents at the centre going High, each with an independent random heading and the same turning rule. Each wins half the races; a race lasts the faster arrival | [user] | Reference race 1.93 s, score per second 0.0590 (0.0554 before the 2026-10-23 switch to the human time outside the race) |

### How random headings and search directions enter the score (2026-10-26)

Nothing is sampled. Every random element is enumerated over a fine grid, or over both of its cases, and
averaged exactly. A race whose winner depends on the central agent's heading counts as a fractional win.

- **Central heading.** The central agent's time to High is computed for 3,600 evenly spaced headings
  (every 0.1°), each equally likely. This is done once for each Low position relative to High, because
  the inferred search depends on where Low is (`central_times_inferred`). Where the inferred direction
  is a tie, both directions are kept at half weight.
- **Peripheral start.** The position and heading are the logged slice-onset values, so the peripheral
  agent has a single time per race row (heading rounded to the 0.5° search grid).
- **Peripheral search direction.** Each race row is one start position × one layout × one search
  direction (two rows per position and layout). A row's weight is the layout probability, shared
  equally over start positions and the two directions. When inference picks a direction, that row takes
  the whole weight. When the two directions tie, both keep half.
- **Win and race time** (`_against_central`). Each peripheral row is compared with every central time for
  its layout. The peripheral win is the share of central times later than the peripheral time (an exact
  tie counts half). The race time is the mean of the earlier arrival over all central times.
- **Score.** A row's expected score is win share × wall value for the peripheral agent, and
  (1 − win share) × 1.0 for the central agent. Score per second is the weighted total score over the
  weighted total of (race time + 6.545 s).

Example (central times with inference, Low one wall from High): a peripheral arrival at 2.0 s is later
than the central agent for 39.8% of headings, so it counts 0.398 of a win. At 2.2 s it is 0.203, at
2.4 s 0.035. The central agent never arrives before 1.82 s, so a peripheral time under that always wins.

How much the central heading decides (final Δ rule, Δ 1.05 s): the peripheral agent always wins 0.564
of trials and always loses 0.395. The remaining 0.041 depend on the central heading; there it wins
0.418 on average. These contested trials give 0.014 of the peripheral agent's 0.416 score per trial and
0.024 of the central agent's 0.419. Most races are decided by geometry alone, mainly whether the lit wall
is next to the peripheral agent's alcove. Shown in `geometric_model_assembly.ipynb`, section 8.

## Rules

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 30 | "Closest" compares the peripheral agent's own travel times. Equal times go to High | [claude] | Ties would otherwise flip on rounding |
| 31 | The threshold rule uses only the agent's own two travel times, not the opponent | [claude] | Matches the rule as stated |
| 32 | Δ swept from 0 to 3.8 s (step 0.05). Δ = 0 is "closest", Δ = 3.8 s is "always High" | [claude] | The best Δ is reported with the range within 0.5% of it |

## Outputs and validation

Notebook output was figures only **[user]**; section 9 adds tables at the user's request (2026-10-20). Validation against hand-counted answers lives in the
unit tests and is described in the notebook text **[user]**.

## Peripheral start positions and headings (designed; 2026-10-26)

The logged agents waited almost all at the back corner of an alcove, facing roughly out (median 13.5°
from facing the centre). Neither is a considered choice. Here the start is designed: a grid of positions,
each with the onset heading that gives the highest expected score per second. Notebook
`demos/analytical_modelling/peripheral_start_positions.ipynb`; code `analysis/start_positions.py`
(tests `tests/test_start_positions.py`); cache `start_position_map_260929`.

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| S1 | **Rays, not human vision.** Both agents keep the RL agents' 7 rays for this analysis | [user] (consistency with the rest of the model, and comparison with the RL agents) | Humans see a continuous field of view, so a few degrees of heading matter far less to them. With 7 rays some views exist only over a few degrees (e.g. one adjacent wall from the back corner, through a window of about 5°). **Rerun with a human field of view (e.g. `get_wall_visible`) before applying this to human players** |
| S2 | Positions: a grid over the clockwise half of alcove 0's sector (147 positions): 7 × 5 inside the alcove (17.0-19.7 units from the centre, lateral 0-1.85), rows 13.0-16.5 units in front of the wall face 0.5 units apart out to the line shared with wall 1, the wall face line (16.70), and the wall corner (just inside the point where walls 0 and 1 meet) | [user] (in front of the alcove, the lip, wall corners) + [claude] (grid, 0.5-unit spacing, 13-unit inner limit) | The arena is the same at every alcove and mirror-symmetric about each alcove's centre line, and layouts are uniform over High's wall, so this half-sector stands for every position. Mirror check in the notebook |
| S3 | Heading: every heading on the 0.5° grid (720) is scored at every position; the one with the highest expected score per second is kept | [user] | Chosen before onset from own position and the arena layout only (information rule). The notebook also shows which walls, and how many, are seen at every position and heading |
| S4 | Δ is chosen per position together with the heading (grid 0-3.8 s, 0.05 s steps) | [claude] (the agent knows where it stands) | Results at the fixed Δ of 1.05 s are shown beside them |
| S5 | The central agent is unchanged (centre, random heading, always High) | [user] | From the centre every heading is equally good before onset (the arena is the same in every direction), so a random heading is already its best. The backpedal (D11) applies only inside an alcove, so it never applies to the central agent |
| S7 | The rule is the final Δ rule (D6-D10) with the backpedal (D11, threshold 0.8) and own-trigger experience (D12) | [user] | Map cache stamp ('v2', 0.8, ...) |
| S6 | Run time: about 1.4 s per position and heading on one core (80 races, motion traces cached across layouts); 147 × 720 on 12 workers took 6.8 hours (2026-10-27) | [claude] | `_TRACE_CACHE` in analytical_race.py |

**Status (2026-10-27): run.** D11 and D12 are on. The logged races are rebuilt with them (cache stamp v10) and the position map is built. With D11 and D12 off, the model reproduces the v8 race cache exactly (2,400 sampled races, 0 differences); the v8 cache is kept as `periph_ray_races_260929_v8`.

**Effect of D11-D12 on the logged positions** (Δ 1.05 s; `nearer_wall_choice.ipynb` section 12, third table). Races change in 0.0100 of trials. In 0.0099 the own wall is lit (High 0.0051, Low 0.0048) and the agent backs into it, so race time falls by 0.0019 s per trial. In 0.00026 of trials (own wall Low) it backs into Low where before it turned, saw High and won it, so peripheral score per trial falls by 0.00016. No backpedal at a logged position lost the gamble. D12 alone changes almost nothing (0.00005 of trials). Net, peripheral score/s 0.05268 -> 0.05267, central 0.05304 -> 0.05305, margin -0.0004 unchanged; best Δ still 1.05 s (within 0.5%: 0.25-1.15 s, was 0.30-1.15 s).

**Results** (`peripheral_start_positions.ipynb`; each position at its best heading and Δ, time outside the race 6.545 s).

| Region | Peripheral score/s | Margin over central | Headings within 0.5% of best (median) |
|---|---|---|---|
| Alcove, in trigger (20) | 0.0540-0.0547 | +0.0024 to +0.0037 | 119° |
| Alcove, not in trigger (15) | 0.0558-0.0592 | +0.0035 to +0.0099 | 13° |
| In front of the wall (111) | 0.0538-0.0686 | +0.0017 to +0.0368 | 14° |
| Wall corner (1) | 0.0534 | +0.0010 | 41.5° |
| Logged positions (reference) | 0.0527 | -0.0004 | |
| Central vs central (reference) | 0.0591 | 0 | |

- Every designed position out-scores its central opponent. Trigger positions beat the logged positions mainly through instant wins (0.250 of trials against 0.233; 0.934 of logged starts are on the trigger).
- In front of the wall, the score rises toward the centre. The best per row goes from 0.0603 at the wall face to 0.0686 at 13 units. The best position (13 units out, 4.5 across) scores 0.0686 against 0.0324. **It is on the grid's inner edge, so the overall best position lies further in and was not found.** Moving out of the alcove gives up instant wins but wins every adjacent-High and own-wall-High layout and more far Lows.
- The wall corner is the weakest position outside the alcove (far from both neighbouring triggers).
- Both adjacent walls are seen together only from 12 of 147 positions (13-14.5 units out, 3.5-5.5 across), and at most 0.036 of headings. From the alcove back, rays reach an adjacent wall (the far one) only off the centre line, and over few headings (share 0 to 0.124 at 19.5-19.7 units).
- Limits: rays not human vision (S1); central opponent fixed (always High, from the exact centre, ignores and cannot block the peripheral agent); grid inner limit at 13 units.

## Open points

- Inference re-decided during the search (e.g. reversing when Low appears), and from the opponent's position.
- Opponent visibility.
- Overhead: agent delays overran by up to 1/3 s per wait (frame-timed waits in sped-up runs; 7.09 s per trial against 6.5 s set). Fixed in Octagon `OctagonArenaSettings.ITI()` (physics-clock waits, applied 2026-10-27; in builds/261027). Human sessions use separate waits in `TrialHandler.cs`, unaffected. Check the first 261027 logs: mean about 6.5 s, no 1/3 s bunching.
- Human version: human onset positions and measured human speed.
