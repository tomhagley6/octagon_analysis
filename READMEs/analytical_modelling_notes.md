# Geometric race model: assumptions and decisions

Covers the current model in `analysis/analytical_race.py` (tests in `tests/test_analytical_race.py`).
The notebook `demos/analytical_modelling/nearer_wall_choice.ipynb` (built by
`build_nearer_wall_choice.py` beside it) shows the model without the central field of view in
sections 3-6. Section 7 shows the logged central turning behaviour that sets the field-of-view
inputs. Model figures with the field of view are still to be added.

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
| 22b | **Assumption:** the empirical closing speed (7.57 units/s) is below the best the control scheme allows (9.24-10 units/s, mean 9.46 over a 77.5 deg in-view turn; 7.57 is 0.80 of that). The model keeps the empirical value, so its central agent closes on High as slowly as logged agents do, not as fast as the controls would permit | [user] | Logged agents lose speed by moving at an angle to High (median 23 deg off the line, observation 7), not by moving slowly |
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
| P5 | If no heading from the current position senses the target, first step toward the arena centre (along the alcove axis) at 10 units/s, by the shortest distance after which some heading senses it, then search | [user] (step toward the mouth) + [claude] (direction, speed, shortest step found by bisection to 0.01 units) | Needed mainly for adjacent walls from the alcove back |
| P6 | Once the target is sensed, keep turning to centre the first straight leg of the shortest path (e.g. toward the alcove mouth corner), not the direct bearing to the wall. If the search turned past that leg before sensing, turn back the short way | [user] (first leg) + [claude] (turning back) | The direct line from an alcove often crosses walls |
| P7 | After centring, follow the shortest path at 10 units/s with no extra time at bends | [user] | The control scheme allows moving in 8 directions without turning |
| P8 | Headings are searched on a 0.5 deg grid | [claude] | 0.5 deg at 331 deg/s is 1.5 ms |
| P9 | An agent overlapping a lit trigger at onset still wins it at once | [data] | As #13 |
| P10 | "Low, if Low seen first": both lit walls sensed at onset is a tie and goes High. Otherwise the first wall a ray hits during the search decides. A wall hit at the same heading as the other is a tie and goes High | [user] | |
| P11 | Under "Low, if Low seen first", the agent steps out only if no heading from where it stands senses either lit wall. It steps the shortest distance after which one becomes sensable, so the wall needing the shorter step is chosen (equal steps go High) | [user] (agreed 2026-10-20) | Consequence: from the alcove back an adjacent High is hidden, while a Low further away often is not. The agent then turns, finds Low and goes there without stepping out to look for High |
| P12 | When High is the first wall sensed, the time to High is identical to "High, always" | [claude] | Checked in tests |
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

## Scoring

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 25 | Score 1.0 for a High win, 0.4 for a Low win, 0 for a loss | [data] | Task rules |
| 26 | Overhead per trial = whole trial − onset-to-trigger time, mean 7.09 s (sd 0.91), from 5 central 260929 models, used for every trial | [data] + [claude] | Covers trial start → onset, trigger → end, and the ITI. These delays match the human task [user]. Not checked whether overhead depends on who won |
| 27 | Primary measure = score per second = total expected score / total expected (race + overhead). Secondary = score per trial | [user] | Players are told to maximise score over time |
| 28 | Relative score = peripheral minus central score per second. A central player against the same opponent ties (margin 0) | [user] | Human players may try to beat the other player as well as maximise their own score |
| 29 | Central-vs-central reference = both agents at the centre going High, each with an independent random heading and the same turning rule. Each wins half the races; a race lasts the faster arrival | [user] | Reference race 1.93 s, score per second 0.0554 |

## Rules

| # | Decision | Tag | Basis / note |
|---|---|---|---|
| 30 | "Closest" compares the peripheral agent's own travel times. Equal times go to High | [claude] | Ties would otherwise flip on rounding |
| 31 | The threshold rule uses only the agent's own two travel times, not the opponent | [claude] | Matches the rule as stated |
| 32 | Δ swept from 0 to 3.8 s (step 0.05). Δ = 0 is "closest", Δ = 3.8 s is "always High" | [claude] | The best Δ is reported with the range within 0.5% of it |

## Outputs and validation

Notebook output is figures only **[user]**. Validation against hand-counted answers lives in the
unit tests and is described in the notebook text **[user]**.

## Open points

- Central turn direction from Low: use the Low wall's position (when in view) to infer which way to turn to High. Logged agents turn the short way more often than chance (observation under #21).
- Opponent visibility.
- Overhead: is the agent ITI fixed by design or variable? If it depends on outcome, #26 needs revisiting.
- Human version: human onset positions and measured human speed.
