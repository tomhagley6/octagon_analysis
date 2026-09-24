# Wall visibility with occlusion

Wall visibility used to be an angle test. A wall counted as visible when the angle from the
player's head direction to that wall was under half the field of view. Nothing checked whether
another wall stood in the way.

That matters most for peripheral players. Standing inside an alcove, the alcove side walls cut the
view of the arena down to what fits through the mouth. From the middle of an alcove's back the view
out spans ±46°, narrower than the ±55° field of view. Pressed against one side of the alcove it is
14° on that side and 61° on the other. The walls lost that way are the neighbouring ones, which are
exactly the walls a peripheral player would otherwise catch at the edge of view.

Agents see through a Unity camera, so blocking is already part of what they see. The analysis was
the only place ignoring it.

## What counts as visible

Four decisions set the measure (Tom, 2026-09-14/15).

- **The colour is on the wall face either side of the alcove**, not inside the alcove.
- **Any visible part is enough**, because the colour is salient.
- **Only physical walls block sight.** These sit at the reachable distances in `globals` plus the
  player radius of 0.61, because a player's body does not limit their sight.
- **The camera sits at the logged position.**

## Where the old and new methods differ

`diagnose_method_difference.py` changes one thing at a time between the two methods, on frames where
the camera is outside every alcove and nothing can block. It runs on 260723 models 23 (peripheral) and
34 (central), 60 trials each, 22,994 frames. Each row gives the share of wall × frame values that
change at that step.

| Step | Change | Changed | Direction |
|---|---|---|---|
| S1 → S2 | reachable → physical wall positions | 0.93% | mostly newly visible |
| S2 → S3 | test both corners of a wall, not just the one chosen | 0.28% | all newly visible |
| S3 → S4 | any point along the wall, not only its corners | 0.68% | all newly visible |
| S4 → S5 | coloured faces only, excluding the alcove mouth | 0.19% | all newly hidden |
| S5 → S6 | the exact implementation | 0% | – |

Steps S2 and S3 are errors in the old method, and both happen within about 2 units of a wall.

- **The corner tested was chosen from the alcove centre.** The old method decided which side of the
  heading a wall was on by looking at its alcove centre, then tested that side's corner. Close to a
  wall that picks the far corner while the near one is in plain view.
- **Only corners were tested.** Facing a wall from close up, both corners can be more than 55° away
  while the wall ahead fills the view.

`method_difference_examples.png` draws both, from real frames.

Together the two methods disagree on 1.69% of wall × frame values outside alcoves, 2.12% for the
peripheral model and 1.28% for the central one.

## The solution

The octagon without its alcoves is convex, so only alcove walls can block sight, and only when the
camera is inside that alcove. Then the camera sees out through the mouth alone.

Both limits are wedges from the camera narrower than 180°, one for the field of view and one for the
mouth. A wedge is two straight boundaries through the camera, and each boundary cuts a wall face along
a straight line, so it limits that face to a range of positions along it. A wall is visible when any
part of either coloured face survives both wedges.

This is exact rather than sampled, and runs as fast as the old angle test. It needs no special case
for the alcove's own wall, whose faces lie on the mouth line and so touch the mouth wedge at a
single point.

An earlier version sampled 20 points along each face. It missed slivers at the edge of view on 0.28%
of values, as large an effect as the errors being fixed, so it was replaced.

**Frames with a missing position or head angle, or a head angle of zero length, show no walls.** This
matches the old method. Without it a missing value left the visible range untouched and every wall
came out visible, which happened on 185 frames of 3 human social trials.

## Checks

56 tests across `tests/test_visibility_occlusion.py`, `tests/test_arena_geometry.py` and
`tests/test_opponent_visibility.py`.

The main ones compare against an independent reference that samples points along the faces and
decides blocking by a mouth-crossing rule rather than by intersecting line segments. The exact
method never misses a wall the reference sees, and the walls it sees that the reference misses fall
from 2,949 to 139 as the reference samples more finely, which is what an exact method should show.
Other tests cover hand-worked views, slivers 0.05° either side of both the field-of-view edge and
the mouth corner, and the geometry constants.

`human_data_checks.py` runs both methods over every human trial, 100 social sessions with both
players and 400 solo sessions.

- No errors, and the same output shape and too-short trials as the old method.
- Players sit inside an alcove on 3.8% of social frames and 4.9% of solo frames.
- Occlusion changes 1.63% of social values and 1.93% of solo values. Inside alcoves it changes 7.55%
  and 5.79%.
- Solo positions never leave the reachable arena, so human and agent geometry match. Social positions
  do overshoot the walls on 0.3-0.5% of frames, which is recorded in the data-quality notes and
  mostly falls outside the visibility window.

## What changed in the repo

- **`trajectory_analysis/visibility_occlusion.py`** holds the geometry, the wedge clipping and
  `is_line_of_sight_blocked` for point targets.
- **`get_wall_visible(..., occlusion=True)` is the default** since 2026-09-15. Every existing caller
  picks it up on its next run. `occlusion=False` still gives the old angle test.
- **Opponent visibility got the same treatment.** `get_other_visible_trial` and
  `get_other_visible_session_from_trials` in `analysis/opponent_visibility.py` require a clear line
  to Other's centre, and `conditioned_player_choice.py` uses them. Over all human social sessions
  this hides Other in 92 of 11,802 slice onsets where the angle alone said visible, taking visibility
  from 44.6% to 44.2%. The old angle-only `get_other_visible_session` is unchanged, and the GLM
  sandbox and `opponent_visibility_analysis_demo.ipynb` still call it.
- **`globals.PLAYER_RADIUS`** (0.61) turns reachable distances into physical wall positions. It was
  fitted from the rounded reach around alcove mouths, against 0.6 on the prefab's `CharacterController`.
- **`utils/cosine_similarity.py`** now clips to [-1, 1]. `math.acos` raised an error when a player
  faced their opponent exactly.

Any visibility result from before 2026-09-15 used the old method and will shift when rerun, most for
peripheral players.

## Known limits

- **The opponent's body does not block sight**, for walls or for the other player.
- **Only horizontal head direction is used**, with no up or down tilt.
- **Logged positions past the physical walls** are treated as if inside that alcove, which is
  meaningless for those frames. They are rare and the result stays bounded.

## Files here

| File | What it does |
|---|---|
| `diagnose_method_difference.py` | the step-by-step comparison above |
| `plot_method_difference_examples.py` | draws `method_difference_examples.png` from two real frames |
| `method_difference_examples.png` | why the methods differ when nothing blocks sight |
| `human_data_checks.py` | geometry, sweep and exactness checks on human data |

Run each with the conda env python from the repo root.
