''' Replicate the RL agent's ray sensor from its logged position and yaw.

The agents do not see a continuous field of view. `SelectivePassThroughRaycast` (Unity,
`Assets/Scripts/RaycastLogic`) casts 7 horizontal rays from the agent's position at 0, +/-18.3,
+/-36.7 and +/-55 degrees from its heading, every physics step. Each ray reports only its first hit
(the opponent is passed through). What a hit tells the agent:
  - Any part of a lit wall reads HighWall or LowWall. A wall's two faces and its alcove-back block
    are three box colliders on one GameObject, so the inside of a lit alcove reads as that wall.
  - A lit wall's trigger box reads HighWallTrigger or LowWallTrigger. Trigger boxes are switched on
    at trial start, sit in the back half of each alcove, and are hit by rays (the ray mask is 507,
    which includes their layer, and queriesHitTriggers is on).
  - An unlit wall or trigger returns a distance with no tag, so the agent can tell it is not lit.
  - A ray starting inside a collider does not detect it, so an agent standing in a trigger box does
    not see that box, but does see the alcove walls around it.

Geometry (physical units, from the prefab `OctagonWallAlcovesHalfDepthFull` at arena scale 1.5):
the wall face is at 17.355 from the centre, the alcove back at 20.355, and the alcove is 5.02 wide.
These match `visibility_occlusion.get_physical_geometry` (reach + player radius). The trigger box
runs from 18.855 to the alcove back, across the full alcove width.

Wall ids are 1-8, wall 1 facing north (+z), numbered clockwise, matching the logs.
'''

import math

import numpy as np

import globals
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import trajectory_analysis.visibility_occlusion as vo

NUM_WALLS = globals.NUM_WALLS
RAY_OFFSETS_DEG = np.linspace(-55, 55, 7)          # 2 * raysPerDirection + 1 rays across +/- maxRayDegrees
TRIGGER_DEPTH = 1.5                                 # trigger box depth from the alcove back (prefab y extent x scale)
EPS = 1e-9

KIND_WALL, KIND_TRIGGER, KIND_AGENT = 0, 1, 2
AGENT_RADIUS = 0.6                  # CharacterController radius on the PlayerAgent prefab
RAY_LENGTH = 60.0                   # rayLength on the PlayerAgent prefab


def get_trigger_boxes():
    ''' Return each wall's trigger box as (p_front, p_back, half_width) in that wall's frame.
        Takes no inputs. p is the distance toward the wall, as in visibility_occlusion. '''

    g = vo.get_physical_geometry()
    return g['alcove_back'] - TRIGGER_DEPTH, g['alcove_back'], g['alcove_half_width']


def get_sensor_segments():
    ''' Return every surface a ray can hit, with the wall it belongs to and whether it is a trigger.
        Takes no inputs. Returns segments (N*2*2), wall ids (N, 1-8) and kinds (N, KIND_WALL or
        KIND_TRIGGER). '''

    # take the five wall sections per wall (two faces, two alcove sides, alcove back)
    walls = vo.get_wall_segments()
    wall_ids = np.repeat(np.arange(1, NUM_WALLS + 1), 5)

    # add the four edges of each trigger box
    front, back, half = get_trigger_boxes()
    trig = []
    for w in range(NUM_WALLS):
        corners = [vo.wall_frame_to_arena(p, q, w) for p, q in ((front, -half), (front, half), (back, half), (back, -half))]
        for i in range(4):
            trig.append([corners[i], corners[(i + 1) % 4]])
    trig = np.array(trig)
    trig_ids = np.repeat(np.arange(1, NUM_WALLS + 1), 4)

    segments = np.concatenate([walls, trig])
    ids = np.concatenate([wall_ids, trig_ids])
    kinds = np.concatenate([np.full(len(walls), KIND_WALL), np.full(len(trig), KIND_TRIGGER)])
    return segments, ids, kinds


def inside_trigger(eyes):
    ''' Return, per frame, the id of the trigger box the eye is inside, or 0 if none.
        Takes eye positions (timepoints*2). '''

    front, back, half = get_trigger_boxes()
    out = np.zeros(len(eyes), int)
    for w in range(NUM_WALLS):
        p, q = vo.arena_to_wall_frame(eyes, w)
        out[(p >= front) & (p <= back) & (np.abs(q) <= half)] = w + 1
    return out


def ray_circle_t(eyes, d, centres, radius=AGENT_RADIUS):
    ''' Return the distance along each ray to where it enters a circle, or inf if it misses.
        Takes eye positions (T*2), ray directions (T*R*2), circle centres (T*2) and the radius. '''

    rel = (centres - eyes)[:, None, :]                                       # T*1*2
    along = (rel * d).sum(-1)                                               # T*R
    perp2 = (rel ** 2).sum(-1) - along ** 2
    half = np.sqrt(np.clip(radius ** 2 - perp2, 0, None))
    t = along - half
    hit = (perp2 <= radius ** 2) & (t > EPS)
    return np.where(hit, t, np.inf)


def ray_directions(yaw_rad, offsets_deg=RAY_OFFSETS_DEG):
    ''' Return every ray's unit direction (T*R*2; x = sin, z = cos of the clockwise angle from +z). '''

    ang = np.asarray(yaw_rad, float)[:, None] + np.radians(offsets_deg)[None, :]
    return np.stack([np.sin(ang), np.cos(ang)], axis=-1)


def cast_rays(eyes, yaw_rad, offsets_deg=RAY_OFFSETS_DEG, blocker=None, triggers=True):
    ''' Return the first hit of every ray on every frame. Takes eye positions (timepoints*2), yaw in
        radians (timepoints, clockwise from +z as logged), the ray offsets in degrees, and optionally
        the positions (timepoints*2) of an agent body that blocks the rays. triggers=False leaves the
        trigger boxes out, as between the trigger and the next slice onset, when they are switched off.

        Returns hit ids (timepoints*rays; wall 1-8, or 0 for the blocking agent), hit kinds (KIND_WALL,
        KIND_TRIGGER or KIND_AGENT) and hit distances. The arena is closed, so every ray hits something. '''

    eyes = np.asarray(eyes, float)
    segments, ids, kinds = get_sensor_segments()
    if not triggers:
        keep = kinds != KIND_TRIGGER
        segments, ids, kinds = segments[keep], ids[keep], kinds[keep]
    d = ray_directions(yaw_rad, offsets_deg)                                # T*R*2

    # solve E + t*d = A + u*e for every ray and segment
    A = segments[None, None, :, 0, :]
    e = (segments[:, 1, :] - segments[:, 0, :])[None, None]
    w = A - eyes[:, None, None, :]
    D = d[:, :, None, :]
    denom = vo.cross_2d(D, e)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = vo.cross_2d(w, e) / denom
        u = vo.cross_2d(w, D) / denom
    hit = (np.abs(denom) > EPS) & (t > EPS) & (u >= -EPS) & (u <= 1 + EPS)

    # ignore the trigger box the eye stands in, as a ray does not detect a collider it starts inside
    own = inside_trigger(eyes)
    skip = (kinds[None, :] == KIND_TRIGGER) & (ids[None, :] == own[:, None])     # T*N
    hit &= ~skip[:, None, :]

    # keep the nearest hit of each ray
    t = np.where(hit, t, np.inf)
    first = np.argmin(t, axis=2)
    dist = np.take_along_axis(t, first[..., None], axis=2)[..., 0]
    hit_ids, hit_kinds = ids[first], kinds[first]

    # let a blocking agent body take the ray if it comes first
    if blocker is not None:
        tb = ray_circle_t(eyes, d, np.asarray(blocker, float))
        nearer = tb < dist
        hit_ids = np.where(nearer, 0, hit_ids)
        hit_kinds = np.where(nearer, KIND_AGENT, hit_kinds)
        dist = np.where(nearer, tb, dist)
    return hit_ids, hit_kinds, dist


# opponent sensing rules of each build generation. Builds before 261014 pass through only the
# OpponentAgent tag, with no occlusion. From 261014 (source after commit f524ff5, with the occlusion edit)
# both agent tags are passed through, and an opponent counts only if it is nearer than the first solid
# (non-agent, non-trigger) hit on that ray
BUILD_PRE_261014 = dict(opponent_channel_seats=(0,), occlude=False)
BUILD_261014 = dict(opponent_channel_seats=(0, 1), occlude=True)


def opponent_ray_distance(eyes, yaw_rad, opponent, occlude=False, offsets_deg=RAY_OFFSETS_DEG):
    ''' Return, per frame and ray, the distance at which the ray enters the opponent's body, or inf where
        it misses or is beyond RAY_LENGTH. Takes the agent's positions and yaw and the opponent's positions.

        With occlude=True, an opponent behind a wall (or an alcove side) is missed, as in builds from 261014.
        Trigger boxes never occlude. '''

    eyes, opponent = np.asarray(eyes, float), np.asarray(opponent, float)
    t = ray_circle_t(eyes, ray_directions(yaw_rad, offsets_deg), opponent)
    t = np.where(t < RAY_LENGTH, t, np.inf)
    if occlude:
        _, _, wall_t = cast_rays(eyes, yaw_rad, offsets_deg, triggers=False)
        t = np.where(t < wall_t, t, np.inf)
    return t


def opponent_sensed(eyes, yaw_rad, opponent, player_id, offsets_deg=RAY_OFFSETS_DEG,
                    opponent_channel_seats=BUILD_PRE_261014['opponent_channel_seats'],
                    occlude=BUILD_PRE_261014['occlude']):
    ''' Return, per frame, whether an agent senses its opponent. Takes the agent's positions and yaw,
        the opponent's positions (all timepoints*2), the agent's logged player id, and the build's rules
        (pass **BUILD_PRE_261014 or **BUILD_261014; the default is the older builds).

        A seat with the opponent channel senses the opponent if any ray passes through its body (behind
        walls too, unless occlude). In older builds player 1 has no channel, and senses player 0 only
        where player 0 is a ray's first hit. '''

    eyes, opponent = np.asarray(eyes, float), np.asarray(opponent, float)
    if player_id in opponent_channel_seats:
        return np.isfinite(opponent_ray_distance(eyes, yaw_rad, opponent, occlude, offsets_deg)).any(axis=1)
    ids, kinds, _ = cast_rays(eyes, yaw_rad, offsets_deg, blocker=opponent)
    return (kinds == KIND_AGENT).any(axis=1)


def sensed_walls(hit_ids, n_walls=NUM_WALLS):
    ''' Return a walls*timepoints boolean array, True where any ray hits any part of that wall
        (faces, alcove or trigger box). Takes the hit ids from cast_rays. '''

    out = np.zeros((n_walls, hit_ids.shape[0]), bool)
    for w in range(1, n_walls + 1):
        out[w - 1] = (hit_ids == w).any(axis=1)
    return out


def trial_sensing(trial, player_id):
    ''' Return the walls one agent's rays hit on each frame from slice onset to the trigger, and
        whether it senses its opponent. Takes a trial dataframe and a player id.
        Returns (sensed walls*timepoints, opponent sensed per frame, eyes, yaw, opponent positions).

        Uses the raw logged yaw, with no smoothing, as the sensor reads the current heading. Player 1's
        rays are blocked by player 0's body. '''

    eyes = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_id).T.astype(float)
    yaw = np.asarray(trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=player_id), float)
    opp = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=1 - player_id).T.astype(float)
    n = min(len(eyes), len(yaw), len(opp))
    eyes, yaw, opp = eyes[:n], yaw[:n], opp[:n]
    ok = np.isfinite(eyes).all(axis=1) & np.isfinite(yaw) & np.isfinite(opp).all(axis=1)
    e, y, o = np.where(ok[:, None], eyes, 0.0), np.where(ok, yaw, 0.0), np.where(ok[:, None], opp, 1e3)
    ids, _, _ = cast_rays(e, y, blocker=o if player_id == 1 else None)
    sensed = sensed_walls(ids)
    sensed[:, ~ok] = False
    opp_sensed = opponent_sensed(e, y, o, player_id) & ok
    return sensed, opp_sensed, eyes, yaw, opp


# ---------------------------------------------------------------------------
# what an ideal observer could infer about where High is
# ---------------------------------------------------------------------------

SEP_PRIOR = {1: 0.5, 2: 0.25, 4: 0.25}             # wall separation weights (General.wallSeparations*)


def wall_pair_prior(sep_prior=SEP_PRIOR, n_walls=NUM_WALLS):
    ''' Return the prior probability of every (High, Low) wall pair. Takes the separation weights.

        High is equally likely at any wall. For separations 1 and 2, Low is equally likely on
        either side; separation 4 has one Low position. '''

    prior = {}
    for sep, p in sep_prior.items():
        sides = (sep,) if sep * 2 == n_walls else (sep, -sep)
        for h in range(1, n_walls + 1):
            for s in sides:
                l = (h - 1 + s) % n_walls + 1
                prior[(h, l)] = prior.get((h, l), 0) + p / n_walls / len(sides)
    return prior


def high_posterior(seen_high=None, seen_low=None, seen_unlit=(), sep_prior=SEP_PRIOR):
    ''' Return the posterior probability that High is at each wall (dict wall -> p).
        Takes the High wall if sensed, the Low wall if sensed, and the walls sensed unlit. '''

    unlit = set(seen_unlit)
    post = {}
    for (h, l), p in wall_pair_prior(sep_prior).items():
        if h in unlit or l in unlit:
            continue
        if seen_high is not None and h != seen_high:
            continue
        if seen_low is not None and l != seen_low:
            continue
        post[h] = post.get(h, 0) + p
    total = sum(post.values())
    return {h: p / total for h, p in post.items()} if total > 0 else {}


# ---------------------------------------------------------------------------
# per-trial table for a batch
# ---------------------------------------------------------------------------

COMMIT_MARGIN = 1.0                 # units; how much further one lit wall's distance must fall than the other's


def commit_frame(d_target, d_other, margin=COMMIT_MARGIN):
    ''' Return the first frame where the agent has closed on the target wall by `margin` more than
        on the other lit wall, or -1 if it never does. Takes the two distance series (from slice onset).

        Leaving an alcove brings an agent nearer most walls at once, so a fall in one distance alone
        does not show which wall it is heading for. The difference between the two does. '''

    gain = (d_target[0] - d_target) - (d_other[0] - d_other)
    idx = np.flatnonzero(gain >= margin)
    return int(idx[0]) if len(idx) else -1


def first_frames(mask):
    ''' Return the first True frame of each row of a walls*timepoints array (-1 if never). '''

    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)


def trial_sensing_rows(trial, players=(0, 1), fov=110.0):
    ''' Return one dict per agent for a trial: lit walls, outcome, the first frame each wall is hit by
        a ray and enters the human-model wedge, the frames the agent commits to High and to Low, and
        the frames of its closest approach to each lit wall. Which lit wall the rays sense at onset and
        first are categorised with trajectory_headangle.wall_visibility_player_slice_onset and
        get_first_visible_wall, the functions the human visibility analysis uses.
        Takes a trial dataframe, the agent ids and the FoV for the wedge model. Frame 0 is slice onset. '''

    import contextlib, io
    import data_extraction.get_indices as get_indices
    import trajectory_analysis.trajectory_headangle as th
    from analysis.race_estimate import ALCOVE_CENTRES

    trig = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    if len(trig) == 0 or len(onset) == 0:
        return []
    # take the walls, winner, wall reached and separation with the shared get_indices helpers
    high, low = get_indices.get_walls(trial)
    winner = get_indices.get_trigger_activator(trial)
    chosen = int(get_indices.get_chosen_walls([trial])[0])
    wall_sep = get_indices.get_wall_difference(trial=trial)
    rt_s = (trig['timeApplication'].iloc[0] - onset['timeApplication'].iloc[0]) / np.timedelta64(1, 's')

    rows = []
    for pid in players:
        sensed, opp_sensed, eyes, _, opp = trial_sensing(trial, pid)
        if len(eyes) == 0:
            continue
        # take the human-model wedge for comparison (5-frame smoothed heading, coloured faces only)
        with contextlib.redirect_stdout(io.StringIO()):
            wedge = th.get_wall_visible(trial=trial, player_id=pid, current_fov=fov, full_trial_window=True)
        wedge_first = first_frames(wedge) if not isinstance(wedge, float) else np.full(NUM_WALLS, -2)

        # get the distance to each lit wall's alcove centre on every frame
        d_high = np.linalg.norm(eyes - ALCOVE_CENTRES[high - 1], axis=1)
        d_low = np.linalg.norm(eyes - ALCOVE_CENTRES[low - 1], axis=1)
        ray_first = first_frames(sensed)

        # categorise what the rays sense with the same functions the human visibility analysis uses
        high_seen0, low_seen0 = th.wall_visibility_player_slice_onset(sensed, trial)
        first_sensed = th.get_first_visible_wall(sensed, high_seen0, low_seen0, trial)

        row = dict(trial_num=float(trial[globals.TRIAL_NUM].dropna().iloc[0]), player=pid,
                   x0=eyes[0, 0], z0=eyes[0, 1], high_wall=high, low_wall=low,
                   won=(pid == winner), chosen_wall=chosen, chose_high=(chosen == high), rt_s=float(rt_s),
                   wall_sep=int(wall_sep), high_seen0=bool(high_seen0), low_seen0=bool(low_seen0),
                   first_sensed={'wall1': 'High', 'wall2': 'Low'}.get(first_sensed, first_sensed),
                   n_frames=len(eyes), d_high0=d_high[0], d_low0=d_low[0],
                   d_high_end=d_high[-1], d_low_end=d_low[-1],
                   commit_high=commit_frame(d_high, d_low), commit_low=commit_frame(d_low, d_high),
                   closest_high=int(np.argmin(d_high)), closest_low=int(np.argmin(d_low)),
                   opp_x0=opp[0, 0], opp_z0=opp[0, 1],
                   opp_first=int(np.argmax(opp_sensed)) if opp_sensed.any() else -1)
        for w in range(1, NUM_WALLS + 1):
            row[f'ray_first_{w}'] = int(ray_first[w - 1])
            row[f'wedge_first_{w}'] = int(wedge_first[w - 1])
        rows.append(row)
    return rows


def collect_sensing_table(sim_data_folder, models, verbose=True):
    ''' Return one row per trial x agent with ray sensing, wedge visibility and commit frames.
        Takes a batch's simulation folder and the model list. Drops the session's first trial. '''

    import contextlib, io
    import pandas as pd
    import parse_data.prepare_sim_data as prepare_sim_data

    records = []
    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            _, trial_list = prepare_sim_data.prepare_single_model_data(model, sim_data_folder=sim_data_folder)
        for trial in trial_list[1:]:
            for r in trial_sensing_rows(trial):
                r['model'] = model
                records.append(r)
        if verbose:
            print(f"model {model} done", end='  ')
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# tournament matchups (two different models)
# ---------------------------------------------------------------------------

def matchup_trial_list(base, folder):
    ''' Return the trial list of one tournament matchup, first trial dropped. Takes the tournament
        directory and the matchup folder ("<A>__vs__<B>"; A drives player 0, B player 1). '''

    import contextlib, io, os
    import parse_data.loading as loading
    import parse_data.preprocess_sim as preprocess_sim
    import parse_data.split_session_by_trial as split_session_by_trial

    d = os.path.join(base, folder)
    fn = next(f for f in os.listdir(d) if f.endswith('.json') and not f.startswith('._') and 'config' not in f)
    with contextlib.redirect_stdout(io.StringIO()):
        df = loading.loading_pipeline(d, fn)
        df = preprocess_sim.standard_preprocessing_sim(df)
        trial_list = split_session_by_trial.split_session_by_trial(df, drop_trial_zero=True)
    return trial_list[1:]


def collect_matchup_tables(base, folders, verbose=True):
    ''' Return (sensing table, trajectory table) for a set of tournament matchups, one row per trial x
        agent. Each row carries the matchup, its own model ('model') and its opponent's ('opp_model').
        Takes the tournament directory and the matchup folder names. '''

    import pandas as pd
    from analysis.agent_loser_choice import trajectory_rows

    sense_rows, traj_rows = [], []
    for folder in folders:
        a, b = folder.split('__vs__')
        names = {0: a, 1: b}
        for trial in matchup_trial_list(base, folder):
            for r in trial_sensing_rows(trial):
                r.update(matchup=folder, model=names[r['player']], opp_model=names[1 - r['player']])
                sense_rows.append(r)
            for r in trajectory_rows(trial):
                r.update(matchup=folder, model=names[r['player']], opp_model=names[1 - r['player']])
                traj_rows.append(r)
        if verbose:
            print(f"{folder} done", end='  ')
    return pd.DataFrame(sense_rows), pd.DataFrame(traj_rows)
