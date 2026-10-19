''' Test whether an agent's wall choice responds to where its opponent is.

Built for cross-model tournament matchups, where the two agents were trained apart. Used by Stage 6
of the 260929 model characterisation notebook.

Choices that depend on the trial's outcome cannot be used here. A winner's wall, or a loser whose
choice can be called, is picked by what both agents did, and an opponent near High changes who wins.
Sensing the opponent during the trial cannot be used either, as an agent heading for High is the one
whose rays sweep an opponent standing near High. So:
  - the outcome is the agent's heading at a fixed time (`heading_high_at`), taken on every trial
    long enough to have one, winner or loser;
  - the opponent counts as seen only if a ray senses it at slice onset, before any movement;
  - comparisons are made within strata of the agent's own position and what it sees at onset.
'''

import numpy as np
import pandas as pd

import globals
import analysis.agent_raycast as ar
import trajectory_analysis.trajectory_direction as trajectory_direction
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import trajectory_analysis.visibility_occlusion as vo
from analysis.race_estimate import ALCOVE_CENTRES

HEADING_FRAME = 50                  # frame (1.0 s after slice onset) at which heading is read
HEADING_WINDOW = 12                 # frames of movement the heading is averaged over
SMOOTH_WINDOW = 11                  # Savitzky-Golay window for the movement direction (trajectory_direction default)
MIN_FRAME_STEP = 0.02               # smoothed units moved per frame; below this no heading is read
NEAR_HIGH = 15.0                    # units from the High alcove centre that count as near High


def heading_high_at(xy, high, low, frame=HEADING_FRAME, window=HEADING_WINDOW, smooth_window=SMOOTH_WINDOW):
    ''' Return 1.0 if the agent is moving more toward the High alcove than the Low one at a set frame,
        0.0 if the reverse, and NaN if the trial is too short or the agent is not moving.
        Takes the trajectory from slice onset (timepoints*2) and the High and Low wall ids.

        The movement direction is trajectory_direction's Savitzky-Golay smoothed direction, and its cosine
        to each alcove comes from cosine_similarity_throughout_trajectory, averaged over the `window`
        frames before `frame`. '''

    if len(xy) <= frame or frame <= window + smooth_window:
        return np.nan
    seg = xy[frame - window - smooth_window:frame].T
    moved = np.linalg.norm(trajectory_direction.get_smoothed_player_direction_vectors_for_trajectory(
        seg, window_size=smooth_window), axis=0)[-window:]
    if moved.mean() < MIN_FRAME_STEP:
        return np.nan
    cos = trajectory_direction.cosine_similarity_throughout_trajectory(seg, window_size=smooth_window)[:, -window:]
    return float(np.nanmean(cos[high - 1]) > np.nanmean(cos[low - 1]))


def own_alcove(x, z):
    ''' Return the wall id (1-8) of the alcove nearest each position. Takes x and z arrays. '''

    pts = np.stack([np.asarray(x, float), np.asarray(z, float)], axis=1)
    return np.argmin(np.linalg.norm(ALCOVE_CENTRES[None] - pts[:, None], axis=2), axis=1) + 1


def signed_steps(wall, own):
    ''' Return the signed number of alcoves from `own` to `wall`, in -3..4 (positive = clockwise). '''

    d = (np.asarray(wall) - np.asarray(own)) % 8
    return np.where(d > 4, d - 8, d)


def position_cell(high, low, own):
    ''' Return each agent's position relative to the two lit walls as (steps to High, steps to Low),
        mirrored so High is never anticlockwise. Mirror-image layouts are the same geometry. '''

    kh, kl = signed_steps(high, own), signed_steps(low, own)
    flip = kh < 0
    kh, kl = np.where(flip, -kh, kh), np.where(flip, -kl, kl)
    kl = np.where(kh == 0, np.abs(kl), kl)
    return kh, kl


def on_lit_trigger(x, z, high, low):
    ''' Return whether an agent's body overlaps either lit wall's trigger box. Takes positions and walls. '''

    front, back, half = ar.get_trigger_boxes()
    out = []
    for xi, zi, h, l in zip(x, z, high, low):
        hit = False
        for w in (h, l):
            p, q = vo.arena_to_wall_frame(np.array([[xi, zi]]), w - 1)
            if np.hypot(max(front - p[0], 0, p[0] - back), max(abs(q[0]) - half, 0)) < globals.PLAYER_RADIUS:
                hit = True
        out.append(hit)
    return np.array(out)


def facing_angle(yaw0, xy0, wall):
    ''' Return the angle (degrees) between an agent's onset heading and the direction to a wall's alcove.
        Takes onset yaws (radians), onset positions (n*2) and wall ids.

        Uses the trajectory_vectors chain the human head-angle analyses use (alcove direction vectors,
        dot products, norms, cosine, angle). '''

    heads = trajectory_headangle.get_player_headangle_vectors_for_trial(np.asarray(yaw0, float))
    to_alcoves = trajectory_vectors.get_player_to_alcove_direction_vectors_for_trajectory(np.asarray(xy0, float).T)
    dots = trajectory_vectors.calculate_vector_dot_products_for_trajectory(to_alcoves, heads)
    head_norms, alcove_norms = trajectory_vectors.calculate_vector_norms_for_trajectory(to_alcoves, heads)
    thetas = trajectory_vectors.calculate_thetas_for_trajectory(
        trajectory_vectors.calculate_cosine_similarity_for_trajectory(dots, head_norms, alcove_norms))
    return np.degrees(thetas[np.asarray(wall) - 1, np.arange(len(heads.T))])


def build_trial_table(sense, traj):
    ''' Return one row per agent-trial with the measures used here. Takes the sensing and trajectory
        tables from agent_raycast.collect_matchup_tables, or from agent_raycast.collect_sensing_table and
        agent_loser_choice.collect_trajectories for self-play.

        Leaves out trials where either agent starts on a lit trigger box, as those end at once. '''

    # join on the session keys both tables carry (matchup for tournaments, model for self-play)
    keys = [k for k in ('matchup', 'model', 'trial_num', 'player') if k in sense.columns and k in traj.columns]
    t = sense.merge(traj[keys + ['xy', 'yaw']], on=keys)
    lit = on_lit_trigger(t.x0, t.z0, t.high_wall, t.low_wall) | on_lit_trigger(t.opp_x0, t.opp_z0, t.high_wall, t.low_wall)
    t = t[~lit].copy()
    t['own'] = own_alcove(t.x0, t.z0)
    t['opp_own'] = own_alcove(t.opp_x0, t.opp_z0)
    t['kh'], t['kl'] = position_cell(t.high_wall.values, t.low_wall.values, t.own.values)
    t['opp_d_high'] = np.hypot(t.opp_x0 - ALCOVE_CENTRES[t.high_wall - 1, 0], t.opp_z0 - ALCOVE_CENTRES[t.high_wall - 1, 1])
    t['opp_seen0'] = t['opp_first'] == 0
    # take which lit wall the rays sense at onset from the sensing table, where it was categorised with
    # trajectory_headangle.wall_visibility_player_slice_onset
    if 'high_seen0' not in t.columns:
        t['high_seen0'] = [r[f'ray_first_{h}'] == 0 for h, (_, r) in zip(t.high_wall, t.iterrows())]
        t['low_seen0'] = [r[f'ray_first_{l}'] == 0 for l, (_, r) in zip(t.low_wall, t.iterrows())]
    xy0 = np.stack([t.x0.values, t.z0.values], axis=1)
    yaw0 = np.array([y[0] for y in t.yaw])
    t['face_high'] = facing_angle(yaw0, xy0, t.high_wall.values)
    t['face_low'] = facing_angle(yaw0, xy0, t.low_wall.values)
    t['heading_high'] = [heading_high_at(r.xy, r.high_wall, r.low_wall) for r in t.itertuples()]
    return t.drop(columns=['xy', 'yaw'])


def mh_odds_ratio(df, exposed, outcome, strata):
    ''' Return the Mantel-Haenszel odds ratio of `outcome` for exposed against unexposed rows, pooled
        over strata, and the number of exposed rows in strata that hold both groups.
        Takes a table, a boolean exposure Series, the boolean outcome column and the strata columns. '''

    num = den = 0.0
    used = 0
    for _, g in df.groupby(strata):
        e = exposed.loc[g.index].astype(bool)
        y = g[outcome].astype(bool)
        if e.all() or (~e).all():
            continue
        n = len(g)
        num += (y & e).sum() * (~y & ~e).sum() / n
        den += (~y & e).sum() * (y & ~e).sum() / n
        used += int(e.sum())
    return (num / den if den > 0 else np.nan), used


def bootstrap_or(df, exposed, outcome, strata, n_boot=500, seed=0):
    ''' Return the 2.5 and 97.5 percentiles of the Mantel-Haenszel odds ratio over row resamples. '''

    rng = np.random.default_rng(seed)
    idx = df.index.values
    out = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx))
        d = df.loc[s].reset_index(drop=True)
        out.append(mh_odds_ratio(d, pd.Series(exposed.loc[s].values), outcome, strata)[0])
    return np.nanpercentile(out, [2.5, 97.5])


def within_strata_difference(df, exposed, col, strata):
    ''' Return the mean of (exposed - unexposed) in `col` within strata, weighted by exposed rows. '''

    tot = w = 0.0
    for _, g in df.groupby(strata):
        e = exposed.loc[g.index].astype(bool)
        if e.all() or (~e).all():
            continue
        tot += e.sum() * (g.loc[e, col].astype(float).mean() - g.loc[~e, col].astype(float).mean())
        w += e.sum()
    return tot / w if w else np.nan
