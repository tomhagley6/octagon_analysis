''' Model a race between two agents from arena geometry alone, to ask how wall choice sets score.

Each agent's time to a wall is a fixed start-up delay, plus the shortest path to that wall's
trigger at a fixed speed, plus an optional timing residual. The faster agent wins and scores
1.0 at High or 0.4 at Low. Every High and Low layout is listed with its exact probability, so
results are expectations, not samples.

Method:
  1. Build the space an agent's centre can reach (octagon with eight alcoves, from globals).
  2. Take the shortest path from a start point to each wall's trigger. A path may bend only
     at the alcove mouth corners, so these 16 corners are the only turning points.
  3. List the 40 layouts (High on any of 8 walls, Low 1, 2 or 4 walls away with weights
     0.5/0.25/0.25 and a random side), as drawn in OctagonArenaSettings.cs.
  4. For every pair of start points and every layout, score the race under a wall-choice
     policy for the focal agent. The opponent goes High with a fixed probability.

Assumptions:
  - Both agents move at the same speed, with the same start-up delay and residual spread.
  - Neither agent blocks the other, and neither changes behaviour for a different opponent.
  - Perception is ignored: each agent knows both lit walls at slice onset.
  - An agent whose body overlaps a lit trigger at slice onset wins that wall at once.
'''

import contextlib
import dataclasses
import io
import math

import numpy as np
import pandas as pd

import globals
import analysis.agent_raycast as agent_raycast
import data_extraction.get_indices as get_indices
import parse_data.prepare_sim_data as prepare_sim_data
import trajectory_analysis.visibility_occlusion as vo

HIGH_SCORE, LOW_SCORE = 1.0, 0.4
NUM_WALLS = globals.NUM_WALLS
SEPARATIONS = {1: 0.5, 2: 0.25, 4: 0.25}     # wall separation weights, GeneralGlobals.cs

W = globals.OCTAGON_WALLFACE_RADIUS
H = globals.ALCOVE_HALF_WIDTH
BACK = globals.OCTAGON_ALCOVE_RADIUS
# the agent centre fires the trigger when the body first overlaps the box front
TRIGGER_REACH = BACK + globals.PLAYER_RADIUS - agent_raycast.TRIGGER_DEPTH - globals.PLAYER_RADIUS
EPS = 0.05                                     # tolerance for points on or just past the boundary


@dataclasses.dataclass
class Timing:
    ''' Hold the timing model shared by both agents. delay and overhead are seconds, speed is
        units per second, resid is a sample of timing residuals (seconds). '''

    delay: float
    speed: float
    overhead: float
    resid: np.ndarray


# geometry

def _pt(p, q, wall):
    ''' Return a wall-frame point as arena x, z. Takes p, q and a wall row index. '''

    return np.array(vo.wall_frame_to_arena(p, q, wall))


def reach_polygon():
    ''' Return the reachable region as a clockwise polygon (40 x 2). Takes no inputs. '''

    L = W * math.tan(math.radians(22.5))
    # take each wall's vertex, mouth corner, two back corners and other mouth corner
    return np.array([_pt(p, q, w) for w in range(NUM_WALLS)
                     for p, q in ((W, -L), (W, -H), (BACK, -H), (BACK, H), (W, H))])


def mouth_corners():
    ''' Return the 16 alcove mouth corners (the only turning points of a shortest path).
        Takes no inputs. Row 2w is wall w's counterclockwise corner, row 2w + 1 its clockwise one. '''

    return np.array([_pt(W, q, w) for w in range(NUM_WALLS) for q in (-H, H)])


def _boundary_dist(poly, pts):
    ''' Return each point's distance to the polygon boundary. Takes the polygon and points (n x 2). '''

    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((pts[:, None] - a) * ab).sum(-1) / (ab ** 2).sum(-1), 0, 1)
    return np.linalg.norm(pts[:, None] - (a + t[..., None] * ab), axis=-1).min(1)


def inside(poly, pts, eps=EPS):
    ''' Return whether points lie inside the polygon, counting points within eps of the boundary.
        Takes the polygon and one point or an array of points. '''

    pts = np.atleast_2d(pts)
    x, z = pts[:, 0:1], pts[:, 1:2]
    a, b = poly, np.roll(poly, -1, axis=0)
    # count edge crossings of a ray running in +x
    cross = ((a[:, 1] > z) != (b[:, 1] > z)) & \
            (x < a[:, 0] + (z - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1] + 1e-300))
    res = (cross.sum(1) % 2 == 1) | (_boundary_dist(poly, pts) < eps)
    return res if len(res) > 1 else bool(res[0])


def _visible(poly, a, b, n_samples=24):
    ''' Return whether the straight segment a-b stays in the reachable region.
        Takes the polygon and two points. '''

    # reject a segment that properly crosses any edge
    c, d = poly, np.roll(poly, -1, axis=0)
    cr = lambda o, p, q: (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])
    d1, d2 = cr(a, b, c), cr(a, b, d)
    d3, d4 = cr(c, d, a), cr(c, d, b)
    tol = 1e-9
    if np.any((d1 * d2 < -tol) & (d3 * d4 < -tol)):
        return False
    # check points along the segment, which catches a pass through a convex vertex
    s = np.linspace(0, 1, n_samples + 2)[1:-1, None]
    return bool(np.all(inside(poly, a + s * (b - a), eps=1e-6)))


def _goal_point(pt, wall):
    ''' Return the nearest point on a wall's trigger line to pt. Takes a point and a wall row index. '''

    _, q = vo.arena_to_wall_frame(pt, wall)
    return _pt(TRIGGER_REACH, float(np.clip(q, -H, H)), wall)


def _in_trigger(pt, wall):
    ''' Return whether an agent centred at pt overlaps a wall's trigger. Takes a point and a wall row index. '''

    p, q = vo.arena_to_wall_frame(pt, wall)
    return bool(p >= TRIGGER_REACH - 1e-9 and abs(q) <= H + EPS)


def _corner_to_goal():
    ''' Return the shortest path from every mouth corner to every wall's trigger (16 x 8).
        Takes no inputs. '''

    poly, cs = reach_polygon(), mouth_corners()
    n = len(cs)
    # get corner-to-corner distances over visible links, then all shortest paths
    cc = np.full((n, n), np.inf)
    for i in range(n):
        cc[i, i] = 0.0
        for j in range(i + 1, n):
            if _visible(poly, cs[i], cs[j]):
                cc[i, j] = cc[j, i] = np.linalg.norm(cs[i] - cs[j])
    for k in range(n):
        cc = np.minimum(cc, cc[:, k:k + 1] + cc[k:k + 1, :])
    # get each corner's direct link to each trigger, then route through the corner graph
    direct = np.full((n, NUM_WALLS), np.inf)
    for i in range(n):
        for w in range(NUM_WALLS):
            g = _goal_point(cs[i], w)
            if _visible(poly, cs[i], g):
                direct[i, w] = np.linalg.norm(cs[i] - g)
    return (cc[:, :, None] + direct[None, :, :]).min(1)


_CORNER_GOAL = None


def _snap_inside(poly, pt, margin=0.02):
    ''' Return pt moved just inside the polygon if it lies outside it. Takes the polygon and a
        point. Logged positions can sit up to ~0.08 units past the polygon, where the real
        boundary is rounded by the body. A point on the boundary is kept. '''

    if _boundary_dist(poly, pt[None])[0] < 1e-9 or inside(poly, pt, eps=0):
        return pt
    # try points around pt at growing distance, and keep the nearest one clear of the boundary
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    for r in np.arange(0.01, 0.51, 0.01):
        cand = pt + r * np.c_[np.sin(ang), np.cos(ang)]
        ok = inside(poly, cand, eps=0) & (_boundary_dist(poly, cand) >= margin)
        if ok.any():
            return cand[np.argmax(ok)]
    raise ValueError(f"point {pt} is more than 0.5 units outside the reachable space")


def path_lengths(points):
    ''' Return the shortest path length from each point to each wall's trigger (n x 8, columns are
        wall row indices). Zero if the agent already overlaps that trigger. Takes points (n x 2). '''

    global _CORNER_GOAL
    if _CORNER_GOAL is None:
        _CORNER_GOAL = _corner_to_goal()
    poly, cs = reach_polygon(), mouth_corners()
    points = np.atleast_2d(np.asarray(points, float))
    out = np.full((len(points), NUM_WALLS), np.inf)
    for i, pt in enumerate(points):
        start = pt
        pt = _snap_inside(poly, pt)
        # find the corners in sight, as the first turning point of any bent path
        seen = np.array([_visible(poly, pt, c) for c in cs])
        via = np.where(seen, np.linalg.norm(cs - pt, axis=1), np.inf)[:, None] + _CORNER_GOAL
        for w in range(NUM_WALLS):
            if _in_trigger(start, w):
                out[i, w] = 0.0
                continue
            g = _goal_point(pt, w)
            direct = np.linalg.norm(pt - g) if _visible(poly, pt, g) else np.inf
            out[i, w] = min(direct, via[:, w].min())
    return out


def path_points(pt, wall):
    ''' Return the corner points of the shortest path from pt to a wall's trigger (k x 2).
        Takes a start point and a wall row index. '''

    poly, cs = reach_polygon(), mouth_corners()
    if _in_trigger(np.asarray(pt, float), wall):
        return np.array([pt], float)
    pts = [_snap_inside(poly, np.asarray(pt, float))]
    while True:
        cur = pts[-1]
        if _in_trigger(cur, wall):
            return np.array(pts)
        g = _goal_point(cur, wall)
        if _visible(poly, cur, g):
            return np.array(pts + [g])
        # step to the visible corner that starts the shortest remaining path
        d = path_lengths(cur[None])[0, wall]
        cost = [np.linalg.norm(c - cur) + _CORNER_GOAL[i, wall]
                if np.linalg.norm(c - cur) > 1e-9 and _visible(poly, cur, c) else np.inf
                for i, c in enumerate(cs)]
        i = int(np.argmin(cost))
        assert abs(cost[i] - d) < 1e-6
        pts.append(cs[i])


def alcove_back_positions():
    ''' Return the centre of each alcove back (8 x 2), where peripheral agents wait. Takes no inputs. '''

    return np.array([_pt(BACK, 0.0, w) for w in range(NUM_WALLS)])


def nearest_wall(points):
    ''' Return the wall row index whose direction is closest to each point. Takes points (n x 2). '''

    ang = np.degrees(np.arctan2(points[:, 0], points[:, 1])) % 360
    return np.round(ang / 45).astype(int) % NUM_WALLS


def signed_offset(k):
    ''' Return a wall offset folded to -3..4 (positive = clockwise). Takes an offset in walls. '''

    return (np.asarray(k) + 3) % NUM_WALLS - 3


# layouts and races

def layout_table():
    ''' Return the 40 High/Low layouts with their probabilities. Takes no inputs.
        high and low are wall row indices. '''

    rows = []
    for high in range(NUM_WALLS):
        for sep, w_sep in SEPARATIONS.items():
            # a 4-wall separation lands on the same wall from either side
            signs = (1,) if sep == 4 else (1, -1)
            for s in signs:
                rows.append(dict(high=high, low=(high + s * sep) % NUM_WALLS, sep=sep,
                                 prob=w_sep / NUM_WALLS / len(signs)))
    return pd.DataFrame(rows)


def build_races(focal_pos, opp_pos):
    ''' Return one row per focal start x opponent start x layout, with both agents' path lengths
        to High and Low and the row's probability. Takes focal and opponent start points (n x 2). '''

    focal_pos, opp_pos = np.atleast_2d(focal_pos), np.atleast_2d(opp_pos)
    df_, do_ = path_lengths(focal_pos), path_lengths(opp_pos)
    lay = layout_table()
    fi, oi, li = np.meshgrid(np.arange(len(focal_pos)), np.arange(len(opp_pos)), np.arange(len(lay)), indexing='ij')
    fi, oi, li = fi.ravel(), oi.ravel(), li.ravel()
    high, low = lay['high'].to_numpy()[li], lay['low'].to_numpy()[li]
    home = nearest_wall(focal_pos)[fi]
    return pd.DataFrame(dict(
        focal=fi, opp=oi, high=high, low=low, sep=lay['sep'].to_numpy()[li],
        high_off=signed_offset(high - home), low_off=signed_offset(low - home),
        focal_d_high=df_[fi, high], focal_d_low=df_[fi, low],
        opp_d_high=do_[oi, high], opp_d_low=do_[oi, low],
        prob=lay['prob'].to_numpy()[li] / (len(focal_pos) * len(opp_pos))))


def _diff_table(resid):
    ''' Return the sorted pairwise residual differences (focal minus opponent) and their prefix sums.
        Takes a residual sample. '''

    resid = np.asarray(resid, float)
    d = np.sort((resid[:, None] - resid[None, :]).ravel())
    return d, np.concatenate([[0.0], np.cumsum(d)])


def _pair_outcome(df, do, timing, diffs, focal_extra_s=0.0):
    ''' Return the focal win probability and expected race time for given path lengths.
        Takes focal and opponent path lengths (arrays), the timing model, the residual differences
        and an extra start-up delay for the focal agent (seconds).

        A zero path length is an instant win at time 0, with no delay. Ties are split evenly. '''

    d, csum = diffs
    n = len(d)
    f0, o0 = df == 0, do == 0
    # focal is first when its residual difference falls below the gap in travel time
    gap = (do - df) / timing.speed - focal_extra_s
    lo, hi = np.searchsorted(d, gap, 'left'), np.searchsorted(d, gap, 'right')
    win = (lo + 0.5 * (hi - lo)) / n
    # take the faster arrival, E[min] = opponent time - E[max(0, gap - D)]
    t_opp = timing.delay + do / timing.speed + np.mean(timing.resid)
    race = t_opp - (lo * gap - csum[lo]) / n
    win = np.where(f0 & o0, 0.5, np.where(f0, 1.0, np.where(o0, 0.0, win)))
    race = np.where(f0 | o0, 0.0, race)
    return win, race


def _choice_outcomes(races, timing, opp_p_high, diffs, focal_extra_s=0.0):
    ''' Return the expected focal score, opponent score and race time for each focal choice.
        Takes the race rows, timing model, opponent High probability, residual differences and the
        focal agent's extra delay.
        Returns a dict keyed by 'high' and 'low', each a dict of row arrays. '''

    fH, fL = races['focal_d_high'].to_numpy(), races['focal_d_low'].to_numpy()
    oH, oL = races['opp_d_high'].to_numpy(), races['opp_d_low'].to_numpy()

    def effective(dH, dL, go_low):
        # an agent standing on a lit trigger takes that wall at once, whatever it intended
        d = np.where(go_low, dL, dH)
        val = np.where(go_low, LOW_SCORE, HIGH_SCORE)
        d = np.where(dH == 0, 0.0, np.where(dL == 0, 0.0, d))
        val = np.where(dH == 0, HIGH_SCORE, np.where(dL == 0, LOW_SCORE, val))
        return d, val

    out = {}
    for choice, go_low in (('high', False), ('low', True)):
        df, vf = effective(fH, fL, go_low)
        acc = dict(focal_win=0.0, focal_score=0.0, opp_score=0.0, race_s=0.0)
        for p_opp, opp_low in ((opp_p_high, False), (1 - opp_p_high, True)):
            if p_opp == 0:
                continue
            do, vo_ = effective(oH, oL, opp_low)
            win, race = _pair_outcome(df, do, timing, diffs, focal_extra_s)
            acc['focal_win'] = acc['focal_win'] + p_opp * win
            acc['focal_score'] = acc['focal_score'] + p_opp * win * vf
            acc['opp_score'] = acc['opp_score'] + p_opp * (1 - win) * vo_
            acc['race_s'] = acc['race_s'] + p_opp * race
        out[choice] = acc
    return out


def _go_low(races, policy, outcomes, timing, w):
    ''' Return each row's probability that the focal agent heads for Low.
        Takes the race rows, the policy, the per-choice outcomes, the timing model and row weights. '''

    # count Low as nearer only beyond rounding, so mirror-image ties go High
    lower = (races['focal_d_low'] < races['focal_d_high'] - 1e-9).to_numpy(float)
    if isinstance(policy, str):
        if policy == 'high':
            return np.zeros(len(races))
        if policy == 'nearer':
            return lower
        if policy == 'best_trial':
            return (outcomes['low']['focal_score'] > outcomes['high']['focal_score']).astype(float)
        if policy == 'best_rate':
            # find the rate-maximising choice by Dinkelbach iteration on score - rate x time
            S = {c: outcomes[c]['focal_score'] for c in ('high', 'low')}
            T = {c: timing.overhead + outcomes[c]['race_s'] for c in ('high', 'low')}
            g = np.zeros(len(races))
            for _ in range(100):
                rate = (w * (g * S['low'] + (1 - g) * S['high'])).sum() / \
                       (w * (g * T['low'] + (1 - g) * T['high'])).sum()
                g_new = ((S['low'] - rate * T['low']) > (S['high'] - rate * T['high']) + 1e-12).astype(float)
                if np.array_equal(g_new, g):
                    break
                g = g_new
            return g
        raise ValueError(f"unknown policy {policy}")
    if np.ndim(policy) == 0:
        # take Low with probability q whenever Low is nearer
        return float(policy) * lower
    return np.asarray(policy, float)


def evaluate(races, timing, policy='high', opp_p_high=1.0, focal_extra_s=0.0):
    ''' Return the expected outcome of each race row under a focal wall-choice policy.
        Takes the race rows, timing model, policy, the opponent's High probability and an extra
        start-up delay for the focal agent (seconds, not applied to instant wins).

        policy is 'high', 'nearer', 'best_trial' (most score per trial), 'best_rate' (most score
        per second), a number q (take Low with probability q when Low is nearer), or an array
        giving each row's Low probability. '''

    diffs = _diff_table(timing.resid)
    outcomes = _choice_outcomes(races, timing, opp_p_high, diffs, focal_extra_s)
    g = _go_low(races, policy, outcomes, timing, races['prob'].to_numpy())
    res = races.copy()
    res['go_low'] = g
    for k in ('focal_win', 'focal_score', 'opp_score', 'race_s'):
        res[k] = g * outcomes['low'][k] + (1 - g) * outcomes['high'][k]
    res['trial_s'] = timing.overhead + res['race_s']
    return res


def race_summary(races, timing, policy='high', opp_p_high=1.0, focal_extra_s=0.0):
    ''' Return the probability-weighted summary of a policy over all race rows.
        Takes the race rows, timing model, policy, the opponent's High probability and the focal
        agent's extra delay.

        Score per second is total expected score over total expected trial time. score_s uses the
        whole trial (overhead included), score_s_race only slice onset to trigger. '''

    r = evaluate(races, timing, policy, opp_p_high, focal_extra_s)
    w = r['prob'].to_numpy()
    tot = lambda c: float((w * r[c]).sum())
    return dict(focal_win=tot('focal_win'), focal_score_trial=tot('focal_score'),
                opp_score_trial=tot('opp_score'), race_s=tot('race_s'), trial_s=tot('trial_s'),
                focal_score_s=tot('focal_score') / tot('trial_s'),
                opp_score_s=tot('opp_score') / tot('trial_s'),
                focal_score_s_race=tot('focal_score') / tot('race_s') if tot('race_s') > 0 else np.inf,
                p_go_low=tot('go_low'))


# timing from the logs

def collect_trial_timing(sim_data_folder, models, verbose=True):
    ''' Return one row per trial with the slice-onset-to-trigger time, the whole trial length, the
        winner's depth into the alcove at the trigger, and the agents' moving speeds. Takes a batch's simulation folder and the model list. Drops the first logged trial. '''

    rows = []
    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            _, trial_list = prepare_sim_data.prepare_single_model_data(model, sim_data_folder=sim_data_folder)
        for trial in trial_list[1:]:
            ev = trial['eventDescription']
            onset, trig = trial[ev == globals.SLICE_ONSET], trial[ev == globals.SELECTED_TRIGGER_ACTIVATION]
            if len(onset) == 0 or len(trig) == 0:
                continue
            t = trial['timeApplication']
            # get the winner's depth into the chosen wall's alcove at the trigger
            k = globals.PLAYER_LOC_DICT
            winner, chosen = get_indices.get_trigger_activator(trial), int(get_indices.get_chosen_walls([trial])[0])
            end = np.array([trig[k[winner]['xloc']].iloc[0], trig[k[winner]['yloc']].iloc[0]], dtype=float)
            trigger_p, _ = vo.arena_to_wall_frame(end, chosen - 1)
            # get both agents' frame-to-frame speeds between slice onset and the trigger
            seg = trial.loc[onset.index[0]:trig.index[0]]
            dt = seg['timeApplication'].diff().dt.total_seconds().to_numpy()[1:]
            speeds = []
            for pid in (0, 1):
                step = np.hypot(np.diff(seg[k[pid]['xloc']].to_numpy(float)), np.diff(seg[k[pid]['yloc']].to_numpy(float)))
                speeds.append(step[dt > 0] / dt[dt > 0])
            speeds = np.concatenate(speeds)
            moving = speeds[speeds > 0.5]
            rows.append(dict(model=model, trial_num=float(trial[globals.TRIAL_NUM].dropna().iloc[0]),
                             rt_s=(trig['timeApplication'].iloc[0] - onset['timeApplication'].iloc[0]).total_seconds(),
                             trial_s=(t.iloc[-1] - t.iloc[0]).total_seconds(), trigger_p=float(trigger_p),
                             speed_median=float(np.median(moving)) if len(moving) else np.nan,
                             speed_p99=float(np.percentile(moving, 99)) if len(moving) else np.nan))
        if verbose:
            print(f"model {model} done", end='  ')
    out = pd.DataFrame(rows)
    out['overhead_s'] = out['trial_s'] - out['rt_s']
    return out


def fit_timing(won_rows, speed, overhead, min_rt=0.05):
    ''' Return the timing model from won trials, with speed fixed. Takes race-table rows of won
        trials (x, z, chosen_wall as wall id, rt_s), the speed and the per-trial overhead.

        delay is the median of trial time less travel time along the shortest path. resid is each
        trial's remainder. Instant wins (rt under min_rt) are left out. '''

    won_rows = won_rows[won_rows['rt_s'] >= min_rt]
    d = path_lengths(won_rows[['x', 'z']].to_numpy(float))
    d_chosen = d[np.arange(len(won_rows)), won_rows['chosen_wall'].to_numpy(int) - 1]
    lag = won_rows['rt_s'].to_numpy(float) - d_chosen / speed
    delay = float(np.median(lag))
    return Timing(delay=delay, speed=speed, overhead=overhead, resid=lag - delay), d_chosen
