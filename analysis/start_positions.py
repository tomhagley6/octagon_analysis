''' Designed start positions and onset headings for the peripheral agent of the geometric race model.

The race model (analytical_race.py) otherwise starts the peripheral agent at logged positions and
headings of trained RL agents. Here the start is a design choice instead: a grid of positions over one
alcove and the floor in front of it, each with the onset heading that gives the highest expected score
per second under the final rule (delta2). Every heading on the 0.5 deg grid is scored at every position.

The arena is the same at every alcove and mirror-symmetric about each alcove's centre line, and the
layouts are uniform over High's wall, so the clockwise half of alcove 0's sector covers every position.
Positions are given in alcove 0's frame: p along the alcove's centre line from the arena centre, q to
the clockwise side (vo.arena_to_wall_frame).

The rule is the final delta2 rule, with the backpedal into the own trigger (analytical_race._motion_kind).
Sensing uses the agents' 7 rays (agent_raycast), kept for consistency with the rest of the model and
for comparison with the RL agents. Humans see a continuous field of view, so this should be rerun
with human vision in future.
'''

import concurrent.futures
import multiprocessing

import numpy as np
import pandas as pd

import analysis.analytical_race as ar
import trajectory_analysis.visibility_occlusion as vo

DELTAS = np.round(np.arange(0, 3.81, 0.05), 2)     # delta grid scored for every position and heading (s)
P_FRONT = np.round(np.arange(13.0, 16.51, 0.5), 2)  # rows in front of the wall face (units from the centre)
P_ALCOVE = np.array([17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 19.7])   # rows inside the alcove
Q_ALCOVE = np.array([0.0, 0.5, 1.0, 1.5, 1.85])     # lateral offsets inside the alcove (half-width 1.9)
Q_STEP = 0.5                                        # lateral spacing in front of the wall face
EDGE = 0.05                                         # margin kept from the walls and the sector edge (units)
MIRROR_WALLS = [0, 7, 6, 5, 4, 3, 2, 1]             # wall row index after mirroring across alcove 0's centre line
METRICS = ('periph_trial', 'central_trial', 'race_s', 'p_low', 'periph_win')


def designed_positions():
    ''' Return the designed start positions, one row each, in alcove 0's clockwise half-sector.
        Takes no inputs.

        Columns: p, q (alcove 0 frame), x, z (arena), region, in_trigger. Regions are 'alcove, in
        trigger' (overlapping alcove 0's trigger, so a lit own wall is won at once), 'alcove, not in
        trigger', 'in front of the wall' (open floor between the wall face and 13 units from the
        centre) and 'wall corner' (where walls 0 and 1 meet, on the line between them). '''

    tan = np.tan(np.radians(22.5))
    pts = []
    # take a grid inside the alcove
    for p in P_ALCOVE:
        for q in Q_ALCOVE:
            pts.append((p, q))
    # take a grid on the floor in front of the wall face, up to the line shared with wall 1
    for p in P_FRONT:
        for q in np.arange(0.0, p * tan - EDGE + 1e-9, Q_STEP):
            pts.append((p, round(q, 3)))
    # add the face line itself, from the alcove mouth to just short of the corner
    p_face = ar.W - EDGE
    for q in np.arange(ar.H + Q_STEP, p_face * tan - EDGE + 1e-9, Q_STEP):
        pts.append((p_face, round(q, 3)))
    pts.append((p_face, 0.0))
    # add the wall corner, just inside the vertex where walls 0 and 1 meet
    r_corner = ar.W / np.cos(np.radians(22.5)) - EDGE
    pts.append((r_corner * np.cos(np.radians(22.5)), r_corner * np.sin(np.radians(22.5))))

    p, q = np.array(pts).T
    x, z = vo.wall_frame_to_arena(p, q, 0)
    out = pd.DataFrame(dict(p=p, q=q, x=x, z=z))
    xy = out[['x', 'z']].to_numpy()
    out['in_trigger'] = [bool(ar._in_trigger(pt, 0)) for pt in xy]
    corner = np.isclose(np.degrees(np.arctan2(q, p)), 22.5, atol=1e-6)
    inside = p > ar.W + 1e-9
    out['region'] = np.where(corner, 'wall corner', np.where(inside & out['in_trigger'], 'alcove, in trigger',
                             np.where(inside, 'alcove, not in trigger', 'in front of the wall')))
    return out.reset_index(drop=True)


def position_races(pos, heading, rot_speed, close_speed, speed=ar.SPEED):
    ''' Return the race rows (one per layout and search direction) for one start position and onset
        heading, with the delta2 outcome pieces, in the form ar.evaluate reads. Takes the position,
        heading (degrees, on the 0.5 deg grid), the rotation and closing speeds and the speed. '''

    pos = np.asarray(pos, float)
    lay = ar.layout_table()
    S0 = ar.onset_sensed(pos[None], [heading])[0]
    rows, pieces = [], []
    for li in range(len(lay)):
        hi, lo = int(lay['high'].iat[li]), int(lay['low'].iat[li])
        for d in (1, -1):
            pcs = ar._delta2_pieces(pos, heading, hi, lo, d, S0, rot_speed, close_speed, speed)
            if len(pcs) > ar.MAX_PIECES:
                raise RuntimeError(f'{len(pcs)} delta2 pieces, more than MAX_PIECES')
            arr = np.full((ar.MAX_PIECES, 3), np.nan)
            arr[:len(pcs)] = pcs
            pieces.append(arr)
            # mark instant wins for evaluate (any positive time otherwise; delta2 times come from the pieces)
            rows.append(dict(pos=0, group=0, direction=d, high=hi, low=lo,
                             t_high=0.0 if ar._in_trigger(pos, hi) else 1.0,
                             t_low=0.0 if ar._in_trigger(pos, lo) else 1.0,
                             high_sensed0=bool(S0[hi]), low_sensed0=bool(S0[lo]),
                             t_central=ar.TRIGGER_REACH / speed, prob=float(lay['prob'].iat[li]) / 2))
    races = pd.DataFrame(rows)
    pc = np.array(pieces)
    d2 = pd.DataFrame({f'd2_{f}_{k}': pc[:, k, m] for k in range(ar.MAX_PIECES)
                       for m, f in enumerate(('hi', 't', 'low'))})
    return pd.concat([races, d2], axis=1)


def _summarise_pieces(HI, T, L, prob, off, central_t, deltas):
    ''' Return the expected outcome per trial at each delta for one position and heading (dict of
        arrays over deltas). Takes the race rows' delta2 pieces (upper delta bound, time, ends at Low),
        probabilities, Low offsets from High, the central times by Low offset and the deltas. '''

    out = {m: np.zeros(len(deltas)) for m in METRICS}
    idx = np.arange(len(prob))
    HIn = np.nan_to_num(HI, nan=-np.inf)
    for j, d in enumerate(deltas):
        # take each race's piece at this delta (a gap within GAP_TIE of delta goes High, as in delta2_outcome)
        k = (d + ar.GAP_TIE < HIn).argmax(axis=1)
        t_p, go_low = T[idx, k], L[idx, k] > 0.5
        val = np.where(go_low, ar.LOW_SCORE, ar.HIGH_SCORE)
        win, race = np.zeros(len(prob)), np.zeros(len(prob))
        for o, ct in central_t.items():
            m = off == o
            win[m], race[m] = ar._against_central(t_p[m], ct)
        out['periph_trial'][j] = (prob * win * val).sum()
        out['central_trial'][j] = (prob * (1 - win) * ar.HIGH_SCORE).sum()
        out['race_s'][j] = (prob * race).sum()
        out['p_low'][j] = (prob * go_low).sum()
        out['periph_win'][j] = (prob * win).sum()
    return out


def heading_scores(pos, headings, rot_speed, close_speed, central_t, deltas=DELTAS, speed=ar.SPEED):
    ''' Return the final rule's expected outcome for one position at each onset heading and delta.
        Takes the position, the headings (degrees, on the 0.5 deg grid), the rotation and closing
        speeds, the central times by Low offset (ar.central_times_inferred), the deltas and the speed.

        Returns a dict: sensed (headings x 8, walls hit at onset) and, for each of METRICS, an array
        (headings x deltas) of expected values per trial (score, race time, share going Low, win). '''

    pos = np.asarray(pos, float)
    lay = ar.layout_table()
    hi_l, lo_l = lay['high'].to_numpy(int), lay['low'].to_numpy(int)
    prob = np.repeat(lay['prob'].to_numpy() / 2, 2)
    off = np.repeat((lo_l - hi_l) % ar.NUM_WALLS, 2)
    sensed = ar.onset_sensed(np.repeat(pos[None], len(headings), axis=0), headings)
    out = {m: np.zeros((len(headings), len(deltas))) for m in METRICS}
    for i, h in enumerate(headings):
        pieces = np.full((len(prob), ar.MAX_PIECES, 3), np.nan)
        r = 0
        for hi, lo in zip(hi_l, lo_l):
            for d in (1, -1):
                pcs = ar._delta2_pieces(pos, float(h), int(hi), int(lo), d, sensed[i], rot_speed, close_speed, speed)
                if len(pcs) > ar.MAX_PIECES:
                    raise RuntimeError(f'{len(pcs)} delta2 pieces, more than MAX_PIECES')
                pieces[r, :len(pcs)] = pcs
                r += 1
        res = _summarise_pieces(pieces[..., 0], pieces[..., 1], pieces[..., 2], prob, off, central_t, deltas)
        for m in METRICS:
            out[m][i] = res[m]
        # free the traces of this heading (they are not reused at the next one)
        ar._TRACE_CACHE.clear()
    out['sensed'] = sensed
    return out


def _position_job(args):
    ''' Return heading_scores for one position (worker entry point). Takes a tuple of its arguments. '''

    pos, headings, rot_speed, close_speed, central_t, deltas = args
    res = heading_scores(pos, headings, rot_speed, close_speed, central_t, deltas)
    return {m: res[m].astype(np.float32) for m in METRICS} | {'sensed': res['sensed']}


def build_position_map(positions, rot_speed, close_speed, central_t, headings=None, deltas=DELTAS,
                       n_jobs=ar.N_JOBS, verbose=False):
    ''' Return the final rule's expected outcome for every designed position, onset heading and delta.
        Takes the positions table (designed_positions), the rotation and closing speeds, the central
        times by Low offset, the headings (default every 0.5 deg), the deltas, the number of worker
        processes and a progress flag.

        Returns a dict: positions, headings, deltas, sensed (positions x headings x 8) and each of
        METRICS (positions x headings x deltas, float32). '''

    headings = ar.heading_grid() if headings is None else np.asarray(headings, float)
    args = [(np.array([x, z]), headings, rot_speed, close_speed, central_t, deltas)
            for x, z in positions[['x', 'z']].to_numpy()]
    res = []
    if n_jobs > 1 and len(args) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs,
                                                    mp_context=multiprocessing.get_context('spawn')) as ex:
            for k, r in enumerate(ex.map(_position_job, args)):
                res.append(r)
                if verbose:
                    print(k, end=' ', flush=True)
    else:
        for k, a in enumerate(args):
            res.append(_position_job(a))
            if verbose:
                print(k, end=' ', flush=True)
    out = dict(positions=positions.reset_index(drop=True), headings=headings, deltas=np.asarray(deltas))
    for m in METRICS + ('sensed',):
        out[m] = np.stack([r[m] for r in res])
    return out


def best_per_position(res, overhead, delta=None):
    ''' Return one row per position with the onset heading (and delta) that give the highest peripheral
        score per second, and the outcome there. Takes a build_position_map result, the time outside the
        race per trial (s) and optionally a fixed delta (otherwise delta is chosen per position too).

        Columns: pos, heading, delta, periph_score_s, central_score_s, margin, the per-trial values,
        sensed walls at that heading (n_sensed and walls_sensed as row indices), and near_best_deg (how
        many degrees of heading come within 0.5% of the best score per second, with delta at its best
        for each heading). '''

    trial_s = overhead + res['race_s']
    ps = res['periph_trial'] / trial_s
    cs = res['central_trial'] / trial_s
    if delta is not None:
        j_fix = int(np.argmin(np.abs(res['deltas'] - delta)))
        mask = np.full(ps.shape, -np.inf)
        mask[..., j_fix] = 0.0
        ps_sel = ps + mask
    else:
        ps_sel = ps
    rows = []
    for i in range(ps.shape[0]):
        flat = int(np.argmax(ps_sel[i]))
        h_i, d_i = np.unravel_index(flat, ps_sel[i].shape)
        per_head = ps_sel[i].max(axis=1)
        sens = res['sensed'][i, h_i]
        rows.append(dict(pos=i, heading=float(res['headings'][h_i]), delta=float(res['deltas'][d_i]),
                         periph_score_s=float(ps[i, h_i, d_i]), central_score_s=float(cs[i, h_i, d_i]),
                         margin=float(ps[i, h_i, d_i] - cs[i, h_i, d_i]),
                         periph_trial=float(res['periph_trial'][i, h_i, d_i]),
                         central_trial=float(res['central_trial'][i, h_i, d_i]),
                         race_s=float(res['race_s'][i, h_i, d_i]), p_low=float(res['p_low'][i, h_i, d_i]),
                         periph_win=float(res['periph_win'][i, h_i, d_i]),
                         n_sensed=int(sens.sum()), walls_sensed=tuple(int(w) for w in np.flatnonzero(sens)),
                         near_best_deg=float((per_head >= 0.995 * per_head.max()).sum() * ar.HEAD_STEP)))
    return pd.DataFrame(rows)
