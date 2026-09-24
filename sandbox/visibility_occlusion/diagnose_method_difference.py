# Find why get_wall_visible(occlusion=True) differs from the angular method on frames where
# nothing blocks sight (eye outside every alcove). Changes one thing at a time between them:
#
#   S0  angular method as implemented (get_wall_visible, occlusion=False)
#   S1  copy of the angular rule, reachable geometry (must equal S0)
#   S2  angular rule, physical geometry
#   S3  test both corners of each wall, not only the one picked by the alcove-centre side test
#   S4  any point on the whole wall, corner to corner (exact)
#   S5  any point on the coloured faces only, excluding the alcove mouth (exact)
#   S6  occlusion implementation (exact)
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python sandbox/visibility_occlusion/diagnose_method_difference.py

import sys, os, math, contextlib, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import globals, data_strings
from plotting import plot_octagon
from parse_data import prepare_sim_data
from trajectory_analysis import trajectory_headangle as th, trajectory_vectors as tv

FOLDER = os.path.join(data_strings.SIM_DATA_FOLDER, '2607', '260723')
MODELS = {23: 'peripheral', 34: 'central'}
N_TRIALS = 60
FOV = 110

R = globals.PLAYER_RADIUS
W_REACH = globals.OCTAGON_WALLFACE_RADIUS
W_PHYS = W_REACH + R
T22 = math.tan(math.radians(22.5))


def wall_corners(wall_face, half_width):
    ''' Return per-wall points in the wall frame converted to arena x, z: CCW vertex, CCW alcove edge,
        alcove centre on the face line, CW alcove edge, CW vertex. Shape num_walls*5*2.
        Takes the wall-face distance and alcove half-width. '''

    L = wall_face * T22
    pq = [(wall_face, -L), (wall_face, -half_width), (wall_face, 0.0), (wall_face, half_width), (wall_face, L)]
    out = np.zeros((8, 5, 2))
    for k in range(8):
        a = math.radians(45 * k)
        for j, (p, q) in enumerate(pq):
            out[k, j] = (p * math.sin(a) + q * math.cos(a), p * math.cos(a) - q * math.sin(a))
    return out


def angle_to(eyes, headings, points):
    ''' Return angles (degrees) between headings and eye-to-point vectors, shape num_points*timepoints.
        Takes eyes (T*2), unit headings (T*2) and points (num_points*2). '''

    d = points[:, None, :] - eyes[None, :, :]
    cos = np.einsum('ntk,tk->nt', d, headings) / np.linalg.norm(d, axis=2)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def cross(a, b):
    ''' Return the 2D cross product (z component). Takes arrays with x, z in the last axis. '''

    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def angular_rule(eyes, headings, vertices_ccw, vertices_cw, alcove_centres, both_corners=False):
    ''' Return num_walls*T visibility from the angular rule. Picks the corner on the heading side using
        the alcove-centre direction (or tests both corners). Takes eyes, unit headings, per-wall CCW and CW
        vertices (8*2), alcove centres (8*2) and whether to test both corners. '''

    ang_ccw = angle_to(eyes, headings, vertices_ccw)
    ang_cw = angle_to(eyes, headings, vertices_cw)
    if both_corners:
        return np.minimum(ang_ccw, ang_cw) < FOV / 2

    # wall is clockwise of the heading when cross(heading, to_alcove) < 0, then its CCW corner is nearer
    to_alcove = alcove_centres[:, None, :] - eyes[None, :, :]
    wall_is_cw = cross(headings[None, :, :], to_alcove) < 0
    return np.where(wall_is_cw, ang_ccw, ang_cw) < FOV / 2


def segment_in_view(eyes, headings, start, end):
    ''' Return num_segments*T True where any point of a segment is within FoV/2 of the heading (exact).
        Takes eyes, unit headings and segment start and end points (num_segments*2 each). '''

    a_start = angle_to(eyes, headings, start)
    a_end = angle_to(eyes, headings, end)
    # the heading ray passes between the endpoints when they sit on opposite sides of it and in front
    ds = start[:, None, :] - eyes[None, :, :]
    de = end[:, None, :] - eyes[None, :, :]
    h = headings[None, :, :]
    between = (np.sign(cross(h, ds)) != np.sign(cross(h, de))) & (cross(ds, de) * cross(ds, h) > 0)
    return between | (np.minimum(a_start, a_end) < FOV / 2)


def main():
    # load frames outside every alcove, with the angular method's own output
    rows = []
    for model, strategy in MODELS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            _, trial_list = prepare_sim_data.prepare_single_model_data(model, sim_data_folder=FOLDER)
        for trial in trial_list[1:N_TRIALS + 1]:
            for pid in (0, 1):
                s0 = th.get_wall_visible(trial=trial, player_id=pid, current_fov=FOV, occlusion=False)
                if isinstance(s0, float):
                    continue
                s6 = th.get_wall_visible(trial=trial, player_id=pid, current_fov=FOV, occlusion=True)
                traj = tv.extract_trial_player_trajectory(trial=trial, player_id=pid)
                sm = th.get_smoothed_player_head_angle_vectors_for_trial(tv.extract_trial_player_headangles(trial=trial, player_id=pid))
                n = sm.shape[1]
                rows.append(dict(model=model, strategy=strategy, eyes=traj[:, :n].T, headings=(sm / np.linalg.norm(sm, axis=0)).T,
                                 s0=s0, s6=s6))

    reach = wall_corners(W_REACH, globals.ALCOVE_HALF_WIDTH)
    phys = wall_corners(W_PHYS, globals.ALCOVE_HALF_WIDTH + R)
    reach_alcove_centres = plot_octagon.return_alcove_centre_points().T          # 8*2, north first
    phys_alcove_centres = reach_alcove_centres * (W_PHYS + (globals.ALCOVE_DEPTH / 2)) / np.linalg.norm(reach_alcove_centres, axis=1, keepdims=True)

    # check the copied vertices and alcove centres match the angular method's own geometry
    old_vertices = th.get_octagon_vertex_coordinates().T
    assert np.allclose(old_vertices, reach[:, 0]), 'CCW vertices differ from get_octagon_vertex_coordinates'

    steps = {k: [] for k in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']}
    meta = []
    for r in rows:
        eyes, heads = r['eyes'], r['headings']
        # keep frames outside every physical alcove
        p = np.stack([eyes[:, 0] * math.sin(math.radians(45 * k)) + eyes[:, 1] * math.cos(math.radians(45 * k)) for k in range(8)])
        keep = p.max(axis=0) <= W_PHYS
        e, h = eyes[keep], heads[keep]
        steps['S0'].append(r['s0'][:, keep])
        steps['S1'].append(angular_rule(e, h, reach[:, 0], reach[:, 4], reach_alcove_centres))
        steps['S2'].append(angular_rule(e, h, phys[:, 0], phys[:, 4], phys_alcove_centres))
        steps['S3'].append(angular_rule(e, h, phys[:, 0], phys[:, 4], phys_alcove_centres, both_corners=True))
        steps['S4'].append(segment_in_view(e, h, phys[:, 0], phys[:, 4]))
        steps['S5'].append(segment_in_view(e, h, phys[:, 0], phys[:, 1]) | segment_in_view(e, h, phys[:, 3], phys[:, 4]))
        steps['S6'].append(r['s6'][:, keep])
        # record, per wall and frame, distance from the wall face line and model
        meta.append(dict(strategy=np.full(keep.sum(), r['strategy']), dist=(W_PHYS - p[:, keep])))

    S = {k: np.concatenate(v, axis=1) for k, v in steps.items()}
    strategy = np.concatenate([m['strategy'] for m in meta])
    dist = np.concatenate([m['dist'] for m in meta], axis=1)
    total = S['S0'].size
    print(f"frames outside alcoves: {S['S0'].shape[1]}  (wall x frame values: {total})\n")

    print('step-by-step change (wall x frame values that change between consecutive steps)')
    order = list(S)
    for a, b in zip(order[:-1], order[1:]):
        gained = (~S[a] & S[b]).sum(); lost = (S[a] & ~S[b]).sum()
        print(f'  {a} -> {b}: {gained + lost:6d} ({(gained + lost) / total:.3%})   became visible {gained:6d}   became hidden {lost:6d}')
    print(f"\n  S0 -> S6 overall: {(S['S0'] != S['S6']).sum()} ({(S['S0'] != S['S6']).mean():.3%})")
    for strat in MODELS.values():
        m = strategy == strat
        print(f"    {strat:10s} S0 vs S6: {(S['S0'][:, m] != S['S6'][:, m]).mean():.3%}")

    # describe the largest step: where do the S3 -> S4 changes happen
    for a, b in [('S2', 'S3'), ('S3', 'S4'), ('S4', 'S5')]:
        changed = S[a] != S[b]
        if changed.any():
            print(f'\n{a} -> {b} changes: distance from the eye to that wall face line  median {np.median(dist[changed]):.2f}, '
                  f'90th pct {np.percentile(dist[changed], 90):.2f}  (all wall-frames median {np.median(dist):.2f})')


if __name__ == '__main__':
    main()
