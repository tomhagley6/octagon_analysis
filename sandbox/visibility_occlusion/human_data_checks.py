# Check occlusion-aware wall visibility on human data (RAW/standard pickles in data_strings.DATAFRAME_PATH).
#   1. geometry: reachable wall face and alcove back measured from human positions match globals
#   2. sweep: get_wall_visible with occlusion on and off for every trial (social both players, solo player 0).
#      Counts errors, matching shapes and too-short trials, timing, and how often the result changes
#   3. exactness: occlusion result contains the sampled reference (tests/test_visibility_occlusion.py)
#      and converges to it, on a subset of sessions
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python sandbox/visibility_occlusion/human_data_checks.py

import sys, os, math, time, contextlib, io, traceback
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
import numpy as np
import pandas as pd
import globals
import data_extraction.data_saving as data_saving
from trajectory_analysis import trajectory_headangle as th, trajectory_vectors as tv
from trajectory_analysis import visibility_occlusion as vo
import test_visibility_occlusion as T

FOV = 110.36
REFERENCE_SESSIONS = 5            # social sessions used for the exactness check

social = data_saving.load_data('social_trials_list')
solo = data_saving.load_data('solo_trials_list')
print(f'social sessions {len(social)}, solo sessions {len(solo)}')

# ---- 1. geometry from human positions ----
# Solo sessions give the true reach. Social sessions contain short overshoots past the walls
# (logged positions, mostly just after a trigger), so the social maximum is not a clean measure.
print('\n1. geometry from human positions')
for label, sessions, players in (('solo', solo, (0,)), ('social', social, (0, 1))):
    xs = []
    for session in sessions:
        df = pd.concat(session)
        for pid in players:
            xz_p = df[[globals.PLAYER_LOC_DICT[pid]['xloc'], globals.PLAYER_LOC_DICT[pid]['yloc']]].to_numpy(float)
            xs.append(xz_p[np.isfinite(xz_p).all(axis=1)])
    xz = np.vstack(xs)
    alcove_max, alcove_q = [], []
    for wall in range(8):
        p, q = T.to_wall_frame(xz, wall)
        # take positions inside the alcove
        inside = p[p > 17.2]
        alcove_max.append(inside.max()); alcove_q.append(np.quantile(inside, 0.9999))
    print(f'   {label:6s} ({len(xz)} frames): alcove back reach max {np.mean(alcove_max):.4f}, '
          f'99.99th pct {np.mean(alcove_q):.4f} (mean over walls), globals {globals.OCTAGON_ALCOVE_RADIUS}')

# ---- 2. sweep every trial ----
def sweep(sessions, players, label):
    ''' Run both methods on every trial and summarise. Takes sessions, player ids and a label. '''

    stats = dict(calls=0, errors=0, error_examples=[], nan_mismatch=0, shape_mismatch=0, too_short=0,
                 values=0, changed=0, alcove_values=0, alcove_changed=0, open_values=0, open_changed=0,
                 nan_position_frames=0, t_angular=0.0, t_occlusion=0.0)
    for s_idx, session in enumerate(sessions):
        for t_idx, trial in enumerate(session):
            for pid in players:
                stats['calls'] += 1
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        t0 = time.time(); a = th.get_wall_visible(trial=trial, player_id=pid, current_fov=FOV, occlusion=False)
                        t1 = time.time(); b = th.get_wall_visible(trial=trial, player_id=pid, current_fov=FOV)
                        t2 = time.time()
                    stats['t_angular'] += t1 - t0; stats['t_occlusion'] += t2 - t1
                except Exception as e:
                    stats['errors'] += 1
                    if len(stats['error_examples']) < 3:
                        stats['error_examples'].append((s_idx, t_idx, pid, type(e).__name__, str(e)[:120]))
                    continue
                a_nan, b_nan = isinstance(a, float), isinstance(b, float)
                if a_nan != b_nan:
                    stats['nan_mismatch'] += 1; continue
                if a_nan:
                    stats['too_short'] += 1; continue
                if a.shape != b.shape:
                    stats['shape_mismatch'] += 1; continue
                # split changes by whether the eye is inside an alcove
                traj = tv.extract_trial_player_trajectory(trial=trial, player_id=pid)[:, :a.shape[1]].T
                stats['nan_position_frames'] += (~np.isfinite(traj).all(axis=1)).sum()
                in_alcove = vo.find_eye_alcove(traj) >= 0
                diff = a != b
                stats['values'] += a.size; stats['changed'] += diff.sum()
                stats['alcove_values'] += a[:, in_alcove].size; stats['alcove_changed'] += diff[:, in_alcove].sum()
                stats['open_values'] += a[:, ~in_alcove].size; stats['open_changed'] += diff[:, ~in_alcove].sum()

    frames_alcove = stats['alcove_values'] / 8; frames_total = stats['values'] / 8
    print(f'\n2. sweep: {label}')
    print(f"   calls {stats['calls']}, errors {stats['errors']}, too-short trials mismatched {stats['nan_mismatch']}, "
          f"shapes mismatched {stats['shape_mismatch']}, too short (both methods) {stats['too_short']}")
    for ex in stats['error_examples']:
        print('   error example (session, trial, player, type, message):', ex)
    print(f"   frames {int(frames_total)}, in an alcove {frames_alcove / frames_total:.1%}, frames with missing position {stats['nan_position_frames']}")
    print(f"   wall x frame values changed by occlusion=True: all {stats['changed'] / stats['values']:.2%}, "
          f"eye in alcove {stats['alcove_changed'] / max(stats['alcove_values'], 1):.2%}, "
          f"eye outside alcoves {stats['open_changed'] / stats['open_values']:.2%}")
    print(f"   time: angular {stats['t_angular']:.1f}s, occlusion {stats['t_occlusion']:.1f}s")

sweep(social, (0, 1), 'social, both players')
sweep(solo, (0,), 'solo, player 0')

# ---- 3. exactness against the sampled reference ----
missed = {20: 0, 400: 0}; extra = {20: 0, 400: 0}; total = 0
for session in social[:REFERENCE_SESSIONS]:
    for trial in session:
        for pid in (0, 1):
            wv = th.get_wall_visible(trial=trial, player_id=pid, current_fov=FOV)
            if isinstance(wv, float):
                continue
            traj = tv.extract_trial_player_trajectory(trial=trial, player_id=pid)
            sm = th.get_smoothed_player_head_angle_vectors_for_trial(tv.extract_trial_player_headangles(trial=trial, player_id=pid))
            n = sm.shape[1]
            eyes = traj[:, :n].T
            keep = np.isfinite(eyes).all(axis=1)
            headings = (sm / np.linalg.norm(sm, axis=0)).T
            for density in missed:
                ref = T.reference_wall_visible(eyes[keep], headings[keep], FOV, density)
                missed[density] += (ref & ~wv[:, keep]).sum()
                extra[density] += (wv[:, keep] & ~ref).sum()
            total += wv[:, keep].size
print(f'\n3. exactness on {REFERENCE_SESSIONS} social sessions ({total // 8} frames)')
print(f'   walls the reference sees but occlusion misses: {missed}')
print(f'   walls occlusion sees but the reference misses (slivers): {extra} ({extra[400] / total:.4%} at 400 points per face)')
