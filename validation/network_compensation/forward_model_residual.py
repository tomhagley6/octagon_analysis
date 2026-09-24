# Test whether the forward path describes the real log.
#
# The round trip in lambda_sweep.py runs a known trace through network_forward and recovers it, so it
# tests the solve but takes the forward path as given. If the path itself were wrong, that test would
# still pass. What the path cannot explain shows up instead in the fit residual on real player 1 frames.
#
# The comparison is against player 0's exact trace put through the same path, where the residual is the
# floor the solve can reach, and against that same case with the log's own time-axis errors applied.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python validation/network_compensation/forward_model_residual.py

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (social_trial_lists, frame_times, player_series, unrounded, forward_path,
                     render_clock, write_table)
import data_extraction.network_compensation.network_compensation as nr

SESSIONS = (0, 1, 2)
WINDOW_S = 300
RENDER_HZ = 59.0
MIN_FRAMES = 500


def block_residual(t, y, lam=nr.LAMBDA):
    ''' Return the best-phase fit residual per frame, for each block of a trace. Takes frame times and a
        logged trace. '''

    out = []
    edges = np.arange(t[0], t[-1] + nr.BLOCK_S, nr.BLOCK_S)
    for lo, hi in zip(edges, edges[1:]):
        m = (t >= lo - nr.EDGE_S) & (t < hi + nr.EDGE_S)
        if m.sum() < MIN_FRAMES:
            continue
        best = min(nr.solve_block(t[m], y[m], ph, lam=lam)[2]
                   for ph in np.linspace(0, nr.TICK_S, nr.N_PHASE, endpoint=False))
        out.append(best / np.sqrt(m.sum()))
    return out


def main():
    tl = social_trial_lists()
    cases = {'real player 1': [], 'player 0 through the path, exact times': [],
             'player 0, millisecond timestamps': [], f'player 0, {RENDER_HZ:.0f} Hz render clock': []}

    for s in SESSIONS:
        df = pd.concat(tl[s])
        t = frame_times(df)
        sel = t < WINDOW_S
        t = t[sel]
        yaw1 = player_series(df, 1)['yaw'][sel]
        truth = {'yaw': player_series(df, 0)['yaw'][sel]}

        cases['real player 1'] += block_residual(t, yaw1)
        for name, t_true, render_t in (
                ('player 0 through the path, exact times', t, None),
                ('player 0, millisecond timestamps', unrounded(t), None),
                (f'player 0, {RENDER_HZ:.0f} Hz render clock', t, render_clock(t, RENDER_HZ))):
            logged, _, _ = forward_path(t_true, truth, render_t=render_t)
            cases[name] += block_residual(t, logged['yaw'])
        print(f'session {s} done', flush=True)

    lines = ['# Does the forward path describe the real log?', '',
             f'Yaw fit residual per frame, in degrees, over {nr.BLOCK_S:.0f} s blocks from the first '
             f'{WINDOW_S} s of sessions {list(SESSIONS)}. Regenerate with',
             '`validation/network_compensation/forward_model_residual.py`.', '',
             '| case | median | max | n blocks |', '|---|---|---|---|']
    for name, v in cases.items():
        v = np.array(v)
        lines.append(f'| {name} | {np.median(v):.4f} | {v.max():.4f} | {len(v)} |')
        print(f'{name:>45}  median {np.median(v):.4f}  max {v.max():.4f}')

    write_table('forward_model_residual.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
