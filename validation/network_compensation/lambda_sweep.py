# Choose the second-difference penalty for network_compensation.solve_block.
#
# The penalty exists to stop the solve chasing error in the time axis. Tuning it on a round trip that
# uses the same frame times forwards and backwards therefore measures nothing: that case has no error to
# absorb, and the sweep picks the smallest penalty offered. Three tests are run instead, each against a
# time-axis error the log really has.
#
#   1. quantised   the log writes whole milliseconds, so a frame interval is an integer number of ms
#   2. render      the logger samples in FixedUpdate, the interpolator steps in Update, so the interval
#                  between two rows is not the delta the engine smoothed with
#   3. held out    real player 1 frames, fitted with a share of them left out and scored on those. No
#                  forward model stands between the answer and the data, so every model error counts
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python validation/network_compensation/lambda_sweep.py

import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, lsqr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (social_trial_lists, frame_times, player_series, unrounded, forward_path,
                     render_clock, step_metrics, write_table)
import data_extraction.network_compensation.network_compensation as nr

LAMBDAS = (0.0, 0.003, 0.01, 0.03, 0.1)
SESSIONS = (0, 1, 2)
WINDOW_S = 300          # kept short so the sweep runs in minutes rather than hours
RENDER_HZ = 59.0        # a render clock close to, but not locked to, the ~48 Hz logging rate
HOLD_OUT = 5            # every fifth frame is scored, the rest are fitted


def round_trip(sessions, lam, kind):
    ''' Return the mean recovery error against a known trace. Takes the sessions, the penalty and which
        time-axis error to apply.

        Player 0's logged trace is exact, so running it through the network path gives a case where the
        owner's values are known. The solver is then handed only what the log would hold. '''

    rows = []
    tl = social_trial_lists()
    for s in sessions:
        df = pd.concat(tl[s])
        t = frame_times(df)
        sel = t < WINDOW_S
        t, truth_series = t[sel], {k: v[sel] for k, v in player_series(df, 0).items()}

        # the network path runs on the true times, the solver only ever sees the logged ones
        t_true = unrounded(t) if kind == 'quantised' else t
        render_t = render_clock(t, RENDER_HZ) if kind == 'render' else None
        logged, tick_t, truth = forward_path(t_true, truth_series, render_t=render_t)

        rt, rv = nr.recover_traces(t, logged, lam=lam)
        ref = {k: np.interp(rt, tick_t, truth[k]) for k in truth}
        got, want = step_metrics(rt, rv), step_metrics(rt, ref)
        rows.append(dict(distance=got['distance'] / want['distance'],
                         rotation=got['rotation'] / want['rotation'],
                         no_rotational=got['no_rotational'] - want['no_rotational'],
                         resting=got['resting'] - want['resting'],
                         position=float(np.median(np.hypot(rv['x'] - ref['x'], rv['z'] - ref['z'])))))
    return pd.DataFrame(rows).mean()


def solve_held_out(t, y, phase, lam, hold_out=HOLD_OUT):
    ''' Return the error on the frames left out of the fit. Takes frame times, a logged trace, the tick
        phase and the penalty.

        The held-out frames stay in the recursion, as the smoothing runs through them, but their residual
        is given no weight in the objective. What the fit predicts there is then compared with the log. '''

    a = nr._alphas(t)
    A, _ = nr._interp_matrix(t, phase)
    D = nr._second_difference(A.shape[1]) * np.sqrt(lam)
    w = np.ones(len(t))
    w[::hold_out] = 0.0

    def matvec(s):
        return np.r_[w * nr.smooth_forward(a, A @ s), D @ s]

    def rmatvec(r):
        return A.T @ nr.smooth_adjoint(a, w * r[:len(t)]) + D.T @ r[len(t):]

    op = LinearOperator((len(t) + D.shape[0], A.shape[1]), matvec=matvec, rmatvec=rmatvec)
    s = lsqr(op, np.r_[w * y, np.zeros(D.shape[0])], atol=1e-8, btol=1e-8, iter_lim=400)[0]
    pred = nr.smooth_forward(a, A @ s)
    out = w == 0
    return float(np.sqrt(np.mean((pred[out] - y[out]) ** 2)))


def held_out(sessions, lam):
    ''' Return the held-out error on real player 1 frames, averaged over blocks. Takes the sessions and
        the penalty. '''

    tl = social_trial_lists()
    errs = []
    for s in sessions:
        df = pd.concat(tl[s])
        t = frame_times(df)
        series = player_series(df, 1)
        edges = np.arange(t[0], min(t[-1], WINDOW_S), nr.BLOCK_S)
        for lo, hi in zip(edges, edges[1:]):
            m = (t >= lo) & (t < hi)
            if m.sum() < 500:
                continue
            for name in ('x', 'z', 'yaw'):
                # scale by the trace's own spread, so degrees and units are comparable
                y = series[name][m]
                errs.append(solve_held_out(t[m], y, 0.0, lam) / max(float(np.std(y)), 1e-9))
    return float(np.mean(errs))


def main():
    lines = ['# Choosing LAMBDA', '',
             f'Sessions {list(SESSIONS)}, first {WINDOW_S} s of each. Regenerate with',
             '`validation/network_compensation/lambda_sweep.py`.', '']

    for kind, title, note in (
            ('exact', 'Round trip, exact frame times',
             'No time-axis error, so nothing for the penalty to absorb. This is the test the original '
             'tuning table used, and it cannot do anything but favour the smallest penalty.'),
            ('quantised', 'Round trip, millisecond timestamps',
             'Every time column in the log is written to the millisecond, so the engine smoothed with '
             'times the solver never sees. The path runs on times inside that millisecond and the solver '
             'is handed the rounded ones.'),
            ('render', f'Round trip, interpolator stepped on a {RENDER_HZ:.0f} Hz render clock',
             'The logger samples in FixedUpdate and the interpolator steps in Update, so one logged row '
             'does not correspond to one smoothing step.')):
        lines += [f'## {title}', '', note, '',
                  '| lambda | distance ratio | rotation ratio | no-rotation error | resting error | position error |',
                  '|---|---|---|---|---|---|']
        for lam in LAMBDAS:
            r = round_trip(SESSIONS, lam, kind)
            lines.append(f'| {lam} | {r["distance"]:.3f} | {r["rotation"]:.3f} | '
                         f'{r["no_rotational"]:+.3f} | {r["resting"]:+.3f} | {r["position"]:.4f} |')
            print(f'{kind:>10} lam={lam:<6} dist {r["distance"]:.3f}  rot {r["rotation"]:.3f}  '
                  f'no_rot {r["no_rotational"]:+.3f}  pos {r["position"]:.4f}', flush=True)
        lines.append('')

    lines += ['## Held out frames, real player 1', '',
              f'Every {HOLD_OUT}th frame is left out of the fit and scored on what the fit predicts '
              'there, as a share of the trace\'s own spread. No forward model stands between the answer '
              'and the data, so this is the only test that sees every model error at once.', '',
              '| lambda | held-out error |', '|---|---|']
    for lam in LAMBDAS:
        e = held_out(SESSIONS, lam)
        lines.append(f'| {lam} | {e:.5f} |')
        print(f'  held out lam={lam:<6} {e:.5f}', flush=True)

    write_table('lambda_sweep.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
