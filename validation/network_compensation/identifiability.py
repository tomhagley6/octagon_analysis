# What can the recovery actually know, and what is it assuming?
#
# The natural objection to undoing a low-pass filter is that a filter destroys what it removes, so there
# is nothing left to recover. That is not quite what happens here, and these two tests say where the line
# falls.
#
#   1. by frequency   a known oscillation is run through the network path and recovered. How much of it
#                     survives in the log, and how much comes back
#   2. by penalty     the real player 1 rotation total, recovered at a range of penalties. A total that
#                     depends on the penalty is a total the penalty invented
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python validation/network_compensation/identifiability.py

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import social_trial_lists, frame_times, player_series, unrounded, write_table
import globals
import data_extraction.network_compensation.network_compensation as nr

FREQS = (0.25, 0.5, 1, 2, 4, 8, 12, 14)
LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)
SESSIONS = (0, 1, 2)
WINDOW_S = 600


def by_frequency(window_s=120):
    ''' Return how much of an oscillation survives the network path and how much is recovered, per
        frequency. Takes the window length.

        A pure oscillation of known amplitude is placed on the tick grid, run through the path, and
        recovered. The path runs on the true times and the solve sees the rounded ones, as in a real log. '''

    t = frame_times(pd.concat(social_trial_lists()[0]))
    t = t[t < window_s]
    ticks = np.arange(t[0], t[-1], nr.TICK_S)
    t_true = unrounded(t)
    A, sample_times = nr._interp_matrix(t_true, 0.0)
    alphas = nr._alphas(t_true)

    rows = []
    for f in FREQS:
        truth = np.sin(2 * np.pi * f * ticks)
        sent = np.interp(sample_times, ticks, truth)
        logged = nr.smooth_forward(alphas, A @ sent)
        row = dict(hz=f, surviving=np.std(logged) / np.std(sent))
        for lam in (0.0, nr.LAMBDA):
            rt, rv = nr.recover_trace(t, logged, lam=lam)
            row[f'rec_{lam}'] = np.std(rv) / np.std(np.interp(rt, ticks, truth))
        rows.append(row)
        print('.', end='', flush=True)
    return pd.DataFrame(rows)


def by_penalty():
    ''' Return the real player 1 rotation total at each penalty, against the logged total. Takes no
        inputs.

        The penalty prefers calmer answers, so it can only hold the total down. A total that keeps rising
        as the penalty is relaxed is being fed by amplified error, and one that settles is in the data. '''

    tl = social_trial_lists()
    got = {}
    for s in SESSIONS:
        df = pd.concat(tl[s])
        t = frame_times(df)
        sel = t < WINDOW_S
        t = t[sel]
        iti = df['trial_epoch'].isin([globals.ITI, globals.TRIAL_STARTED]).to_numpy()[sel]
        yaw = player_series(df, 1)['yaw'][sel]
        got.setdefault('logged', []).append(np.abs(np.diff(yaw))[iti[1:] & iti[:-1]].sum())
        for lam in LAMBDAS:
            rt, rv = nr.recover_trace(t, yaw, lam=lam)
            ok = np.abs(np.diff(rt) - nr.TICK_S) < 0.5 * nr.TICK_S
            it = iti[np.clip(np.searchsorted(t, rt), 0, len(t) - 1)]
            got.setdefault(lam, []).append(np.abs(np.diff(rv))[ok & it[1:] & it[:-1]].sum())
        print('.', end='', flush=True)
    return {k: float(np.mean(v)) for k, v in got.items()}


def main():
    freq = by_frequency()
    print()
    pen = by_penalty()
    print()

    lines = ['# What is recoverable, and what is assumed', '',
             'Regenerate with `validation/network_compensation/identifiability.py`.', '',
             '## The filter attenuates, it does not erase', '',
             'A pure oscillation of known size is put on the tick grid, run through the network path and '
             'recovered. `surviving` is how much of it is still in the logged trace. The two recovered '
             'columns are how much the solve gets back, without a penalty and with the one in use.', '',
             '| Hz | surviving in the log | recovered, lambda 0 | recovered, lambda 0.003 |',
             '|---|---|---|---|']
    for _, r in freq.iterrows():
        lines.append(f'| {r.hz:g} | {r.surviving:.3f} | {r["rec_0.0"]:.3f} | {r[f"rec_{nr.LAMBDA}"]:.3f} |')

    lines += ['', 'Two things to read here. Even at 14 Hz, near the fastest the tick grid can carry, a '
                  'tenth of the movement is still in the logged trace. The path multiplies movement by a '
                  'small number, it does not throw it away, and with no penalty the solve returns every '
                  'frequency at full size. So the information is in the log and the question is whether it '
                  'can be pulled out without pulling out error with it, which is conditioning rather than '
                  'identifiability.', '',
              'The second thing is what the penalty costs. Below about 4 Hz it takes almost nothing. By 8 '
              'Hz it is holding back a fifth of the movement and by 12 Hz most of it. **The recovery is '
              'therefore conservative at the fast end, not inflationary.** Whatever it reports for fast '
              'turning is a floor.', '',
              'This test flatters the solve, as a clean oscillation gives error nothing to hide behind. It '
              'shows the ceiling of what is identifiable, not what happens on a real trace.', '',
              '## Does the penalty invent the extra turning?', '',
              "Player 1's between-trial rotation total, recovered at a range of penalties. The penalty "
              'prefers calmer answers, so it can only hold a total down. If the uplift were the penalty '
              'inventing movement, relaxing the penalty would shrink it.', '',
              '| lambda | ITI rotation (deg) | against logged |', '|---|---|---|',
              f'| logged, on frames | {pen["logged"]:.0f} | |']
    for lam in LAMBDAS:
        lines.append(f'| {lam} | {pen[lam]:.0f} | {100 * (pen[lam] / pen["logged"] - 1):+.1f}% |')

    plateau = 100 * (pen[0.1] / pen['logged'] - 1)
    chosen = 100 * (pen[nr.LAMBDA] / pen['logged'] - 1)
    lines += ['', 'It goes the other way. The total falls as the penalty rises and then settles: at '
                  f'lambda 0.03 and 0.1, penalties ten and thirty times the one in use, the uplift is '
                  f'still {plateau:+.0f}%. **An uplift that survives any penalty you care to apply is in '
                  'the data, not in the penalty.** No penalty at all gives '
                  f'{100 * (pen[0.0] / pen["logged"] - 1):+.0f}%, which is the unregularised inverse '
                  'amplifying error, and is the failure the penalty exists to stop.', '',
              f'At the chosen 0.003 the uplift is {chosen:+.1f}%, about {chosen - plateau:.1f} points above '
              'where it settles. That gap is the part of the uplift that is amplified error rather than '
              'recovered movement. If rotation totals are the main interest, 0.01 to 0.03 sits closer to '
              'the settled value at no measurable cost on the held-out test.']

    write_table('identifiability.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
