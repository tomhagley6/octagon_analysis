# How much does each shaky assumption in the forward path cost?
#
# The method reads the network path off the engine source, but two parts of that reading are known not to
# match the log exactly, and a third cannot be checked against the data at all. This puts numbers on them.
#
#   1. render rate   the interpolator smooths once per render frame, and targetFrameRate is 144 while the
#                    log runs near 48 Hz, so a logged row spans about three smoothing steps and the model
#                    applies one
#   2. tick rate     TICK_S is read from the scene file. The fit cannot choose it, so the question is
#                    whether being wrong about it would change any answer
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python validation/network_compensation/assumption_errors.py

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (social_trial_lists, frame_times, player_series, forward_path, render_clock,
                     step_metrics, write_table)
import globals
import data_extraction.network_compensation.network_compensation as nr

SESSIONS = (0, 1, 2)
WINDOW_S = 300
RENDER_RATES = (48, 60, 90, 144, 240)
TICK_RATES = (27.0, 28.5, 30.0, 31.5, 33.0)


def render_sweep():
    ''' Return the recovery error against the truth for each render rate. Takes no inputs.

        The path is stepped on a render clock and the solve is given only the log times, which is the
        mismatch the real data carries. `None` steps once per logged row, as the model assumes. '''

    tl = social_trial_lists()
    rows = []
    for rate in list(RENDER_RATES) + [None]:
        per = []
        for s in SESSIONS:
            df = pd.concat(tl[s])
            t = frame_times(df)
            t = t[t < WINDOW_S]
            truth = {k: v[:len(t)] for k, v in player_series(df, 0).items()}
            logged, tick_t, truth_ticks = forward_path(
                t, truth, render_t=render_clock(t, rate) if rate else None)
            rt, rv = nr.recover_traces(t, logged)
            ref = {k: np.interp(rt, tick_t, truth_ticks[k]) for k in truth_ticks}
            got, want = step_metrics(rt, rv), step_metrics(rt, ref)
            per.append(dict(distance=got['distance'] / want['distance'],
                            rotation=got['rotation'] / want['rotation'],
                            no_rotational=got['no_rotational'] - want['no_rotational'],
                            position=float(np.median(np.hypot(rv['x'] - ref['x'], rv['z'] - ref['z'])))))
        rows.append(dict(rate=rate, **pd.DataFrame(per).mean().to_dict()))
        print('.', end='', flush=True)
    return pd.DataFrame(rows)


def tick_sweep():
    ''' Return player 1's recovered rotation total and stillness share at each tick rate. Takes no inputs.

        Totals are given per 100 ticks as well, since a finer grid has more of them. The stillness cutoff
        is rescaled with the tick so it keeps the same meaning in degrees per second. '''

    tl = social_trial_lists()
    rows = []
    for rate in TICK_RATES:
        per = []
        for s in SESSIONS:
            df = pd.concat(tl[s])
            t = frame_times(df)
            sel = t < 600
            t = t[sel]
            iti = df['trial_epoch'].isin([globals.ITI, globals.TRIAL_STARTED]).to_numpy()[sel]
            yaw = player_series(df, 1)['yaw'][sel]
            rt, rv = nr.recover_trace(t, yaw, tick_s=1.0 / rate)
            ok = np.abs(np.diff(rt) - 1.0 / rate) < 0.5 / rate
            it = iti[np.clip(np.searchsorted(t, rt), 0, len(t) - 1)]
            keep = ok & it[1:] & it[:-1]
            step = np.abs(np.diff(rv))[keep]
            per.append(dict(total=step.sum(), per100=100 * step.sum() / keep.sum(),
                            still=(step < nr.STILL_ROT_PER_TICK * (30.0 / rate)).mean()))
        rows.append(dict(rate=rate, **pd.DataFrame(per).mean().to_dict()))
        print('.', end='', flush=True)
    return pd.DataFrame(rows)


def main():
    rend = render_sweep()
    print()
    tick = tick_sweep()
    print()

    one = rend[rend.rate.isna()].iloc[0]
    real = rend[rend.rate == 144].iloc[0]

    lines = ['# What the shaky assumptions cost', '',
             'Regenerate with `validation/network_compensation/assumption_errors.py`.', '',
             '## One smoothing step per logged row', '',
             'The model applies one smoothing step per logged row, with the weight read from the logged '
             'timestamps. The interpolator really steps once per render frame, and `targetFrameRate` is '
             '144 against a log near 48 Hz, so a logged row spans about three steps. This runs the path '
             'on a render clock and gives the solve only the log times.', '',
             '| render rate | distance | rotation | no-rotation error | position error |',
             '|---|---|---|---|---|']
    for _, r in rend.iterrows():
        name = 'one step per row (what the model assumes)' if pd.isna(r.rate) else f'{r.rate:g} Hz'
        lines.append(f'| {name} | {r.distance:.3f} | {r.rotation:.3f} | {r.no_rotational:+.3f} | '
                     f'{r.position:.4f} |')

    lines += ['', f'At the real 144 Hz the totals come back at {real.rotation:.3f} and '
                  f'{real.distance:.3f}, so this costs well under half a percent, and in the direction of '
                  'reporting slightly too little rather than too much. The stillness share is unaffected, '
                  f'{real.no_rotational:+.3f} against {one.no_rotational:+.3f} when the assumption holds '
                  'exactly.', '',
              f'What it does cost is position: {real.position:.4f} units against {one.position:.4f}, a '
              'hundredfold increase, though still a quarter of a percent of the arena radius.', '',
              'The reason is that several small smoothing steps compose into very nearly one larger one, '
              'so the shape of the trace survives while its effective lag shifts a little. A timing error '
              'moves a trace without changing its path length or how often it holds still, which is '
              'exactly the pattern in the table. **The assumption that is known to be wrong is the one '
              'that matters least for what this data is used for.** It lands on the same weak axis as the '
              'whole-tick lag, which is timing, and leaves the strong axis, which is shape, alone.', '',
              '## The tick rate', '',
              '`TICK_S` comes from `TickRate: 30` in the scene file. Held-out frames cannot choose it, '
              'scoring flat from 26 to 38 Hz, because the movement carries little near any of those '
              'Nyquist limits. So the question is not whether the data confirms it but whether being '
              'wrong would matter.', '',
              '| tick rate | rotation total (deg) | per 100 ticks | stillness share |', '|---|---|---|---|']
    for _, r in tick.iterrows():
        mark = ' (in use)' if r.rate == 30 else ''
        lines.append(f'| {r.rate:g} Hz{mark} | {r.total:.0f} | {r.per100:.1f} | {r.still:.3f} |')

    spread = 100 * (tick.per100.max() / tick.per100.min() - 1)
    lines += ['', f'Barely. A ten percent error in the tick rate moves the rotation total by '
                  f'{spread:.2f}%. The stillness share drifts more, about '
                  f'{100 * (tick.still.max() - tick.still.min()):.1f} points across the same range, as the '
                  'number of ticks changes and rescaling the cutoff only approximately preserves its '
                  'meaning. Since the rate is read from the scene file rather than guessed, neither is a '
                  'live concern.', '',
              '## What is still untested', '',
              'Two assumptions have no number here. Sends are threshold gated at 0.001 units and 0.01 '
              'degrees and packets can be lost, so not every tick necessarily carries a sample. And the '
              'buffer can consume several items at once or extrapolate up to three ticks, neither of '
              'which the interpolation matrix can represent. Both are network side, and they are the '
              'most likely home for the unexplained residual in the validation section.']

    write_table('assumption_errors.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
