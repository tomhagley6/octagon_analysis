# Shared helpers for the network compensation validation scripts.
#
# Each script in this folder regenerates one table in
# READMEs/network_compensation.md. They load the saved session pickles, so
# they are slow and are not part of the unit test suite. Run them from the repo root with the conda env
# python and commit the files they write to results/.

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import data_strings
import globals
import data_extraction.network_compensation.network_compensation as nr

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# the log writes every timestamp to the millisecond, so a frame interval is an integer number of ms
LOG_TIME_RESOLUTION_S = 1e-3


def social_trial_lists():
    ''' Return the saved social trial lists. Takes nothing. '''

    return pd.read_pickle(os.path.join(data_strings.DATAFRAME_PATH, 'social_trials_list.pkl'))


def frame_times(df, time_col='timeLocal'):
    ''' Return frame times in seconds from the session's first frame. Takes a session dataframe. '''

    return np.maximum.accumulate((df[time_col] - df[time_col].iloc[0]).dt.total_seconds().to_numpy())


def player_series(df, log_id):
    ''' Return a player's logged x, z and yaw as a dict, yaw unwrapped. Takes a session dataframe and
        the player's log id. '''

    out = {}
    for name, col, angle in (('x', globals.PLAYER_LOC_DICT[log_id]['xloc'], False),
                             ('z', globals.PLAYER_LOC_DICT[log_id]['yloc'], False),
                             ('yaw', globals.PLAYER_ROT_DICT[log_id]['yrot'], True)):
        v = pd.Series(df[col].to_numpy(float)).ffill().bfill().to_numpy()
        out[name] = np.rad2deg(np.unwrap(np.deg2rad(v))) if angle else v
    return out


def unrounded(t, resolution=LOG_TIME_RESOLUTION_S, seed=0):
    ''' Return a true time axis consistent with the logged one. Takes the logged frame times.

        The log writes whole milliseconds, so a logged frame time is the engine's own time rounded to the
        nearest millisecond, and the true time sits somewhere in that millisecond. The round trip runs the
        network path on these times and hands the solver the logged ones, which is the error the solver
        really faces. Rounding the logged times instead does nothing, as they are already on the grid. '''

    rng = np.random.default_rng(seed)
    return np.maximum.accumulate(t + rng.uniform(-0.5, 0.5, len(t)) * resolution)


def forward_path(t, series, phase=0.0, render_t=None):
    ''' Return the trace the host would log, the tick times and the owner's tick values.
        Takes frame times and a dict of traces.

        With render_t given, the smoothing runs on that clock instead of the frame times, then the result
        is read back at the frame times. The logger samples in FixedUpdate while the interpolator steps in
        Update, so the two clocks are not the same one. '''

    logged, ticks, truth = {}, None, {}
    for name, v in series.items():
        if render_t is None:
            lg, st, s = nr.network_forward(t, v, phase=phase)
        else:
            # the interpolator steps on the render clock, so the trace is read there first
            lg_r, st, s = nr.network_forward(render_t, np.interp(render_t, t, v), phase=phase)
            lg = np.interp(t, render_t, lg_r)       # the logger reads whatever the last Update wrote
        logged[name], ticks, truth[name] = lg, st, s
    return logged, ticks, truth


def render_clock(t, rate_hz, seed=0):
    ''' Return render frame times spanning the same window at a different rate. Takes frame times.

        Models the interpolator stepping on the render clock while the log is written from FixedUpdate. '''

    rng = np.random.default_rng(seed)
    n = int((t[-1] - t[0]) * rate_hz)
    step = rng.uniform(0.85, 1.15, n) / rate_hz
    return t[0] + np.cumsum(step)


def step_metrics(times, vals, rot_cut=nr.STILL_ROT_PER_TICK, speed_cut=nr.STILL_SPEED_PER_TICK,
                 tick_s=nr.TICK_S):
    ''' Return total distance, total rotation and the stillness shares for a tick trace.
        Takes tick times and a dict with x, z and yaw.

        Pairs spanning a block join are dropped, as a join can place two knots less than a tick apart and
        that pair would read as a smaller step than it was. '''

    ok = np.abs(np.diff(times) - tick_s) < 0.5 * tick_s
    step = np.hypot(np.diff(vals['x']), np.diff(vals['z']))[ok]
    turn = np.abs(np.diff(vals['yaw']))[ok]
    return dict(distance=step.sum(), rotation=turn.sum(),
                no_translational=(step < speed_cut).mean(), no_rotational=(turn < rot_cut).mean(),
                resting=((step < speed_cut) & (turn < rot_cut)).mean())


def write_table(name, text):
    ''' Write a results file and echo where it went. Takes the file name and its contents. '''

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, 'w') as f:
        f.write(text)
    print(f'wrote {path}')
    return path
