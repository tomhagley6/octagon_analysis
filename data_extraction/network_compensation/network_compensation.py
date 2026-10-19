#!/usr/bin/env python
# coding: utf-8

# Recover a networked player's real movement from the host's log.
#
# In a social session the host logs its own player directly, but the other player's transform arrives
# over the network. Netcode's ClientNetworkTransform sends the owner's transform each network tick, the
# host renders it one tick behind with linear interpolation between ticks, and then smooths the result
# toward that target every frame (BufferedLinearInterpolator.cs:220). The logged trace is therefore
# delayed and low-pass filtered, which mostly destroys frame-by-frame stillness.
#
# Both stages are linear, so the logged trace is `logged = S A s`, with s the owner's values on the tick
# grid, A the interpolation and S the smoothing. This module solves for s directly, with a penalty on the
# second difference of s to stop the solution chasing noise. Motion between ticks was never sent, so the
# tick grid is the best that can be recovered.
#
# Validation runs the same path forward on the host's own player, where the truth is known.

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lsqr

import globals


TICK_S = 1 / 30             # NetworkManager TickRate 30 (OctagonStage.unity)
LAG_TICKS = 1               # the host is the server, so it renders one tick behind
SMOOTH_S = 0.1              # BufferedLinearInterpolator.MaximumInterpolationTime
LAMBDA = 0.003            # second-difference penalty; set by tuning against player 0 round trips (see docs)
BLOCK_S = 60                # solved in blocks of this length, so a clock drift cannot build up
EDGE_S = 2                  # discarded from each end of a block, where the solution is least constrained
N_PHASE = 8                 # tick grid offsets tried per block

# Stillness cutoffs for tick-grid traces. These are the agent pipeline's per-frame thresholds
# (mc.STATIONARY_ROT, mc.STATIONARY_SPEED at about 48 Hz) rescaled to a 33 ms tick, so they keep the same
# physical meaning: about 24 deg/s and about 2.4 units/s. A recovered trace never sits at exactly zero,
# as the penalty that keeps its totals honest also stops it holding perfectly still, so anything finer
# separates a recovered player from a locally logged one. At these values it does not (rotation share
# differs by 0.009, p .75, over the 100 social sessions). See READMEs/network_compensation.md.
LOGGED_FPS = 48.3           # typical logged frame rate of a clean session
STILL_ROT_PER_TICK = 0.8    # degrees per tick below which a step counts as no rotation
STILL_SPEED_PER_TICK = 0.08     # units per tick below which a step counts as no translation
MIN_SAFE_ROT_PER_TICK = 0.4     # below this, recovery and local logging stop agreeing


def _alphas(t, smooth_s=SMOOTH_S):
    ''' Return the per-frame smoothing weight a_k = min(dt / smooth_s, 1). Takes frame times (s).

        The first frame takes the target outright, as the interpolator starts there. '''

    a = np.minimum(np.diff(t, prepend=t[0]) / smooth_s, 1.0)
    a[0] = 1.0
    return a


def _first_order(u, d, chunk=500):
    ''' Return h with h_m = u_m + d_m h_(m-1) and h_(-1) = 0. Takes the input and the coefficients.

        Solved in chunks rather than element by element. With P the running product of d,
        h_m = P_m * sum(u_i / P_i) for i <= m. A chunk ends at any small d, so P stays away from zero. '''

    n = len(u)
    out = np.empty(n)
    starts = sorted(set(range(0, n, chunk)) | set(np.flatnonzero(d <= 0.5).tolist()))
    h = 0.0
    for s, e in zip(starts, starts[1:] + [n]):
        h = u[s] + d[s] * h
        out[s] = h
        if e - s > 1:
            dd, uu = d[s + 1:e], u[s + 1:e]
            P = np.cumprod(dd)
            out[s + 1:e] = P * (h + np.cumsum(uu / P))
            h = out[e - 1]
    return out


def smooth_forward(a, x):
    ''' Return the smoothed trace y_k = y_(k-1) + a_k (x_k - y_(k-1)). Takes the weights and the target. '''

    return _first_order(a * x, 1 - a)


def smooth_adjoint(a, r):
    ''' Return S^T r, the adjoint of smooth_forward. Takes the weights and a residual vector.

        S is lower triangular with S[j, k] = a_k prod(1 - a_i) over k < i <= j, so
        (S^T r)_k = a_k g_k with g_k = r_k + (1 - a_(k+1)) g_(k+1), run backwards. Reversing that
        recursion turns it into the same first-order form as the forward pass. '''

    n = len(r)
    d = np.empty(n)
    d[0] = 0.0
    if n > 1:
        d[1:] = 1 - a[n - 1:0:-1]
    return a * _first_order(r[::-1], d)[::-1]


def _interp_matrix(t, phase, tick_s=TICK_S, lag_ticks=LAG_TICKS):
    ''' Return the sparse matrix mapping tick-grid samples to the interpolation target at each frame,
        and the sample times. Takes frame times, the tick grid phase.

        The host renders lag_ticks behind, so the frame at time t reads the grid at t - lag. '''

    u = (t - lag_ticks * tick_s - phase) / tick_s
    j = np.floor(u).astype(int)
    j -= j.min()
    frac = u - np.floor(u)
    n_knots = int(j.max()) + 2
    rows = np.repeat(np.arange(len(t)), 2)
    A = sparse.csr_matrix((np.c_[1 - frac, frac].ravel(), (rows, np.c_[j, j + 1].ravel())),
                          shape=(len(t), n_knots))
    sample_times = phase + tick_s * (np.arange(n_knots) + np.floor((t[0] - lag_ticks * tick_s - phase) / tick_s))
    return A, sample_times


def _second_difference(n):
    ''' Return the second-difference operator for a length-n vector (each row giving
        s[k] - 2 s[k+1] + s[k+2], the discrete counterpart of a second derivative). Takes the length.

        Left unnormalised, with no division by the tick spacing, so LAMBDA is tied to a 30 Hz grid. '''

    if n < 3:
        return sparse.csr_matrix((0, n))
    return sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))


def solve_block(t, y, phase, lam=LAMBDA, smooth_s=SMOOTH_S, **kw):
    ''' Return the tick-grid samples that best explain a logged block, their times, and the fit residual.
        Takes frame times, the logged values, the tick phase and the penalty weight.

        Minimises ||S A s - y||^2 + lam ||D2 s||^2, where S A is the network path and D2 the second
        difference. The penalty is what keeps the recovered trace from chasing the noise that dividing
        out the smoothing would otherwise amplify. '''

    a = _alphas(t, smooth_s)
    A, sample_times = _interp_matrix(t, phase, **kw)
    n_knots = A.shape[1]
    D = _second_difference(n_knots) * np.sqrt(lam)

    def matvec(s):
        return np.r_[smooth_forward(a, A @ s), D @ s]

    def rmatvec(r):
        return A.T @ smooth_adjoint(a, r[:len(t)]) + D.T @ r[len(t):]

    op = LinearOperator((len(t) + D.shape[0], n_knots), matvec=matvec, rmatvec=rmatvec)
    s = lsqr(op, np.r_[y, np.zeros(D.shape[0])], atol=1e-8, btol=1e-8, iter_lim=400)[0]
    resid = float(np.linalg.norm(smooth_forward(a, A @ s) - y))
    return s, sample_times, resid


def recover_traces(t, series, lam=LAMBDA, block_s=BLOCK_S, edge_s=EDGE_S, n_phase=N_PHASE, tick_s=TICK_S):
    ''' Return the owner's values on the tick grid for several traces at once, as (times, dict of values).
        Takes frame times and a dict of traces.

        Solved in blocks, as the tick grid phase is not logged and the two clocks can drift apart over a
        session. One phase is fitted per block across all the traces together, since a player's position
        and rotation travel in the same transform updates. Blocks overlap by edge_s at each end and only
        the interior of each is kept. '''

    names = list(series)
    times, vals = [], {n: [] for n in names}
    edges = np.arange(t[0], t[-1] + block_s, block_s)
    for lo, hi in zip(edges, edges[1:]):
        m = (t >= lo - edge_s) & (t < hi + edge_s)
        if m.sum() < 20:
            continue
        tb = t[m]
        yb = {n: series[n][m] for n in names}
        # scale each trace's residual by its own spread, so one signal's units cannot decide the phase
        scale = {n: max(float(np.std(yb[n])), 1e-9) for n in names}

        best = None
        for phase in np.linspace(0, tick_s, n_phase, endpoint=False):
            fits = {n: solve_block(tb, yb[n], phase, lam=lam) for n in names}
            cost = sum(fits[n][2] / scale[n] for n in names)
            if best is None or cost < best[0]:
                best = (cost, fits)
        fits = best[1]
        st = fits[names[0]][1]
        keep = (st >= lo) & (st < hi)
        times.append(st[keep])
        for n in names:
            vals[n].append(fits[n][0][keep])
    if not times:
        return np.empty(0), {n: np.empty(0) for n in names}
    out_t = np.concatenate(times)
    out_v = {n: np.concatenate(vals[n]) for n in names}
    # each block fits its own phase, so the first knot of a block can land less than a tick after the
    # last of the one before. Drop it, or that pair would read as a smaller step than it was
    keep = np.r_[True, np.diff(out_t) > 0.5 * tick_s]
    return out_t[keep], {n: v[keep] for n, v in out_v.items()}


def recover_trace(t, y, **kw):
    ''' Return one trace's owner values on the tick grid, and their times. Takes frame times and the
        logged trace. A convenience wrapper on recover_traces. '''

    times, vals = recover_traces(t, {'y': y}, **kw)
    return times, vals['y']


def network_forward(t, x, phase=0.0, smooth_s=SMOOTH_S, **kw):
    ''' Return what the host would log for a trace sampled on the tick grid. Takes frame times and the
        frame-by-frame values. Used to test recovery against a known answer. '''

    A, sample_times = _interp_matrix(t, phase, **kw)
    s = np.interp(sample_times, t, x)
    return smooth_forward(_alphas(t, smooth_s), A @ s), sample_times, s


def recover_player(df, log_id, lam=LAMBDA, time_col='timeLocal', **kw):
    ''' Return a player's recovered x, z and yaw on the tick grid, as a dataframe. Takes a session
        dataframe and the player's log id.

        The three traces are solved together on one tick grid, as they travel in the same transform
        updates. Yaw is unwrapped before recovery and returned unwrapped, so rotation can be differenced
        directly. Missing frames are filled before solving, as the filter spans them. '''

    t = np.maximum.accumulate((df[time_col] - df[time_col].iloc[0]).dt.total_seconds().to_numpy())
    series = {}
    for name, col, angle in (('x', globals.PLAYER_LOC_DICT[log_id]['xloc'], False),
                             ('z', globals.PLAYER_LOC_DICT[log_id]['yloc'], False),
                             ('yaw', globals.PLAYER_ROT_DICT[log_id]['yrot'], True)):
        v = pd.Series(df[col].to_numpy(float)).ffill().bfill().to_numpy()
        series[name] = np.rad2deg(np.unwrap(np.deg2rad(v))) if angle else v
    times, vals = recover_traces(t, series, lam=lam, **kw)
    return pd.DataFrame({'t': times, **vals})


def sample_player(df, log_id, times, time_col='timeLocal'):
    ''' Return a locally logged player's x, z and yaw at the owner's tick times, as a dataframe.
        Takes a session dataframe, the player's log id and the tick times recover_player returned.

        Those times are when the owner sampled its transform, not when the host drew it, so the local
        trace is read at the same instants. This is how a locally logged player is put on the same
        footing as a recovered one: sampled at 30 Hz, but never delayed or smoothed. '''

    t = np.maximum.accumulate((df[time_col] - df[time_col].iloc[0]).dt.total_seconds().to_numpy())
    at = times
    out = {'t': times}
    for name, col, angle in (('x', globals.PLAYER_LOC_DICT[log_id]['xloc'], False),
                             ('z', globals.PLAYER_LOC_DICT[log_id]['yloc'], False),
                             ('yaw', globals.PLAYER_ROT_DICT[log_id]['yrot'], True)):
        v = pd.Series(df[col].to_numpy(float)).ffill().bfill().to_numpy()
        if angle:
            v = np.rad2deg(np.unwrap(np.deg2rad(v)))
        out[name] = np.interp(at, t, v)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# recovered traces for every social session, saved beside the session pickles
# ---------------------------------------------------------------------------

RECOVERED_SOCIAL = 'social_recovered_ticks'     # saved in data_strings.DATAFRAME_PATH


def session_tick_table(trial_list, experiment, lam=LAMBDA, network_ids=(1,), local_ids=(0,)):
    ''' Return one row per network tick for one social session, with both players' x, z and yaw.
        Takes the session's trial list and its experiment index.

        Players in network_ids are recovered from the host's log, and those in local_ids are read
        straight off the local trace at the same instants. Each row also carries the trial number and
        whether that tick falls in an ITI, so the table stands on its own. '''

    df = pd.concat(trial_list)
    t = np.maximum.accumulate((df['timeLocal'] - df['timeLocal'].iloc[0]).dt.total_seconds().to_numpy())

    out = None
    for log_id in network_ids:
        rec = recover_player(df, log_id, lam=lam)
        out = rec if out is None else out
        for c in ('x', 'z', 'yaw'):
            out[f'p{log_id}_{c}'] = rec[c]
    out = out.drop(columns=[c for c in ('x', 'z', 'yaw') if c in out])
    for log_id in local_ids:
        loc = sample_player(df, log_id, out['t'].to_numpy())
        for c in ('x', 'z', 'yaw'):
            out[f'p{log_id}_{c}'] = loc[c]

    # carry the trial number and epoch of the frame each tick falls in
    frame = np.clip(np.searchsorted(t, out['t'].to_numpy()), 0, len(t) - 1)
    out.insert(0, 'experiment', experiment)
    out['trial_num'] = df[globals.TRIAL_NUM].to_numpy()[frame]
    out['is_iti'] = pd.Series(df['trial_epoch'].to_numpy()[frame]).isin(
        [globals.ITI, globals.TRIAL_STARTED]).to_numpy()
    return out


def build_recovered_social(trial_lists, lam=LAMBDA, verbose=True):
    ''' Return the tick table for every social session, stacked. Takes the social trial lists. '''

    parts = []
    for i, tl in enumerate(trial_lists):
        parts.append(session_tick_table(tl, i, lam=lam))
        if verbose:
            print(f'{i + 1}/{len(trial_lists)}', end='  ', flush=True)
    out = pd.concat(parts, ignore_index=True)
    floats = out.select_dtypes('float64').columns
    out[floats] = out[floats].astype('float32')     # halves the file, far finer than the data warrants
    return out


def frames_from_ticks(df, log_id, ticks, time_col='timeLocal'):
    ''' Return a copy of a session dataframe with one player's x, z and yaw replaced by recovered values,
        read at each frame's time. Takes the session dataframe, the player's log id, and that player's
        recovered tick table (columns t, x, z, yaw).

        Frame-level analysis (wall visibility, head angle, position at slice onset) needs a value per
        frame, so the recovered 30 Hz trace is interpolated back up. That cannot restore what happened
        between ticks, but it removes the delay and the smoothing, which is what those measures suffer
        from. Yaw is interpolated unwrapped and wrapped back to 0-360, as the log holds it. '''

    t = np.maximum.accumulate((df[time_col] - df[time_col].iloc[0]).dt.total_seconds().to_numpy())
    out = df.copy()
    tt = ticks['t'].to_numpy()
    for col, name, angle in ((globals.PLAYER_LOC_DICT[log_id]['xloc'], 'x', False),
                             (globals.PLAYER_LOC_DICT[log_id]['yloc'], 'z', False),
                             (globals.PLAYER_ROT_DICT[log_id]['yrot'], 'yaw', True)):
        v = np.interp(t, tt, ticks[name].to_numpy())
        out[col] = np.mod(v, 360.0) if angle else v
    return out


RECOVERED_SOLO = 'solo_local_ticks'         # solo sessions sampled onto the same grid, saved alongside


def local_tick_table(trial_list, experiment, player, session_type, tick_s=TICK_S):
    ''' Return one row per tick for a locally logged session. Takes the trial list, the experiment index,
        the player and the session type.

        Solo players are logged locally, so there is nothing to recover. They are sampled onto a regular
        30 Hz grid only so that their movement is measured over the same interval as a social player's. '''

    df = pd.concat(trial_list)
    t = np.maximum.accumulate((df['timeLocal'] - df['timeLocal'].iloc[0]).dt.total_seconds().to_numpy())
    times = np.arange(t[0], t[-1], tick_s)
    loc = sample_player(df, 0, times)
    frame = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
    return pd.DataFrame(dict(experiment=experiment, player=player, session_type=session_type, t=times,
                             x=loc['x'], z=loc['z'], yaw=loc['yaw'],
                             trial_num=df[globals.TRIAL_NUM].to_numpy()[frame],
                             is_iti=pd.Series(df['trial_epoch'].to_numpy()[frame]).isin(
                                 [globals.ITI, globals.TRIAL_STARTED]).to_numpy(),
                             source='local'))


# ---------------------------------------------------------------------------
# testing what downsampling costs, before choosing to downsample
# ---------------------------------------------------------------------------

def downsampling_effect(native, resampled, keys, metrics):
    ''' Return the change each metric takes from the native rate to the resampled one.
        Takes a native table, a resampled table, the keys identifying one unit on both sides, and
        the metrics to test.

        Downsampling throws away the precision the high logging rate bought, so it is worth doing
        only when it costs nothing. This measures the cost. Both tables must cover the SAME units
        (the merge is an inner join on keys), so the ratio is a within-unit change rather than a
        difference between two sets of rows.

        Run it on any trace that has a native rate to compare against: locally logged player 0,
        solo players, or agent sessions. Player 1 has no native trace, which is why the question
        arises at all.

        ratio near 1   the metric survives downsampling, so that group can be downsampled freely
        ratio far from 1   it does not, and downsampling changes what the metric means

        Ratios near each other but far from 1 are a third case: a shared factor cancels out of a
        standardised effect or a between-group ratio, so the comparison survives while the absolute
        values do not. That is weaker than every ratio being near 1, and which case holds should be
        stated rather than left implied.

        A ratio depends on the signal, not only on the two rates, so it has to be measured per
        group and per metric rather than assumed to cancel. See READMEs/network_compensation.md,
        "Downsampling: test it before you do it". '''

    metrics = list(metrics)
    merged = native[keys + metrics].merge(resampled[keys + metrics], on=keys,
                                          suffixes=('_native', '_resampled'))
    rows = []
    for m in metrics:
        a = merged[f'{m}_native'].to_numpy(float)
        b = merged[f'{m}_resampled'].to_numpy(float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        native_mean, resampled_mean = np.mean(a[ok]), np.mean(b[ok])
        rows.append(dict(metric=m, native=native_mean, resampled=resampled_mean,
                         diff=resampled_mean - native_mean,
                         ratio=resampled_mean / native_mean if native_mean else np.nan,
                         n_units=int(ok.sum())))
    return pd.DataFrame(rows)


def social_ticks_long(wide):
    ''' Return the saved social tick table as one row per player per tick. Takes the wide table.

        The wide table holds both players side by side, as they share one grid. Analysis wants one row
        per player, matching the solo table. '''

    parts = []
    for p, source in ((0, 'local'), (1, 'recovered')):
        parts.append(pd.DataFrame(dict(
            experiment=wide['experiment'], player=p, session_type='social', t=wide['t'],
            x=wide[f'p{p}_x'], z=wide[f'p{p}_z'], yaw=wide[f'p{p}_yaw'],
            trial_num=wide['trial_num'], is_iti=wide['is_iti'], source=source)))
    return pd.concat(parts, ignore_index=True)
