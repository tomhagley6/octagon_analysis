# Tests for data_extraction/network_compensation/network_compensation.py. The forward path is checked against a plain frame loop, the
# adjoint against the explicit matrix, and recovery against a synthetic trace with a known answer.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_network_compensation -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
import data_extraction.network_compensation.network_compensation as nr


def frame_times(n, dt=0.0207, jitter=0.0, seed=0):
    ''' Return frame times with optional jitter. Takes the frame count and mean interval. '''

    rng = np.random.default_rng(seed)
    steps = np.full(n, dt) if jitter == 0 else rng.uniform(dt - jitter, dt + jitter, n)
    return np.cumsum(steps)


def walk(t, seed=0, speed=6.0, hold=0.4):
    ''' Return a trace that alternates between standing still and moving, like a player between trials.
        Takes frame times. '''

    rng = np.random.default_rng(seed)
    v = np.zeros(len(t))
    k = 0
    while k < len(t):
        n = rng.integers(20, 60)
        if rng.random() > hold:
            v[k:k + n] = rng.normal(0, speed) * (t[k:k + n] - t[k])
        v[k:] += v[min(k + n, len(t)) - 1] - v[k:][0] if False else 0
        k += n
    return np.cumsum(np.diff(v, prepend=v[0]))


class TestSmoothing(unittest.TestCase):

    def test_forward_matches_frame_loop(self):
        t = frame_times(2000, jitter=0.004, seed=1)
        x = np.cumsum(np.random.default_rng(2).normal(0, 0.1, 2000))
        a = nr._alphas(t)
        out = nr.smooth_forward(a, x)
        # the same recursion, one frame at a time
        ref = np.empty_like(x)
        y = 0.0
        for k in range(len(x)):
            y = y + (x[k] - y) * a[k]
            ref[k] = y
        np.testing.assert_allclose(out, ref, atol=1e-9)

    def test_adjoint_matches_explicit_matrix(self):
        t = frame_times(60, jitter=0.003, seed=3)
        a = nr._alphas(t)
        # build S column by column, then compare S^T r with the adjoint
        S = np.column_stack([nr.smooth_forward(a, e) for e in np.eye(len(t))])
        r = np.random.default_rng(4).normal(size=len(t))
        np.testing.assert_allclose(nr.smooth_adjoint(a, r), S.T @ r, atol=1e-9)

    def test_adjoint_inner_product(self):
        # <S x, r> == <x, S^T r> for any x and r
        t = frame_times(300, jitter=0.004, seed=5)
        a = nr._alphas(t)
        rng = np.random.default_rng(6)
        x, r = rng.normal(size=300), rng.normal(size=300)
        self.assertAlmostEqual(nr.smooth_forward(a, x) @ r, x @ nr.smooth_adjoint(a, r), places=9)


class TestRecovery(unittest.TestCase):

    def round_trip(self, x, t, lam=nr.LAMBDA, phase=0.0):
        ''' Return the recovered and true tick-grid values for a trace. Takes the trace and frame times. '''

        y, sample_times, truth = nr.network_forward(t, x, phase=phase)
        times, vals = nr.recover_trace(t, y, lam=lam)
        keep = (times > t[0] + nr.EDGE_S) & (times < t[-1] - nr.EDGE_S)
        return vals[keep], np.interp(times[keep], t, x)

    def test_constant_is_recovered(self):
        t = frame_times(3000)
        got, want = self.round_trip(np.full(3000, 4.2), t)
        np.testing.assert_allclose(got, want, atol=1e-3)

    def test_moving_trace_is_recovered(self):
        # with frame times known exactly, the path is invertible and needs no penalty
        t = frame_times(4000, jitter=0.004, seed=7)
        x = np.cumsum(np.random.default_rng(8).normal(0, 0.05, 4000))
        got, want = self.round_trip(x, t, lam=0.0)
        self.assertLess(np.median(np.abs(got - want)), 0.01)
        self.assertLess(abs(np.abs(np.diff(got)).sum() / np.abs(np.diff(want)).sum() - 1), 0.05)

    def test_recovers_stillness_the_logged_trace_loses(self):
        # bursts of walking between spells of standing still, the case the smoothing destroys
        t = frame_times(4000, jitter=0.003, seed=9)
        rng = np.random.default_rng(10)
        step = np.zeros(4000)
        for k in range(0, 4000, 60):
            if rng.random() < 0.5:
                step[k:k + 30] = rng.normal(0, 2.0)
        x = np.cumsum(step)
        y, times, _ = nr.network_forward(t, x)
        got, want = self.round_trip(x, t, lam=0.0)
        still = lambda v: np.mean(np.abs(np.diff(v)) < 0.5)
        logged_still = np.mean(np.abs(np.diff(np.interp(times, t, y))) < 0.5)
        self.assertGreater(still(got), logged_still - 1e-9)
        self.assertLess(abs(still(got) - still(want)), 0.05)

    def test_phase_is_found(self):
        # the tick grid is not aligned to the first frame in real data
        t = frame_times(3000, jitter=0.004, seed=11)
        x = np.cumsum(np.random.default_rng(12).normal(0, 0.05, 3000))
        got, want = self.round_trip(x, t, phase=0.017)
        self.assertLess(np.median(np.abs(got - want)), 0.08)

    def test_penalty_helps_when_frame_times_are_uncertain(self):
        """ The solver only has logged timestamps, which are not Unity's own frame deltas. With that
            mismatch, no penalty lets the solution chase noise and inflates total distance. """

        true_t = frame_times(3000, jitter=0.004, seed=13)
        x = np.repeat(np.random.default_rng(14).normal(0, 1, 30), 100)[:3000]
        y, _, _ = nr.network_forward(true_t, x)
        # the times the solver sees are the true ones with a millisecond of noise
        seen_t = true_t + np.random.default_rng(15).normal(0, 0.001, 3000)
        seen_t = np.maximum.accumulate(seen_t)
        want = np.interp(true_t, true_t, x)
        total = lambda v: np.abs(np.diff(v)).sum()
        want_total = total(np.interp(np.arange(true_t[0], true_t[-1], nr.TICK_S), true_t, x))
        errs = {}
        for lam in (0.0, nr.LAMBDA):
            times, vals = nr.recover_trace(seen_t, y, lam=lam)
            keep = (times > true_t[0] + nr.EDGE_S) & (times < true_t[-1] - nr.EDGE_S)
            errs[lam] = abs(total(vals[keep]) / want_total - 1)
        self.assertLess(errs[nr.LAMBDA], errs[0.0])


class TestPlayerFrames(unittest.TestCase):

    def session(self, n=3000):
        t0 = pd.Timestamp('1900-01-01 12:00:00')
        t = frame_times(n, jitter=0.003, seed=15)
        rng = np.random.default_rng(16)
        df = pd.DataFrame({'timeLocal': t0 + pd.to_timedelta(t, unit='s')})
        for p in (0, 1):
            df[globals.PLAYER_LOC_DICT[p]['xloc']] = np.cumsum(rng.normal(0, 0.05, n))
            df[globals.PLAYER_LOC_DICT[p]['yloc']] = np.cumsum(rng.normal(0, 0.05, n))
            df[globals.PLAYER_ROT_DICT[p]['yrot']] = np.mod(np.cumsum(rng.normal(0, 1.0, n)), 360)
        return df

    def test_recover_and_sample_share_a_grid(self):
        df = self.session()
        rec = nr.recover_player(df, 1)
        loc = nr.sample_player(df, 0, rec['t'].to_numpy())
        self.assertEqual(list(rec.columns), ['t', 'x', 'z', 'yaw'])
        np.testing.assert_allclose(rec['t'], loc['t'])
        # tick spacing, give or take the phase shift where two blocks meet
        gaps = np.diff(rec['t'].to_numpy())
        self.assertAlmostEqual(np.median(gaps), nr.TICK_S, places=9)
        self.assertTrue(np.all((gaps > 0.5 * nr.TICK_S) & (gaps < 1.5 * nr.TICK_S)))
        self.assertGreater(loc['yaw'].max() - loc['yaw'].min(), 0)

    def test_sample_player_reads_the_local_trace(self):
        df = self.session()
        rec = nr.recover_player(df, 1)
        loc = nr.sample_player(df, 0, rec['t'].to_numpy())
        t = (df['timeLocal'] - df['timeLocal'].iloc[0]).dt.total_seconds().to_numpy()
        want = np.interp(rec['t'].to_numpy(), t, df[globals.PLAYER_LOC_DICT[0]['xloc']])
        np.testing.assert_allclose(loc['x'], want, atol=1e-9)


class TestTickTable(unittest.TestCase):

    def session(self, n=3000, n_trials=3):
        df = TestPlayerFrames().session(n)
        # label the frames as three trials, each ending in an ITI
        per = n // n_trials
        df[globals.TRIAL_NUM] = np.repeat(np.arange(1.0, n_trials + 1), per)[:n]
        epoch = np.tile(np.array(['slices active'] * (per // 2) + [globals.ITI] * (per - per // 2)), n_trials)
        df['trial_epoch'] = epoch[:n]
        return df

    def test_columns_and_grid(self):
        tab = nr.session_tick_table([self.session()], 7)
        self.assertEqual(tab.columns.tolist(),
                         ['experiment', 't', 'p1_x', 'p1_z', 'p1_yaw', 'p0_x', 'p0_z', 'p0_yaw',
                          'trial_num', 'is_iti'])
        self.assertTrue((tab['experiment'] == 7).all())
        gaps = np.diff(tab['t'].to_numpy())
        self.assertAlmostEqual(np.median(gaps), nr.TICK_S, places=9)

    def test_trial_and_epoch_follow_the_frames(self):
        df = self.session()
        tab = nr.session_tick_table([df], 0)
        t = (df['timeLocal'] - df['timeLocal'].iloc[0]).dt.total_seconds().to_numpy()
        # every tick takes the trial number and epoch of the frame it falls in
        frame = np.clip(np.searchsorted(t, tab['t'].to_numpy()), 0, len(t) - 1)
        np.testing.assert_allclose(tab['trial_num'], df[globals.TRIAL_NUM].to_numpy()[frame])
        want_iti = pd.Series(df['trial_epoch'].to_numpy()[frame]).isin([globals.ITI, globals.TRIAL_STARTED])
        np.testing.assert_array_equal(tab['is_iti'], want_iti.to_numpy())
        self.assertTrue(0 < tab['is_iti'].mean() < 1)

    def test_player_0_is_not_recovered(self):
        # player 0 is read straight off its own trace, so it matches the logged values at those instants
        df = self.session()
        tab = nr.session_tick_table([df], 0)
        t = (df['timeLocal'] - df['timeLocal'].iloc[0]).dt.total_seconds().to_numpy()
        want = np.interp(tab['t'].to_numpy(), t, df[globals.PLAYER_LOC_DICT[0]['xloc']])
        np.testing.assert_allclose(tab['p0_x'], want, atol=1e-6)


class TestFramesFromTicks(unittest.TestCase):

    def test_only_the_named_player_changes_and_values_follow_the_ticks(self):
        df = TestPlayerFrames().session(2000)
        rec = nr.recover_player(df, 1)
        out = nr.frames_from_ticks(df, 1, rec)
        x1 = globals.PLAYER_LOC_DICT[1]['xloc']
        x0 = globals.PLAYER_LOC_DICT[0]['xloc']
        # player 0 untouched, player 1 replaced
        pd.testing.assert_series_equal(out[x0], df[x0])
        self.assertFalse(np.allclose(out[x1], df[x1]))
        # each frame reads the recovered trace at its own time
        t = (df['timeLocal'] - df['timeLocal'].iloc[0]).dt.total_seconds().to_numpy()
        want = np.interp(t, rec['t'].to_numpy(), rec['x'].to_numpy())
        np.testing.assert_allclose(out[x1], want, atol=1e-9)
        # yaw comes back inside the logged range, and the frame count is unchanged
        yaw = out[globals.PLAYER_ROT_DICT[1]['yrot']].to_numpy()
        self.assertTrue(((yaw >= 0) & (yaw < 360)).all())
        self.assertEqual(len(out), len(df))


class TestStillnessCutoffs(unittest.TestCase):

    def test_cutoffs_match_the_frame_thresholds(self):
        # the tick cutoffs are the agent per-frame thresholds over the same physical time
        import analysis.model_characterisation as mc
        per_tick = nr.TICK_S * nr.LOGGED_FPS
        self.assertAlmostEqual(nr.STILL_ROT_PER_TICK, mc.STATIONARY_ROT * per_tick, places=2)
        self.assertAlmostEqual(nr.STILL_SPEED_PER_TICK, mc.STATIONARY_SPEED * per_tick, places=2)

    def test_cutoffs_are_above_the_safe_floor(self):
        # finer cutoffs still separate a recovered player from a locally logged one
        self.assertGreater(nr.STILL_ROT_PER_TICK, nr.MIN_SAFE_ROT_PER_TICK)


if __name__ == '__main__':
    unittest.main()
