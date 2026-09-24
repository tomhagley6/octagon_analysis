# Tests for the timing of the visibility metrics in analysis/model_characterisation.py. Synthetic
# trials check that a visibility array column is converted to seconds with the trial's logged times,
# which matters because the human logs run at about 48 Hz rather than the nominal 50.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_visibility_timing -v

import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
import analysis.model_characterisation as mc


HUMAN_STEP = 1 / 48.25          # a human log's row spacing, slower than the nominal 50 Hz


def make_trial(n_frames=60, step_s=HUMAN_STEP, walls=(1, 2), start_row=0, times=True):
    ''' Return a trial with trial start, slice onset, n_frames time rows, trigger and trial end.
        Takes the frame count, the row spacing and the trial's two walls.

        Rows are spaced by step_s, so the slice-onset row sits at index 1 and the trigger at
        n_frames + 1. times=False drops the timeApplication column, to check the fallback. '''

    events = [globals.TRIAL_START, globals.SLICE_ONSET] + ['time'] * (n_frames - 1) \
             + [globals.SELECTED_TRIGGER_ACTIVATION, globals.TRIAL_END]
    rows = []
    for i, ev in enumerate(events):
        row = {'eventDescription': ev, globals.TRIAL_NUM: 1.0,
               globals.WALL_1: walls[0], globals.WALL_2: walls[1]}
        if times:
            row['timeApplication'] = pd.Timedelta(seconds=100 + i * step_s)
        rows.append(row)
    return pd.DataFrame(rows, index=range(start_row, start_row + len(rows)))


def fake_visible(n_cols, first_visible=None, walls=(1, 2)):
    ''' Return an 8 x n_cols visibility array with wall 1 visible from first_visible on.
        Takes the column count and the first visible column (None for never visible). '''

    wv = np.zeros((8, n_cols), dtype=bool)
    if first_visible is not None:
        wv[walls[0] - 1, first_visible:] = True
    return wv


class TestWindowTimes(unittest.TestCase):

    def test_seconds_from_slice_onset(self):
        trial = make_trial(n_frames=10)
        t = mc.window_times(trial, 11)                  # slice onset row to the trigger row
        self.assertEqual(len(t), 11)
        self.assertAlmostEqual(t[0], 0.0)
        np.testing.assert_allclose(t, np.arange(11) * HUMAN_STEP, atol=1e-9)

    def test_numeric_times_accepted(self):
        trial = make_trial(n_frames=10)
        trial['timeApplication'] = trial['timeApplication'].dt.total_seconds()
        np.testing.assert_allclose(mc.window_times(trial, 11), np.arange(11) * HUMAN_STEP, atol=1e-9)

    def test_missing_column_returns_none(self):
        self.assertIsNone(mc.window_times(make_trial(times=False), 11))

    def test_missing_slice_onset_returns_none(self):
        trial = make_trial()
        trial['eventDescription'] = trial['eventDescription'].replace(globals.SLICE_ONSET, 'time')
        self.assertIsNone(mc.window_times(trial, 11))

    def test_too_few_rows_returns_none(self):
        # asking for more columns than the trial has rows after slice onset
        self.assertIsNone(mc.window_times(make_trial(n_frames=10), 500))

    def test_missing_timestamp_returns_none(self):
        trial = make_trial(n_frames=10)
        trial.loc[trial.index[4], 'timeApplication'] = pd.NaT
        self.assertIsNone(mc.window_times(trial, 11))


class TestVisibilityMetrics(unittest.TestCase):
    ''' The visibility array runs from the slice-onset row to the trigger row, so its column count is
        n_frames + 1 on these trials. get_wall_visible is replaced by a known array. '''

    def metrics(self, trial, wv):
        with mock.patch('trajectory_analysis.trajectory_headangle.get_wall_visible', return_value=wv):
            return mc.visibility_metrics(trial, 0)

    def test_latency_uses_logged_times(self):
        trial = make_trial(n_frames=100)
        res = self.metrics(trial, fake_visible(101, first_visible=48))
        # 48 rows after slice onset, at the log's own spacing rather than 48 / 50
        self.assertAlmostEqual(res['latency_to_wall_visible'], 48 * HUMAN_STEP, places=9)
        self.assertGreater(res['latency_to_wall_visible'], 48 / mc.FS)
        self.assertFalse(res['visible_at_onset'])
        self.assertTrue(res['ever_visible'])

    def test_trial_s_is_onset_to_trigger(self):
        trial = make_trial(n_frames=100)
        res = self.metrics(trial, fake_visible(101, first_visible=0))
        self.assertAlmostEqual(res['trial_s'], 100 * HUMAN_STEP, places=9)
        self.assertTrue(res['visible_at_onset'])
        self.assertEqual(res['latency_to_wall_visible'], 0.0)

    def test_never_visible(self):
        trial = make_trial(n_frames=100)
        res = self.metrics(trial, fake_visible(101, first_visible=None))
        self.assertTrue(np.isnan(res['latency_to_wall_visible']))
        self.assertFalse(res['ever_visible'])

    def test_falls_back_to_nominal_rate(self):
        # without logged times, the frame index is converted with FS as before
        trial = make_trial(n_frames=100, times=False)
        res = self.metrics(trial, fake_visible(101, first_visible=48))
        self.assertAlmostEqual(res['latency_to_wall_visible'], 48 / mc.FS, places=9)
        self.assertAlmostEqual(res['trial_s'], 100 / mc.FS, places=9)

    def test_fifty_hz_log_unchanged(self):
        # an agent sim logs at exactly 50 Hz, so the logged times give the old answer
        trial = make_trial(n_frames=100, step_s=1 / 50)
        res = self.metrics(trial, fake_visible(101, first_visible=48))
        self.assertAlmostEqual(res['latency_to_wall_visible'], 48 / mc.FS, places=9)

    def test_too_short_trial_returns_none(self):
        self.assertIsNone(self.metrics(make_trial(), np.nan))


if __name__ == '__main__':
    unittest.main()
