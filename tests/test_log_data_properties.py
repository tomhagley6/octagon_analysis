# Tests for properties of the logged data that analysis code relies on, and that would change silently if
# the sessions were rebuilt from a different Unity build.
#
# These load the saved session pickles, so they are slower than the rest of the suite. Each one pins a
# fact that is argued from the Unity source in READMEs/network_compensation.md.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_log_data_properties -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_strings
import globals

SOCIAL = os.path.join(data_strings.DATAFRAME_PATH, 'social_trials_list.pkl')
SOLO = os.path.join(data_strings.DATAFRAME_PATH, 'solo_trials_list.pkl')
TIME_COLS = ('timeLocal', 'timeApplication', 'timeReferenced')


@unittest.skipUnless(os.path.exists(SOCIAL), 'session pickles not available')
class TestSocialLogProperties(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.trial_lists = pd.read_pickle(SOCIAL)

    def test_player_1_pitch_and_roll_are_a_copy_of_player_0(self):
        ''' The camera is not networked and ArenaLogger finds the host's for both players, so player 1
            has no head pitch of its own. Pinned so nothing starts reading it as if it did. '''

        for i, session in enumerate(self.trial_lists):
            df = pd.concat(session)
            for axis in ('xrot', 'zrot'):
                a = df[globals.PLAYER_ROT_DICT[0][axis]].to_numpy(float)
                b = df[globals.PLAYER_ROT_DICT[1][axis]].to_numpy(float)
                keep = ~np.isnan(a) & ~np.isnan(b)
                self.assertTrue(np.array_equal(a[keep], b[keep]),
                                f'session {i} {axis} is no longer a copy of player 0')

    def test_player_yaw_differs_between_players(self):
        ''' Yaw is the one rotation axis that is networked, so it must not match. This is the control for
            the test above: it shows the copy is a logging fault and not identical players. '''

        for i, session in enumerate(self.trial_lists):
            df = pd.concat(session)
            a = df[globals.PLAYER_ROT_DICT[0]['yrot']].to_numpy(float)
            b = df[globals.PLAYER_ROT_DICT[1]['yrot']].to_numpy(float)
            keep = ~np.isnan(a) & ~np.isnan(b)
            self.assertFalse(np.array_equal(a[keep], b[keep]), f'session {i} yaw is a copy of player 0')

    def test_timestamps_are_written_to_the_millisecond(self):
        ''' Every time column is whole milliseconds, so a frame interval carries half a millisecond of
            rounding. network_compensation reads the smoothing weight from that interval, and the penalty
            it uses is tuned against this. '''

        df = pd.concat(self.trial_lists[0])
        for col in TIME_COLS:
            # timeLocal is a datetime and the others are timedeltas, so all three are read as an offset
            ns = (df[col] - df[col].iloc[0]).dt.total_seconds().to_numpy() * 1e9
            self.assertTrue(np.allclose(np.round(ns) % 1e6, 0),
                            f'{col} is no longer millisecond resolution')

    def test_logging_runs_near_the_fixed_timestep(self):
        ''' The logger writes from FixedUpdate against a 0.02 s accumulator, so rows land near 50 Hz and
            not at the render rate. Frame intervals matching the render clock would mean the smoothing
            model in network_compensation should be read frame by frame instead. '''

        df = pd.concat(self.trial_lists[0])
        t = (df['timeLocal'] - df['timeLocal'].iloc[0]).dt.total_seconds().to_numpy()
        dt = np.diff(t)
        self.assertGreater(np.median(dt), 0.018)
        self.assertLess(np.median(dt), 0.024)


@unittest.skipUnless(os.path.exists(SOLO), 'session pickles not available')
class TestSoloLogProperties(unittest.TestCase):

    def test_solo_sessions_have_no_player_1_columns(self):
        ''' A solo session logs one player, so nothing in it goes through the network path. '''

        df = pd.concat(pd.read_pickle(SOLO)[0])
        for axis in ('xrot', 'yrot', 'zrot'):
            self.assertNotIn(globals.PLAYER_ROT_DICT[1][axis], df.columns)


if __name__ == '__main__':
    unittest.main()
