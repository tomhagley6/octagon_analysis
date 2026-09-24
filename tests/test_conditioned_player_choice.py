# Tests for analysis/conditioned_player_choice.py, covering the screen view that opponent visibility is
# measured on. A player can only act on what was drawn, so the visibility condition uses that player's own
# screen, while choices, walls and outcomes stay on the logged trials.
#
# The risk these cover is index alignment: the screen view is cut by the same filters as the logged trials,
# and a slip there would silently pair a trial with another trial's visibility.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_conditioned_player_choice -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
import analysis.conditioned_player_choice as cpc
import analysis.opponent_visibility as opponent_visibility


def trial(trial_num, walls=(1, 2), positions=((0.0, 0.0), (5.0, 5.0)), yaws=(0.0, 180.0), winner=0):
    ''' Return a minimal trial with a slice onset carrying both players' positions and head angles.
        Takes the trial number, the two walls, each player's position and yaw, and the winner. '''

    events = [globals.TRIAL_START, globals.SLICE_ONSET, globals.SELECTED_TRIGGER_ACTIVATION,
              globals.TRIAL_END]
    t0 = pd.Timestamp('1900-01-01 12:00:00') + pd.Timedelta(seconds=10 * trial_num)
    rows = []
    for i, ev in enumerate(events):
        row = {'timeLocal': t0 + pd.Timedelta(seconds=i), 'eventDescription': ev,
               globals.TRIAL_NUM: float(trial_num), globals.WALL_1: walls[0], globals.WALL_2: walls[1],
               globals.TRIAL_TYPE: globals.HIGH_LOW,
               globals.TRIGGER_CLIENT: winner if ev == globals.SELECTED_TRIGGER_ACTIVATION else np.nan}
        for p in (0, 1):
            row[globals.PLAYER_LOC_DICT[p]['xloc']] = positions[p][0]
            row[globals.PLAYER_LOC_DICT[p]['yloc']] = positions[p][1]
            row[globals.PLAYER_ROT_DICT[p]['yrot']] = yaws[p]
        rows.append(row)
    return pd.DataFrame(rows, index=range(100 * trial_num, 100 * trial_num + len(rows)))


def session(n=6, yaws=(0.0, 180.0)):
    ''' Return a session of n trials. Takes the trial count and the two players' head angles. '''

    return [trial(i + 1, yaws=yaws) for i in range(n)]


class TestScreenViewPlumbing(unittest.TestCase):

    def test_visibility_is_measured_on_the_view_when_one_is_given(self):
        ''' The view decides the visibility answer, so turning the viewer around in the view alone must
            change what the function sees. '''

        logged = session(yaws=(0.0, 180.0))
        # in the view, player 0 faces the other way, so the opponent falls outside its field of view
        turned = session(yaws=(180.0, 180.0))
        on_logged = opponent_visibility.get_other_visible_session_from_trials(0, logged, 110.0)
        on_view = opponent_visibility.get_other_visible_session_from_trials(0, turned, 110.0)
        self.assertFalse(np.array_equal(on_logged, on_view),
                         'the fixture does not distinguish the two views, so the test proves nothing')

    def test_no_views_leaves_the_result_unchanged(self):
        ''' Passing nothing must behave exactly as before, so existing analyses are unaffected. '''

        trial_lists = [session(), session()]
        args = dict(inferred_choice=False, current_fov=110.0, other_visible=True, debug=False)
        before = cpc.probability_trial_outcome_given_conditions_all_sessions(trial_lists, **args)[0]
        after = cpc.probability_trial_outcome_given_conditions_all_sessions(
            trial_lists, screen_views=None, **args)[0]
        np.testing.assert_array_equal(before, after)

    def test_a_view_identical_to_the_log_changes_nothing(self):
        ''' The plumbing itself must be neutral. Handing back the same trials must not move the answer. '''

        trial_lists = [session(), session()]
        views = [{0: tl, 1: tl} for tl in trial_lists]
        args = dict(inferred_choice=False, current_fov=110.0, other_visible=True, debug=False)
        plain = cpc.probability_trial_outcome_given_conditions_all_sessions(trial_lists, **args)[0]
        viewed = cpc.probability_trial_outcome_given_conditions_all_sessions(
            trial_lists, screen_views=views, **args)[0]
        np.testing.assert_array_equal(plain, viewed)

    def test_the_view_is_cut_by_the_same_filters_as_the_logged_trials(self):
        ''' The view is cut to match by tracked indices. If that slipped, a trial would be paired with
            another trial's visibility.

            The view here is identical on the trials the filter keeps and turned around on the trials it
            drops. Correct indexing therefore reproduces the logged answer exactly, and any slip pulls a
            dropped trial's view into a kept trial and changes it. '''

        # alternate the wall separation, so a wall_sep filter keeps every other trial
        logged = [trial(i + 1, walls=(1, 2) if i % 2 == 0 else (1, 5)) for i in range(16)]
        view = [trial(i + 1, walls=(1, 2) if i % 2 == 0 else (1, 5),
                      yaws=(0.0, 180.0) if i % 2 == 0 else (180.0, 180.0)) for i in range(16)]
        args = dict(inferred_choice=False, current_fov=110.0, other_visible=True, wall_sep=1, debug=False)

        plain, _, plain_denom = cpc.probability_trial_outcome_given_conditions_all_sessions([logged], **args)
        viewed, _, view_denom = cpc.probability_trial_outcome_given_conditions_all_sessions(
            [logged], screen_views=[{0: view, 1: view}], **args)
        # nan compares equal to nan, so check the comparison is not vacuous before trusting it
        self.assertTrue(np.isfinite(plain).any(), 'no probability was computed, so this proves nothing')
        np.testing.assert_array_equal(plain, viewed)
        np.testing.assert_array_equal(plain_denom[0][0], view_denom[0][0])

    def test_a_view_that_differs_on_kept_trials_does_change_the_answer(self):
        ''' The control for the test above. If the view were ignored, everything here would pass while
            doing nothing, so a view that differs where it matters must move the result. '''

        logged = session(yaws=(0.0, 180.0))
        turned = session(yaws=(180.0, 180.0))
        args = dict(inferred_choice=False, current_fov=110.0, other_visible=True, debug=False)

        plain = cpc.probability_trial_outcome_given_conditions_all_sessions([logged], **args)[2]
        viewed = cpc.probability_trial_outcome_given_conditions_all_sessions(
            [logged], screen_views=[{0: turned, 1: turned}], **args)[2]
        # the denominator is the set of trials that passed the visibility filter, so it must differ
        self.assertNotEqual(len(plain[0][0]), len(viewed[0][0]))

if __name__ == '__main__':
    unittest.main()
