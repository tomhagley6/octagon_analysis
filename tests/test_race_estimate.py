# Tests for analysis/race_estimate.py. Synthetic race tables and time lists with known answers
# check the target rule, the time fit, the race arithmetic and the winner check.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_race_estimate -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
import analysis.race_estimate as re_


def make_rows(d_high, d_low, won, chose_high, rt_s, trial_num=None, player=None):
    ''' Return a race-table slice from lists of per-agent values. Takes the column lists. '''

    n = len(d_high)
    rm = pd.DataFrame(dict(d_high=d_high, d_low=d_low, won=won, chose_high=chose_high, rt_s=rt_s,
                           trial_num=trial_num if trial_num is not None else np.arange(n),
                           player=player if player is not None else np.zeros(n, int)))
    rm['d_chosen'] = np.where(rm['chose_high'], rm['d_high'], rm['d_low'])
    return rm


class TestAlcoveCentres(unittest.TestCase):

    def test_eight_walls_on_one_ring(self):
        # every alcove centre sits the same distance from the arena centre, between the wall
        # face and the alcove back
        r = np.linalg.norm(re_.ALCOVE_CENTRES, axis=1)
        self.assertEqual(re_.ALCOVE_CENTRES.shape, (8, 2))
        self.assertTrue(np.allclose(r, r[0]))
        self.assertTrue(globals.OCTAGON_WALLFACE_RADIUS < r[0] < globals.OCTAGON_ALCOVE_RADIUS)

    def test_wall_one_faces_north_and_walls_are_45_degrees_apart(self):
        x, z = re_.ALCOVE_CENTRES[0]
        self.assertAlmostEqual(x, 0, places=6)
        self.assertGreater(z, 0)
        ang = np.degrees(np.arctan2(re_.ALCOVE_CENTRES[:, 0], re_.ALCOVE_CENTRES[:, 1])) % 360
        self.assertTrue(np.allclose((np.diff(ang) + 180) % 360 - 180, 45))


class TestTargets(unittest.TestCase):

    def setUp(self):
        # agent 0 nearer High, agent 1 nearer Low; agent 1 won at High on a far trip
        self.rm = make_rows(d_high=[5, 20], d_low=[10, 3], won=[False, True],
                            chose_high=[False, True], rt_s=[2.0, 2.0])

    def test_nearest(self):
        d, high = re_.target_distance(self.rm, 'nearest')
        self.assertEqual(d.tolist(), [5, 3])
        self.assertEqual(high.tolist(), [True, False])

    def test_high(self):
        d, high = re_.target_distance(self.rm, 'high')
        self.assertEqual(d.tolist(), [5, 20])
        self.assertTrue(high.all())

    def test_observed_uses_real_choice_only_for_the_winner(self):
        d, high = re_.target_distance(self.rm, 'observed')
        self.assertEqual(d.tolist(), [5, 20])
        self.assertEqual(high.tolist(), [True, True])

    def test_p_high_draws_high_at_the_given_rate(self):
        rm = make_rows(d_high=[5.0] * 4000, d_low=[10.0] * 4000, won=[True] * 4000,
                       chose_high=[True] * 3000 + [False] * 1000, rt_s=[1.0] * 4000)
        # default p_high is the share of won trials at High (0.75)
        d, high = re_.target_distance(rm, 'p_high', rng=np.random.default_rng(0))
        self.assertAlmostEqual(high.mean(), 0.75, delta=0.02)
        self.assertTrue(np.all(d == np.where(high, 5.0, 10.0)))
        # a given p_high overrides it, and the ends are exact
        _, high = re_.target_distance(rm, 'p_high', p_high=1.0)
        self.assertTrue(high.all())
        _, high = re_.target_distance(rm, 'p_high', p_high=0.0)
        self.assertFalse(high.any())

    def test_bad_target(self):
        with self.assertRaises(ValueError):
            re_.target_distance(self.rm, 'furthest')


class TestFitAndPredict(unittest.TestCase):

    def test_fit_recovers_a_line_on_won_trials_only(self):
        d = np.array([2.0, 10, 20, 30])
        rm = make_rows(d_high=d, d_low=d + 5, won=[True] * 4, chose_high=[True] * 4,
                       rt_s=0.3 + 0.1 * d)
        # add a lost trial that would break the line if it were used
        rm = pd.concat([rm, make_rows([5.0], [6.0], [False], [True], [99.0])], ignore_index=True)
        fit = re_.fit_time_distance(rm)
        self.assertAlmostEqual(fit['slope'], 0.1)
        self.assertAlmostEqual(fit['intercept'], 0.3)
        self.assertAlmostEqual(fit['r'], 1.0)
        self.assertEqual(fit['n'], 4)

    def test_prediction_is_floored_at_the_fastest_win(self):
        fit = dict(slope=0.1, intercept=-0.5, r=1.0, min_rt=0.2, n=1)
        rm = make_rows(d_high=[1.0, 20.0], d_low=[50.0, 50.0], won=[True, True],
                       chose_high=[True, True], rt_s=[0.2, 1.5])
        t, high = re_.predict_times(rm, fit, 'nearest')
        self.assertTrue(np.allclose(t, [0.2, 1.5]))
        self.assertTrue(high.all())

    def test_noise_adds_the_won_trial_residuals(self):
        # won trials sit 0.1 s either side of the line, so every noisy time is the line +/- 0.1
        d = np.array([10.0, 20.0, 10.0, 20.0])
        rm = make_rows(d_high=d, d_low=d + 5, won=[True] * 4, chose_high=[True] * 4,
                       rt_s=0.1 * d + np.array([0.1, 0.1, -0.1, -0.1]))
        fit = re_.fit_time_distance(rm)
        t, _ = re_.predict_times(rm, fit, 'high', noise=True, rng=np.random.default_rng(1))
        off = np.round(t - 0.1 * d, 6)
        self.assertTrue(set(off) <= {0.1, -0.1})

    def test_default_target_is_p_high(self):
        # a model that always took High in self-play goes to High every time by default
        fit = dict(slope=0.1, intercept=0.0, r=1.0, min_rt=0.0, n=1)
        rm = make_rows(d_high=[20.0] * 10, d_low=[2.0] * 10, won=[True] * 10, chose_high=[True] * 10,
                       rt_s=[2.0] * 10)
        t, high = re_.predict_times(rm, fit)
        self.assertTrue(np.allclose(t, 2.0))
        self.assertTrue(high.all())


class TestRace(unittest.TestCase):

    def test_always_faster_wins_everything(self):
        out = re_.race([1.0, 1.0], [True, False], [2.0, 3.0], [1.0, 1.0])
        self.assertEqual(out['win_share'], 1.0)
        self.assertAlmostEqual(out['score_per_trial'], 0.7)        # half at High, half at Low
        self.assertEqual(out['opp_score_per_trial'], 0.0)
        self.assertAlmostEqual(out['score_rate'], 1.4 * 2 / 4.0)  # 2.8 score over 4 trials of 1 s

    def test_hand_worked_mixed_case(self):
        # focal 1 s (High) and 3 s (Low) against opponent 2 s (scores 1.0)
        out = re_.race([1.0, 3.0], [True, False], [2.0], [1.0])
        self.assertEqual(out['win_share'], 0.5)
        self.assertAlmostEqual(out['score_per_trial'], 0.5)
        self.assertAlmostEqual(out['opp_score_per_trial'], 0.5)
        self.assertAlmostEqual(out['score_rate'], 1.0 / 3.0)       # trial times 1 and 2
        self.assertAlmostEqual(out['opp_score_rate'], 1.0 / 3.0)

    def test_ties_split_and_same_distribution_is_even(self):
        out = re_.race([2.0], [True], [2.0], [1.0])
        self.assertEqual(out['win_share'], 0.5)
        rng = np.random.default_rng(0)
        t = rng.uniform(1, 3, 2000)
        out = re_.race(t, np.ones(2000, bool), t, np.ones(2000))
        self.assertAlmostEqual(out['win_share'], 0.5, places=6)


class TestEstimateMatchup(unittest.TestCase):

    def test_model_against_itself_is_even_and_nearer_model_wins(self):
        d = np.linspace(5, 30, 40)
        near = make_rows(d_high=d, d_low=d + 5, won=[True] * 40, chose_high=[True] * 40,
                         rt_s=0.1 * d + 0.2)
        far = near.copy()
        far[['d_high', 'd_low', 'd_chosen']] += 10
        far['rt_s'] += 1.0
        self.assertAlmostEqual(re_.estimate_matchup(near, near, noise=False)['win_share'], 0.5)
        self.assertGreater(re_.estimate_matchup(near, far, noise=False)['win_share'], 0.5)


class TestSelfplayWinnerCheck(unittest.TestCase):

    def test_perfect_and_reversed_predictions(self):
        fit = dict(slope=0.1, intercept=0.0, r=1.0, min_rt=0.0, n=2)
        # trial 0: agent 0 nearer and wins; trial 1: agent 1 nearer and wins
        rm = make_rows(d_high=[5, 15, 15, 5], d_low=[30] * 4, won=[True, False, False, True],
                       chose_high=[True] * 4, rt_s=[0.5] * 4,
                       trial_num=[0, 0, 1, 1], player=[0, 1, 0, 1])
        self.assertEqual(re_.selfplay_winner_check(rm, fit), 1.0)
        rm['won'] = ~rm['won']
        self.assertEqual(re_.selfplay_winner_check(rm, fit), 0.0)


if __name__ == '__main__':
    unittest.main()
