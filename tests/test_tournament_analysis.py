# Tests for the score decomposition in analysis/tournament_analysis.py. Synthetic perspective
# tables with known win rates and wall choices check each quantity against a hand-worked answer.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_tournament_analysis -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.tournament_analysis as ta


HIGH, LOW = ta.HIGH, ta.LOW


def make_perspective(rows):
    ''' Return a perspective-style table from a list of (focus, opp, outcome) triples.
        Takes outcome as 'high', 'low' or 'loss', meaning the focus model won at High, won at
        Low, or did not win. Both sides of a trial are the caller's job, since every function
        under test groups by focus. '''

    score = {'high': HIGH, 'low': LOW, 'loss': 0.0}
    recs = [dict(matchup=f"{f}__vs__{o}", focus=f, opp=o, f_score=score[out],
                 f_reward=score[out] - 0.45, f_win=float(out != 'loss'), sep=1)
            for f, o, out in rows]
    return pd.DataFrame(recs, columns=['matchup', 'focus', 'opp', 'f_score', 'f_reward',
                                       'f_win', 'sep'])


def repeat(focus, opp, n_high, n_low, n_loss):
    ''' Return the (focus, opp, outcome) triples for a given count of each outcome.
        Takes the two models and the three counts. '''

    return ([(focus, opp, 'high')] * n_high + [(focus, opp, 'low')] * n_low
            + [(focus, opp, 'loss')] * n_loss)


class TestScoreDecomposition(unittest.TestCase):
    ''' Mean score must split exactly into win rate times the value of a win, and the split
        must be able to disagree with the score ranking. '''

    def setUp(self):
        # model 20 wins 6 of 10 and always at High; model 34 wins 4 of 10, half at each wall
        rows = repeat(20, 34, n_high=6, n_low=0, n_loss=4) + repeat(34, 20, n_high=2, n_low=2, n_loss=6)
        self.selfless = make_perspective(rows)
        self.strategy = {20: 'A', 34: 'B'}
        self.decomp = ta.score_decomposition(self.selfless, self.strategy, models=[20, 34],
                                             order=['A', 'B'], verbose=False)

    def row(self, model):
        return self.decomp.set_index('model').loc[model]

    def test_win_rate(self):
        self.assertAlmostEqual(self.row(20).win_rate, 0.6)
        self.assertAlmostEqual(self.row(34).win_rate, 0.4)

    def test_value_per_win(self):
        # every win at High is worth 1.0; two High and two Low wins average 0.7
        self.assertAlmostEqual(self.row(20).value_per_win, HIGH)
        self.assertAlmostEqual(self.row(34).value_per_win, (2 * HIGH + 2 * LOW) / 4)

    def test_high_share_counts_won_trials_only(self):
        self.assertAlmostEqual(self.row(20).high_share, 1.0)
        self.assertAlmostEqual(self.row(34).high_share, 0.5)

    def test_identity_holds_exactly(self):
        # mean score = win rate x value per win, since a loss pays 0
        for m in (20, 34):
            r = self.row(m)
            self.assertAlmostEqual(r.win_rate * r.value_per_win, r.mean_score, places=12)

    def test_mean_score(self):
        self.assertAlmostEqual(self.row(20).mean_score, 0.6)
        self.assertAlmostEqual(self.row(34).mean_score, 0.4 * 0.7)

    def test_sorted_by_mean_score(self):
        self.assertEqual(list(self.decomp.model), [20, 34])
        self.assertEqual(list(self.decomp.score_rank), [1, 2])

    def test_strategy_carried_through(self):
        self.assertEqual(list(self.decomp.strategy), ['A', 'B'])

    def test_win_rate_se(self):
        # binomial standard error on the win rate, over the model's own trials
        r = self.row(20)
        self.assertAlmostEqual(r.win_rate_se, np.sqrt(0.6 * 0.4 / 10))


class TestRankShift(unittest.TestCase):
    ''' The whole point of the decomposition is that the two factors can disagree, so a model
        that wins less often but always at High can still out-score. '''

    def setUp(self):
        # 20 wins 4 of 10, all at High (score 0.4); 34 wins 6 of 10, all at Low (score 0.24)
        rows = repeat(20, 34, n_high=4, n_low=0, n_loss=6) + repeat(34, 20, n_high=0, n_low=6, n_loss=4)
        self.decomp = ta.score_decomposition(make_perspective(rows), {20: 'A', 34: 'B'},
                                             models=[20, 34], order=['A', 'B'], verbose=False)

    def test_score_and_win_rate_disagree(self):
        d = self.decomp.set_index('model')
        self.assertLess(d.loc[20].win_rate, d.loc[34].win_rate)         # 20 wins less often
        self.assertGreater(d.loc[20].mean_score, d.loc[34].mean_score)  # but scores more

    def test_rank_shift_signs(self):
        # positive rank_shift = the model climbs when judged on win rate alone
        d = self.decomp.set_index('model')
        self.assertEqual(d.loc[20].score_rank, 1)
        self.assertEqual(d.loc[20].win_rate_rank, 2)
        self.assertEqual(d.loc[20].rank_shift, -1)
        self.assertEqual(d.loc[34].rank_shift, 1)


class TestNeverWins(unittest.TestCase):
    ''' A model that never wins has no value per win to report, and must not break the table. '''

    def setUp(self):
        rows = repeat(20, 34, n_high=10, n_low=0, n_loss=0) + repeat(34, 20, n_high=0, n_low=0, n_loss=10)
        self.decomp = ta.score_decomposition(make_perspective(rows), {20: 'A', 34: 'B'},
                                             models=[20, 34], order=['A', 'B'], verbose=False)

    def test_value_per_win_is_nan_not_an_error(self):
        d = self.decomp.set_index('model')
        self.assertTrue(np.isnan(d.loc[34].value_per_win))
        self.assertTrue(np.isnan(d.loc[34].high_share))

    def test_mean_score_is_zero(self):
        self.assertAlmostEqual(self.decomp.set_index('model').loc[34].mean_score, 0.0)


class TestValuePerWinMatrix(unittest.TestCase):
    ''' The head-to-head value of a win is mean score divided by win rate, and must stay
        inside the two payouts. '''

    def setUp(self):
        rows = (repeat(20, 34, n_high=6, n_low=0, n_loss=4) + repeat(34, 20, n_high=2, n_low=2, n_loss=6)
                + repeat(20, 39, n_high=2, n_low=3, n_loss=5) + repeat(39, 20, n_high=5, n_low=0, n_loss=5))
        self.selfless = make_perspective(rows)
        self.models = [20, 34, 39]
        self.mats = ta.head_to_head(self.selfless, self.models)
        self.vpw = ta.value_per_win(self.mats)

    def test_known_cells(self):
        self.assertAlmostEqual(self.vpw.loc[20, 34], HIGH)                      # all six wins at High
        self.assertAlmostEqual(self.vpw.loc[34, 20], 0.7)                       # two High, two Low
        self.assertAlmostEqual(self.vpw.loc[20, 39], (2 * HIGH + 3 * LOW) / 5)  # two High, three Low

    def test_bounded_by_the_two_payouts(self):
        vals = self.vpw.to_numpy(float)
        vals = vals[~np.isnan(vals)]
        self.assertTrue(np.all((vals >= LOW - 1e-12) & (vals <= HIGH + 1e-12)))

    def test_unplayed_pairs_are_nan(self):
        # 34 and 39 never met, so their cell has no value of a win
        self.assertTrue(np.isnan(self.vpw.loc[34, 39]))


class TestGroupSummary(unittest.TestCase):
    ''' The group means are the headline comparison, so they must weight by trials, not by model. '''

    def test_trial_weighted_group_means(self):
        # two models in group A with different trial counts, so a model-mean and a trial-mean differ
        rows = (repeat(20, 34, n_high=8, n_low=0, n_loss=2)                 # 10 trials, win rate 0.8
                + repeat(14, 34, n_high=2, n_low=0, n_loss=28)              # 30 trials, win rate 0.2
                + repeat(34, 20, n_high=0, n_low=2, n_loss=8) + repeat(34, 14, n_high=0, n_low=28, n_loss=2))
        strategy = {20: 'A', 14: 'A', 34: 'B'}
        summary = ta.group_summary(make_perspective(rows), strategy, order=['A', 'B'], verbose=False)
        a = summary.set_index('strategy').loc['A']
        self.assertEqual(a.n_trials, 40)
        self.assertAlmostEqual(a.win_rate, 10 / 40)                         # 8 + 2 wins over 40 trials
        self.assertAlmostEqual(a.value_per_win, HIGH)
        self.assertEqual(a.n_models, 2)


if __name__ == '__main__':
    unittest.main()
