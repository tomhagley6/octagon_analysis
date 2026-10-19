# Tests for analysis/opponent_response.py. Hand-worked tables and straight-line trajectories check the
# stratified odds ratio, the position cells and the fixed-time heading.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_opponent_response -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.opponent_response as orsp
from analysis.race_estimate import ALCOVE_CENTRES


class TestOddsRatio(unittest.TestCase):

    def table(self, cells):
        ''' Return a table from (stratum, exposed, outcome, count) tuples. '''

        rows = [dict(s=s, e=e, y=y) for s, e, y, n in cells for _ in range(n)]
        return pd.DataFrame(rows)

    def test_single_stratum_matches_the_crude_odds_ratio(self):
        # exposed 8 yes / 2 no, unexposed 5 yes / 5 no: OR = (8*5)/(2*5) = 4
        df = self.table([(0, True, True, 8), (0, True, False, 2), (0, False, True, 5), (0, False, False, 5)])
        orr, used = orsp.mh_odds_ratio(df, df['e'], 'y', ['s'])
        self.assertAlmostEqual(orr, 4.0)
        self.assertEqual(used, 10)

    def test_strata_remove_a_confounder(self):
        # within each stratum the exposure does nothing, but exposure and outcome both pile into stratum 1
        df = self.table([(0, True, True, 1), (0, True, False, 4), (0, False, True, 10), (0, False, False, 40),
                         (1, True, True, 40), (1, True, False, 10), (1, False, True, 4), (1, False, False, 1)])
        orr, _ = orsp.mh_odds_ratio(df, df['e'], 'y', ['s'])
        self.assertAlmostEqual(orr, 1.0)
        crude = (41 * 41) / (14 * 14)
        self.assertGreater(crude, 5)

    def test_strata_with_one_group_are_skipped(self):
        df = self.table([(0, True, True, 3), (1, True, True, 2), (1, False, False, 2), (1, True, False, 1), (1, False, True, 1)])
        orr, used = orsp.mh_odds_ratio(df, df['e'], 'y', ['s'])
        self.assertEqual(used, 3)
        self.assertAlmostEqual(orr, 4.0)

    def test_within_strata_difference(self):
        df = pd.DataFrame(dict(s=[0, 0, 0, 1, 1], e=[True, False, False, True, False], v=[5.0, 1.0, 3.0, 10.0, 4.0]))
        # stratum 0: 5 - 2 = 3 (one exposed); stratum 1: 10 - 4 = 6 (one exposed)
        self.assertAlmostEqual(orsp.within_strata_difference(df, df['e'], 'v', ['s']), 4.5)


class TestPositions(unittest.TestCase):

    def test_position_cell_mirrors_to_clockwise_high(self):
        kh, kl = orsp.position_cell(np.array([2, 8, 3]), np.array([5, 5, 7]), np.array([1, 1, 3]))
        self.assertEqual(kh.tolist(), [1, 1, 0])
        self.assertEqual(kl.tolist(), [4, -4, 4])

    def test_facing_angle(self):
        # at the centre facing north, wall 1 is straight ahead, wall 3 at 90 degrees and wall 5 behind
        a = orsp.facing_angle(np.zeros(3), np.zeros((3, 2)), np.array([1, 3, 5]))
        self.assertTrue(np.allclose(a, [0, 90, 180], atol=1e-6))

    def test_heading_high_at(self):
        # move from the centre straight at wall 1's alcove, with Low at wall 5
        xy = np.linspace([0, 0], ALCOVE_CENTRES[0] * 0.8, 80)
        self.assertEqual(orsp.heading_high_at(xy, high=1, low=5), 1.0)
        self.assertEqual(orsp.heading_high_at(xy, high=5, low=1), 0.0)
        self.assertTrue(np.isnan(orsp.heading_high_at(xy[:40], high=1, low=5)))
        self.assertTrue(np.isnan(orsp.heading_high_at(np.zeros((80, 2)), high=1, low=5)))


if __name__ == '__main__':
    unittest.main()
