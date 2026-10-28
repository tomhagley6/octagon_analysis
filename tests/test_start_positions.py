''' Tests for analysis/start_positions.py (designed peripheral start positions and onset headings).

Run with: /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_start_positions
'''

import unittest

import numpy as np

import analysis.analytical_race as ar
import analysis.start_positions as sp
import trajectory_analysis.visibility_occlusion as vo


ROT, CLOSE = 331.0, 7.57


def _central():
    return ar.central_times_inferred(ROT, CLOSE, n=360)


class TestDesignedPositions(unittest.TestCase):
    ''' The designed grid covers alcove 0's half-sector and stays in reachable space. '''

    def setUp(self):
        self.pos = sp.designed_positions()

    def test_inside_reachable_space(self):
        xy = self.pos[['x', 'z']].to_numpy()
        self.assertTrue(ar.inside(ar.reach_polygon(), xy).all())

    def test_half_sector(self):
        # every position lies on the clockwise half of alcove 0's sector (lateral >= 0, within 22.5 deg)
        self.assertTrue((self.pos['q'] >= 0).all())
        bearing = np.degrees(np.arctan2(self.pos['q'], self.pos['p']))
        self.assertTrue((bearing <= 22.5 + 1e-6).all())

    def test_regions(self):
        regions = set(self.pos['region'])
        for r in ('alcove, in trigger', 'alcove, not in trigger', 'in front of the wall', 'wall corner'):
            self.assertIn(r, regions)
        corner = self.pos[self.pos['region'] == 'wall corner'].iloc[0]
        # the wall corner sits where walls 0 and 1 meet, on the line between them
        self.assertAlmostEqual(np.degrees(np.arctan2(corner['q'], corner['p'])), 22.5, places=3)
        self.assertTrue((self.pos.loc[self.pos['region'] == 'alcove, in trigger', 'in_trigger']).all())

    def test_frame(self):
        x, z = vo.wall_frame_to_arena(self.pos['p'].to_numpy(), self.pos['q'].to_numpy(), 0)
        np.testing.assert_allclose(np.c_[x, z], self.pos[['x', 'z']].to_numpy())


class TestHeadingScores(unittest.TestCase):
    ''' heading_scores gives the final rule's expected outcome for each onset heading. '''

    @classmethod
    def setUpClass(cls):
        cls.ct = _central()
        cls.deltas = np.array([0.0, 1.05, 3.8])

    def test_matches_evaluate(self):
        p = np.array(ar._pt(15.5, 1.0, 0), float)
        h = 200.0
        res = sp.heading_scores(p, [h], ROT, CLOSE, self.ct, self.deltas)
        races = sp.position_races(p, h, ROT, CLOSE)
        for k, d in enumerate(self.deltas):
            ev = ar.evaluate(races, 'delta2', d, self.ct)
            w = races['prob'].to_numpy()
            self.assertAlmostEqual(res['periph_trial'][0, k], (w * ev['periph_score']).sum(), places=5)
            self.assertAlmostEqual(res['central_trial'][0, k], (w * ev['central_score']).sum(), places=5)
            self.assertAlmostEqual(res['race_s'][0, k], (w * ev['race_s']).sum(), places=5)

    def test_mirror_symmetry(self):
        # mirror a position and heading across alcove 0's centre line: the outcome is the same
        p = np.array(ar._pt(16.0, 2.0, 0), float)
        pm = np.array(ar._pt(16.0, -2.0, 0), float)
        for h in (150.0, 190.5, 240.0):
            hm = (-h) % 360
            a = sp.heading_scores(p, [h], ROT, CLOSE, self.ct, self.deltas)
            b = sp.heading_scores(pm, [hm], ROT, CLOSE, self.ct, self.deltas)
            np.testing.assert_allclose(a['periph_trial'], b['periph_trial'], atol=1e-6)
            np.testing.assert_allclose(a['sensed'], b['sensed'][:, sp.MIRROR_WALLS])

    def test_in_trigger_wins_own_wall(self):
        # at the alcove back, a lit own wall is an instant win whatever the heading
        p = np.array(ar._pt(19.6, 0.0, 0), float)
        races = sp.position_races(p, 180.0, ROT, CLOSE)
        ev = ar.evaluate(races, 'delta2', 1.05, self.ct)
        own_high = races['high'] == 0
        self.assertTrue((ev.loc[own_high, 'periph_win'] == 1).all())
        self.assertTrue((ev.loc[own_high, 'race_s'] == 0).all())


class TestBest(unittest.TestCase):
    ''' best_per_position picks the heading and delta with the highest score per second. '''

    def test_best(self):
        n_pos, n_head, n_d = 2, 3, 2
        res = dict(headings=np.array([0.0, 10.0, 20.0]), deltas=np.array([0.5, 1.0]),
                   periph_trial=np.zeros((n_pos, n_head, n_d)), central_trial=np.full((n_pos, n_head, n_d), 0.4),
                   race_s=np.ones((n_pos, n_head, n_d)), p_low=np.zeros((n_pos, n_head, n_d)),
                   periph_win=np.zeros((n_pos, n_head, n_d)), sensed=np.zeros((n_pos, n_head, 8), bool))
        res['periph_trial'][0, 1, 1] = 0.5
        res['periph_trial'][1, 2, 0] = 0.6
        best = sp.best_per_position(res, overhead=6.0)
        self.assertEqual(best.loc[0, 'heading'], 10.0)
        self.assertEqual(best.loc[0, 'delta'], 1.0)
        self.assertEqual(best.loc[1, 'heading'], 20.0)
        self.assertAlmostEqual(best.loc[1, 'periph_score_s'], 0.6 / 7.0)


if __name__ == '__main__':
    unittest.main()
