''' Tests for analysis/analytical_race.py, the geometric race model of wall choice.

Run with the conda env python:
    /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_analytical_race
'''

import math
import unittest

import numpy as np

import globals
import analysis.analytical_race as ar
import trajectory_analysis.visibility_occlusion as vo

W = globals.OCTAGON_WALLFACE_RADIUS
H = globals.ALCOVE_HALF_WIDTH
BACK = globals.OCTAGON_ALCOVE_RADIUS
TRIG = ar.TRIGGER_REACH


def wall_point(p, q, wall):
    ''' Return a point in one wall's frame as arena x, z. Takes p, q and a wall row index. '''

    return np.array(vo.wall_frame_to_arena(p, q, wall))


class TestGeometry(unittest.TestCase):

    def test_trigger_reach_matches_box_front_minus_body(self):
        # box front (physical back 20.357 less 1.5 depth) less the body radius
        self.assertAlmostEqual(TRIG, BACK + globals.PLAYER_RADIUS - 1.5 - globals.PLAYER_RADIUS, places=6)

    def test_polygon_closes_between_walls(self):
        # the last vertex of one wall's run meets the first vertex of the next clockwise wall
        L = W * math.tan(math.radians(22.5))
        for w in range(8):
            np.testing.assert_allclose(wall_point(W, L, w), wall_point(W, -L, (w + 1) % 8), atol=1e-9)

    def test_alcove_back_and_centre_inside(self):
        poly = ar.reach_polygon()
        self.assertTrue(ar.inside(poly, np.array([0.0, 0.0])))
        self.assertTrue(ar.inside(poly, wall_point(BACK - 0.01, 0, 3)))
        self.assertFalse(ar.inside(poly, wall_point(W + 1.0, H + 1.0, 3)))

    def test_centre_to_every_wall(self):
        d = ar.path_lengths(np.zeros((1, 2)))
        np.testing.assert_allclose(d, TRIG, atol=1e-9)

    def test_own_alcove_is_zero(self):
        d = ar.path_lengths(wall_point(BACK, 0, 2)[None])
        self.assertEqual(d[0, 2], 0.0)

    def test_opposite_alcove_is_straight_through_centre(self):
        d = ar.path_lengths(wall_point(BACK, 0, 0)[None])
        self.assertAlmostEqual(d[0, 4], BACK + TRIG, places=6)

    def test_adjacent_alcove_goes_round_the_mouth_corners(self):
        # back -> own mouth corner -> next wall's near mouth corner -> into its trigger
        start = wall_point(BACK, 0, 0)
        c1, c2 = wall_point(W, H, 0), wall_point(W, -H, 1)
        goal = wall_point(TRIG, -H, 1)
        expect = np.linalg.norm(c1 - start) + np.linalg.norm(c2 - c1) + np.linalg.norm(goal - c2)
        d = ar.path_lengths(start[None])
        self.assertAlmostEqual(d[0, 1], expect, places=6)
        self.assertAlmostEqual(d[0, 7], expect, places=6)

    def test_path_never_shorter_than_straight_line(self):
        rng = np.random.default_rng(1)
        pts = rng.uniform(-12, 12, (40, 2))
        d = ar.path_lengths(pts)
        for w in range(8):
            straight = np.linalg.norm(pts - wall_point(TRIG, 0, w), axis=1)
            # the nearest trigger point can sit off the alcove axis, so allow the half-width
            self.assertTrue(np.all(d[:, w] >= straight - H - 1e-9))
            self.assertTrue(np.all(d[:, w] > 0))


class TestPathPoints(unittest.TestCase):

    def test_path_points_sum_to_path_length(self):
        starts = [ar.alcove_back_positions()[0], np.array([3.0, -2.0])]
        for s in starts:
            d = ar.path_lengths(s[None])[0]
            for w in range(8):
                pts = ar.path_points(s, w)
                self.assertAlmostEqual(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum(), d[w], places=6)


class TestSnap(unittest.TestCase):

    def test_points_just_past_the_boundary_get_finite_paths(self):
        # logged positions overshoot alcove sides and wall faces by up to ~0.08
        pts = np.array([wall_point(W + 0.3, -H - 0.011, 1), wall_point(W + 0.06, H + 0.18, 2),
                        wall_point(BACK - 0.36, -H - 0.004, 4)])
        d = ar.path_lengths(pts)
        self.assertTrue(np.isfinite(d).all())
        # snapping moves a point only a few hundredths of a unit
        inner = np.array([wall_point(W + 0.3, -H + 0.05, 1)])
        self.assertLess(np.abs(ar.path_lengths(inner) - d[:1]).max(), 0.2)


class TestLayouts(unittest.TestCase):

    def test_probabilities(self):
        lay = ar.layout_table()
        self.assertAlmostEqual(lay['prob'].sum(), 1.0)
        # High is uniform over the 8 walls, and so is Low
        np.testing.assert_allclose(lay.groupby('high')['prob'].sum(), 1 / 8)
        np.testing.assert_allclose(lay.groupby('low')['prob'].sum(), 1 / 8)
        sep = ((lay['low'] - lay['high']) % 8).map({1: 1, 7: 1, 2: 2, 6: 2, 4: 4})
        np.testing.assert_allclose(lay.groupby(sep)['prob'].sum().to_numpy(), [0.5, 0.25, 0.25])


class TestRace(unittest.TestCase):

    def setUp(self):
        self.timing = ar.Timing(delay=0.2, speed=10.0, overhead=7.0, resid=np.zeros(1))
        self.periph = ar.alcove_back_positions()
        self.centre = np.zeros((1, 2))

    def test_identical_positions_tie(self):
        res = ar.race_summary(ar.build_races(self.centre, self.centre), self.timing, 'high')
        self.assertAlmostEqual(res['focal_win'], 0.5)
        self.assertAlmostEqual(res['focal_score_trial'], 0.5)

    def test_identical_positions_tie_with_noise(self):
        t = ar.Timing(delay=0.2, speed=10.0, overhead=7.0, resid=np.random.default_rng(0).normal(0, 0.3, 400))
        res = ar.race_summary(ar.build_races(self.centre, self.centre), t, 'high')
        self.assertAlmostEqual(res['focal_win'], 0.5, places=6)

    def test_always_high_matches_hand_count(self):
        # wins own alcove (High 1/8, Low 1/8, instant) and adjacent High unless standing on Low
        res = ar.race_summary(ar.build_races(self.periph, self.centre), self.timing, 'high')
        self.assertAlmostEqual(res['focal_win'], 1 / 8 + 1 / 8 + 2 / 8 * 0.75)
        self.assertAlmostEqual(res['focal_score_trial'], 1 / 8 + 0.4 / 8 + 2 / 8 * 0.75)

    def test_nearer_matches_hand_count(self):
        # adds adjacent Low when High is two or more walls away (probability 5/32)
        res = ar.race_summary(ar.build_races(self.periph, self.centre), self.timing, 'nearer')
        self.assertAlmostEqual(res['focal_score_trial'], 0.3625 + 0.4 * 5 / 32)
        self.assertAlmostEqual(res['opp_score_trial'], 1 - (0.4375 + 5 / 32))

    def test_instant_low_win_whatever_the_policy(self):
        races = ar.build_races(self.periph, self.centre)
        on_low = races['focal_d_low'] == 0
        for pol in ('high', 'nearer', 'best_trial'):
            out = ar.evaluate(races, self.timing, pol)
            np.testing.assert_allclose(out.loc[on_low, 'focal_score'], 0.4)
            np.testing.assert_allclose(out.loc[on_low, 'race_s'], 0.0)

    def test_race_time_is_the_faster_arrival(self):
        res = ar.evaluate(ar.build_races(self.centre, self.centre), self.timing, 'high')
        np.testing.assert_allclose(res['race_s'], 0.2 + TRIG / 10)

    def test_sweep_ends_match_policies(self):
        races = ar.build_races(self.periph, self.centre)
        self.assertAlmostEqual(ar.race_summary(races, self.timing, 0.0)['focal_score_trial'],
                               ar.race_summary(races, self.timing, 'high')['focal_score_trial'])
        self.assertAlmostEqual(ar.race_summary(races, self.timing, 1.0)['focal_score_trial'],
                               ar.race_summary(races, self.timing, 'nearer')['focal_score_trial'])

    def test_best_policies_beat_the_rest(self):
        t = ar.Timing(delay=0.2, speed=10.0, overhead=7.0, resid=np.random.default_rng(0).normal(0, 0.3, 300))
        races = ar.build_races(self.periph, self.centre)
        res = {p: ar.race_summary(races, t, p) for p in ('high', 'nearer', 0.5, 'best_trial', 'best_rate')}
        for p in ('high', 'nearer', 0.5):
            self.assertGreaterEqual(res['best_trial']['focal_score_trial'] + 1e-12, res[p]['focal_score_trial'])
            self.assertGreaterEqual(res['best_rate']['focal_score_s'] + 1e-12, res[p]['focal_score_s'])

    def test_focal_extra_delay(self):
        races = ar.build_races(self.centre, self.centre)
        # any extra delay loses every tie when timing is exact
        self.assertAlmostEqual(ar.race_summary(races, self.timing, 'high', focal_extra_s=0.1)['focal_win'], 0.0)
        # an instant win ignores the extra delay
        out = ar.evaluate(ar.build_races(self.periph, self.centre), self.timing, 'high', focal_extra_s=5.0)
        np.testing.assert_allclose(out.loc[out['focal_d_high'] == 0, 'focal_win'], 1.0)

    def test_extra_delay_matches_monte_carlo(self):
        rng = np.random.default_rng(4)
        resid = rng.normal(0, 0.4, 500)
        t = ar.Timing(delay=0.2, speed=10.0, overhead=7.0, resid=resid)
        out = ar.evaluate(ar.build_races(self.centre, self.centre), t, 'high', focal_extra_s=0.3)
        tf = 0.5 + TRIG / 10 + rng.choice(resid, 200000)
        to = 0.2 + TRIG / 10 + rng.choice(resid, 200000)
        self.assertAlmostEqual(out['focal_win'].iloc[0], (tf < to).mean(), places=2)
        self.assertAlmostEqual(out['race_s'].iloc[0], np.minimum(tf, to).mean(), places=2)

    def test_noise_win_probability_matches_monte_carlo(self):
        rng = np.random.default_rng(3)
        resid = rng.normal(0, 0.4, 500)
        t = ar.Timing(delay=0.2, speed=10.0, overhead=7.0, resid=resid)
        races = ar.build_races(self.periph, self.centre)
        out = ar.evaluate(races, t, 'high')
        # simulate one adjacent-High race directly
        row = races.index[(races['focal_d_high'] > 0) & (races['focal_d_high'] < 15) & (races['focal_d_low'] > 0)][0]
        tf = 0.2 + races.loc[row, 'focal_d_high'] / 10 + rng.choice(resid, 200000)
        to = 0.2 + races.loc[row, 'opp_d_high'] / 10 + rng.choice(resid, 200000)
        self.assertAlmostEqual(out.loc[row, 'focal_win'], (tf < to).mean(), places=2)
        self.assertAlmostEqual(out.loc[row, 'race_s'], np.minimum(tf, to).mean(), places=2)


if __name__ == '__main__':
    unittest.main()
