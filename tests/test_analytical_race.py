''' Tests for analysis/analytical_race.py, the geometric race model of wall choice.

Run with the conda env python:
    /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_analytical_race
'''

import math
import unittest

import numpy as np
import pandas as pd

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


class TestCentralVisibility(unittest.TestCase):

    def test_rays_from_the_centre(self):
        # the outer ray (55 deg) reaches a wall up to its vertex, 22.5 deg from the wall's centre
        self.assertAlmostEqual(ar.central_visible_half_angle(), 77.5, places=3)
        c = np.zeros((1, 2))
        self.assertTrue(ar.high_sensed(c, 90.0, 2)[0])
        self.assertTrue(ar.high_sensed(c, 90.0 + 77.4, 2)[0])
        self.assertFalse(ar.high_sensed(c, 90.0 + 77.6, 2)[0])
        self.assertFalse(ar.high_sensed(c, 90.0 - 77.6, 2)[0])

    def test_rays_from_the_alcove_back_miss_adjacent_walls(self):
        # from the alcove back no ray reaches the adjacent walls at any heading (agent_raycast notes)
        back = np.repeat(ar.alcove_back_positions()[:1], 360, axis=0)
        hd = np.arange(360.0)
        for w in (1, 7):
            self.assertFalse(ar.high_sensed(back, hd, w).any())
        # the agent stands in its own trigger box but still senses its own alcove walls
        self.assertTrue(ar.high_sensed(back[:1], 180.0, 0)[0])

    def test_rays_are_blocked_by_a_body_in_front(self):
        c = np.zeros((1, 2))
        # a body just ahead, directly toward High, blocks the middle rays only, so High is still sensed
        self.assertTrue(ar.high_sensed(c, 0.0, 0, blocker=np.array([[0.0, 2.0]]))[0])

    def test_central_times(self):
        w = 165.0
        t = ar.central_times(rot_speed=w)
        t0 = TRIG / ar.SPEED
        V = 77.5
        # every heading appears twice, once per turn direction
        self.assertEqual(len(t), 2 * ar.N_HEADINGS)
        # mean turn: short way when in view, half phi and half 360 - phi (mean 180) when not
        mean_turn = (V ** 2 / 2 + 180 * (180 - V)) / 180 / w
        self.assertAlmostEqual(t.mean() - t0, mean_turn, places=4)
        # the longest turn is just past the edge of view, the long way round
        self.assertAlmostEqual(t.max(), t0 + (360 - V) / w, places=2)
        self.assertLess(t.min() - t0, 0.001)
        # turns within V degrees: every in-view heading (twice) plus the out-of-view short turns up to V (none)
        self.assertAlmostEqual(np.mean(t - t0 < V / w), V / 180, places=3)
        np.testing.assert_allclose(ar.central_times(rot_speed=None), t0)

    def test_moving_while_high_is_in_view(self):
        w, c, V, D = 331.0, 8.0, 77.5, TRIG
        # one heading at 90 deg from High (out of view): short turn 90, long turn 270, the last 77.5 of each in view
        t = ar.central_times(rot_speed=w, close_speed=c, n=1)
        expect = [A / w + (D - c * V / w) / ar.SPEED for A in (90.0, 270.0)]
        np.testing.assert_allclose(np.sort(t), expect)
        # an in-view heading moves for the whole turn
        t_in = ar.central_times(rot_speed=w, close_speed=c, n=3)       # headings 30, 90, 150 deg
        self.assertAlmostEqual(t_in[0], 30 / w + (D - c * 30 / w) / ar.SPEED)
        # no closing speed is the turn-then-move model
        np.testing.assert_allclose(ar.central_times(rot_speed=w, close_speed=0.0), ar.central_times(rot_speed=w))

    def test_distance_to_trigger_line(self):
        pts = np.array([[0.0, 0.0], wall_point(10.0, 1.0, 5), wall_point(10.0, 4.0, 5)])
        d = ar._dist_to_trigger(pts, 5)
        np.testing.assert_allclose(d, [TRIG, TRIG - 10.0, np.hypot(TRIG - 10.0, 4.0 - H)])

    def test_zigzag_closing_speed(self):
        # full speed along an allowed direction, slowest midway between two (22.5 deg)
        v = ar.zigzag_closing_speed(np.array([0.0, 22.5, 45.0, 90.0, -30.0]))
        expect_mid = 10 * math.sin(math.radians(45)) / (2 * math.sin(math.radians(22.5)))
        np.testing.assert_allclose(v, [10.0, expect_mid, 10.0, 10.0, ar.zigzag_closing_speed(30.0)])
        self.assertAlmostEqual(expect_mid, 9.2388, places=4)

    def test_closing_speed_is_median_rate(self):
        tab = pd.DataFrame(dict(closing_inview=[8.0, 3.0, 1.0, 5.0], t_inview=[1.0, 0.5, 0.0, np.nan]))
        self.assertAlmostEqual(ar.closing_speed(tab), 7.0)

    def test_central_always_turns_so_a_peripheral_agent_at_the_centre_wins(self):
        races = ar.build_races(np.zeros((1, 2)))
        res = ar.summarise(races, 7.0, 'high', central_t=ar.central_times(rot_speed=165.0)).iloc[0]
        self.assertAlmostEqual(res['periph_win'], 1.0)

    def test_rotation_matches_a_loop_over_headings(self):
        races = ar.build_races(ar.alcove_back_positions())
        ct = ar.central_times(rot_speed=165.0)
        out = ar.evaluate(races, 'closest', central_t=ct)
        # rescore each row against every central time directly
        base = ar.evaluate(races, 'closest')
        t_p = np.where(base['periph_win'] == 1, base['race_s'], np.nan)
        for i in range(0, len(races), 7):
            r = base.iloc[i]
            tp = np.where(races.iloc[i]['t_high'] == 0, 0.0, np.where(races.iloc[i]['t_low'] == 0, 0.0,
                          races.iloc[i]['t_low'] if r['go_low'] else races.iloc[i]['t_high']))
            win = np.mean((tp < ct - 1e-9) + 0.5 * (np.abs(tp - ct) <= 1e-9))
            self.assertAlmostEqual(out.iloc[i]['periph_win'], win, places=9)
            self.assertAlmostEqual(out.iloc[i]['race_s'], np.mean(np.minimum(tp, ct)), places=9)

    def test_reference_with_rotation(self):
        ct = ar.central_times(rot_speed=165.0)
        ref = ar.central_reference(7.0, central_t=ct)
        self.assertAlmostEqual(ref['score_trial'], 0.5)
        self.assertAlmostEqual(ref['race_s'], np.minimum(ct[:, None], ct[None, :]).mean(), places=9)
        self.assertAlmostEqual(ref['score_s'], 0.5 / (7.0 + ref['race_s']))

    def test_rotation_speed_is_median_turning_rate(self):
        # rates 300, 200, 400 deg/s; the agent-trial already facing High and the unaligned one are left out
        tab = pd.DataFrame(dict(visible=[False, True, False, True, False], turned=[300.0, 100.0, 200.0, 0.0, 50.0],
                                t_align=[1.0, 0.5, 0.5, 0.0, np.nan]))
        self.assertAlmostEqual(ar.rotation_speed(tab), 300.0)


class TestPeripheralSensing(unittest.TestCase):

    def test_heading_sensing_at_the_centre_matches_the_central_rule(self):
        heads = ar.heading_grid()
        sensed = ar.heading_sensing(np.zeros(2))
        # wall 2 lies at bearing 90 deg; sensed within 77.5 deg of it
        rel = np.abs(ar._wrap(heads - 90.0))
        np.testing.assert_array_equal(sensed[:, 2], rel <= 77.5 + 1e-9)

    def test_first_leg(self):
        # from the centre the first leg points straight at each wall
        np.testing.assert_allclose(ar._wrap(ar.first_leg_bearing(np.zeros(2)) - 45.0 * np.arange(8)), 0, atol=1e-9)
        # from the alcove back, the leg to an adjacent wall heads for its own mouth corner
        back = ar.alcove_back_positions()[0]
        corner = wall_point(W, H, 0)
        self.assertAlmostEqual(ar.first_leg_bearing(back)[1], ar._bearing(corner - back), places=6)

    def test_step_out_to_sense_an_adjacent_wall(self):
        back = ar.alcove_back_positions()[0]
        step, pos = ar.step_to_sense(back, 1)
        self.assertGreater(step, 0.0)
        self.assertTrue(ar.heading_sensing(pos)[:, 1].any())
        # a little less is not enough
        less = back + (pos - back) * (step - 0.05) / step
        self.assertFalse(ar.heading_sensing(less)[:, 1].any())
        # no step is needed where some heading already senses the wall
        self.assertEqual(ar.step_to_sense(back, 4)[0], 0.0)

    def test_target_time_matches_the_central_agent_from_the_centre(self):
        w, c = 331.0, 7.57
        D = TRIG
        for phi in (30.0, 120.0):
            for direction in (1, -1):
                t = ar.target_time(np.zeros(2), 90.0 - phi, 2, direction, w, c)
                # short-way turns match central_times; a long-way turn goes 360 - phi
                A = phi if (phi < 77.5 or direction == 1) else 360 - phi
                expect = A / w + (D - c * min(A, 77.5) / w) / ar.SPEED
                self.assertAlmostEqual(t, expect, places=2)

    def test_target_time_when_already_facing_the_wall(self):
        back = ar.alcove_back_positions()[0]
        # facing the opposite wall (bearing 180) from the alcove back: no turn, straight path
        t = ar.target_time(back, 180.0, 4, 1, 331.0, 7.57)
        self.assertAlmostEqual(t, ar.path_lengths(back[None])[0, 4] / ar.SPEED, places=3)

    def test_target_time_adds_the_step(self):
        back = ar.alcove_back_positions()[0]
        step, pos = ar.step_to_sense(back, 1)
        t = ar.target_time(back, 180.0, 1, 1, 331.0, 7.57)
        self.assertGreater(t, step / ar.SPEED + ar.path_lengths(pos[None])[0, 1] / ar.SPEED - 0.3)


class TestLowFirst(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # one agent at the centre facing north (wall 0), one at the alcove back of wall 0 facing the centre
        cls.centre = ar.build_races_rays(np.zeros((1, 2)), [0.0], 331.0, 7.57)
        cls.back = ar.build_races_rays(ar.alcove_back_positions()[:1], [180.0], 331.0, 7.57)

    @staticmethod
    def row(races, high, low, direction):
        r = races[(races['high'] == high) & (races['low'] == low) & (races['direction'] == direction)]
        assert len(r) == 1
        return r.iloc[0]

    def test_both_sensed_at_onset_goes_high(self):
        # walls 1 and 7 sit 45 deg either side of the heading
        r = self.row(self.centre, 1, 7, 1)
        self.assertTrue(r['high_sensed0'] and r['low_sensed0'])
        self.assertFalse(r['low_first'])

    def test_only_low_sensed_at_onset_goes_low(self):
        for d in (1, -1):
            r = self.row(self.centre, 2, 1, d)
            self.assertTrue(r['low_sensed0'] and not r['high_sensed0'])
            self.assertTrue(r['low_first'])
            r = self.row(self.centre, 1, 2, d)
            self.assertFalse(r['low_first'])

    def test_neither_at_onset_takes_the_first_wall_the_search_finds(self):
        # High east (wall 2), Low west (wall 6), both 90 deg off the heading
        self.assertFalse(self.row(self.centre, 2, 6, 1)['low_first'])
        self.assertTrue(self.row(self.centre, 2, 6, -1)['low_first'])

    def test_no_step_when_a_lit_wall_can_already_be_sensed(self):
        # from the alcove back an adjacent High is hidden, but a Low three walls away is not
        r = self.row(self.back, 1, 3, 1)
        self.assertGreater(r['high_step'], 0.0)
        self.assertEqual(r['low_step'], 0.0)
        self.assertTrue(r['low_first'])
        # and the reverse: an adjacent Low is hidden while High two walls away is not
        r = self.row(self.back, 2, 1, 1)
        self.assertFalse(r['low_first'])

    def test_equal_steps_tie_to_high(self):
        # both adjacent walls need the same step, by symmetry
        r = self.row(self.back, 1, 7, 1)
        self.assertAlmostEqual(r['high_step'], r['low_step'], places=6)
        self.assertFalse(r['low_first'])

    def test_high_first_matches_always_high(self):
        races = pd.concat([self.centre, self.back], ignore_index=True)
        a = ar.evaluate(races, 'low_first')
        b = ar.evaluate(races, 'high')
        keep = ~races['low_first']
        np.testing.assert_allclose(a.loc[keep, 'race_s'], b.loc[keep, 'race_s'])
        np.testing.assert_allclose(a.loc[keep, 'periph_score'], b.loc[keep, 'periph_score'])


class TestNearest(unittest.TestCase):

    W_, C_ = 331.0, 7.57

    def test_trace_arrival_matches_target_time(self):
        for pos, hd, wall, d in ((np.zeros(2), 0.0, 4, 1), (ar.alcove_back_positions()[0], 180.0, 1, -1),
                                 (ar.alcove_back_positions()[0], 90.0, 3, 1), (np.array([2.0, 0.0]), 30.0, 6, -1)):
            tr = ar.high_trace(pos, hd, wall, d, self.W_, self.C_)
            self.assertAlmostEqual(tr['arrive'], ar.target_time(pos, hd, wall, d, self.W_, self.C_), places=6)
            # the trace ends on the wall's trigger line
            p_end, _ = tr['state'](tr['arrive'])
            self.assertAlmostEqual(ar._dist_to_trigger(p_end[None], wall)[0], 0.0, places=4)

    def test_heading_turns_to_each_new_stretch(self):
        # from the alcove back to an adjacent wall the path bends at the far mouth corner
        back = ar.alcove_back_positions()[0]
        tr = ar.high_trace(back, 180.0, 1, 1, self.W_, self.C_)
        pts = tr['path']
        last = ar._bearing(pts[-1] - pts[-2])
        _, h_end = tr['state'](tr['arrive'])
        _, h_mid = tr['state'](tr['bend_times'][0])
        self.assertLess(abs(ar._wrap(h_end - last)), abs(ar._wrap(h_mid - last)) + 1e-9)

    def test_nearest_at_the_centre_matches_always_high(self):
        races = ar.build_races_rays(np.zeros((1, 2)), [0.0], self.W_, self.C_)
        a, b = ar.evaluate(races, 'nearest'), ar.evaluate(races, 'high')
        self.assertFalse(races['nearest_low'].any())
        np.testing.assert_allclose(a['race_s'], b['race_s'])

    def test_both_sensed_at_onset_takes_the_nearer(self):
        back = ar.alcove_back_positions()[:1]
        races = ar.build_races_rays(back, [180.0], self.W_, self.C_)
        r = races[(races.high == 4) & (races.low == 3) & (races.direction == 1)].iloc[0]
        self.assertTrue(r['high_sensed0'] and r['low_sensed0'])
        self.assertTrue(r['nearest_low'])
        self.assertAlmostEqual(r['t_nearest'], ar.target_time(back[0], 180.0, 3, 1, self.W_, self.C_), places=6)
        r = races[(races.high == 3) & (races.low == 4) & (races.direction == 1)].iloc[0]
        self.assertFalse(r['nearest_low'])
        self.assertAlmostEqual(r['t_nearest'], r['t_high'], places=9)

    def test_hidden_adjacent_high_is_kept(self):
        # the trap of 'Low, if Low seen first': adjacent High hidden, far Low visible
        back = ar.alcove_back_positions()[:1]
        races = ar.build_races_rays(back, [180.0], self.W_, self.C_)
        r = races[(races.high == 1) & (races.low == 3) & (races.direction == 1)].iloc[0]
        self.assertTrue(r['low_first'])
        self.assertFalse(r['nearest_low'])
        self.assertAlmostEqual(r['t_nearest'], r['t_high'], places=9)

    def test_switch_when_low_turns_out_nearer(self):
        # just east of the centre, facing north, High behind (wall 4), Low east (wall 2)
        pos = np.array([2.0, 0.0])
        races = ar.build_races_rays(pos[None], [0.0], self.W_, self.C_)
        # turning clockwise passes Low first, then finds High; Low is nearer, so it switches
        r = races[(races.high == 4) & (races.low == 2) & (races.direction == 1)].iloc[0]
        self.assertTrue(r['nearest_low'])
        tr = ar.high_trace(pos, 0.0, 4, 1, self.W_, self.C_)
        p, h = tr['state'](tr['sensed'])
        expect = tr['sensed'] + ar.time_from_sensed(p, h, 2, self.W_, self.C_)
        self.assertAlmostEqual(r['t_nearest'], expect, places=6)
        self.assertLess(r['t_nearest'], r['t_high'])
        # turning anticlockwise finds High without passing Low, and Low stays out of view
        r = races[(races.high == 4) & (races.low == 2) & (races.direction == -1)].iloc[0]
        self.assertFalse(r['nearest_low'])


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
        self.back = ar.alcove_back_positions()
        self.races = ar.build_races(self.back)
        self.tc = TRIG / ar.SPEED
        self.t_adj = ar.path_lengths(self.back[:1])[0, 1] / ar.SPEED

    def test_central_time_is_the_same_to_every_wall(self):
        np.testing.assert_allclose(self.races['t_central'], self.tc)

    def test_shared_start_ties(self):
        races = ar.build_races(np.zeros((1, 2)))
        res = ar.summarise(races, 7.0, 'high').iloc[0]
        self.assertAlmostEqual(res['periph_win'], 0.5)
        self.assertAlmostEqual(res['periph_score_trial'], 0.5)

    def test_always_high_matches_hand_count(self):
        # wins own alcove (High 1/8, Low 1/8, instant) and an adjacent High unless standing on Low
        res = ar.summarise(self.races, 7.0, 'high').iloc[0]
        self.assertAlmostEqual(res['periph_win'], 1 / 8 + 1 / 8 + 2 / 8 * 0.75)
        self.assertAlmostEqual(res['periph_score_trial'], 1 / 8 + 0.4 / 8 + 2 / 8 * 0.75)
        self.assertAlmostEqual(res['central_score_trial'], 1 - 7 / 16)
        # race lasts 0 on instant wins (1/4), the adjacent time on adjacent wins (3/16), else central time
        race = 3 / 16 * self.t_adj + 9 / 16 * self.tc
        self.assertAlmostEqual(res['race_s'], race)
        self.assertAlmostEqual(res['periph_score_s'], 0.3625 / (7.0 + race))

    def test_closest_matches_hand_count(self):
        # adds an adjacent Low whenever High is two or more walls away (probability 5/32)
        res = ar.summarise(self.races, 7.0, 'closest').iloc[0]
        self.assertAlmostEqual(res['periph_score_trial'], 0.3625 + 0.4 * 5 / 32)
        self.assertAlmostEqual(res['central_score_trial'], 1 - (7 / 16 + 5 / 32))

    def test_threshold_ends_match_the_other_policies(self):
        for delta, pol in ((0.0, 'closest'), (10.0, 'high')):
            a = ar.summarise(self.races, 7.0, 'threshold', delta).iloc[0]
            b = ar.summarise(self.races, 7.0, pol).iloc[0]
            self.assertAlmostEqual(a['periph_score_s'], b['periph_score_s'])
            self.assertAlmostEqual(a['p_go_low'], b['p_go_low'])

    def test_threshold_between(self):
        # at 1 s every winnable Low is still taken (gap 1.17 s or more), so the score matches closest
        a = ar.summarise(self.races, 7.0, 'threshold', 1.0).iloc[0]
        b = ar.summarise(self.races, 7.0, 'closest').iloc[0]
        self.assertAlmostEqual(a['periph_score_trial'], b['periph_score_trial'])
        self.assertLess(a['p_go_low'], b['p_go_low'])

    def test_instant_win_whatever_the_policy(self):
        on_low = self.races['t_low'] == 0
        for pol, delta in (('high', None), ('closest', None), ('threshold', 0.5)):
            out = ar.evaluate(self.races, pol, delta)
            np.testing.assert_allclose(out.loc[on_low, 'periph_score'], 0.4)
            np.testing.assert_allclose(out.loc[on_low, 'race_s'], 0.0)

    def test_groups_weigh_their_own_positions(self):
        races = ar.build_races(np.vstack([self.back, np.zeros((1, 2))]), groups=[0] * 8 + [1])
        np.testing.assert_allclose(races.groupby('group')['prob'].sum(), 1.0)
        res = ar.summarise(races, 7.0, 'high').set_index('group')
        self.assertAlmostEqual(res.loc[0, 'periph_win'], 7 / 16)
        self.assertAlmostEqual(res.loc[1, 'periph_win'], 0.5)

    def test_central_reference(self):
        ref = ar.central_reference(7.0)
        self.assertAlmostEqual(ref['score_s'], 0.5 / (7.0 + self.tc))


if __name__ == '__main__':
    unittest.main()
