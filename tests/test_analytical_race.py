''' Tests for analysis/analytical_race.py, the geometric race model of wall choice.

Run with the conda env python:
    /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_analytical_race
'''

import math
import unittest

import numpy as np
import pandas as pd

import globals
import analysis.agent_raycast as agent_raycast
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

    def test_both_hidden_takes_the_first_found_while_stepping(self):
        # both adjacent walls hidden: the agent steps out while turning, and finds the side it turns toward first
        r_cw, r_acw = self.row(self.back, 1, 7, 1), self.row(self.back, 1, 7, -1)
        self.assertGreater(r_cw['high_step'], 0.0)
        self.assertNotEqual(bool(r_cw['low_first']), bool(r_acw['low_first']))

    def test_equal_steps_tie_to_high_when_stepping_first(self):
        # the earlier version (step, then search) compares the steps, and equal steps go High
        back = ar.build_races_rays(ar.alcove_back_positions()[:1], [180.0], 331.0, 7.57, merge_step=False, move_out=False)
        r = self.row(back, 1, 7, 1)
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


class TestSearchWhileStepping(unittest.TestCase):

    W_, C_ = 331.0, 7.57

    def test_wall_is_sensed_when_the_combined_search_stops(self):
        back = ar.alcove_back_positions()[0]
        ms = ar.step_search_trace(back, 180.0, 1, self.W_)
        t_s = ms['first'][1]
        self.assertTrue(np.isfinite(t_s) and t_s > 0)
        q, h = ms['state'](t_s)
        self.assertTrue(ar.high_sensed(q[None], h, 1)[0])
        # it moved toward the centre and turned clockwise at the given rates
        self.assertAlmostEqual(np.linalg.norm(q - back), min(t_s * ar.SPEED, ar.MAX_STEP), places=6)
        self.assertAlmostEqual(h, 180.0 + self.W_ * t_s, places=6)

    def test_search_while_stepping_is_no_slower(self):
        back = ar.alcove_back_positions()[0]
        for d in (1, -1):
            merged = ar.target_time(back, 180.0, 1, d, self.W_, self.C_, move_out=False)
            separate = ar.target_time(back, 180.0, 1, d, self.W_, self.C_, merge_step=False, move_out=False)
            self.assertLessEqual(merged, separate + 1e-9)
            tr = ar.high_trace(back, 180.0, 1, d, self.W_, self.C_, move_out=False)
            self.assertAlmostEqual(tr['arrive'], merged, places=6)

    def test_no_step_needed_is_unchanged(self):
        back = ar.alcove_back_positions()[0]
        self.assertAlmostEqual(ar.target_time(back, 180.0, 3, 1, self.W_, self.C_, move_out=False),
                               ar.target_time(back, 180.0, 3, 1, self.W_, self.C_, merge_step=False, move_out=False), places=9)

    def test_known_wall_times(self):
        back = ar.alcove_back_positions()[0]
        D, TK = ar.known_wall_times(back[None], [180.0], self.W_, self.C_)
        np.testing.assert_allclose(D[0], ar.path_lengths(back[None])[0])
        for w in (1, 4):
            self.assertAlmostEqual(TK[0, w], ar.time_from_sensed(back, 180.0, w, self.W_, self.C_), places=9)


class TestParallelBuild(unittest.TestCase):

    def test_parallel_matches_serial(self):
        pos = np.vstack([ar.alcove_back_positions()[[0, 3]], [[2.0, -1.0]]])
        heads = [180.0, 20.0, 300.0]
        a = ar.build_races_rays(pos, heads, 331.0, 7.57, n_jobs=1)
        b = ar.build_races_rays(pos, heads, 331.0, 7.57, n_jobs=2)
        pd.testing.assert_frame_equal(a, b)

    def test_default_leaves_cores_free(self):
        self.assertGreaterEqual(ar.N_JOBS, 1)
        self.assertLessEqual(ar.N_JOBS, 12)


class TestMoveOut(unittest.TestCase):
    ''' The default search: move out while turning whenever the target is not sensed at onset. '''

    W_, C_ = 331.0, 7.57

    def setUp(self):
        self.back = ar.alcove_back_positions()[0]

    def test_wall_sensable_from_the_spot_is_searched_while_moving_out(self):
        # facing 250 deg, the wall three away (bearing ~157) is out of view but a turn in place would find it
        h = 250.0
        self.assertFalse(ar.high_sensed(self.back[None], h, 3)[0])
        self.assertEqual(ar.step_to_sense(self.back, 3)[0], 0.0)
        for d in (1, -1):
            ms = ar.step_search_trace(self.back, h, d, self.W_)
            q, hh = ms['state'](ms['first'][3])
            expect = ms['first'][3] + ar.time_from_sensed(q, hh, 3, self.W_, self.C_)
            t = ar.target_time(self.back, h, 3, d, self.W_, self.C_)
            self.assertAlmostEqual(t, expect, places=9)
            # moving out while searching is no slower than searching in place
            self.assertLessEqual(t, ar.target_time(self.back, h, 3, d, self.W_, self.C_, move_out=False) + 1e-9)
            self.assertAlmostEqual(ar.high_trace(self.back, h, 3, d, self.W_, self.C_)['arrive'], t, places=6)

    def test_behaviour_uses_only_what_was_sensed(self):
        # two walls not sensed at onset follow the same motion until each is first sensed,
        # whether or not a turn in place could have found them
        h = 250.0
        for d in (1, -1):
            ms = ar.step_search_trace(self.back, h, d, self.W_)
            for w in (1, 3):
                self.assertFalse(ar.high_sensed(self.back[None], h, w)[0])
                tr = ar.high_trace(self.back, h, w, d, self.W_, self.C_)
                t_s = tr['sensed']
                self.assertAlmostEqual(t_s, ms['first'][w], places=9)
                for t in np.linspace(0, t_s, 5):
                    np.testing.assert_allclose(tr['state'](t)[0], ms['state'](t)[0])

    def test_outside_an_alcove_the_search_turns_in_place(self):
        pos = np.array([2.0, 0.0])
        self.assertFalse(ar.in_alcove(pos))
        self.assertTrue(ar.in_alcove(self.back))
        ms = ar.step_search_trace(pos, 0.0, 1, self.W_, move=ar.search_moves(pos, True))
        np.testing.assert_allclose(ms['state'](0.5)[0], pos)

    def test_sensed_at_onset_is_unchanged(self):
        self.assertAlmostEqual(ar.target_time(self.back, 180.0, 4, 1, self.W_, self.C_),
                               ar.target_time(self.back, 180.0, 4, 1, self.W_, self.C_, move_out=False), places=9)


class TestInference(unittest.TestCase):

    W_, C_ = 331.0, 7.57

    def test_posterior_matches_agent_raycast(self):
        # nothing seen but walls 3, 4, 5 unlit; and Low seen at wall 2 with walls 3, 4 unlit
        S = np.zeros(8, bool); S[[3, 4, 5]] = True
        post = ar.high_posterior_given(S, None)
        ref = agent_raycast.high_posterior(seen_unlit=(4, 5, 6))
        np.testing.assert_allclose(post, [ref.get(w + 1, 0.0) for w in range(8)])
        S = np.zeros(8, bool); S[[2, 3, 4]] = True
        post = ar.high_posterior_given(S, 2)
        ref = agent_raycast.high_posterior(seen_low=3, seen_unlit=(4, 5))
        np.testing.assert_allclose(post, [ref.get(w + 1, 0.0) for w in range(8)])

    def test_central_nothing_informative_is_unchanged(self):
        # High dead behind, Low behind too: the walls in view are unlit and placed symmetrically, so both directions tie
        inf = ar.central_times_inferred(self.W_, self.C_, n=720)
        plain = ar.central_times(self.W_, close_speed=self.C_, n=360)
        self.assertEqual(set(inf), {1, 2, 4, 6, 7})
        for k, v in inf.items():
            self.assertEqual(len(v), 2 * 720)
        # without Low information the inferred central agent is never slower on average
        w = ar.low_offset_weights()
        mean_inf = sum(w[k] * inf[k].mean() for k in inf)
        self.assertLessEqual(mean_inf, plain.mean() + 1e-3)

    def test_central_turns_toward_the_side_low_points_to(self):
        # High at wall 0; Low one wall clockwise (wall 1) and in view; heading 100 deg so High is out of view.
        # Walls 1-3 are in view (Low and two unlit), so High is at wall 0 (p 0.4), 5 (0.4) or 7 (0.2).
        # Expected search: anticlockwise 85.5 deg, clockwise 119.5 deg, so it turns anticlockwise, which is
        # the short way to the true High (100 deg, sensed after 22.5)
        t, d = ar.central_time_one(100.0, 1, self.W_, self.C_)
        self.assertEqual(d, -1)
        expect = 100.0 / self.W_ + (ar.TRIGGER_REACH - self.C_ * ar.central_visible_half_angle() / self.W_) / ar.SPEED
        self.assertAlmostEqual(t, expect, places=6)

    def test_evaluate_accepts_central_times_by_layout(self):
        races = ar.build_races(ar.alcove_back_positions())
        flat = ar.central_times(self.W_, close_speed=self.C_)
        same = {k: flat for k in (1, 2, 4, 6, 7)}
        a = ar.summarise(races, 7.0, 'high', central_t=flat).iloc[0]
        b = ar.summarise(races, 7.0, 'high', central_t=same).iloc[0]
        self.assertAlmostEqual(a['periph_score_s'], b['periph_score_s'], places=12)

    def test_peripheral_direction_weights(self):
        races = ar.build_races_rays(ar.alcove_back_positions()[:2], [180.0, 200.0], self.W_, self.C_, n_jobs=1)
        sensed = ar.onset_sensed(ar.alcove_back_positions()[:2], [180.0, 200.0])
        inf = ar.infer_direction(races, sensed)
        # each position and layout keeps its total probability
        tot = lambda r: r.groupby(['pos', 'high', 'low'])['prob'].sum()
        pd.testing.assert_series_equal(tot(inf), tot(races))
        # inference never makes the expected search for High longer than a random direction
        w0, w1 = races['prob'].to_numpy(), inf['prob'].to_numpy()
        hidden = ~races['high_sensed0'].to_numpy()
        self.assertLessEqual((w1 * races['high_tsense'])[hidden].sum(), (w0 * races['high_tsense'])[hidden].sum() + 1e-9)


class TestDeltaRule(unittest.TestCase):

    W_, C_ = 331.0, 7.57

    @classmethod
    def setUpClass(cls):
        cls.pos = np.vstack([ar.alcove_back_positions()[[0, 3]], [[0.0, 0.0], [2.0, 0.0]]])
        cls.heads = [180.0, 20.0, 0.0, 0.0]
        cls.races = ar.build_races_rays(cls.pos, cls.heads, cls.W_, cls.C_, n_jobs=1)

    def test_large_delta_is_always_high(self):
        a = ar.evaluate(self.races, 'delta', delta=10.0)
        b = ar.evaluate(self.races, 'high')
        np.testing.assert_allclose(a['race_s'], b['race_s'])
        np.testing.assert_allclose(a['periph_score'], b['periph_score'])

    def test_centre_never_switches(self):
        # from the exact centre every wall is equally far, so no gap exceeds 0
        r = self.races[self.races['pos'] == 2]
        go_low, t = ar.delta_outcome(r, 0.0)
        self.assertFalse(go_low.any())
        np.testing.assert_allclose(t, r['t_high'])

    def test_table_matches_direct_simulation(self):
        rows = self.races[(self.races['t_high'] > 0) & (self.races['t_low'] > 0)].sample(25, random_state=1)
        for _, r in rows.iterrows():
            for delta in (0.0, 0.3, 1.0):
                go_low, t = ar.delta_outcome(r.to_frame().T.infer_objects(), delta)
                gl, tt = ar.delta_direct(self.pos[int(r['pos'])], self.heads[int(r['pos'])], int(r['high']), int(r['low']),
                                         int(r['direction']), delta, self.W_, self.C_)
                self.assertEqual(bool(go_low[0]), gl)
                self.assertAlmostEqual(float(t[0]), tt, places=6)

    def test_outcomes_are_finite(self):
        for delta in (0.0, 0.5, 2.0):
            go_low, t = ar.delta_outcome(self.races, delta)
            self.assertTrue(np.all(np.isfinite(t)) and np.all(t >= 0))


class TestDeltaBoth(unittest.TestCase):
    ''' The delta rule with inference about both walls ('delta2'). '''

    W_, C_ = 331.0, 7.57

    @classmethod
    def setUpClass(cls):
        cls.pos = np.vstack([ar.alcove_back_positions()[[0, 3]], [[0.0, 0.0], [2.0, 0.0]]])
        cls.heads = [180.0, 20.0, 0.0, 0.0]
        cls.races = ar.build_races_rays(cls.pos, cls.heads, cls.W_, cls.C_, n_jobs=1)

    def test_joint_posterior(self):
        S = np.zeros(8, bool); S[[2, 3, 4]] = True
        J = ar.joint_posterior(S, None, 2)
        # Low seen at wall 2 matches the High posterior used before
        np.testing.assert_allclose(J.sum(axis=1), ar.high_posterior_given(S, 2))
        self.assertAlmostEqual(J.sum(), 1.0)
        # High seen at wall 6: Low can be at 5, 7, 0 (not seen) but not at 4 (seen unlit) or 2 (seen unlit)
        S = np.zeros(8, bool); S[[6, 4, 2]] = True
        J = ar.joint_posterior(S, 6, None)
        self.assertTrue(np.all(J[np.arange(8) != 6] == 0))
        self.assertEqual(J[6, 4], 0.0); self.assertEqual(J[6, 2], 0.0)
        self.assertGreater(J[6, 5], 0.0); self.assertGreater(J[6, 0], 0.0)

    def test_table_matches_direct_simulation(self):
        rows = self.races[(self.races['t_high'] > 0) & (self.races['t_low'] > 0)].sample(30, random_state=2)
        for _, r in rows.iterrows():
            for delta in (0.0, 0.4, 1.0, 2.5):
                go_low, t = ar.delta2_outcome(r.to_frame().T.infer_objects(), delta)
                gl, tt = ar.delta2_direct(self.pos[int(r['pos'])], self.heads[int(r['pos'])], int(r['high']), int(r['low']),
                                          int(r['direction']), delta, self.W_, self.C_)
                self.assertEqual(bool(go_low[0]), gl)
                self.assertAlmostEqual(float(t[0]), tt, places=6)

    def test_centre_never_takes_low(self):
        r = self.races[self.races['pos'] == 2]
        go_low, _ = ar.delta2_outcome(r, 0.0)
        self.assertFalse(go_low.any())

    def test_far_high_seen_adjacent_low_inferred(self):
        # alcove back facing out: High three walls away in view, Low adjacent and hidden.
        # With a small delta the agent infers Low is probably near, searches for it and goes there, faster than to High
        r = self.races[(self.races['pos'] == 0) & (self.races['high'] == 3) & (self.races['low'] == 1)]
        self.assertTrue(r['high_sensed0'].all() and not r['low_sensed0'].any())
        go_low, t = ar.delta2_outcome(r, 1.0)
        self.assertTrue(go_low.all())
        self.assertTrue(np.all(t < r['t_high'].to_numpy()))

    def test_outcomes_are_finite(self):
        for delta in (0.0, 0.7, 3.0):
            _, t = ar.delta2_outcome(self.races, delta)
            self.assertTrue(np.all(np.isfinite(t)) and np.all(t >= 0))


class TestAuditFixes(unittest.TestCase):
    ''' Fixes from the 2026-10-22 audit. '''

    W_, C_ = 331.0, 7.57

    def test_first_sense_time_uses_the_start_position(self):
        # from the alcove back facing out, an adjacent wall cannot be sensed at onset, so its first-sense
        # time must come from the search, never 0 (it was 0 when a stepped-out position could see it)
        back = ar.alcove_back_positions()[:1]
        races = ar.build_races_rays(back, [180.0], self.W_, self.C_, n_jobs=1)
        for d in (1, -1):
            r = races[(races.high == 1) & (races.direction == d)].iloc[0]
            ms = ar.step_search_trace(back[0], 180.0, d, self.W_, move=True)
            self.assertFalse(r['high_sensed0'])
            self.assertGreater(r['high_tsense'], 0.0)
            self.assertAlmostEqual(r['high_tsense'], ms['first'][1], places=9)

    def test_delta2_route_matches_direct(self):
        back = ar.alcove_back_positions()[0]
        for h, l, d, delta in ((3, 1, 1, 1.0), (1, 3, -1, 1.0), (2, 1, 1, 0.0), (4, 3, 1, 2.0)):
            r = ar.delta2_route(back, 180.0, h, l, d, delta, self.W_, self.C_)
            gl, t = ar.delta2_direct(back, 180.0, h, l, d, delta, self.W_, self.C_)
            self.assertEqual(r['go_low'], gl)
            self.assertAlmostEqual(r['time'], t, places=9)
            # segments join up in time
            ends = [t0 + (e - s0) for _, s0, e, t0, _ in r['segments']]
            self.assertAlmostEqual(ends[-1], t, places=9)

    def test_mirror_symmetry_of_delta2(self):
        # an alcove back facing straight out is symmetric: a mirrored layout with the opposite search
        # direction gives the same wall and time, including exact ties at delta = 0
        back = ar.alcove_back_positions()[0]
        for h, l in ((1, 7), (2, 6), (3, 1), (1, 3), (2, 1)):
            for d in (1, -1):
                for delta in (0.0, 1.0):
                    a = ar.delta2_direct(back, 180.0, h, l, d, delta, self.W_, self.C_)
                    b = ar.delta2_direct(back, 180.0, (-h) % 8, (-l) % 8, -d, delta, self.W_, self.C_)
                    self.assertEqual(a[0], b[0])
                    self.assertAlmostEqual(a[1], b[1], places=2)


class TestBackpedal(unittest.TestCase):
    """ An agent in its own alcove, off its trigger, backpedals into the trigger without turning when its
        target is its own wall with probability >= BACKPEDAL_P. Nothing else moves without turning. """

    def setUp(self):
        # make sure the rules are on for these tests
        self._flags = (ar.BACKPEDAL, ar.OWN_TRIGGER_KNOWN)
        ar.BACKPEDAL, ar.OWN_TRIGGER_KNOWN = True, True
        ar._TRACE_CACHE.clear()
        self.rot, self.close = 331.0, 7.57
        self.mid = np.array(ar._pt(17.5, 0.0, 0), float)          # in alcove 0, off its trigger
        self.back = np.array(ar._pt(19.6, 0.0, 0), float)         # on alcove 0's trigger
        self.out = np.array(ar._pt(15.5, 0.0, 0), float)          # just outside the alcove

    def tearDown(self):
        ar.BACKPEDAL, ar.OWN_TRIGGER_KNOWN = self._flags
        ar._TRACE_CACHE.clear()

    def test_on_by_default(self):
        # the rules are on in the module defaults
        self.assertEqual(self._flags, (True, True))

    def test_switched_off(self):
        # with the rules off, the agent never backpedals and knows only what its rays hit
        ar.BACKPEDAL, ar.OWN_TRIGGER_KNOWN = False, False
        sensed = np.ones(8, bool)
        sensed[0] = False
        self.assertNotEqual(ar._motion_kind(self.mid, sensed, 0, 0, 4), 'backpedal')
        self.assertFalse(ar.onset_known(self.back, np.zeros(8, bool)).any())

    def test_blind_trace(self):
        tr = ar.blind_trace(self.mid, 180.0, 0)
        self.assertAlmostEqual(tr['arrive'], ar.path_lengths(self.mid[None])[0][0] / ar.SPEED, places=6)
        for t in np.linspace(0, tr['arrive'], 7):
            self.assertAlmostEqual(tr['state'](t)[1], 180.0)
        self.assertTrue(ar._in_trigger(tr['state'](tr['arrive'])[0], 0))

    def test_kind_threshold(self):
        sensed = np.ones(8, bool)
        sensed[0] = False
        # High at the unseen own wall, Low seen at the opposite wall: High is certain there
        self.assertEqual(ar._motion_kind(self.mid, sensed, 0, 0, 4), 'backpedal')
        # a second possible wall for High: own wall 0.25 / (0.25 + 0.125) = 0.67 < 0.8, so search
        sensed[2] = False
        self.assertAlmostEqual(ar._own_wall_p(self.mid, sensed, 0, 0, 4)[1], 2 / 3)
        self.assertEqual(ar._motion_kind(self.mid, sensed, 0, 0, 4), 'search')

    def test_only_in_own_alcove_off_trigger(self):
        sensed = np.ones(8, bool)
        sensed[0] = False
        self.assertNotEqual(ar._motion_kind(self.back, sensed, 0, 0, 4), 'backpedal')
        self.assertNotEqual(ar._motion_kind(self.out, sensed, 0, 0, 4), 'backpedal')

    def test_backpedal_route(self):
        sensed = np.ones(8, bool)
        sensed[0] = False
        m = ar._motion_to(self.mid, 30.0, 0, sensed, 0, 4, 1, self.rot, self.close)
        self.assertEqual(m['kind'], 'backpedal')
        self.assertEqual(m['wall'], 0)
        self.assertAlmostEqual(m['arrive'], (ar.TRIGGER_REACH - 17.5) / ar.SPEED, places=6)
        self.assertAlmostEqual(m['state'](m['arrive'] / 2)[1], 30.0)

    def test_onset_known(self):
        # on its own trigger the agent knows its own wall is unlit, whatever the rays hit
        known = ar.onset_known(self.back, np.zeros(8, bool))
        self.assertTrue(known[0])
        self.assertEqual(int(known.sum()), 1)
        self.assertFalse(ar.onset_known(self.mid, np.zeros(8, bool)).any())

    def test_no_backpedal_from_trigger(self):
        # starting on the trigger, the agent never backpedals in any layout
        lay = ar.layout_table()
        for hi, lo in zip(lay['high'], lay['low']):
            r = ar.delta2_route(self.back, 180.0, int(hi), int(lo), 1, 1.05, self.rot, self.close)
            self.assertNotIn('backpedal', r['kinds'])

    def test_adjacent_wall_not_driven_blind(self):
        # a hidden adjacent High that inference places exactly is still searched for, not driven to blindly
        sensed = ar.onset_known(self.back, ar.onset_sensed(self.back[None], [170.0])[0])
        self.assertFalse(sensed[1])
        self.assertEqual(ar._motion_kind(self.back, sensed, 1, 1, 2), 'search')

    def test_pieces_match_direct(self):
        # outcomes from the pieces match a direct simulation, including backpedals that turn out wrong
        lay = ar.layout_table()
        n_back = 0
        for pos, h in ((self.mid, 180.0), (np.array(ar._pt(18.0, 1.2, 0), float), 150.0)):
            S0 = ar.onset_sensed(pos[None], [h])[0]
            for hi, lo in zip(lay['high'], lay['low']):
                pcs = ar._delta2_pieces(pos, h, int(hi), int(lo), 1, S0, self.rot, self.close)
                for d in (0.0, 1.05, 3.0):
                    k = next(i for i, (ub, _, _) in enumerate(pcs) if d + ar.GAP_TIE < ub)
                    go_low, t = ar.delta2_direct(pos, h, int(hi), int(lo), 1, d, self.rot, self.close)
                    self.assertEqual(bool(pcs[k][2]), bool(go_low))
                    self.assertAlmostEqual(pcs[k][1], t, places=6)
                r = ar.delta2_route(pos, h, int(hi), int(lo), 1, 1.05, self.rot, self.close)
                n_back += 'backpedal' in r['kinds']
        self.assertGreater(n_back, 0)


class TestTrialDelays(unittest.TestCase):
    """ trial_delays splits a logged trial's time outside the race into its three parts. """

    def _trial(self, events):
        # build a trial log at 0.02 s frames, with events at the given times (s)
        t = np.round(np.arange(0, 10.0, 0.02), 2)
        ev = np.full(len(t), None, dtype=object)
        for name, at in events.items():
            ev[int(round(at / 0.02))] = name
        return pd.DataFrame({'timeApplication': pd.to_timedelta(t, unit='s'), 'eventDescription': ev})

    def test_parts(self):
        trial = self._trial({globals.TRIAL_START: 0.0, globals.SLICE_ONSET: 1.2,
                             globals.SELECTED_TRIGGER_ACTIVATION: 3.0, globals.TRIAL_END: 5.0})
        d = ar.trial_delays(trial)
        self.assertAlmostEqual(d['trial start to slice onset'], 1.2)
        self.assertAlmostEqual(d['trigger to trial end'], 2.0)
        self.assertAlmostEqual(d['trial end to next trial'], 9.98 - 5.0)

    def test_missing_event(self):
        trial = self._trial({globals.TRIAL_START: 0.0, globals.SLICE_ONSET: 1.2, globals.SELECTED_TRIGGER_ACTIVATION: 3.0})
        self.assertIsNone(ar.trial_delays(trial))


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
