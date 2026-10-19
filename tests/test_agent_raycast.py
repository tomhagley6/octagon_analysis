# Tests for analysis/agent_raycast.py. Set positions and headings with hand-worked answers check the
# ray geometry, the trigger boxes and the ideal-observer posterior over the High wall.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_agent_raycast -v

import os
import sys
import unittest

import numpy as np

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.agent_raycast as ar
import trajectory_analysis.visibility_occlusion as vo


def alcove_back_eye(wall, depth_from_back=0.61):
    ''' Return an eye position on the centre line of a wall's alcove, a set distance in front of the
        alcove back. Takes the wall id (1-8) and the distance. '''

    g = vo.get_physical_geometry()
    return np.array(vo.wall_frame_to_arena(g['alcove_back'] - depth_from_back, 0.0, wall - 1))


def facing_centre(wall):
    ''' Return the yaw (radians, clockwise from +z) of an agent in a wall's alcove facing the centre. '''

    return np.radians(45 * (wall - 1) + 180)


class TestGeometry(unittest.TestCase):

    def test_trigger_box_matches_prefab(self):
        front, back, half = ar.get_trigger_boxes()
        self.assertAlmostEqual(front, 18.855, places=2)
        self.assertAlmostEqual(back, 20.355, places=2)
        self.assertAlmostEqual(half, 2.51, places=2)

    def test_sensor_segments(self):
        seg, ids, kinds = ar.get_sensor_segments()
        self.assertEqual(len(seg), 8 * 5 + 8 * 4)
        self.assertEqual(set(ids), set(range(1, 9)))
        self.assertEqual((kinds == ar.KIND_TRIGGER).sum(), 32)

    def test_inside_trigger(self):
        eyes = np.array([[0.0, 0.0], alcove_back_eye(3), alcove_back_eye(3, depth_from_back=2.5)])
        self.assertEqual(ar.inside_trigger(eyes).tolist(), [0, 3, 0])


class TestRays(unittest.TestCase):

    def test_seven_rays_from_the_centre(self):
        # facing north from the centre, the middle ray goes down the alcove of wall 1 and hits its trigger
        ids, kinds, dist = ar.cast_rays(np.zeros((1, 2)), np.zeros(1))
        self.assertEqual(ids.shape, (1, 7))
        self.assertEqual(ids[0, 3], 1)
        self.assertEqual(kinds[0, 3], ar.KIND_TRIGGER)
        self.assertAlmostEqual(dist[0, 3], 18.855, places=2)
        # the +/-55 degree rays hit the face of the wall either side
        self.assertEqual(ids[0, 6], 2)
        self.assertEqual(ids[0, 0], 8)
        self.assertEqual(kinds[0, 6], ar.KIND_WALL)

    def test_switched_off_triggers_let_the_ray_reach_the_alcove_back(self):
        ids, kinds, dist = ar.cast_rays(np.zeros((1, 2)), np.zeros(1), triggers=False)
        self.assertEqual(ids[0, 3], 1)
        self.assertEqual(kinds[0, 3], ar.KIND_WALL)
        self.assertAlmostEqual(dist[0, 3], 20.355, places=2)

    def test_face_hit_off_the_alcove(self):
        # from the centre, 15 degrees off north misses the alcove mouth and hits wall 1's face
        ids, kinds, dist = ar.cast_rays(np.zeros((1, 2)), np.radians([15.0]), offsets_deg=[0.0])
        self.assertEqual(ids[0, 0], 1)
        self.assertEqual(kinds[0, 0], ar.KIND_WALL)
        self.assertAlmostEqual(dist[0, 0], 17.355 / np.cos(np.radians(15)), places=2)

    def test_own_alcove_is_sensed_from_inside_its_trigger(self):
        # standing in wall 3's trigger box facing out, the outer rays hit wall 3's alcove sides, and
        # no ray reports wall 3's trigger box
        eye = alcove_back_eye(3)[None]
        ids, kinds, _ = ar.cast_rays(eye, [facing_centre(3)])
        self.assertEqual(ids[0, 0], 3)
        self.assertEqual(ids[0, 6], 3)
        self.assertFalse(((ids == 3) & (kinds == ar.KIND_TRIGGER)).any())
        # the middle ray crosses the arena to the opposite wall
        self.assertEqual(ids[0, 3], 7)

    def test_adjacent_walls_unreachable_from_the_alcove_back(self):
        # no heading lets any ray from the back of wall 3's alcove reach walls 2 or 4
        yaws = np.radians(np.arange(0, 360, 0.5))
        eyes = np.repeat(alcove_back_eye(3)[None], len(yaws), axis=0)
        ids, _, _ = ar.cast_rays(eyes, yaws)
        self.assertFalse(np.isin(ids, [2, 4]).any())

    def test_sensed_walls(self):
        ids = np.array([[1, 1, 2], [3, 3, 3]])
        s = ar.sensed_walls(ids)
        self.assertEqual(s.shape, (8, 2))
        self.assertEqual(s[:, 0].nonzero()[0].tolist(), [0, 1])
        self.assertEqual(s[:, 1].nonzero()[0].tolist(), [2])


class TestOpponent(unittest.TestCase):

    def test_body_in_front_blocks_the_wall_behind(self):
        # facing north from the centre, a body 5 units ahead takes the middle ray
        ids, kinds, dist = ar.cast_rays(np.zeros((1, 2)), np.zeros(1), blocker=np.array([[0.0, 5.0]]))
        self.assertEqual(ids[0, 3], 0)
        self.assertEqual(kinds[0, 3], ar.KIND_AGENT)
        self.assertAlmostEqual(dist[0, 3], 5.0 - ar.AGENT_RADIUS, places=6)
        # the other rays still reach the walls
        self.assertTrue((kinds[0, [0, 6]] == ar.KIND_WALL).all())

    def test_player_0_senses_through_walls_player_1_does_not(self):
        # player 1 stands in wall 7's alcove; player 0 stands inside wall 3's alcove facing it,
        # so player 0's middle ray runs out of its alcove and across the arena into wall 7's alcove
        eye = alcove_back_eye(3)[None]
        opp = alcove_back_eye(7)[None]
        yaw = [facing_centre(3)]
        self.assertTrue(ar.opponent_sensed(eye, yaw, opp, player_id=0)[0])
        self.assertFalse(ar.opponent_sensed(eye, yaw, opp, player_id=1)[0])
        # an opponent straight ahead in the open is sensed from either seat
        opp_open = np.array([[eye[0, 0] - 8.0, eye[0, 1]]])
        self.assertTrue(ar.opponent_sensed(eye, yaw, opp_open, player_id=0)[0])
        self.assertTrue(ar.opponent_sensed(eye, yaw, opp_open, player_id=1)[0])

    def test_261014_rules_both_seats_and_walls_occlude(self):
        # the opponent in another alcove, seen across the arena from the back of wall 3's alcove
        eye = alcove_back_eye(3)[None]
        opp = alcove_back_eye(7)[None]
        yaw = [facing_centre(3)]
        # older builds: player 0 senses it through wall 7's alcove walls, player 1 does not
        self.assertTrue(ar.opponent_sensed(eye, yaw, opp, 0, **ar.BUILD_PRE_261014)[0])
        # 261014: the middle ray runs down wall 7's alcove, so nothing solid is in the way and both seats sense it
        self.assertTrue(ar.opponent_sensed(eye, yaw, opp, 0, **ar.BUILD_261014)[0])
        self.assertTrue(ar.opponent_sensed(eye, yaw, opp, 1, **ar.BUILD_261014)[0])
        # an opponent hidden behind a wall face is sensed only without occlusion
        # (a point beyond wall 1's face, beside its alcove, at about 12 degrees from the centre)
        hidden = np.array([vo.wall_frame_to_arena(18.5, 4.0, 0)])
        centre = np.zeros((1, 2))
        self.assertTrue(ar.opponent_sensed(centre, [np.radians(12.0)], hidden, 0, **ar.BUILD_PRE_261014)[0])
        self.assertFalse(ar.opponent_sensed(centre, [np.radians(12.0)], hidden, 0, **ar.BUILD_261014)[0])

    def test_opponent_behind_is_not_sensed(self):
        eye, opp = np.zeros((1, 2)), np.array([[0.0, -5.0]])
        self.assertFalse(ar.opponent_sensed(eye, [0.0], opp, player_id=0)[0])
        self.assertFalse(ar.opponent_sensed(eye, [0.0], opp, player_id=1)[0])


class TestPosterior(unittest.TestCase):

    def test_prior_sums_to_one_and_matches_separations(self):
        prior = ar.wall_pair_prior()
        self.assertAlmostEqual(sum(prior.values()), 1.0)
        self.assertAlmostEqual(prior[(1, 2)], 0.5 / 8 / 2)
        self.assertAlmostEqual(prior[(1, 5)], 0.25 / 8)

    def test_low_seen_alone(self):
        # Low at wall 5: High is at 4 or 6 (0.25 each), 3 or 7 (0.125 each), or 1 (0.25)
        post = ar.high_posterior(seen_low=5)
        self.assertAlmostEqual(post[4], 0.25)
        self.assertAlmostEqual(post[6], 0.25)
        self.assertAlmostEqual(post[3], 0.125)
        self.assertAlmostEqual(post[1], 0.25)
        self.assertNotIn(5, post)

    def test_unlit_walls_remove_candidates(self):
        post = ar.high_posterior(seen_low=5, seen_unlit=[6, 7, 1])
        self.assertAlmostEqual(post[4], 0.25 / 0.375)
        self.assertAlmostEqual(post[3], 0.125 / 0.375)

    def test_high_seen_is_certain(self):
        self.assertEqual(ar.high_posterior(seen_high=2), {2: 1.0})

    def test_impossible_observation_is_empty(self):
        # walls 2 and 5 are three apart, which no trial uses
        self.assertEqual(ar.high_posterior(seen_high=2, seen_low=5), {})


if __name__ == '__main__':
    unittest.main()
