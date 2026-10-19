# Tests for analysis/policy_replay.py. Hand-set weights check the forward pass, and set positions check the
# rebuilt observations, including the seat rules for the opponent channel.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_policy_replay -v

import os
import sys
import unittest

import numpy as np

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.policy_replay as pr


def zero_weights(hidden=4, body=8):
    ''' Return a tiny policy with every weight zero except chosen head biases. '''

    return dict(W0=np.zeros((body, 43)), b0=np.zeros(body), W2=np.zeros((body, body)), b2=np.zeros(body),
                LW=np.zeros((1, 4 * hidden, body)), LR=np.zeros((1, 4 * hidden, hidden)), LB=np.zeros((1, 8 * hidden)),
                H0=np.zeros((3, hidden)), h0=np.array([0.0, np.log(2), 0.0]),
                H1=np.zeros((3, hidden)), h1=np.zeros(3), H2=np.zeros((3, hidden)), h2=np.zeros(3))


class TestPolicy(unittest.TestCase):

    def test_head_biases_set_the_probabilities(self):
        # with no input path, each head is a softmax of its bias
        P = pr.Policy(zero_weights()).run(np.zeros((3, 42)))
        self.assertTrue(np.allclose(P[:, 0], [0.25, 0.5, 0.25]))
        self.assertTrue(np.allclose(P[:, 1], 1 / 3))

    def test_lstm_matches_a_hand_step(self):
        # one hidden unit fed by the cell gate only, all gates otherwise open at sigmoid(0) = 0.5
        w = zero_weights(hidden=1, body=1)
        w['b0'] = np.array([1.0])                     # body unit 1 -> swish(1) after layer 0
        w['W2'] = np.array([[1.0]])                   # layer 2 passes it on, then swish again
        w['LW'] = np.zeros((1, 4, 1)); w['LW'][0, 3, 0] = 1.0     # cell input weight (gate order i, o, f, c)
        w['H0'] = np.array([[0.0], [1.0], [0.0]]); w['h0'] = np.zeros(3)
        x = 1 / (1 + np.exp(-1.0)); x = x * 1 / (1 + np.exp(-x))   # swish(swish(1))
        c = 0.5 * np.tanh(x); h = 0.5 * np.tanh(c)
        P = pr.Policy(w).run(np.zeros((1, 42)))
        expect = np.exp([0, h, 0]) / np.exp([0, h, 0]).sum()
        self.assertTrue(np.allclose(P[0, 0], expect))


class TestObservations(unittest.TestCase):

    def setUp(self):
        # agent at the centre facing wall 1 (High), Low at wall 5, opponent 5 units ahead
        self.T = 4
        self.eyes = np.zeros((self.T, 2))
        self.yaw = np.zeros(self.T)
        self.opp = np.tile([0.0, 5.0], (self.T, 1))

    def obs(self, seat, seats=(0,), lit_from=2):
        return pr.build_obs(self.eyes, self.yaw, self.opp, seat, 1, 5, lit_from, seats).reshape(self.T, 7, 6)

    def test_walls_light_from_lit_from(self):
        o = self.obs(0)
        # the middle ray enters wall 1's alcove: no tag before lighting, the High trigger tag after
        self.assertEqual(o[0, 3, 1:5].sum(), 0)
        self.assertEqual(o[2, 3, 3], 1)
        # before lighting the trigger is off, so the ray reaches the alcove back
        self.assertAlmostEqual(o[0, 3, 0] * 60, 20.355, places=2)
        self.assertAlmostEqual(o[2, 3, 0] * 60, 18.855, places=2)

    def test_seat_rules_for_the_opponent(self):
        # seat 0 passes through and reports the opponent on the middle ray
        o0 = self.obs(0)
        self.assertAlmostEqual(o0[0, 3, 5] * 60, 5.0 - 0.6, places=4)
        # seat 1 before 261014 has no opponent channel, and its middle ray stops at the opponent's body
        o1 = self.obs(1)
        self.assertEqual(o1[:, :, 5].sum(), 0)
        self.assertAlmostEqual(o1[0, 3, 0] * 60, 5.0 - 0.6, places=4)
        # seat 1 from 261014 behaves like seat 0
        o1_new = self.obs(1, seats=(0, 1))
        self.assertTrue(np.allclose(o1_new, o0))

    def test_261014_channel_holds_proximity(self):
        # the opponent 5 units ahead in the open: distance 4.4 before 261014, proximity 1 - 4.4 / 60 after
        old = self.obs(0)
        new = pr.build_obs(self.eyes, self.yaw, self.opp, 1, 1, 5, 2, **pr.OBS_261014).reshape(self.T, 7, 6)
        self.assertAlmostEqual(old[0, 3, 5], 4.4 / 60, places=4)
        self.assertAlmostEqual(new[0, 3, 5], 1 - 4.4 / 60, places=4)
        # rays that miss the opponent read 0 in both
        self.assertEqual(new[0, 0, 5], 0)
        self.assertEqual(old[0, 0, 5], 0)

    def test_occlusion_hides_an_opponent_behind_a_wall(self):
        import analysis.agent_raycast as ar
        import trajectory_analysis.visibility_occlusion as vo
        # the opponent sits beyond wall 1's face, beside its alcove, about 12 degrees off the agent's heading
        opp = np.tile(vo.wall_frame_to_arena(18.5, 4.0, 0), (self.T, 1))
        yaw = np.full(self.T, np.radians(12.2))
        seen = pr.build_obs(self.eyes, yaw, opp, 0, 1, 5, 2, **pr.OBS_PRE_261014).reshape(self.T, 7, 6)
        hidden = pr.build_obs(self.eyes, yaw, opp, 0, 1, 5, 2, **pr.OBS_261014).reshape(self.T, 7, 6)
        self.assertGreater(seen[:, :, 5].max(), 0)
        self.assertEqual(hidden[:, :, 5].max(), 0)


if __name__ == '__main__':
    unittest.main()
