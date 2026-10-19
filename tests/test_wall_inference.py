# Tests for analysis/wall_inference.py. Set wall layouts check the ideal observer's onset reading,
# including the 180 degree inference from unlit walls and the 'closer, side uncertain' case.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_wall_inference -v

import os
import sys
import unittest

import numpy as np

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.wall_inference as wi
from analysis.race_estimate import ALCOVE_CENTRES


class TestOnsetInference(unittest.TestCase):

    def setUp(self):
        # an agent at the back of wall 3's alcove
        self.xy = ALCOVE_CENTRES[2] * 1.05

    def test_180_inferred_when_every_wall_near_low_is_unlit(self):
        # Low at wall 7, walls 5, 6, 8 and 1 (within 90 degrees of it) seen unlit: High must be wall 3
        r = wi.onset_inference([7, 5, 6, 8, 1], high=3, low=7, xy0=self.xy)
        self.assertFalse(r['high_seen0'])
        self.assertAlmostEqual(r['p_high_true'], 1.0)
        self.assertEqual(r['n_candidates'], 1)

    def test_closer_but_side_uncertain(self):
        # Low at wall 6 with walls 5, 7 and 8 unlit leaves wall 4 (90 degrees) or wall 2 (180 degrees),
        # both neighbours of the agent's alcove and both closer than Low
        r = wi.onset_inference([6, 5, 7, 8], high=2, low=6, xy0=self.xy)
        self.assertEqual(r['n_candidates'], 2)
        self.assertAlmostEqual(r['p_high_true'], 0.25 / 0.375)
        self.assertTrue(r['all_candidates_closer'])
        self.assertEqual(wi.inference_state(r), 'High certainly closer, side uncertain')

    def test_states(self):
        self.assertEqual(wi.inference_state(dict(high_seen0=True, low_seen0=False, p_high_true=1.0,
                                                 n_candidates=1, all_candidates_closer=True)), 'High sensed')
        self.assertEqual(wi.inference_state(dict(high_seen0=False, low_seen0=False, p_high_true=0.1,
                                                 n_candidates=8, all_candidates_closer=False)), 'Low not sensed')
        r = wi.onset_inference([7, 5, 6, 8, 1], high=3, low=7, xy0=self.xy)
        self.assertEqual(wi.inference_state(r), 'High position certain')
        # Low two alcoves away (wall 5) seen alone: the opposite wall (7) is still a candidate and is further
        r = wi.onset_inference([5], high=6, low=5, xy0=self.xy)
        self.assertEqual(wi.inference_state(r), 'uncertain')
        # Low opposite (wall 7) seen alone: every candidate is nearer than Low
        r = wi.onset_inference([7], high=6, low=7, xy0=self.xy)
        self.assertEqual(wi.inference_state(r), 'High certainly closer, side uncertain')


if __name__ == '__main__':
    unittest.main()
