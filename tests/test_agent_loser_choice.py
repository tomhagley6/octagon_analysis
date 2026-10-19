# Tests for analysis/agent_loser_choice.py. Synthetic trajectories with known answers check each rule
# and each reason for making no call.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_agent_loser_choice -v

import os
import sys
import unittest

import numpy as np

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
import analysis.agent_loser_choice as lc
from analysis.race_estimate import ALCOVE_CENTRES

FACE = globals.OCTAGON_WALLFACE_RADIUS


def line(a, b, n):
    ''' Return n evenly spaced points from a to b (n*2). '''

    return np.linspace(np.asarray(a, float), np.asarray(b, float), n)


class TestRules(unittest.TestCase):

    def test_heading_straight_at_a_lit_wall_is_aligned(self):
        # from the centre toward wall 1's alcove, stopping well short
        xy = line([0, 0], [0, 10], 60)
        self.assertEqual(lc.infer_choice(xy, high=1, low=5), (1, 'aligned'))
        self.assertEqual(lc.infer_choice(xy, high=5, low=1), (1, 'aligned'))

    def test_heading_at_an_unlit_wall_makes_no_call(self):
        xy = line([0, 0], [0, 10], 60)
        self.assertEqual(lc.infer_choice(xy, high=3, low=5), (None, 'aligned to unlit wall'))

    def test_entering_a_lit_alcove(self):
        # run along wall 1's face and turn into the alcove
        xy = np.vstack([line([-6, FACE - 1.0], [0, FACE - 1.0], 40), line([0, FACE - 0.9], [0, FACE + 1.5], 10)])
        self.assertEqual(lc.infer_choice(xy, high=1, low=3), (1, 'at alcove'))

    def test_passing_in_front_of_a_lit_alcove_is_not_at_the_alcove(self):
        # cross in front of wall 1's mouth, 2 units short of the face, heading on toward wall 2
        xy = line([-6, FACE - 2.0], [3, FACE - 2.0], 50)
        call, rule = lc.infer_choice(xy, high=1, low=2)
        self.assertNotEqual(rule, 'at alcove')

    def test_stopped_and_too_short(self):
        xy = np.vstack([line([0, 0], [0, 5], 30), np.repeat([[0, 5.0]], 30, axis=0)])
        self.assertEqual(lc.infer_choice(xy, high=1, low=5), (None, 'stopped'))
        self.assertEqual(lc.infer_choice(line([0, 0], [0, 3], 20), high=1, low=5), (None, 'too short'))

    def test_turning_between_two_walls_is_ambiguous(self):
        # swing from heading at wall 1 to heading at wall 3 inside the final window
        a = line([0, 0], [0, 4], 30)
        b = line([0, 4], [6, 4], 20)
        self.assertEqual(lc.infer_choice(np.vstack([a, b]), high=1, low=3)[1], 'ambiguous')


class TestTruncationCheck(unittest.TestCase):

    def test_cuts_of_a_straight_winner_are_correct(self):
        import pandas as pd
        xy = line([0, 0], ALCOVE_CENTRES[0], 120)
        t = pd.DataFrame([dict(model=1, won=True, high_wall=1, low_wall=5, chosen_wall=1, xy=xy),
                          dict(model=1, won=False, high_wall=1, low_wall=5, chosen_wall=1, xy=xy)])
        out = lc.truncation_check(t)
        self.assertEqual(len(out), len(range(31, 120, 10)))
        self.assertTrue(out['accepted'].all())
        self.assertTrue(out['correct'].all())


if __name__ == '__main__':
    unittest.main()
