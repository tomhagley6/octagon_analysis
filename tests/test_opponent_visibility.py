# Tests for opponent visibility at slice onset with line of sight blocked by the arena walls
# (analysis/opponent_visibility.py, get_other_visible_trial and get_other_visible_session_from_trials).
#
# The reference used here does not intersect line segments. The octagon without alcoves is convex, so a
# line between two players can only be blocked by an alcove that one of them is in. It is clear if it
# leaves Self's alcove through the mouth and enters Other's alcove through the mouth (or both players
# are in the same alcove).
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_opponent_visibility -v

import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
import data_strings
from analysis import opponent_visibility

# physical geometry, recomputed from globals
R = globals.PLAYER_RADIUS
W = globals.OCTAGON_WALLFACE_RADIUS + R
H = globals.ALCOVE_HALF_WIDTH + R
FOV = 110.36
HUMAN_SOCIAL_PICKLE = os.path.join(data_strings.DATAFRAME_PATH, data_strings.FILENAME_SOCIAL + '.pkl')


def to_wall_frame(xz, wall):
    ''' Return distance toward a wall (p) and offset along it (q, positive = clockwise).
        Takes an array of x, z in the last axis and a wall row index. '''

    a = math.radians(45 * wall)
    return xz[..., 0] * math.sin(a) + xz[..., 1] * math.cos(a), xz[..., 0] * math.cos(a) - xz[..., 1] * math.sin(a)


def alcove_of(point):
    ''' Return the wall row of the physical alcove a point is in, or -1. Takes an x, z point. '''

    p = np.array([to_wall_frame(np.asarray(point, float), wall)[0] for wall in range(8)])
    return int(p.argmax()) if p.max() > W else -1


def reference_blocked(self_xz, other_xz):
    ''' Return True if an alcove wall blocks the line between two players (alcove-mouth rule).
        Takes Self and Other x, z positions. '''

    a_self, a_other = alcove_of(self_xz), alcove_of(other_xz)
    if a_self >= 0 and a_self == a_other:
        return False
    for inside, outside, wall in ((self_xz, other_xz, a_self), (other_xz, self_xz, a_other)):
        if wall < 0:
            continue
        # find where the line crosses that alcove's mouth line (p = W)
        p_i, q_i = to_wall_frame(np.asarray(inside, float), wall)
        p_o, q_o = to_wall_frame(np.asarray(outside, float), wall)
        s = (p_i - W) / (p_i - p_o)
        if abs(q_i + s * (q_o - q_i)) > H:
            return True
    return False


def make_trial(self_xz, self_heading_deg, other_xz, other_heading_deg=0.0, n_rows=10, onset_row=3, first_index=500):
    ''' Return a minimal trial dataframe with a slice onset row. Player 0 is Self, player 1 is Other.
        Takes positions (x, z) and headings (degrees clockwise from +z) for both players. '''

    df = pd.DataFrame(index=np.arange(first_index, first_index + n_rows))
    df['eventDescription'] = ''
    df.iloc[onset_row, df.columns.get_loc('eventDescription')] = globals.SLICE_ONSET
    for pid, (xz, heading) in enumerate(((self_xz, self_heading_deg), (other_xz, other_heading_deg))):
        df[globals.PLAYER_LOC_DICT[pid]['xloc']] = float(xz[0])
        df[globals.PLAYER_LOC_DICT[pid]['yloc']] = float(xz[1])
        df[globals.PLAYER_ROT_DICT[pid]['yrot']] = float(heading)
    return df


def heading_towards(self_xz, other_xz, offset_deg=0.0):
    ''' Return the heading (degrees clockwise from +z) from Self towards Other, plus an offset. Takes both positions. '''

    d = np.asarray(other_xz, float) - np.asarray(self_xz, float)
    return math.degrees(math.atan2(d[0], d[1])) + offset_deg


def north_alcove(q=0.0, depth=globals.OCTAGON_ALCOVE_RADIUS):
    ''' Return an x, z position inside the north alcove. Takes offset along the wall and distance from centre. '''

    return (q, depth)


class TestOpponentVisibleTrial(unittest.TestCase):

    def visible(self, self_xz, heading, other_xz, occlusion=True, fov=FOV):
        ''' Return get_other_visible_trial for a synthetic trial. Takes positions, Self heading and occlusion flag. '''

        trial = make_trial(self_xz, heading, other_xz)
        return opponent_visibility.get_other_visible_trial(0, current_fov=fov, trial=trial, occlusion=occlusion)

    def test_open_arena_facing(self):
        self_xz, other_xz = (0.0, 0.0), (5.0, 5.0)
        for occlusion in (True, False):
            self.assertTrue(self.visible(self_xz, heading_towards(self_xz, other_xz), other_xz, occlusion))
            self.assertFalse(self.visible(self_xz, heading_towards(self_xz, other_xz, 180), other_xz, occlusion))

    def test_fov_edge(self):
        self_xz, other_xz = (0.0, 0.0), (0.0, 10.0)
        for offset, expected in ((FOV / 2 - 0.5, True), (FOV / 2 + 0.5, False)):
            self.assertEqual(self.visible(self_xz, heading_towards(self_xz, other_xz, offset), other_xz), expected)

    def test_hidden_from_inside_alcove(self):
        # Self at the back of the north alcove, Other beside the alcove mouth in the arena, Self facing Other
        self_xz, other_xz = north_alcove(), (8.0, 14.0)
        heading = heading_towards(self_xz, other_xz)
        self.assertTrue(self.visible(self_xz, heading, other_xz, occlusion=False))
        self.assertFalse(self.visible(self_xz, heading, other_xz, occlusion=True))
        self.assertTrue(reference_blocked(self_xz, other_xz))

    def test_seen_through_mouth(self):
        self_xz, other_xz = north_alcove(), (0.0, -5.0)
        self.assertTrue(self.visible(self_xz, heading_towards(self_xz, other_xz), other_xz))

    def test_other_hidden_inside_alcove(self):
        # Other at the back of the north alcove, Self in the arena off to the side
        self_xz, other_xz = (10.0, 12.0), north_alcove()
        self.assertFalse(self.visible(self_xz, heading_towards(self_xz, other_xz), other_xz))

    def test_same_alcove(self):
        self_xz, other_xz = north_alcove(q=-1.5), north_alcove(q=1.5, depth=18.0)
        self.assertTrue(self.visible(self_xz, heading_towards(self_xz, other_xz), other_xz))

    def test_missing_positions_or_heading_not_visible(self):
        cases = [((np.nan, 0.0), 0.0, (0.0, 5.0)), ((0.0, 0.0), 0.0, (np.nan, 5.0)), ((0.0, 0.0), np.nan, (0.0, 5.0))]
        for self_xz, heading, other_xz in cases:
            for occlusion in (True, False):
                self.assertFalse(self.visible(self_xz, heading, other_xz, occlusion), f'{self_xz}, {heading}, {other_xz}')

    def test_default_is_occlusion(self):
        self_xz, other_xz = north_alcove(), (8.0, 14.0)
        trial = make_trial(self_xz, heading_towards(self_xz, other_xz), other_xz)
        self.assertEqual(opponent_visibility.get_other_visible_trial(0, current_fov=FOV, trial=trial),
                         opponent_visibility.get_other_visible_trial(0, current_fov=FOV, trial=trial, occlusion=True))

    def test_occlusion_matches_reference_random_positions(self):
        from matplotlib.path import Path
        from plotting import plot_octagon
        rng = np.random.default_rng(1)
        outline = Path(np.array(plot_octagon.return_octagon_path_points()))
        pts = rng.uniform(-globals.OCTAGON_ALCOVE_RADIUS, globals.OCTAGON_ALCOVE_RADIUS, size=(20000, 2))
        pts = pts[outline.contains_points(pts)]
        # oversample alcove positions, where blocking happens
        in_alcove = np.array([alcove_of(p) >= 0 for p in pts])
        pts = np.vstack([pts[in_alcove][:600], pts[~in_alcove][:300]])
        pairs = rng.choice(len(pts), size=(1500, 2))
        mismatches = blocked_count = 0
        for i, j in pairs:
            if i == j:
                continue
            self_xz, other_xz = pts[i], pts[j]
            # face Other so only blocking decides visibility
            visible = self.visible(self_xz, heading_towards(self_xz, other_xz), other_xz)
            expected = not reference_blocked(self_xz, other_xz)
            mismatches += visible != expected
            blocked_count += not expected
        self.assertGreater(blocked_count, 50)
        self.assertEqual(mismatches, 0)


class TestOpponentVisibleSession(unittest.TestCase):

    def test_session_matches_trials(self):
        trials = [make_trial((0.0, 0.0), 0.0, (0.0, 5.0)), make_trial(north_alcove(), heading_towards(north_alcove(), (8.0, 14.0)), (8.0, 14.0)),
                  make_trial((0.0, 0.0), 180.0, (0.0, 5.0))]
        for occlusion in (True, False):
            session = opponent_visibility.get_other_visible_session_from_trials(0, trials, current_fov=FOV, occlusion=occlusion)
            self.assertEqual(session.dtype, bool)
            expected = [opponent_visibility.get_other_visible_trial(0, current_fov=FOV, trial=t, occlusion=occlusion) for t in trials]
            self.assertEqual(session.tolist(), expected)
        self.assertEqual(opponent_visibility.get_other_visible_session_from_trials(0, trials, current_fov=FOV).tolist(), [True, False, False])

    def test_angular_matches_old_functions(self):
        # occlusion=False reproduces the existing angle-based pipeline
        rng = np.random.default_rng(2)
        trials = [make_trial(rng.uniform(-12, 12, 2), rng.uniform(0, 360), rng.uniform(-12, 12, 2)) for _ in range(200)]
        angles = opponent_visibility.get_angle_of_opponent_from_player_session(0, trials)
        old = opponent_visibility.get_other_visible_session(angles, FOV)
        new = opponent_visibility.get_other_visible_session_from_trials(0, trials, current_fov=FOV, occlusion=False)
        np.testing.assert_array_equal(old, new)


@unittest.skipUnless(os.path.isfile(HUMAN_SOCIAL_PICKLE), 'human social trial list not available')
class TestHumanTrials(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import data_extraction.data_saving as data_saving
        cls.social = data_saving.load_data(data_strings.FILENAME_SOCIAL)[:10]

    def test_occlusion_only_hides(self):
        hidden = total = 0
        for session in self.social:
            for pid in (0, 1):
                angular = opponent_visibility.get_other_visible_session_from_trials(pid, session, current_fov=FOV, occlusion=False)
                occluded = opponent_visibility.get_other_visible_session_from_trials(pid, session, current_fov=FOV)
                self.assertFalse((occluded & ~angular).any())
                hidden += (angular & ~occluded).sum()
                total += len(session)
        self.assertGreater(total, 0)

    def test_matches_reference_on_logged_positions(self):
        mismatches = checked = 0
        for session in self.social:
            for trial in session:
                for pid in (0, 1):
                    self_pos, other_pos = opponent_visibility.get_two_player_positions_slice_onset(pid, trial=trial)
                    self_xz, other_xz = self_pos[:, 0].astype(float), other_pos[:, 0].astype(float)
                    if not (np.isfinite(self_xz).all() and np.isfinite(other_xz).all()):
                        continue
                    angular = opponent_visibility.get_other_visible_trial(pid, current_fov=FOV, trial=trial, occlusion=False)
                    if not angular:
                        continue
                    # skip logged positions past the reachable arena, where the mouth rule does not apply
                    reach = [max(to_wall_frame(xz, w)[0] for w in range(8)) for xz in (self_xz, other_xz)]
                    if max(reach) > globals.OCTAGON_ALCOVE_RADIUS:
                        continue
                    occluded = opponent_visibility.get_other_visible_trial(pid, current_fov=FOV, trial=trial)
                    mismatches += occluded != (not reference_blocked(self_xz, other_xz))
                    checked += 1
        self.assertGreater(checked, 0)
        self.assertEqual(mismatches, 0)


if __name__ == '__main__':
    unittest.main()
