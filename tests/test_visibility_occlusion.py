# Tests for trajectory_analysis/visibility_occlusion.py and get_wall_visible(occlusion=True).
#
# The reference used here does not intersect line segments. The octagon without alcoves is convex,
# so from the open arena every wall face is in clear sight. From inside an alcove, a point is in
# clear sight only if the line to it leaves through the alcove mouth (crosses the mouth line within
# the alcove half-width). Points on that alcove's own wall are never in sight.
#
# The module is exact, while the reference samples points along each coloured face. So the module
# must see every wall the reference sees, and can only add slivers narrower than the point spacing.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_visibility_occlusion -v

import math
import os
import sys
import unittest

import numpy as np

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
import data_strings
from plotting import plot_octagon
from trajectory_analysis import visibility_occlusion as vo
from trajectory_analysis import trajectory_headangle

# physical geometry, recomputed from globals rather than taken from the module
R = globals.PLAYER_RADIUS
W = globals.OCTAGON_WALLFACE_RADIUS + R
B = globals.OCTAGON_ALCOVE_RADIUS + R
H = globals.ALCOVE_HALF_WIDTH + R
L = W * math.tan(math.radians(22.5))

RNG = np.random.default_rng(0)
SIM_FOLDER = os.path.join(data_strings.SIM_DATA_FOLDER, '2607', '260723')


def to_wall_frame(xz, wall):
    ''' Return distance toward a wall (p) and offset along it (q, positive = clockwise).
        Takes an array of x, z in the last axis and a wall row index. '''

    a = math.radians(45 * wall)
    p = xz[..., 0] * math.sin(a) + xz[..., 1] * math.cos(a)
    q = xz[..., 0] * math.cos(a) - xz[..., 1] * math.sin(a)
    return p, q


def reference_face_points(points_per_face):
    ''' Return coloured-face sample points, shape num_walls*(2*points_per_face)*2, and each point's wall.
        Takes the number of points per face. Uses the same midpoint spacing as the module. '''

    steps = (np.arange(points_per_face) + 0.5) / points_per_face
    q = np.concatenate([-L + steps * (L - H), H + steps * (L - H)])
    points = np.zeros((8, q.size, 2))
    for wall in range(8):
        a = math.radians(45 * wall)
        points[wall, :, 0] = W * math.sin(a) + q * math.cos(a)
        points[wall, :, 1] = W * math.cos(a) - q * math.sin(a)
    return points


def reference_blocked(eye, targets, target_walls):
    ''' Return a boolean array (num_targets) of blocked sight lines, using the alcove-mouth rule.
        Takes one eye position (2), target points (num_targets*2) and each target's wall row. '''

    # find the alcove the eye is in, if any
    in_alcove = [wall for wall in range(8) if to_wall_frame(eye, wall)[0] > W]
    if not in_alcove:
        return np.zeros(len(targets), dtype=bool)
    wall = in_alcove[0]

    # find where each sight line crosses the mouth line (p = W) of that alcove
    p_e, q_e = to_wall_frame(eye, wall)
    p_t, q_t = to_wall_frame(targets, wall)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (p_e - W) / (p_e - p_t)
    q_cross = q_e + s * (q_t - q_e)

    # block points on the alcove's own wall, and lines that cross the mouth line outside the mouth
    return (target_walls == wall) | (np.abs(q_cross) > H)


def reference_wall_visible(eyes, headings, fov, points_per_face):
    ''' Return a num_walls*timepoints boolean array from the reference rule.
        Takes eye positions (timepoints*2), unit headings (timepoints*2), total FoV and points per face. '''

    points = reference_face_points(points_per_face)
    targets = points.reshape(-1, 2)
    target_walls = np.repeat(np.arange(8), points.shape[1])
    cos_half = math.cos(math.radians(fov / 2))

    out = np.zeros((8, len(eyes)), dtype=bool)
    for i, (eye, h) in enumerate(zip(eyes, headings)):
        # test field of view, then the sight line
        d = targets - eye
        in_fov = (d @ h) / np.linalg.norm(d, axis=1) > cos_half
        visible = in_fov & ~reference_blocked(eye, targets, target_walls)
        out[:, i] = visible.reshape(8, -1).any(axis=1)
    return out


def random_reachable_positions(n, alcove_only=False):
    ''' Return n random positions inside the reachable outline (plot_octagon), shape n*2.
        Takes the number of positions and whether to keep only positions inside a physical alcove. '''

    from matplotlib.path import Path
    outline = Path(np.array(plot_octagon.return_octagon_path_points()))
    kept = []
    while len(kept) < n:
        xz = RNG.uniform(-globals.OCTAGON_ALCOVE_RADIUS, globals.OCTAGON_ALCOVE_RADIUS, size=(4 * n, 2))
        xz = xz[outline.contains_points(xz)]
        if alcove_only:
            p_max = np.max([to_wall_frame(xz, k)[0] for k in range(8)], axis=0)
            xz = xz[p_max > W]
        kept.extend(xz)
    return np.array(kept[:n])


def headings_from_degrees(deg):
    ''' Return unit heading vectors (n*2) for angles clockwise from +z. Takes an array of degrees. '''

    rad = np.radians(deg)
    return np.column_stack([np.sin(rad), np.cos(rad)])


class TestGeometry(unittest.TestCase):

    def test_physical_geometry(self):
        g = vo.get_physical_geometry()
        self.assertAlmostEqual(g['wall_face'], W, places=9)
        self.assertAlmostEqual(g['alcove_back'], B, places=9)
        self.assertAlmostEqual(g['alcove_half_width'], H, places=9)
        self.assertAlmostEqual(g['half_wall_length'], L, places=9)
        self.assertAlmostEqual(g['vertex'], W / math.cos(math.radians(22.5)), places=9)

    def test_coloured_face_segments(self):
        faces = vo.get_coloured_face_segments()
        self.assertEqual(faces.shape, (8, 2, 2, 2))
        for wall in range(8):
            p, q = to_wall_frame(faces[wall], wall)
            np.testing.assert_allclose(p, W, atol=1e-9)
            # counterclockwise face runs vertex to alcove edge, clockwise face runs alcove edge to vertex
            np.testing.assert_allclose(q, [[-L, -H], [H, L]], atol=1e-9)

    def test_eye_alcove_lookup(self):
        for wall in range(8):
            a = math.radians(45 * wall)
            inside = np.array([[B - 1, 0.0]]) @ np.array([[math.sin(a), math.cos(a)], [math.cos(a), -math.sin(a)]])
            outside = np.array([[W - 0.01, 0.0]]) @ np.array([[math.sin(a), math.cos(a)], [math.cos(a), -math.sin(a)]])
            self.assertEqual(vo.find_eye_alcove(inside)[0], wall)
            self.assertEqual(vo.find_eye_alcove(outside)[0], -1)

    def test_segments_form_closed_outline(self):
        segments = vo.get_wall_segments()
        self.assertEqual(segments.shape, (40, 2, 2))
        # each segment ends where the next starts, and the last returns to the first
        np.testing.assert_allclose(segments[:, 1], np.roll(segments[:, 0], -1, axis=0), atol=1e-9)

    def test_segments_match_reachable_outline_plus_radius(self):
        # each physical corner sits the player radius beyond the matching reachable corner along the wall normal
        segments = vo.get_wall_segments()
        for wall in range(8):
            p, q = to_wall_frame(segments[5 * wall:5 * wall + 5, 0], wall)
            np.testing.assert_allclose(p, [W, W, B, B, W], atol=1e-9)
            np.testing.assert_allclose(q, [-L, -H, -H, H, H], atol=1e-9)


class TestLineOfSight(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        points = reference_face_points(20)
        cls.targets = points.reshape(-1, 2)
        cls.target_walls = np.repeat(np.arange(8), points.shape[1])

    def test_centre_sees_every_face(self):
        blocked = vo.is_line_of_sight_blocked(np.zeros((1, 2)), self.targets)
        self.assertFalse(blocked.any())

    def test_open_arena_never_blocked(self):
        # take reachable positions outside every alcove
        eyes = random_reachable_positions(2000)
        p_max = np.max([to_wall_frame(eyes, k)[0] for k in range(8)], axis=0)
        eyes = eyes[p_max <= W]
        blocked = vo.is_line_of_sight_blocked(eyes, self.targets)
        self.assertFalse(blocked.any())

    def test_own_wall_hidden_from_alcove(self):
        for wall in range(8):
            a = math.radians(45 * wall)
            eye = np.array([[globals.OCTAGON_ALCOVE_RADIUS * math.sin(a), globals.OCTAGON_ALCOVE_RADIUS * math.cos(a)]])
            blocked = vo.is_line_of_sight_blocked(eye, self.targets)[0]
            self.assertTrue(blocked[self.target_walls == wall].all(), f'wall {wall}')

    def test_matches_mouth_rule_inside_alcoves(self):
        eyes = random_reachable_positions(1500, alcove_only=True)
        blocked = vo.is_line_of_sight_blocked(eyes, self.targets)
        mismatches = 0
        for i, eye in enumerate(eyes):
            expected = reference_blocked(eye, self.targets, self.target_walls)
            mismatches += (blocked[i] != expected).sum()
        self.assertEqual(mismatches, 0)

    def test_mouth_edge_angles(self):
        # a point at the back of the north alcove, centred, sees out to atan(H / depth) either side of south
        depth = globals.OCTAGON_ALCOVE_RADIUS - W
        half_angle = math.degrees(math.atan(H / depth))
        eye = np.array([[0.0, globals.OCTAGON_ALCOVE_RADIUS]])
        # place targets just inside and just outside that angle, past the mouth but inside the arena
        for sign in (1, -1):
            for offset, expect_blocked in ((-0.5, False), (0.5, True)):
                ang = math.radians(180 + sign * (half_angle + offset))
                target = eye + 6 * np.array([[math.sin(ang), math.cos(ang)]])
                self.assertLess(np.hypot(*target[0]), W)
                blocked = vo.is_line_of_sight_blocked(eye, target)[0, 0]
                self.assertEqual(blocked, expect_blocked, f'sign {sign}, offset {offset}')
        self.assertAlmostEqual(half_angle, 46.4, places=1)


class TestWallVisibleOccluded(unittest.TestCase):

    def test_fov_out_of_range_raises(self):
        for fov in (0, 180, 200):
            with self.assertRaises(ValueError):
                vo.get_wall_visible_occluded(np.zeros((2, 25)), headings_from_degrees(np.zeros(25)).T, current_fov=fov)

    def test_output_shape_and_type(self):
        wv = vo.get_wall_visible_occluded(np.zeros((2, 30)), headings_from_degrees(np.zeros(30)).T)
        self.assertEqual(wv.shape, (8, 30))
        self.assertEqual(wv.dtype, bool)

    def test_crops_trajectory_to_heading_length(self):
        # trajectory longer than headings (as in get_wall_visible) is cropped from the end
        traj = np.zeros((2, 35))
        traj[:, 30:] = 99
        wv = vo.get_wall_visible_occluded(traj, headings_from_degrees(np.zeros(30)).T)
        self.assertEqual(wv.shape, (8, 30))

    def test_known_views(self):
        cases = [
            # (eye, heading deg, expected visible rows)
            ((0.0, 0.0), 0, [0, 1, 7]),
            ((0.0, 0.0), 180, [3, 4, 5]),
            ((0.0, globals.OCTAGON_ALCOVE_RADIUS), 180, [2, 3, 4, 5, 6]),
            ((globals.ALCOVE_HALF_WIDTH, globals.OCTAGON_ALCOVE_RADIUS), 180, [3, 4, 5, 6]),
            ((-globals.ALCOVE_HALF_WIDTH, globals.OCTAGON_ALCOVE_RADIUS), 180, [2, 3, 4, 5]),
            # facing into the back of the north alcove sees nothing coloured
            ((0.0, globals.OCTAGON_ALCOVE_RADIUS - 0.5), 0, []),
        ]
        for eye, deg, expected in cases:
            wv = vo.get_wall_visible_occluded(np.array(eye)[:, None].repeat(25, axis=1),
                                              headings_from_degrees(np.full(25, deg)).T, current_fov=110)
            self.assertTrue((wv == wv[:, :1]).all())
            self.assertEqual(np.where(wv[:, 0])[0].tolist(), expected, f'eye {eye}, heading {deg}')

    def test_contains_and_converges_to_reference_random_frames(self):
        eyes = np.vstack([random_reachable_positions(1500), random_reachable_positions(1500, alcove_only=True)])
        headings = headings_from_degrees(RNG.uniform(0, 360, len(eyes)))
        for fov in (60, 110.36, 132):
            module = vo.get_wall_visible_occluded(eyes.T, headings.T, current_fov=fov)
            extra = []
            for density in (20, 400, 4000):
                reference = reference_wall_visible(eyes, headings, fov, density)
                # never miss a wall the reference sees
                self.assertEqual((reference & ~module).sum(), 0, f'fov {fov}, density {density}')
                extra.append((module & ~reference).sum())
            # walls only the module sees shrink as the reference samples more finely
            self.assertTrue(extra[0] >= extra[1] >= extra[2], f'fov {fov}: {extra}')
            self.assertLess(extra[2] / module.size, 0.0005, f'fov {fov}: {extra}')

    def test_sliver_at_fov_edge(self):
        # wall 1's counterclockwise face starts at the vertex 22.5 degrees from the centre
        for offset, expected in ((0.05, True), (-0.05, False)):
            heading = 22.5 + offset - 55
            wv = vo.get_wall_visible_occluded(np.zeros((2, 25)), headings_from_degrees(np.full(25, heading)).T, current_fov=110)
            self.assertEqual(bool(wv[1, 0]), expected, f'offset {offset}')

    def test_sliver_at_mouth_edge(self):
        # from inside the north alcove, rotate a heading so its FoV edge sits just inside or outside the mouth
        eye = np.array([0.0, globals.OCTAGON_ALCOVE_RADIUS])
        corner = np.array([H, W])                                 # clockwise mouth corner of the north alcove
        edge = math.degrees(math.atan2(*(corner - eye))) % 360    # direction to that corner
        for offset, expected in ((0.05, True), (-0.05, False)):
            # put the clockwise FoV edge just past or just short of the corner, with the rest of the FoV on the
            # blocked side. The sliver past the corner looks through the mouth onto the east wall's coloured face.
            heading = edge + offset - 55
            wv = vo.get_wall_visible_occluded(eye[:, None].repeat(25, axis=1), headings_from_degrees(np.full(25, heading)).T,
                                              current_fov=110)
            self.assertEqual(bool(wv[:, 0].any()), expected, f'offset {offset}')

    def test_short_headings_do_not_break_normalisation(self):
        # unsmoothed and smoothed heading vectors of any length give the same result
        eyes = random_reachable_positions(50, alcove_only=True)
        headings = headings_from_degrees(RNG.uniform(0, 360, 50))
        a = vo.get_wall_visible_occluded(eyes.T, headings.T)
        b = vo.get_wall_visible_occluded(eyes.T, 0.3 * headings.T)
        np.testing.assert_array_equal(a, b)

    def test_invalid_frames_hide_every_wall(self):
        # a frame with a missing or unusable eye position or heading shows no walls, as in the angular method
        n, bad = 25, 3
        in_alcove = (0.0, globals.OCTAGON_ALCOVE_RADIUS)
        cases = {
            'NaN eye': ((0.0, 0.0), 'eye', np.nan),
            'NaN heading': ((0.0, 0.0), 'heading', np.nan),
            'NaN eye and heading': ((0.0, 0.0), 'both', np.nan),
            'zero-length heading': ((0.0, 0.0), 'heading', 0.0),
            'infinite eye': ((0.0, 0.0), 'eye', np.inf),
            'NaN heading inside alcove': (in_alcove, 'heading', np.nan),
        }
        for label, (eye, which, value) in cases.items():
            traj = np.array(eye)[:, None].repeat(n, axis=1)
            head = headings_from_degrees(np.full(n, 180.0)).T
            good = vo.get_wall_visible_occluded(traj, head, current_fov=110)
            if which in ('eye', 'both'):
                traj[:, bad] = value
            if which in ('heading', 'both'):
                head[:, bad] = value
            wv = vo.get_wall_visible_occluded(traj, head, current_fov=110)
            self.assertFalse(wv[:, bad].any(), label)
            # leave every other frame unchanged
            others = np.arange(n) != bad
            np.testing.assert_array_equal(wv[:, others], good[:, others], err_msg=label)

    def test_invalid_frames_do_not_warn(self):
        traj = np.zeros((2, 25)); traj[:, 3] = np.nan
        head = headings_from_degrees(np.zeros(25)).T; head[:, 5] = np.nan
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            vo.get_wall_visible_occluded(traj, head)


@unittest.skipUnless(os.path.isdir(SIM_FOLDER), '260723 simulation data not available')
class TestRealTrials(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import contextlib, io
        from parse_data import prepare_sim_data
        from trajectory_analysis import trajectory_vectors
        cls.trajectory_vectors = trajectory_vectors
        with contextlib.redirect_stdout(io.StringIO()):
            # model 23 is peripheral, so many frames are inside alcoves
            _, trial_list = prepare_sim_data.prepare_single_model_data(23, sim_data_folder=SIM_FOLDER)
        cls.trials = trial_list[1:31]

    def test_default_is_occlusion(self):
        for trial in self.trials[:5]:
            default = trajectory_headangle.get_wall_visible(trial=trial, player_id=0, current_fov=110)
            occluded = trajectory_headangle.get_wall_visible(trial=trial, player_id=0, current_fov=110, occlusion=True)
            if isinstance(default, float):
                self.assertTrue(isinstance(occluded, float))
            else:
                np.testing.assert_array_equal(default, occluded)

    def test_same_shape_as_angular_method(self):
        for trial in self.trials:
            for pid in (0, 1):
                for full in (False, True):
                    a = trajectory_headangle.get_wall_visible(trial=trial, player_id=pid, current_fov=110,
                                                              full_trial_window=full, occlusion=False)
                    b = trajectory_headangle.get_wall_visible(trial=trial, player_id=pid, current_fov=110,
                                                              full_trial_window=full, occlusion=True)
                    self.assertEqual(isinstance(a, float), isinstance(b, float))
                    if not isinstance(a, float):
                        self.assertEqual(a.shape, b.shape)

    def test_contains_and_converges_to_reference_on_logged_frames(self):
        missed = {20: 0, 400: 0}
        extra = {20: 0, 400: 0}
        total = 0
        for trial in self.trials:
            for pid in (0, 1):
                wv = trajectory_headangle.get_wall_visible(trial=trial, player_id=pid, current_fov=110, occlusion=True)
                if isinstance(wv, float):
                    continue
                # rebuild the same inputs get_wall_visible uses
                traj = self.trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=pid)
                angles = self.trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=pid)
                smoothed = trajectory_headangle.get_smoothed_player_head_angle_vectors_for_trial(angles)
                n = smoothed.shape[1]
                headings = (smoothed / np.linalg.norm(smoothed, axis=0)).T
                for density in missed:
                    reference = reference_wall_visible(traj[:, :n].T, headings, 110, density)
                    missed[density] += (reference & ~wv).sum()
                    extra[density] += (wv & ~reference).sum()
                total += wv.size
        self.assertGreater(total, 0)
        self.assertEqual(missed, {20: 0, 400: 0})
        self.assertGreaterEqual(extra[20], extra[400])
        self.assertLess(extra[400] / total, 0.0005)


HUMAN_SOCIAL_PICKLE = os.path.join(data_strings.DATAFRAME_PATH, data_strings.FILENAME_SOCIAL + '.pkl')


@unittest.skipUnless(os.path.isfile(HUMAN_SOCIAL_PICKLE), 'human social trial list not available')
class TestHumanTrials(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import data_extraction.data_saving as data_saving
        from trajectory_analysis import trajectory_vectors
        cls.trajectory_vectors = trajectory_vectors
        social = data_saving.load_data(data_strings.FILENAME_SOCIAL)
        # keep trial and player pairs with a missing head angle inside the visibility window
        cls.cases = []
        for session in social:
            for trial in session:
                for pid in (0, 1):
                    angles = trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=pid)
                    if np.isnan(np.asarray(angles, float)).any():
                        cls.cases.append((trial, pid))

    def test_missing_head_angles_hide_walls(self):
        self.assertGreater(len(self.cases), 0)
        checked = 0
        for trial, pid in self.cases:
            wv = trajectory_headangle.get_wall_visible(trial=trial, player_id=pid, current_fov=110.36)
            if isinstance(wv, float):
                continue
            # find frames whose smoothed heading or position is missing
            angles = self.trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=pid)
            smoothed = trajectory_headangle.get_smoothed_player_head_angle_vectors_for_trial(angles)
            traj = self.trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=pid)[:, :smoothed.shape[1]]
            invalid = ~np.isfinite(smoothed).all(axis=0) | ~np.isfinite(traj).all(axis=0)
            self.assertTrue(invalid.any())
            self.assertFalse(wv[:, invalid].any())
            # and agree with the angular method on those frames
            angular = trajectory_headangle.get_wall_visible(trial=trial, player_id=pid, current_fov=110.36, occlusion=False)
            np.testing.assert_array_equal(wv[:, invalid], angular[:, invalid])
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == '__main__':
    unittest.main()
