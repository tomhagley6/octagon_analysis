# Tests for arena geometry. All distances are the space the player centre can reach,
# not wall positions (measured from 260723 agent logs, models 1, 23, 33, 51).
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest discover -s tests -p "test_*.py" -v

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
from plotting import plot_octagon


def wall_frame(x, y, wall):
    ''' Return distance toward a wall (p) and offset along it (q) for each point.
        Takes x and y arrays and a wall index (wall k faces 45*k degrees clockwise from +y). '''

    a = math.radians(45 * wall)
    p = x * math.sin(a) + y * math.cos(a)
    q = x * math.cos(a) - y * math.sin(a)
    return p, q


def outline_arrays():
    ''' Return the plot_octagon outline as x and y arrays (closing point dropped). Takes no inputs. '''

    points = np.array(plot_octagon.return_octagon_path_points())
    return points[:-1, 0], points[:-1, 1]


class TestGlobals(unittest.TestCase):

    def test_vertex_radius_follows_wall_face(self):
        # vertex sits where neighbouring wall-face lines meet
        expected = globals.OCTAGON_WALLFACE_RADIUS / math.cos(math.radians(22.5))
        self.assertAlmostEqual(globals.OCTAGON_VERTEX_RADIUS, expected, places=9)
        self.assertAlmostEqual(globals.OCTAGON_VERTEX_RADIUS, 18.127, places=3)

    def test_radii_are_ordered(self):
        # wall face < vertex < alcove back
        self.assertLess(globals.OCTAGON_WALLFACE_RADIUS, globals.OCTAGON_VERTEX_RADIUS)
        self.assertLess(globals.OCTAGON_VERTEX_RADIUS, globals.OCTAGON_ALCOVE_RADIUS)

    def test_alcove_depth(self):
        self.assertAlmostEqual(globals.ALCOVE_DEPTH, 3.0, places=6)

    def test_alcove_fits_in_wall(self):
        # alcove width must be less than the wall length
        wall_length = 2 * globals.OCTAGON_WALLFACE_RADIUS * math.tan(math.radians(22.5))
        self.assertLess(2 * globals.ALCOVE_HALF_WIDTH, wall_length)


class TestPlotOctagon(unittest.TestCase):

    def test_module_constants_read_globals(self):
        self.assertAlmostEqual(plot_octagon.radius, globals.OCTAGON_VERTEX_RADIUS, places=9)
        self.assertAlmostEqual(plot_octagon.alcove_length_scaled, globals.ALCOVE_DEPTH, places=9)
        self.assertAlmostEqual(plot_octagon.alcove_half_width, globals.ALCOVE_HALF_WIDTH, places=9)

    def test_vertices_at_vertex_radius(self):
        vx, vy = plot_octagon.calculate_coordinates(vertex=True)
        r = np.hypot(vx[:-1], vy[:-1])
        np.testing.assert_allclose(r, globals.OCTAGON_VERTEX_RADIUS, atol=1e-9)

    def test_vertices_offset_from_wall_directions(self):
        # vertices sit at 22.5 + 45k degrees clockwise from +y
        vx, vy = plot_octagon.calculate_coordinates(vertex=True)
        angles = np.degrees(np.arctan2(vx[:-1], vy[:-1])) % 360
        np.testing.assert_allclose(np.sort(angles), 22.5 + 45 * np.arange(8), atol=1e-9)

    def test_alcove_edges_reproduce_thirds(self):
        # half-width of a sixth of the wall gives the old 1/3 and 2/3 points
        vx, vy = plot_octagon.calculate_coordinates(vertex=True)
        wall_length = math.hypot(vx[1] - vx[0], vy[1] - vy[0])
        ex, ey = plot_octagon.calculate_alcove_edge_coordinates(vx, vy, alcove_half_width=wall_length / 6)

        # compute the old thirds points in the same interleaved order
        thirds_x, thirds_y = [], []
        for i in range(8):
            for f in (1 / 3, 2 / 3):
                thirds_x.append((1 - f) * vx[i] + f * vx[i + 1])
                thirds_y.append((1 - f) * vy[i] + f * vy[i + 1])

        np.testing.assert_allclose(ex, thirds_x, atol=1e-9)
        np.testing.assert_allclose(ey, thirds_y, atol=1e-9)

    def test_alcove_edges_on_wall_face(self):
        vx, vy = plot_octagon.calculate_coordinates(vertex=True)
        ex, ey = map(np.array, plot_octagon.calculate_alcove_edge_coordinates(vx, vy))
        self.assertEqual(len(ex), 16)
        for wall in range(8):
            # take the two edge points of this wall
            p, q = wall_frame(ex, ey, wall)
            on_wall = np.isclose(p, globals.OCTAGON_WALLFACE_RADIUS, atol=1e-6) & (np.abs(q) < 3)
            self.assertEqual(on_wall.sum(), 2, f'wall {wall}')
            np.testing.assert_allclose(np.sort(q[on_wall]),
                                       [-globals.ALCOVE_HALF_WIDTH, globals.ALCOVE_HALF_WIDTH], atol=1e-6)

    def test_outline_alcoves(self):
        x, y = outline_arrays()
        for wall in range(8):
            p, q = wall_frame(x, y, wall)
            # take the outline points behind this wall's face (the alcove back corners)
            back = p > globals.OCTAGON_WALLFACE_RADIUS + 0.5
            self.assertEqual(back.sum(), 2, f'wall {wall}')
            np.testing.assert_allclose(p[back], globals.OCTAGON_ALCOVE_RADIUS, atol=1e-6)
            np.testing.assert_allclose(np.sort(q[back]),
                                       [-globals.ALCOVE_HALF_WIDTH, globals.ALCOVE_HALF_WIDTH], atol=1e-6)

    def test_outline_wall_faces(self):
        # every outline point is either on a wall face (or its vertex) or an alcove back
        x, y = outline_arrays()
        p_max = np.max([wall_frame(x, y, k)[0] for k in range(8)], axis=0)
        on_face = np.isclose(p_max, globals.OCTAGON_WALLFACE_RADIUS, atol=1e-6)
        on_back = np.isclose(p_max, globals.OCTAGON_ALCOVE_RADIUS, atol=1e-6)
        self.assertTrue(np.all(on_face | on_back))
        self.assertEqual(on_back.sum(), 16)

    def test_outline_extent(self):
        x, y = outline_arrays()
        self.assertAlmostEqual(np.abs(x).max(), globals.OCTAGON_ALCOVE_RADIUS, places=6)
        self.assertAlmostEqual(np.abs(y).max(), globals.OCTAGON_ALCOVE_RADIUS, places=6)

    def test_maximum_distance(self):
        self.assertAlmostEqual(plot_octagon.return_maximum_distance(),
                               2 * globals.OCTAGON_ALCOVE_RADIUS, places=6)

    def test_alcove_centre_points(self):
        centres = plot_octagon.return_alcove_centre_points()
        self.assertEqual(centres.shape, (2, 8))
        # alcove centres sit halfway into each alcove
        expected_r = globals.OCTAGON_WALLFACE_RADIUS + globals.ALCOVE_DEPTH / 2
        np.testing.assert_allclose(np.hypot(*centres), expected_r, atol=1e-6)
        # first centre is the north (+y) wall
        self.assertAlmostEqual(centres[0, 0], 0.0, places=6)
        self.assertGreater(centres[1, 0], 0)


class TestCentrality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import analysis.model_characterisation as mc
        cls.mc = mc

    def centrality_at(self, x, z):
        ''' Return slice-onset centrality for a player at x, z. Takes x and z. '''

        # replace the slice-onset lookup so no trial data is needed
        with mock.patch.object(self.mc.get_indices, 'get_player_slice_onset_loc', return_value=(x, z)):
            return self.mc.slice_onset_centrality(trial=None, player_id=0)['centrality']

    def test_arena_radius_is_alcove_back(self):
        self.assertEqual(self.mc.ARENA_RADIUS, globals.OCTAGON_ALCOVE_RADIUS)

    def test_centre_is_one(self):
        self.assertAlmostEqual(self.centrality_at(0.0, 0.0), 1.0, places=9)

    def test_alcove_back_is_zero(self):
        for wall in range(8):
            a = math.radians(45 * wall)
            r = globals.OCTAGON_ALCOVE_RADIUS
            self.assertAlmostEqual(self.centrality_at(r * math.sin(a), r * math.cos(a)), 0.0, places=9)

    def test_wall_face_value(self):
        expected = 1 - globals.OCTAGON_WALLFACE_RADIUS / globals.OCTAGON_ALCOVE_RADIUS
        self.assertAlmostEqual(self.centrality_at(0.0, globals.OCTAGON_WALLFACE_RADIUS), expected, places=9)

    def test_minimum_at_alcove_back_corner(self):
        # alcove back corners sit beyond the alcove back radius, so the lowest reachable score is just below 0
        x, y = outline_arrays()
        values = [self.centrality_at(xi, yi) for xi, yi in zip(x, y)]
        corner_r = math.hypot(globals.OCTAGON_ALCOVE_RADIUS, globals.ALCOVE_HALF_WIDTH)
        self.assertAlmostEqual(min(values), 1 - corner_r / globals.OCTAGON_ALCOVE_RADIUS, places=9)
        self.assertGreater(min(values), -0.005)


if __name__ == '__main__':
    unittest.main()
