# %%
# Wall visibility with line of sight blocked by the arena walls.
#
# Uses physical wall positions (the reachable distances in globals plus the player radius),
# because the player's body does not limit sight. Each wall's colour covers its wall face
# either side of the alcove, not the alcove itself. A wall counts as visible on a frame if any
# part of its coloured face is inside the field of view with a clear line of sight from the
# camera (the logged player position).
#
# Visibility is exact, not sampled. The octagon without alcoves is convex, so only alcove walls
# can block sight, and only when the eye is inside that alcove. Then the eye sees out through the
# alcove mouth only. The field of view and the mouth are both wedges from the eye (under 180
# degrees), so each coloured face is clipped to the part inside both wedges.
#
# Wall rows follow trajectory_headangle.get_wall_visible. Row k is the wall centred 45*k degrees
# clockwise from +z (row 0 = north wall = wall 1).

import math
import numpy as np
import globals

NUM_WALLS = globals.NUM_WALLS
EPS = 1e-9                  # tolerance for intersection tests and clipped face length


# %%
## GEOMETRY ##


# %%
def get_physical_geometry():
    ''' Return physical wall positions as distances from the arena centre (and the alcove half-width).
        Takes no inputs. '''

    # add the player radius to each reachable distance
    wall_face = globals.OCTAGON_WALLFACE_RADIUS + globals.PLAYER_RADIUS

    return dict(wall_face=wall_face,
                vertex=wall_face / math.cos(math.radians(22.5)),
                alcove_back=globals.OCTAGON_ALCOVE_RADIUS + globals.PLAYER_RADIUS,
                alcove_half_width=globals.ALCOVE_HALF_WIDTH + globals.PLAYER_RADIUS,
                half_wall_length=wall_face * math.tan(math.radians(22.5)))


# %%
def wall_frame_to_arena(p, q, wall):
    ''' Return arena x, z for points given in one wall's frame.
        Takes distance toward the wall (p), offset along the wall (q, positive = clockwise) and a wall row index. '''

    a = math.radians(45 * wall)
    x = p * math.sin(a) + q * math.cos(a)
    z = p * math.cos(a) - q * math.sin(a)
    return x, z


# %%
def arena_to_wall_frame(xz, wall):
    ''' Return distance toward a wall (p) and offset along it (q, positive = clockwise).
        Takes an array with x, z in the last axis and a wall row index. '''

    a = math.radians(45 * wall)
    p = xz[..., 0] * math.sin(a) + xz[..., 1] * math.cos(a)
    q = xz[..., 0] * math.cos(a) - xz[..., 1] * math.sin(a)
    return p, q


# %%
def get_coloured_face_segments():
    ''' Return the coloured faces as line segments, shape num_walls*2*2*2 (wall; counterclockwise face,
        clockwise face; start, end; x, z). Each face runs from a vertex to the alcove edge, or back. Takes no inputs. '''

    g = get_physical_geometry()
    W, H, L = g['wall_face'], g['alcove_half_width'], g['half_wall_length']

    faces = np.zeros((NUM_WALLS, 2, 2, 2))
    for wall in range(NUM_WALLS):
        # get the counterclockwise face (vertex to alcove edge), then the clockwise face (alcove edge to vertex)
        for f, (q0, q1) in enumerate([(-L, -H), (H, L)]):
            faces[wall, f, 0] = wall_frame_to_arena(W, q0, wall)
            faces[wall, f, 1] = wall_frame_to_arena(W, q1, wall)

    return faces


# %%
def get_wall_segments():
    ''' Return every physical wall section as a line segment, shape (5*num_walls)*2*2 (start, end; x, z).
        Per wall these are the two coloured faces, the two alcove sides and the alcove back. Takes no inputs. '''

    g = get_physical_geometry()
    W, B = g['wall_face'], g['alcove_back']
    H, L = g['alcove_half_width'], g['half_wall_length']

    # list each section's start and end as (p, q) in the wall's frame
    sections = [((W, -L), (W, -H)),      # counterclockwise face
                ((W, -H), (B, -H)),      # counterclockwise alcove side
                ((B, -H), (B, H)),       # alcove back
                ((B, H), (W, H)),        # clockwise alcove side
                ((W, H), (W, L))]        # clockwise face

    segments = []
    for wall in range(NUM_WALLS):
        for (p0, q0), (p1, q1) in sections:
            segments.append([wall_frame_to_arena(p0, q0, wall), wall_frame_to_arena(p1, q1, wall)])

    return np.array(segments)


# %%
## LINE OF SIGHT ##


# %%
def cross_2d(a, b):
    ''' Return the z component of the cross product of 2D vectors (broadcast over leading axes).
        Takes two arrays with x, z in the last axis. '''

    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# %%
def is_line_of_sight_blocked(eyes, targets, segments=None):
    ''' Return a boolean array, shape timepoints*num_targets, that is True where the line from the eye
        to a target point crosses a wall segment. Takes eye positions (timepoints*2), target points
        (num_targets*2) and optionally the wall segments (defaults to get_wall_segments).
        For point targets such as the opponent; wall visibility uses the exact wedge method below. '''

    if segments is None:
        segments = get_wall_segments()

    # broadcast to timepoints*targets*segments
    E = eyes[:, None, None, :]
    P = targets[None, :, None, :]
    A = segments[None, None, :, 0, :]
    e = segments[None, None, :, 1, :] - segments[None, None, :, 0, :]

    # solve E + t*(P - E) = A + u*e for t (along the sight line) and u (along the segment)
    d = P - E
    w = A - E
    denom = cross_2d(d, e)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cross_2d(w, e) / denom
        u = cross_2d(w, d) / denom

    # count a crossing strictly between eye and target. This ignores the target's own segment
    # (t = 1) and lines running parallel to a segment (denom = 0).
    crosses = (np.abs(denom) > EPS) & (t > EPS) & (t < 1 - EPS) & (u >= 0) & (u <= 1)

    return crosses.any(axis=2)


# %%
def find_eye_alcove(eyes):
    ''' Return, per frame, the row of the alcove the eye is inside, or -1 if outside every alcove.
        Takes eye positions (timepoints*2). '''

    W = get_physical_geometry()['wall_face']

    # the eye is inside an alcove when it is beyond that wall's face line (at most one wall at a time)
    p = np.stack([arena_to_wall_frame(eyes, wall)[0] for wall in range(NUM_WALLS)])
    return np.where(p.max(axis=0) > W, p.argmax(axis=0), -1)


# %%
def clip_faces_to_wedge(edge_1, edge_2, face_start, face_direction, lower, upper):
    ''' Narrow each face's visible range [lower, upper] (fractions along the face) to the part inside a wedge.
        Takes the wedge edge directions (timepoints*2 each, less than 180 degrees apart), face start points
        relative to the eye (timepoints*faces*2), face directions (faces*2) and the current lower and
        upper fractions (timepoints*faces). The wedge is the region swept from edge_1 to edge_2. '''

    # orient the wedge so that inside means a non-negative cross product with both edges
    sign = np.sign(cross_2d(edge_1, edge_2))[:, None]

    # a point at fraction t along a face is inside when a + t*b >= 0 for both edges (linear in t)
    for a, b in ((sign * cross_2d(edge_1[:, None, :], face_start), sign * cross_2d(edge_1[:, None, :], face_direction[None])),
                 (sign * cross_2d(face_start, edge_2[:, None, :]), sign * cross_2d(face_direction[None], edge_2[:, None, :]))):
        with np.errstate(divide='ignore', invalid='ignore'):
            limit = -a / b
        # raise the lower bound where b > 0, drop the upper bound where b < 0
        lower = np.where(b > 0, np.maximum(lower, limit), lower)
        upper = np.where(b < 0, np.minimum(upper, limit), upper)
        # close the range where the face runs parallel to the edge on the outside
        upper = np.where((b == 0) & (a < 0), -np.inf, upper)

    return lower, upper


# %%
def get_wall_visible_occluded(trajectory, smoothed_head_angle_vectors, current_fov=110.36):
    ''' Return a num_walls*timepoints boolean array, True where any part of a wall's coloured face is
        inside the field of view with nothing blocking the line of sight.
        Takes a 2*timepoints trajectory, 2*timepoints smoothed head angle vectors and the total FoV in degrees.
        The trajectory is cropped to the smoothed head angle length, as in the angular method. '''

    if not 0 < current_fov < 180:
        raise ValueError(f'current_fov must be between 0 and 180 degrees, got {current_fov}')

    timepoints = smoothed_head_angle_vectors.shape[1]
    eyes = trajectory[:, :timepoints].T.astype(float)                       # timepoints*2
    headings = smoothed_head_angle_vectors.T.astype(float)                  # timepoints*2

    # find frames with a missing or unusable eye position or heading. These show no walls (as in the
    # angular method), so give them placeholder values for the calculation and hide their walls at the end.
    valid = (np.isfinite(eyes).all(axis=1) & np.isfinite(headings).all(axis=1)
             & (np.hypot(headings[:, 0], headings[:, 1]) > EPS))
    eyes = np.where(valid[:, None], eyes, 0.0)
    headings = np.where(valid[:, None], headings, [0.0, 1.0])

    # get the coloured faces as start points relative to each eye, and directions along each face
    faces = get_coloured_face_segments().reshape(-1, 2, 2)                  # (num_walls*2)*2*2
    face_start = faces[None, :, 0, :] - eyes[:, None, :]                    # timepoints*faces*2
    face_direction = faces[:, 1, :] - faces[:, 0, :]                        # faces*2
    lower = np.zeros(face_start.shape[:2])
    upper = np.ones(face_start.shape[:2])

    # clip faces to the field of view (edges at half the FoV either side of the heading)
    half = math.radians(current_fov / 2)
    cos_h, sin_h = math.cos(half), math.sin(half)
    # rotate the heading counterclockwise and clockwise (x = sin, z = cos, angles clockwise from +z)
    fov_ccw = np.column_stack([headings[:, 0] * cos_h - headings[:, 1] * sin_h, headings[:, 0] * sin_h + headings[:, 1] * cos_h])
    fov_cw = np.column_stack([headings[:, 0] * cos_h + headings[:, 1] * sin_h, -headings[:, 0] * sin_h + headings[:, 1] * cos_h])
    lower, upper = clip_faces_to_wedge(fov_ccw, fov_cw, face_start, face_direction, lower, upper)

    # clip faces to the alcove mouth for frames where the eye is inside an alcove
    alcove = find_eye_alcove(eyes)
    in_alcove = alcove >= 0
    if in_alcove.any():
        g = get_physical_geometry()
        # get both mouth corners of each frame's alcove, relative to the eye
        corner_ccw = np.zeros((in_alcove.sum(), 2))
        corner_cw = np.zeros((in_alcove.sum(), 2))
        for wall in np.unique(alcove[in_alcove]):
            rows = alcove[in_alcove] == wall
            corner_ccw[rows] = wall_frame_to_arena(g['wall_face'], -g['alcove_half_width'], wall)
            corner_cw[rows] = wall_frame_to_arena(g['wall_face'], g['alcove_half_width'], wall)
        e = eyes[in_alcove]
        lower[in_alcove], upper[in_alcove] = clip_faces_to_wedge(corner_ccw - e, corner_cw - e, face_start[in_alcove],
                                                                 face_direction, lower[in_alcove], upper[in_alcove])

    # a wall is visible if either coloured face keeps a range of positive length, on a valid frame
    face_visible = ((upper - lower) > EPS) & valid[:, None]
    wall_visible = face_visible.reshape(timepoints, NUM_WALLS, 2).any(axis=2)

    return wall_visible.T
