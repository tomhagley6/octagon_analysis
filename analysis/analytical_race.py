''' Model a peripheral agent racing a central agent from arena geometry alone.

The central agent waits at the arena centre and always goes High (260929 central models take High
on 0.998 of won trials, even when Low is much nearer). The peripheral agent starts from a logged
slice-onset position and follows a fixed wall-choice rule. Each agent's time to a wall is the
shortest path to that wall's trigger divided by the speed. The faster agent wins and scores 1.0 at
High or 0.4 at Low. Every High and Low layout is listed with its exact probability, so results are
expectations, not samples.

Method:
  1. Build the space an agent's centre can reach (octagon with eight alcoves, from globals).
  2. Take the shortest path from a start point to each wall's trigger. A path may bend only at
     the alcove mouth corners, so these 16 corners are the only turning points.
  3. List the 40 layouts (High on any of 8 walls, Low 1, 2 or 4 walls away with weights
     0.5/0.25/0.25 and a random side), as drawn in OctagonArenaSettings.cs.
  4. For every peripheral position and layout, score the race under a wall-choice rule.
  5. Optionally, give the central agent its ray sensor (analysis/agent_raycast.py: 7 rays at 0,
     +-18.3, +-36.7 and +-55 deg from the heading, first hit only; any part of a lit wall or its
     trigger box tells the agent that wall's identity). Its heading at slice onset is uniform at
     random. It does not move until High is centred in front of it (along the middle ray). If any part of High
     is already in view, it turns the short way to centre it. If not, it turns in a random
     direction (half clockwise, half anticlockwise) until High is centred. It then travels as
     before. The central time is a distribution over headings and turn directions, and every race
     is scored against all of it.

Assumptions:
  - Both agents move at the same fixed speed from slice onset. No start-up delay, no timing noise.
  - Without the field of view, both agents know both lit walls at slice onset. With it, only the
    central agent's view is modelled; the peripheral agent still knows both walls.
  - Neither agent blocks the other.
  - An agent whose body overlaps a lit trigger at slice onset wins that wall at once.
'''

import contextlib
import io
import math

import numpy as np
import pandas as pd

import globals
import analysis.agent_raycast as agent_raycast
import data_extraction.get_indices as get_indices
import parse_data.prepare_sim_data as prepare_sim_data
import trajectory_analysis.visibility_occlusion as vo

HIGH_SCORE, LOW_SCORE = 1.0, 0.4
NUM_WALLS = globals.NUM_WALLS
SEPARATIONS = {1: 0.5, 2: 0.25, 4: 0.25}     # wall separation weights, GeneralGlobals.cs
SPEED = 10.0                                   # units per second (moveSpeed, OctagonAgent.cs)
POLICIES = ('high', 'closest', 'threshold', 'low_first', 'nearest')
ALIGN_TOL = 10.0                               # degrees within which an agent counts as facing High
N_HEADINGS = 3600                              # heading grid for the central agent (0.05 deg steps)
HEAD_STEP = 0.5                                # heading grid for the peripheral agent's ray search (deg)
MAX_STEP = 8.0                                 # furthest the peripheral agent steps out to sense a wall (units)
TRACE_DT = 0.005                               # time step for tracing the peripheral agent's route (s)

W = globals.OCTAGON_WALLFACE_RADIUS
H = globals.ALCOVE_HALF_WIDTH
BACK = globals.OCTAGON_ALCOVE_RADIUS
# the agent centre fires the trigger when the body first overlaps the box front
TRIGGER_REACH = BACK + globals.PLAYER_RADIUS - agent_raycast.TRIGGER_DEPTH - globals.PLAYER_RADIUS
EPS = 0.05                                     # tolerance for points on or just past the boundary
TIE = 1e-9                                     # times closer than this count as equal


# geometry

def _pt(p, q, wall):
    ''' Return a wall-frame point as arena x, z. Takes p, q and a wall row index. '''

    return np.array(vo.wall_frame_to_arena(p, q, wall))


def reach_polygon():
    ''' Return the reachable region as a clockwise polygon (40 x 2). Takes no inputs. '''

    L = W * math.tan(math.radians(22.5))
    # take each wall's vertex, mouth corner, two back corners and other mouth corner
    return np.array([_pt(p, q, w) for w in range(NUM_WALLS)
                     for p, q in ((W, -L), (W, -H), (BACK, -H), (BACK, H), (W, H))])


def mouth_corners():
    ''' Return the 16 alcove mouth corners (the only turning points of a shortest path).
        Takes no inputs. Row 2w is wall w's counterclockwise corner, row 2w + 1 its clockwise one. '''

    return np.array([_pt(W, q, w) for w in range(NUM_WALLS) for q in (-H, H)])


def _boundary_dist(poly, pts):
    ''' Return each point's distance to the polygon boundary. Takes the polygon and points (n x 2). '''

    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((pts[:, None] - a) * ab).sum(-1) / (ab ** 2).sum(-1), 0, 1)
    return np.linalg.norm(pts[:, None] - (a + t[..., None] * ab), axis=-1).min(1)


def inside(poly, pts, eps=EPS):
    ''' Return whether points lie inside the polygon, counting points within eps of the boundary.
        Takes the polygon and one point or an array of points. '''

    pts = np.atleast_2d(pts)
    x, z = pts[:, 0:1], pts[:, 1:2]
    a, b = poly, np.roll(poly, -1, axis=0)
    # count edge crossings of a ray running in +x
    cross = ((a[:, 1] > z) != (b[:, 1] > z)) & \
            (x < a[:, 0] + (z - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1] + 1e-300))
    res = (cross.sum(1) % 2 == 1) | (_boundary_dist(poly, pts) < eps)
    return res if len(res) > 1 else bool(res[0])


def _visible(poly, a, b, n_samples=24):
    ''' Return whether the straight segment a-b stays in the reachable region.
        Takes the polygon and two points. '''

    # reject a segment that properly crosses any edge
    c, d = poly, np.roll(poly, -1, axis=0)
    cr = lambda o, p, q: (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])
    d1, d2 = cr(a, b, c), cr(a, b, d)
    d3, d4 = cr(c, d, a), cr(c, d, b)
    tol = 1e-9
    if np.any((d1 * d2 < -tol) & (d3 * d4 < -tol)):
        return False
    # check points along the segment, which catches a pass through a convex vertex
    s = np.linspace(0, 1, n_samples + 2)[1:-1, None]
    return bool(np.all(inside(poly, a + s * (b - a), eps=1e-6)))


def _goal_point(pt, wall):
    ''' Return the nearest point on a wall's trigger line to pt. Takes a point and a wall row index. '''

    _, q = vo.arena_to_wall_frame(pt, wall)
    return _pt(TRIGGER_REACH, float(np.clip(q, -H, H)), wall)


def _in_trigger(pt, wall):
    ''' Return whether an agent centred at pt overlaps a wall's trigger. Takes a point and a wall row index. '''

    p, q = vo.arena_to_wall_frame(pt, wall)
    return bool(p >= TRIGGER_REACH - 1e-9 and abs(q) <= H + EPS)


def _snap_inside(poly, pt, margin=0.02):
    ''' Return pt moved just inside the polygon if it lies outside it. Takes the polygon and a
        point. Logged positions can sit up to ~0.08 units past the polygon, where the real
        boundary is rounded by the body. A point on the boundary is kept. '''

    if _boundary_dist(poly, pt[None])[0] < 1e-9 or inside(poly, pt, eps=0):
        return pt
    # try points around pt at growing distance, and keep the nearest one clear of the boundary
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    for r in np.arange(0.01, 0.51, 0.01):
        cand = pt + r * np.c_[np.sin(ang), np.cos(ang)]
        ok = inside(poly, cand, eps=0) & (_boundary_dist(poly, cand) >= margin)
        if ok.any():
            return cand[np.argmax(ok)]
    raise ValueError(f"point {pt} is more than 0.5 units outside the reachable space")


def _corner_to_goal():
    ''' Return the shortest path from every mouth corner to every wall's trigger (16 x 8).
        Takes no inputs. '''

    poly, cs = reach_polygon(), mouth_corners()
    n = len(cs)
    # get corner-to-corner distances over visible links, then all shortest paths
    cc = np.full((n, n), np.inf)
    for i in range(n):
        cc[i, i] = 0.0
        for j in range(i + 1, n):
            if _visible(poly, cs[i], cs[j]):
                cc[i, j] = cc[j, i] = np.linalg.norm(cs[i] - cs[j])
    for k in range(n):
        cc = np.minimum(cc, cc[:, k:k + 1] + cc[k:k + 1, :])
    # get each corner's direct link to each trigger, then route through the corner graph
    direct = np.full((n, NUM_WALLS), np.inf)
    for i in range(n):
        for w in range(NUM_WALLS):
            g = _goal_point(cs[i], w)
            if _visible(poly, cs[i], g):
                direct[i, w] = np.linalg.norm(cs[i] - g)
    return (cc[:, :, None] + direct[None, :, :]).min(1)


_CORNER_GOAL = None


def path_lengths(points):
    ''' Return the shortest path length from each point to each wall's trigger (n x 8, columns are
        wall row indices). Zero if the agent already overlaps that trigger. Takes points (n x 2). '''

    global _CORNER_GOAL
    if _CORNER_GOAL is None:
        _CORNER_GOAL = _corner_to_goal()
    poly, cs = reach_polygon(), mouth_corners()
    points = np.atleast_2d(np.asarray(points, float))
    out = np.full((len(points), NUM_WALLS), np.inf)
    for i, start in enumerate(points):
        pt = _snap_inside(poly, start)
        # find the corners in sight, as the first turning point of any bent path
        seen = np.array([_visible(poly, pt, c) for c in cs])
        via = np.where(seen, np.linalg.norm(cs - pt, axis=1), np.inf)[:, None] + _CORNER_GOAL
        for w in range(NUM_WALLS):
            if _in_trigger(start, w):
                out[i, w] = 0.0
                continue
            g = _goal_point(pt, w)
            direct = np.linalg.norm(pt - g) if _visible(poly, pt, g) else np.inf
            out[i, w] = min(direct, via[:, w].min())
    return out


def path_points(pt, wall):
    ''' Return the corner points of the shortest path from pt to a wall's trigger (k x 2).
        Takes a start point and a wall row index. '''

    poly, cs = reach_polygon(), mouth_corners()
    if _in_trigger(np.asarray(pt, float), wall):
        return np.array([pt], float)
    pts = [_snap_inside(poly, np.asarray(pt, float))]
    while True:
        cur = pts[-1]
        if _in_trigger(cur, wall):
            return np.array(pts)
        g = _goal_point(cur, wall)
        if _visible(poly, cur, g):
            return np.array(pts + [g])
        # step to the visible corner that starts the shortest remaining path
        d = path_lengths(cur[None])[0, wall]
        cost = [np.linalg.norm(c - cur) + _CORNER_GOAL[i, wall]
                if np.linalg.norm(c - cur) > 1e-9 and _visible(poly, cur, c) else np.inf
                for i, c in enumerate(cs)]
        i = int(np.argmin(cost))
        assert abs(cost[i] - d) < 1e-6
        pts.append(cs[i])


def alcove_back_positions():
    ''' Return the centre of each alcove back (8 x 2). Takes no inputs. '''

    return np.array([_pt(BACK, 0.0, w) for w in range(NUM_WALLS)])


def nearest_wall(points):
    ''' Return the wall row index whose direction is closest to each point. Takes points (n x 2). '''

    ang = np.degrees(np.arctan2(points[:, 0], points[:, 1])) % 360
    return np.round(ang / 45).astype(int) % NUM_WALLS


def signed_offset(k):
    ''' Return a wall offset folded to -3..4 (positive = clockwise). Takes an offset in walls. '''

    return (np.asarray(k) + 3) % NUM_WALLS - 3


# field of view

def _bearing(v):
    ''' Return the arena bearing of vectors in degrees (0 = +z, clockwise positive, as Unity yaw).
        Takes vectors with x, z in the last axis. '''

    return np.degrees(np.arctan2(v[..., 0], v[..., 1]))


def _wrap(a):
    ''' Return angles wrapped to -180..180 degrees. Takes angles in degrees. '''

    return (np.asarray(a) + 180) % 360 - 180


def high_sensed(pos, heading, wall, blocker=None):
    ''' Return whether any of the agent's rays hits a wall (any part of it, or its trigger box).
        Takes positions (n x 2), headings (degrees, Unity yaw; one value or n), a wall row index and
        optionally the positions (n x 2) of a body that blocks the rays.

        Uses agent_raycast.cast_rays, the replica of the agents' SelectivePassThroughRaycast sensor:
        7 rays, first hit only, a ray starting inside a trigger box does not detect that box. '''

    pos = np.atleast_2d(np.asarray(pos, float))
    heading = np.broadcast_to(np.asarray(heading, float), (len(pos),))
    ids, _, _ = agent_raycast.cast_rays(pos, np.radians(heading), blocker=blocker)
    return (ids == wall + 1).any(axis=1)


_CENTRAL_HALF_ANGLE = None


def central_visible_half_angle(step=0.01):
    ''' Return the largest angle between the heading and High's centre at which an agent at the arena
        centre still senses High with its rays (degrees). Takes the grid step for the search.

        Found by casting the rays over a grid of headings, then refining the edge by bisection.
        Sensing is checked to be one unbroken range of angles from 0. '''

    global _CENTRAL_HALF_ANGLE
    if _CENTRAL_HALF_ANGLE is not None:
        return _CENTRAL_HALF_ANGLE
    c = np.zeros((1, 2))
    phi = np.arange(0, 180 + step, step)
    seen = high_sensed(np.repeat(c, len(phi), axis=0), phi, 0)
    last = int(np.flatnonzero(seen).max())
    assert seen[:last + 1].all() and not seen[last + 1:].any(), "sensing is not one range of angles"
    # refine the edge between the last sensed and first unsensed angle
    lo, hi = phi[last], phi[last + 1]
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if high_sensed(c, mid, 0)[0] else (lo, mid)
    _CENTRAL_HALF_ANGLE = float(lo)
    return _CENTRAL_HALF_ANGLE


def zigzag_closing_speed(angle, speed=SPEED):
    ''' Return the best speed along the line to a target under the control scheme (units/s).
        Takes the target's angle from the heading (degrees) and the speed.

        Movement can only point along Up, Down, Left, Right or their diagonals (8 directions 45 deg
        apart, relative to the heading), each at full speed. Mixing the two allowed directions either
        side of the target, in the share that cancels sideways drift, moves the agent straight along
        the line. For a target theta degrees past the nearest allowed direction (0 to 45), the speed
        along the line is speed * sin(45) / (sin(theta) + sin(45 - theta)). '''

    theta = np.radians(np.mod(np.asarray(angle, float), 45.0))
    return speed * math.sin(math.radians(45)) / (np.sin(theta) + np.sin(math.radians(45) - theta))


def central_times(rot_speed=None, speed=SPEED, n=N_HEADINGS, close_speed=None):
    ''' Return the central agent's time to High over a uniform grid of onset headings.
        Takes the rotation speed (degrees per second, None for no field of view), the speed, the
        grid size and the closing speed while turning with High in view (units per second, None
        for no movement while turning).

        With rotation, the agent turns until High is at the centre of its view, at rot_speed. If High
        is sensed by a ray at onset (within central_visible_half_angle), it turns the short way (phi). If not,
        it turns either way with equal chance (phi or 360 - phi). While High is out of view it only
        turns. Once High is sensed (the last central_visible_half_angle degrees of the turn, or the
        whole turn if High starts in view) it also closes on High at close_speed, moving along the
        straight line to High by mixing the allowed directions either side of it (see
        zigzag_closing_speed for the best speed this allows). When High is
        centred it travels the rest of the way at speed. By symmetry only the angle to High (phi, 0 to
        180 deg) matters. Returns 2n equally likely times (each heading once per turn direction). '''

    travel = TRIGGER_REACH / speed
    if rot_speed is None:
        return np.array([travel])
    # take the midpoints of n equal steps of the angle to High
    phi = (np.arange(n) + 0.5) * 180 / n
    V = central_visible_half_angle()
    seen = phi < V
    # turn the short way when High is in view, and either way with equal chance when it is not
    turn = np.concatenate([phi, np.where(seen, phi, 360 - phi)])
    # find the time spent turning with High in view, and the distance closed during it
    t_view = np.minimum(turn, V) / rot_speed
    closed = 0.0 if close_speed is None else close_speed * t_view
    return turn / rot_speed + (TRIGGER_REACH - closed) / speed


# peripheral agent with rays

def heading_grid(step=HEAD_STEP):
    ''' Return the grid of headings (degrees) used for the peripheral ray search. Takes the step. '''

    return np.arange(0.0, 360.0, step)


def heading_sensing(pos, step=HEAD_STEP):
    ''' Return which walls the agent's rays hit at every heading on the grid (headings x 8).
        Takes one position (x, z) and the grid step. Rays pass through the opponent. '''

    heads = heading_grid(step)
    eyes = np.repeat(np.asarray(pos, float)[None], len(heads), axis=0)
    ids, _, _ = agent_raycast.cast_rays(eyes, np.radians(heads))
    return (ids[..., None] == np.arange(1, NUM_WALLS + 1)).any(axis=1)


def first_leg_bearing(pos):
    ''' Return the bearing (degrees) of the first straight leg of the shortest path to each wall's
        trigger (8, NaN where the agent already overlaps that trigger). Takes one position. '''

    global _CORNER_GOAL
    if _CORNER_GOAL is None:
        _CORNER_GOAL = _corner_to_goal()
    poly, cs = reach_polygon(), mouth_corners()
    start = np.asarray(pos, float)
    pt = _snap_inside(poly, start)
    out = np.full(NUM_WALLS, np.nan)
    seen = np.array([np.linalg.norm(c - pt) > 1e-9 and _visible(poly, pt, c) for c in cs])
    for w in range(NUM_WALLS):
        if _in_trigger(start, w):
            continue
        g = _goal_point(pt, w)
        direct = np.linalg.norm(g - pt) if _visible(poly, pt, g) else np.inf
        via = np.where(seen, np.linalg.norm(cs - pt, axis=1) + _CORNER_GOAL[:, w], np.inf)
        # head straight for the trigger if that is shortest, otherwise for the first corner
        nxt = g if direct <= via.min() else cs[int(np.argmin(via))]
        out[w] = float(_bearing(nxt - pt))
    return out


def step_to_sense(pos, wall, max_step=MAX_STEP, tol=0.01, sensing=None):
    ''' Return the shortest step toward the arena centre after which some heading lets the rays
        hit a wall, and the position after it. Takes one position, a wall row index, the longest
        step allowed and the tolerance (units). The step is 0 if a heading already senses the wall.

        From an alcove back no heading reaches the adjacent walls, so the agent must step out
        first. Found by bisection, assuming that once a step is long enough, longer steps are too. '''

    pos = np.asarray(pos, float)
    if (heading_sensing(pos) if sensing is None else sensing)[:, wall].any():
        return 0.0, pos
    u = -pos / np.linalg.norm(pos)
    if not heading_sensing(pos + max_step * u)[:, wall].any():
        raise ValueError(f"wall {wall} not sensed within {max_step} units of {pos}")
    lo, hi = 0.0, max_step
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if heading_sensing(pos + mid * u)[:, wall].any() else (mid, hi)
    return hi, pos + hi * u


def sweep_angle(sens, heading, direction):
    ''' Return how far the agent turns in the search direction before a ray first hits a wall
        (degrees, 0 if hit at the current heading, inf if never). Takes the wall's column of
        heading_sensing, the current heading (degrees) and the direction (1 clockwise, -1 anticlockwise). '''

    n = len(sens)
    k0 = int(round((heading % 360) / HEAD_STEP)) % n
    order = (k0 + direction * np.arange(n)) % n
    hit = sens[order]
    return float(np.argmax(hit) * HEAD_STEP) if hit.any() else np.inf


def target_time(pos, heading, wall, direction, rot_speed, close_speed, speed=SPEED, sensing=None, step_out=None,
                legs=None, lengths=None):
    ''' Return the peripheral agent's time from slice onset to a wall's trigger.
        Takes one position, its heading (degrees), the wall row index, the search turn direction
        (1 = clockwise, -1 = anticlockwise), the rotation speed (deg/s), the closing speed while
        turning with the wall sensed (units/s), the speed, and optionally the precomputed
        heading_sensing at pos, step_to_sense result, and first_leg_bearing and path_lengths at pos.

        Steps:
          1. An agent overlapping the wall's trigger is there at time 0.
          2. If no heading senses the wall from pos, step toward the centre until one does (at speed).
          3. If the rays already sense the wall, turn the short way to centre the first leg of the
             path. Otherwise turn in the search direction until a ray hits the wall, then turn on (or
             back, the short way) to centre the first leg.
          4. While the wall is sensed and the turn continues, close along the path at close_speed.
             Then follow the rest of the shortest path at speed, with no extra time at bends. '''

    pos = np.asarray(pos, float)
    if _in_trigger(pos, wall):
        return 0.0
    heads = heading_grid()
    n = len(heads)
    sens = (heading_sensing(pos) if sensing is None else sensing)[:, wall]
    step, p = (step_to_sense(pos, wall, sensing=sensing) if step_out is None else step_out)
    if step > 0:
        sens = heading_sensing(p)[:, wall]
    leg = (first_leg_bearing(p) if (step > 0 or legs is None) else legs)[wall]
    L = (path_lengths(p[None])[0] if (step > 0 or lengths is None) else lengths)[wall]
    theta_s = sweep_angle(sens, heading, direction)
    if theta_s == 0:
        # turn the short way to the first leg, with the wall in view throughout
        blind, view = 0.0, abs(float(_wrap(leg - heading)))
    else:
        # sweep in the search direction until a ray hits the wall
        theta_a = float((direction * (leg - heading)) % 360)
        blind = theta_s
        back = theta_s - theta_a
        view = theta_a - theta_s if back <= 0 else min(back, 360 - back)
    t_view = view / rot_speed
    closed = close_speed * t_view
    # arrive during the turn if the closing covers the whole path
    move = L / close_speed if closed >= L else t_view + (L - closed) / speed
    return step / speed + blind / rot_speed + move


def _along(pts, arc):
    ''' Return the point at a distance along a polyline, and the index of its segment.
        Takes the polyline (k x 2) and the distance (units, clipped to the polyline). '''

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    arc = float(np.clip(arc, 0.0, cum[-1]))
    k = int(min(np.searchsorted(cum, arc, 'right') - 1, len(seg) - 1))
    f = 0.0 if seg[k] == 0 else (arc - cum[k]) / seg[k]
    return pts[k] + f * (pts[k + 1] - pts[k]), k


def high_trace(pos, heading, wall, direction, rot_speed, close_speed, speed=SPEED, sensing=None, step_out=None):
    ''' Return the peripheral agent's route to a wall under "High, always", as a function of time.
        Takes one position, its heading (degrees), the wall row index, the search direction, the
        rotation and closing speeds, the speed, and optionally the precomputed heading_sensing at pos
        and step_to_sense result.

        Returns a dict: arrive (time at the trigger, equal to target_time), sensed (time the wall is
        first sensed), state (a function of time giving position and heading), path (the shortest
        path's corner points after any step) and bend_times (when each later stretch of the path
        starts).

        Phases: step out (heading fixed), search turn in place, turn to centre the first leg while
        closing along the path, then follow the path. Turning and moving are uncoupled, so at each
        bend the agent keeps moving at full speed and turns the short way to face the new stretch
        at the rotation speed. '''

    pos = np.asarray(pos, float)
    h0 = float(heading)
    if _in_trigger(pos, wall):
        return dict(arrive=0.0, sensed=0.0, state=lambda t: (pos, h0), path=pos[None], bend_times=[])
    S = heading_sensing(pos) if sensing is None else sensing
    step, p1 = step_to_sense(pos, wall, sensing=S) if step_out is None else step_out
    u = -pos / np.linalg.norm(pos) if step > 0 else np.zeros(2)
    sens1 = (S if step == 0 else heading_sensing(p1))[:, wall]
    pts = path_points(p1, wall)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(seg.sum())
    bear = np.array([float(_bearing(pts[k + 1] - pts[k])) for k in range(len(seg))])
    leg = bear[0]

    # time the step, the search turn and the turn to the first leg (as in target_time)
    t_a = step / speed
    theta_s = sweep_angle(sens1, h0, direction)
    if theta_s == 0:
        rel = float(_wrap(leg - h0))
        h_b, t_b, view, dir_c = h0, t_a, abs(rel), (1.0 if rel >= 0 else -1.0)
    else:
        theta_a = float((direction * (leg - h0)) % 360)
        h_b, t_b = h0 + direction * theta_s, t_a + theta_s / rot_speed
        back = theta_s - theta_a
        if back <= 0:
            view, dir_c = theta_a - theta_s, direction
        elif back <= 180:
            view, dir_c = back, -direction
        else:
            view, dir_c = 360 - back, direction
    closed = close_speed * view / rot_speed
    if closed >= L:
        t_c, closed, arrive = t_b + L / close_speed, L, t_b + L / close_speed
    else:
        t_c = t_b + view / rot_speed
        arrive = t_c + (L - closed) / speed

    # find when each later stretch starts, and the heading pieces while following the path
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    _, k_c = _along(pts, closed)
    bend_times = [t_c + (cum[k] - closed) / speed for k in range(k_c + 1, len(seg))]
    pieces = [(t_c, leg, bear[k_c])]
    for k, tk in zip(range(k_c + 1, len(seg)), bend_times):
        t0, hs, tgt = pieces[-1]
        pieces.append((tk, _turn_toward(hs, tgt, rot_speed * (tk - t0)), bear[k]))

    def state(t):
        if t <= t_a and step > 0:
            return pos + u * speed * t, h0
        if t <= t_b:
            return p1, h0 + direction * rot_speed * (t - t_a)
        if t <= t_c:
            q, _ = _along(pts, close_speed * (t - t_b))
            return q, h_b + dir_c * rot_speed * (t - t_b)
        q, _ = _along(pts, closed + speed * (t - t_c))
        t0, hs, tgt = [pc for pc in pieces if pc[0] <= t][-1]
        return q, _turn_toward(hs, tgt, rot_speed * (t - t0))

    return dict(arrive=arrive, sensed=t_b, state=state, path=pts, bend_times=bend_times)


def _turn_toward(h, target, max_turn):
    ''' Return a heading turned the short way toward a target by at most max_turn degrees.
        Takes the heading, target bearing and largest turn (degrees). '''

    rel = float(_wrap(target - h))
    return h + np.sign(rel) * min(abs(rel), max_turn)


def time_from_sensed(pos, heading, wall, rot_speed, close_speed, speed=SPEED):
    ''' Return the time to a wall's trigger for an agent that already knows where the wall is.
        Takes one position, its heading (degrees), the wall row index, the rotation and closing
        speeds and the speed.

        The agent turns the short way to centre the first leg of the shortest path, closing along
        the path meanwhile, then follows the path at speed (as target_time when the wall is in view). '''

    pos = np.asarray(pos, float)
    if _in_trigger(pos, wall):
        return 0.0
    leg = first_leg_bearing(pos)[wall]
    L = path_lengths(pos[None])[0, wall]
    t_view = abs(float(_wrap(leg - heading))) / rot_speed
    closed = close_speed * t_view
    return L / close_speed if closed >= L else t_view + (L - closed) / speed


def trace_first_sensed(trace, dt=TRACE_DT):
    ''' Return the first time each wall is hit by a ray along a traced route (8, inf if never
        before arrival). Takes a high_trace result and the time step. '''

    times = np.append(np.arange(0.0, trace['arrive'], dt), trace['arrive'])
    states = [trace['state'](t) for t in times]
    eyes = np.array([q for q, _ in states])
    yaw = np.radians([hh for _, hh in states])
    ids, _, _ = agent_raycast.cast_rays(eyes, yaw)
    sensed = (ids[..., None] == np.arange(1, NUM_WALLS + 1)).any(axis=1)
    first = np.full(NUM_WALLS, np.inf)
    hit = sensed.any(axis=0)
    first[hit] = times[sensed.argmax(axis=0)[hit]]
    return first


def build_races_rays(periph_pos, periph_heading, rot_speed, close_speed, groups=None, speed=SPEED, verbose=False):
    ''' Return one row per peripheral position x layout x search direction, with the peripheral
        agent's ray-based times to High and Low (target_time) and whether its rays sense each at onset.
        Takes peripheral start points (n x 2), their onset headings (degrees), the rotation and
        closing speeds, an optional group label per point, the speed and a progress flag.

        Each search direction carries half the row's probability. prob sums to 1 within each group. '''

    periph_pos = np.atleast_2d(np.asarray(periph_pos, float))
    heading = np.asarray(periph_heading, float)
    groups = np.zeros(len(periph_pos), int) if groups is None else np.asarray(groups)
    T = np.zeros((len(periph_pos), NUM_WALLS, 2))
    STEP = np.zeros((len(periph_pos), NUM_WALLS))
    THETA = np.zeros((len(periph_pos), NUM_WALLS, 2))
    S0 = np.zeros((len(periph_pos), NUM_WALLS), bool)
    lay = layout_table()
    NEAR_LOW = np.zeros((len(periph_pos), len(lay), 2), bool)
    T_NEAR = np.zeros((len(periph_pos), len(lay), 2))
    for i, (p, h) in enumerate(zip(periph_pos, heading)):
        S = heading_sensing(p)
        S0[i] = S[int(round((h % 360) / HEAD_STEP)) % len(S)]
        legs, lengths = first_leg_bearing(p), path_lengths(p[None])[0]
        for w in range(NUM_WALLS):
            st = step_to_sense(p, w, sensing=S) if not _in_trigger(p, w) else (0.0, p)
            # keep the step and the search angle at which the wall is first sensed
            STEP[i, w] = st[0]
            sens_w = (S if st[0] == 0 else heading_sensing(st[1]))[:, w]
            for j, d in enumerate((1, -1)):
                THETA[i, w, j] = sweep_angle(sens_w, h, d)
                T[i, w, j] = target_time(p, h, w, d, rot_speed, close_speed, speed, sensing=S, step_out=st,
                                         legs=legs, lengths=lengths)
        # trace each High and direction under "High, always", then decide "nearest" for every Low
        for w in range(NUM_WALLS):
            st = step_to_sense(p, w, sensing=S) if not _in_trigger(p, w) else (0.0, p)
            for j, d in enumerate((1, -1)):
                tr = high_trace(p, h, w, d, rot_speed, close_speed, speed, sensing=S, step_out=st)
                first = trace_first_sensed(tr)
                for li in np.flatnonzero(lay['high'].to_numpy() == w):
                    NEAR_LOW[i, li, j], T_NEAR[i, li, j] = _decide_nearest(tr, first, w, lay['low'].iat[li], T[i, w, j],
                                                                           rot_speed, close_speed, speed)
        if verbose and i % 200 == 0:
            print(i, end=' ')
    pi, li, di = np.meshgrid(np.arange(len(periph_pos)), np.arange(len(lay)), np.arange(2), indexing='ij')
    pi, li, di = pi.ravel(), li.ravel(), di.ravel()
    high, low = lay['high'].to_numpy()[li], lay['low'].to_numpy()[li]
    home = nearest_wall(periph_pos)[pi]
    n_in_group = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()[pi]
    races = pd.DataFrame(dict(
        pos=pi, group=groups[pi], direction=np.where(di == 0, 1, -1), high=high, low=low,
        sep=lay['sep'].to_numpy()[li], high_off=signed_offset(high - home), low_off=signed_offset(low - home),
        t_high=T[pi, high, di], t_low=T[pi, low, di],
        high_sensed0=S0[pi, high], low_sensed0=S0[pi, low],
        high_step=STEP[pi, high], low_step=STEP[pi, low],
        high_theta=THETA[pi, high, di], low_theta=THETA[pi, low, di],
        nearest_low=NEAR_LOW[pi, li, di], t_nearest=T_NEAR[pi, li, di],
        t_central=TRIGGER_REACH / speed, prob=lay['prob'].to_numpy()[li] / n_in_group / 2))
    races['low_first'] = low_sensed_first(races)
    return races


def _decide_nearest(trace, first, high, low, t_high, rot_speed, close_speed, speed=SPEED):
    ''' Return whether "nearest, once both are sensed" switches to Low, and the time to the wall
        reached. Takes the High trace, the first time each wall is sensed along it, the High and Low
        row indices, the "High, always" time, the rotation and closing speeds and the speed.

        Both walls are known at the later of High's and Low's first sensing. If that comes before the
        agent reaches High, it compares the remaining shortest paths from where it is then and goes to
        the nearer (equal paths go High). It decides only once. '''

    t_dec = max(trace['sensed'], first[low])
    if not np.isfinite(t_dec) or t_dec >= trace['arrive']:
        return False, t_high
    q, hd = trace['state'](t_dec)
    d = path_lengths(q[None])[0]
    if d[low] < d[high] - 1e-9:
        return True, t_dec + time_from_sensed(q, hd, low, rot_speed, close_speed, speed)
    return False, t_high


def low_sensed_first(races):
    ''' Return whether the peripheral agent's rays hit Low strictly before High on each race row.
        Takes rows from build_races_rays.

        Both sensed at onset counts as a tie (High). Otherwise the agent searches without stepping if
        any heading from where it stands senses either lit wall, and the first wall its search finds
        wins. Only if neither can be sensed from there does it step out, and the wall that needs the
        shorter step is the one it then senses. Ties go to High. '''

    hs, ls = races['high_step'].to_numpy(), races['low_step'].to_numpy()
    ht, lt = races['high_theta'].to_numpy(), races['low_theta'].to_numpy()
    no_step = (hs == 0) | (ls == 0)
    # without a step, a wall that would need one cannot be found by the search
    ht_ns, lt_ns = np.where(hs == 0, ht, np.inf), np.where(ls == 0, lt, np.inf)
    first = np.where(no_step, lt_ns < ht_ns, ls < hs)
    return first & ~(races['high_sensed0'].to_numpy() & races['low_sensed0'].to_numpy())


# layouts and races

def layout_table():
    ''' Return the 40 High/Low layouts with their probabilities. Takes no inputs.
        high and low are wall row indices. '''

    rows = []
    for high in range(NUM_WALLS):
        for sep, w_sep in SEPARATIONS.items():
            # a 4-wall separation lands on the same wall from either side
            signs = (1,) if sep == 4 else (1, -1)
            for s in signs:
                rows.append(dict(high=high, low=(high + s * sep) % NUM_WALLS, sep=sep,
                                 prob=w_sep / NUM_WALLS / len(signs)))
    return pd.DataFrame(rows)


def build_races(periph_pos, groups=None, speed=SPEED):
    ''' Return one row per peripheral position x layout, with the peripheral agent's travel times to
        High and Low, the central agent's travel time to High, and the row's probability.
        Takes peripheral start points (n x 2), an optional group label per point (e.g. model) and
        the speed.

        prob sums to 1 within each group, so every group weighs its own positions equally. '''

    periph_pos = np.atleast_2d(np.asarray(periph_pos, float))
    groups = np.zeros(len(periph_pos), int) if groups is None else np.asarray(groups)
    d = path_lengths(periph_pos)
    t_central = path_lengths(np.zeros((1, 2)))[0] / speed
    lay = layout_table()
    pi, li = np.meshgrid(np.arange(len(periph_pos)), np.arange(len(lay)), indexing='ij')
    pi, li = pi.ravel(), li.ravel()
    high, low = lay['high'].to_numpy()[li], lay['low'].to_numpy()[li]
    home = nearest_wall(periph_pos)[pi]
    # weigh each position equally within its group
    n_in_group = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()[pi]
    return pd.DataFrame(dict(
        pos=pi, group=groups[pi], high=high, low=low, sep=lay['sep'].to_numpy()[li],
        high_off=signed_offset(high - home), low_off=signed_offset(low - home),
        t_high=d[pi, high] / speed, t_low=d[pi, low] / speed, t_central=t_central[high],
        prob=lay['prob'].to_numpy()[li] / n_in_group))


def choose_low(races, policy='high', delta=None):
    ''' Return whether the peripheral agent heads for Low on each race row.
        Takes the race rows, the policy and, for 'threshold', the travel-time gap delta (seconds).

        'high' always heads High. 'closest' heads for the wall with the shorter travel time (equal
        times go High). 'threshold' heads High unless High takes more than delta seconds longer
        than Low, and then heads for the closer wall (Low). delta = 0 matches 'closest'.
        'low_first' heads for Low if the rays sense Low strictly before High (build_races_rays rows
        only), and High otherwise. 'nearest' heads High until both walls are sensed, then for the
        nearer (build_races_rays rows only). '''

    gap = (races['t_high'] - races['t_low']).to_numpy()
    if policy == 'high':
        return np.zeros(len(races), bool)
    if policy == 'closest':
        return gap > TIE
    if policy == 'low_first':
        return races['low_first'].to_numpy(bool)
    if policy == 'nearest':
        return races['nearest_low'].to_numpy(bool)
    if policy == 'threshold':
        if delta is None:
            raise ValueError("threshold needs delta")
        return gap > delta + TIE
    raise ValueError(f"policy must be one of {POLICIES}")


def evaluate(races, policy='high', delta=None, central_t=None):
    ''' Return each race row's outcome under a peripheral wall-choice policy.
        Takes the race rows, the policy, its delta and the central agent's times (an array from
        central_times, each equally likely). Without central_t, the central agent takes its travel
        time to High.

        A peripheral agent already overlapping a lit trigger wins that wall at time 0, whatever
        the policy. Otherwise the faster agent wins, and equal times split the win. '''

    go_low = choose_low(races, policy, delta)
    tH, tL, tC = races['t_high'].to_numpy(), races['t_low'].to_numpy(), races['t_central'].to_numpy()
    # take the wall the peripheral agent reaches, with an instant win overriding the choice
    t_p = np.where(go_low, tL, tH)
    if policy == 'nearest':
        # take the time to the wall reached after the switch decision
        t_p = races['t_nearest'].to_numpy()
    val = np.where(go_low, LOW_SCORE, HIGH_SCORE)
    t_p = np.where(tH == 0, 0.0, np.where(tL == 0, 0.0, t_p))
    val = np.where(tH == 0, HIGH_SCORE, np.where(tL == 0, LOW_SCORE, val))
    if central_t is None:
        win = np.where(np.abs(t_p - tC) <= TIE, 0.5, (t_p < tC).astype(float))
        race = np.minimum(t_p, tC)
    else:
        # score each row against every central time, using the sorted times and their running sums
        ct = np.sort(np.asarray(central_t, float))
        n = len(ct)
        csum = np.concatenate([[0.0], np.cumsum(ct)])
        lo, hi = np.searchsorted(ct, t_p - TIE, 'left'), np.searchsorted(ct, t_p + TIE, 'right')
        win = ((n - hi) + 0.5 * (hi - lo)) / n
        race = (csum[lo] + t_p * (n - lo)) / n
    res = races.copy()
    res['go_low'] = go_low
    res['periph_win'] = win
    res['periph_score'] = win * val
    res['central_score'] = (1 - win) * HIGH_SCORE
    res['race_s'] = race
    return res


def summarise(races, overhead, policy='high', delta=None, central_t=None):
    ''' Return each group's expected outcome under a policy, one row per group.
        Takes the race rows, the fixed time per trial outside the race (seconds), the policy, its
        delta and the central agent's times (see evaluate).

        Score per second is total expected score over total expected trial time, where a trial
        lasts the race plus the overhead. '''

    r = evaluate(races, policy, delta, central_t)
    rows = []
    for g, s in r.groupby('group'):
        w = s['prob'].to_numpy()
        tot = lambda c: float((w * s[c]).sum())
        trial_s = overhead + tot('race_s')
        rows.append(dict(group=g, policy=policy, delta=np.nan if delta is None else delta, periph_win=tot('periph_win'),
                         periph_score_trial=tot('periph_score'), central_score_trial=tot('central_score'),
                         p_go_low=tot('go_low'), race_s=tot('race_s'), trial_s=trial_s,
                         periph_score_s=tot('periph_score') / trial_s,
                         central_score_s=tot('central_score') / trial_s))
    return pd.DataFrame(rows)


def central_reference(overhead, speed=SPEED, central_t=None):
    ''' Return score per trial and per second for central vs central (both at the centre, both
        going High). Takes the overhead, the speed and the central agent's times (see evaluate).

        Without central_t every race ties. With it, the two agents draw their times independently,
        each wins half the races, and a race lasts the faster of the two. '''

    if central_t is None:
        race_s = TRIGGER_REACH / speed
    else:
        # take E[min] of two independent draws: the i-th smallest of n is the minimum in 2(n-i)-1 of n^2 pairs
        ct = np.sort(np.asarray(central_t, float))
        n = len(ct)
        race_s = float((ct * (2 * (n - np.arange(n)) - 1)).sum() / n ** 2)
    return dict(score_trial=0.5, score_s=0.5 / (overhead + race_s), race_s=race_s)


# inputs from the logs

def collect_trial_timing(sim_data_folder, models, verbose=True):
    ''' Return one row per trial with the slice-onset-to-trigger time, the whole trial length, the
        winner's depth into the alcove at the trigger, and the agents' moving speeds. Takes a batch's
        simulation folder and the model list. Drops the first logged trial. '''

    rows = []
    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            _, trial_list = prepare_sim_data.prepare_single_model_data(model, sim_data_folder=sim_data_folder)
        for trial in trial_list[1:]:
            ev = trial['eventDescription']
            onset, trig = trial[ev == globals.SLICE_ONSET], trial[ev == globals.SELECTED_TRIGGER_ACTIVATION]
            if len(onset) == 0 or len(trig) == 0:
                continue
            t = trial['timeApplication']
            # get the winner's depth into the chosen wall's alcove at the trigger
            k = globals.PLAYER_LOC_DICT
            winner, chosen = get_indices.get_trigger_activator(trial), int(get_indices.get_chosen_walls([trial])[0])
            end = np.array([trig[k[winner]['xloc']].iloc[0], trig[k[winner]['yloc']].iloc[0]], dtype=float)
            trigger_p, _ = vo.arena_to_wall_frame(end, chosen - 1)
            # get both agents' frame-to-frame speeds between slice onset and the trigger
            seg = trial.loc[onset.index[0]:trig.index[0]]
            dt = seg['timeApplication'].diff().dt.total_seconds().to_numpy()[1:]
            speeds = []
            for pid in (0, 1):
                step = np.hypot(np.diff(seg[k[pid]['xloc']].to_numpy(float)), np.diff(seg[k[pid]['yloc']].to_numpy(float)))
                speeds.append(step[dt > 0] / dt[dt > 0])
            speeds = np.concatenate(speeds)
            moving = speeds[speeds > 0.5]
            rows.append(dict(model=model, trial_num=float(trial[globals.TRIAL_NUM].dropna().iloc[0]),
                             rt_s=(trig['timeApplication'].iloc[0] - onset['timeApplication'].iloc[0]).total_seconds(),
                             trial_s=(t.iloc[-1] - t.iloc[0]).total_seconds(), trigger_p=float(trigger_p),
                             speed_median=float(np.median(moving)) if len(moving) else np.nan,
                             speed_p99=float(np.percentile(moving, 99)) if len(moving) else np.nan))
        if verbose:
            print(f"model {model} done", end='  ')
    out = pd.DataFrame(rows)
    out['overhead_s'] = out['trial_s'] - out['rt_s']
    return out


def _dist_to_trigger(xy, wall):
    ''' Return the straight-line distance from each point to a wall's trigger line.
        Takes points (n x 2) and a wall row index. '''

    _, q = vo.arena_to_wall_frame(xy, wall)
    x, z = vo.wall_frame_to_arena(TRIGGER_REACH, np.clip(q, -H, H), wall)
    return np.linalg.norm(xy - np.c_[x, z], axis=1)


def collect_central_alignment(sim_data_folder, models, tol=ALIGN_TOL, verbose=True):
    ''' Return one row per agent-trial describing how the agent turns to face High after slice
        onset. Takes a batch's simulation folder, central model list and the alignment tolerance
        (degrees). Drops the first logged trial. Sensing uses the agents' rays (high_sensed), with
        player 1's rays blocked by player 0's body, as in agent_raycast.trial_sensing.

        Columns:
          phi            High's bearing from the onset position minus the heading (degrees, -180..180)
          visible        High sensed by a ray at onset
          t_align        first time after onset facing High within tol (s, NaN if not before the trigger)
          The rest cover slice onset to t_align (NaN if never aligned):
          turned         total heading change, counting back-and-forth (degrees)
          net_turn       signed net heading change (degrees, positive = clockwise)
          short_way      net turn has the same sign as phi
          reversals      changes of turning direction
          t_rotating     time in frames where the heading changed by more than 0.5 deg (s)
          t_still        time in the other frames (s)
          path_moved     distance travelled (units)
          closer         reduction in distance to High's trigger line (units)
          t_inview       time in frames that start with High sensed by a ray (s)
          closing_inview reduction in distance to High's trigger line over those frames (units) '''

    rows = []
    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            _, trial_list = prepare_sim_data.prepare_single_model_data(model, sim_data_folder=sim_data_folder)
        for trial in trial_list[1:]:
            ev = trial['eventDescription']
            onset, trig = trial[ev == globals.SLICE_ONSET], trial[ev == globals.SELECTED_TRIGGER_ACTIVATION]
            if len(onset) == 0 or len(trig) == 0:
                continue
            high, _ = get_indices.get_walls(trial)
            winner = get_indices.get_trigger_activator(trial)
            seg = trial.loc[onset.index[0]:trig.index[0]]
            secs = (seg['timeApplication'] - seg['timeApplication'].iloc[0]).dt.total_seconds().to_numpy()
            target = _pt(TRIGGER_REACH, 0.0, high - 1)
            for pid in (0, 1):
                # get the agent's position and heading over the slice phase
                xy = np.c_[seg[globals.PLAYER_LOC_DICT[pid]['xloc']].to_numpy(float),
                           seg[globals.PLAYER_LOC_DICT[pid]['yloc']].to_numpy(float)]
                heading = seg[globals.PLAYER_ROT_DICT[pid]['yrot']].to_numpy(float)
                # find the frames where a ray hits High (player 1's rays are blocked by player 0's body)
                opp = np.c_[seg[globals.PLAYER_LOC_DICT[1 - pid]['xloc']].to_numpy(float),
                            seg[globals.PLAYER_LOC_DICT[1 - pid]['yloc']].to_numpy(float)]
                sensed = high_sensed(xy, heading, high - 1, blocker=opp if pid == 1 else None)
                # find the first frame facing High within the tolerance
                off = _wrap(_bearing(target - xy) - heading)
                aligned = np.nonzero(np.abs(off) <= tol)[0]
                row = dict(model=model, trial_num=float(trial[globals.TRIAL_NUM].dropna().iloc[0]),
                           player=pid, won=(pid == winner), r0=float(np.hypot(*xy[0])), phi=float(off[0]),
                           visible=bool(sensed[0]), rt_s=float(secs[-1]),
                           t_align=np.nan)
                if len(aligned):
                    i = aligned[0]
                    # take the per-frame heading change, time step and distance to High up to alignment
                    dth, dt = _wrap(np.diff(heading[:i + 1])), np.diff(secs[:i + 1])
                    dist = _dist_to_trigger(xy[:i + 1], high - 1)
                    rotating = np.abs(dth) > 0.5
                    signs = np.sign(dth[rotating])
                    in_view = sensed[:i]
                    row.update(t_align=float(secs[i]), turned=float(np.abs(dth).sum()), net_turn=float(dth.sum()),
                               short_way=bool(np.sign(dth.sum()) == np.sign(off[0])),
                               reversals=int((np.diff(signs) != 0).sum()),
                               t_rotating=float(dt[rotating].sum()), t_still=float(dt[~rotating].sum()),
                               path_moved=float(np.linalg.norm(np.diff(xy[:i + 1], axis=0), axis=1).sum()),
                               closer=float(dist[0] - dist[-1]),
                               t_inview=float(dt[in_view].sum()), closing_inview=float(-np.diff(dist)[in_view].sum()))
                rows.append(row)
        if verbose:
            print(f"model {model} done", end='  ')
    return pd.DataFrame(rows)


def rotation_speed(alignment):
    ''' Return the central agents' turning rate (degrees per second), as the median over agent-trials
        of the total heading change from slice onset to facing High, over the time taken. Takes the
        alignment table, using every agent-trial that faced High after onset and before the trigger.

        The rate does not depend on which way the agent turned. The median is used because the
        per-agent-trial rate is consistent across models and angles, while a few trials with long
        pauses pull a pooled total down. '''

    a = alignment[alignment['t_align'] > 0]
    return float((a['turned'] / a['t_align']).median())


def closing_speed(alignment):
    ''' Return the central agents' speed of closing on High while turning with High in view (units
        per second), as the median over agent-trials of the distance to High closed in frames with
        High in view before alignment, over the time in those frames. Takes the alignment table,
        using agent-trials with some in-view time before alignment. '''

    a = alignment[alignment['t_inview'] > 0]
    return float((a['closing_inview'] / a['t_inview']).median())
