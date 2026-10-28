''' Model a peripheral agent racing a central agent from arena geometry and the agents' ray sensor.

The central agent waits at the arena centre with a random heading and always goes High (260929
central models take High on 0.998 of won trials). It senses walls with the agents' 7 rays
(analysis/agent_raycast.py), turns to find and face High (331 deg/s), closing on High at 7.57 units/s
once High is in view, then travels straight at 10 units/s. Its search direction can use inference.

The peripheral agent starts from a logged slice-onset position and heading of a peripheral model. It
senses with the same rays, from its current position and heading. If the wall it wants is not sensed
it moves out of its alcove while turning until a ray hits it, then turns onto the shortest path
(closing at 7.57 units/s) and follows it at 10 units/s. Its wall choice follows a rule; the main rule
('delta2') infers where unseen walls probably are and goes for Low when High is more than delta
seconds further (expected distance for unseen walls), rechecking whenever a ray hits a new wall.

The faster agent wins and scores 1.0 at High or 0.4 at Low. Every High and Low layout is listed
with its exact probability, so results are expectations, not samples. Score per second is total
expected score over total expected trial time (race plus a fixed overhead).

Assumptions (full list in READMEs/analytical_modelling_notes.md):
  - Both agents move at 10 units/s, turn at 331 deg/s and close at 7.57 units/s; no start-up delay,
    no timing noise.
  - An agent's behaviour depends only on what its rays have sensed, its own position and heading,
    and the arena layout.
  - Neither agent blocks the other or uses the other's position.
  - An agent whose body overlaps a lit trigger at slice onset wins that wall at once.
'''

import concurrent.futures
import contextlib
import io
import math
import multiprocessing
import os

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
POLICIES = ('high', 'closest', 'threshold', 'low_first', 'nearest', 'delta', 'delta2')
ALIGN_TOL = 10.0                               # degrees within which an agent counts as facing High
N_HEADINGS = 3600                              # heading grid for the central agent (0.1 deg steps)
HEAD_STEP = 0.5                                # heading grid for the peripheral agent's ray search (deg)
MAX_STEP = 8.0                                 # furthest the peripheral agent steps out to sense a wall (units)
TRACE_DT = 0.005                               # time step for tracing the peripheral agent's route (s)
N_JOBS = max(1, min(12, (os.cpu_count() or 1) - 4))   # worker processes for build_races_rays (leaves cores free)
MAX_EV = 9                                     # decision points stored per race for the delta rule
EV_FIELDS = ('t', 'g', 'tl', 'tb', 'gb', 'Tb')     # per decision point: time, gap, Low arrival, switch-back time, gap, High arrival
MAX_PIECES = 12                                # delta2 outcome pieces stored per race (ranges of delta)

W = globals.OCTAGON_WALLFACE_RADIUS
H = globals.ALCOVE_HALF_WIDTH
BACK = globals.OCTAGON_ALCOVE_RADIUS
# the agent centre fires the trigger when the body first overlaps the box front
TRIGGER_REACH = BACK + globals.PLAYER_RADIUS - agent_raycast.TRIGGER_DEPTH - globals.PLAYER_RADIUS
EPS = 0.05                                     # tolerance for points on or just past the boundary
TIE = 1e-9                                     # times closer than this count as equal
GAP_TIE = 1e-6                                 # delta2 gaps within this of delta count as ties (go High)
BACKPEDAL_P = 0.8                              # least probability of the target being the own wall to backpedal (arbitrary, user 2026-10-26)
BACKPEDAL = True                               # backpedal into the own trigger (on from cache v10; caches up to v8 have it off)
OWN_TRIGGER_KNOWN = True                       # standing on the own trigger at onset counts as sensing that wall unlit (on with BACKPEDAL)


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


def in_alcove(pos):
    ''' Return whether a position lies inside its nearest alcove (past the wall face line).
        Takes one position. The agent knows its own position, so this uses no hidden information. '''

    pos = np.asarray(pos, float)
    p, q = vo.arena_to_wall_frame(pos, int(nearest_wall(pos[None])[0]))
    return bool(p > W + 1e-9 and abs(q) <= H + EPS)


def search_moves(pos, move_out):
    ''' Return whether the search motion moves toward the centre: always in the earlier merged step,
        and with move_out only when the agent starts inside an alcove. Takes one position and move_out. '''

    return in_alcove(pos) if move_out else True


def step_search_trace(pos, heading, direction, rot_speed, speed=SPEED, max_step=MAX_STEP, dt=TRACE_DT, move=True):
    ''' Return the motion of an agent that steps toward the centre while turning in its search
        direction, and the first time its rays hit each wall. Takes one position, its heading
        (degrees), the search direction, the rotation speed, the speed, the longest step and the
        time step.

        Turning and moving are uncoupled (Control scheme), so the agent moves toward the centre at
        speed (stopping after max_step) and turns at rot_speed at the same time, from slice onset.
        With move=False it only turns (in place).
        Returns a dict: first (8, first time each wall is hit, inf if not within one full turn plus
        the step) and state (a function of time giving position and heading). '''

    pos = np.asarray(pos, float)
    h0 = float(heading)
    r = np.linalg.norm(pos)
    u = -pos / r if (r > 0 and move) else np.zeros(2)
    t_stop = max_step / speed

    def state(t):
        return pos + u * speed * min(t, t_stop), h0 + direction * rot_speed * t

    times = np.arange(0.0, 360.0 / rot_speed + t_stop + dt, dt)
    eyes = pos + u[None] * speed * np.minimum(times, t_stop)[:, None]
    ids, _, _ = agent_raycast.cast_rays(eyes, np.radians(h0 + direction * rot_speed * times))
    sensed = (ids[..., None] == np.arange(1, NUM_WALLS + 1)).any(axis=1)
    first = np.full(NUM_WALLS, np.inf)
    hit = sensed.any(axis=0)
    first[hit] = times[sensed.argmax(axis=0)[hit]]
    return dict(first=first, state=state)


def target_time(pos, heading, wall, direction, rot_speed, close_speed, speed=SPEED, sensing=None, step_out=None,
                legs=None, lengths=None, merge_step=True, search=None, move_out=True):
    ''' Return the peripheral agent's time from slice onset to a wall's trigger.
        Takes one position, its heading (degrees), the wall row index, the search turn direction
        (1 = clockwise, -1 = anticlockwise), the rotation speed (deg/s), the closing speed while
        turning with the wall sensed (units/s), the speed, and optionally the precomputed
        heading_sensing at pos, step_to_sense result, and first_leg_bearing and path_lengths at pos,
        whether the step and search happen together (merge_step), a precomputed step_search_trace, and
        whether the agent searches by moving out while turning whenever the wall is not sensed (move_out).

        With move_out (default) the agent's behaviour depends only on whether its rays sense the wall
        at onset: if they do, it turns onto the path while moving along it; if not, it moves toward the
        centre while turning in the search direction until a ray hits the wall (step_search_trace),
        then turns the short way onto the path. Without move_out, the earlier versions apply (below),
        which use whether a heading from pos could sense the wall, information the agent does not have.

        Steps:
          1. An agent overlapping the wall's trigger is there at time 0.
          2. If no heading senses the wall from pos, step toward the centre. With merge_step, the agent
             steps and turns in the search direction at the same time until a ray hits the wall, then
             turns the short way to centre the first leg (time_from_sensed). Without it, it steps the
             shortest distance after which a heading would sense the wall, then searches.
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
    if move_out and sweep_angle(sens, heading, direction) > 0:
        # not sensed at onset: move out while turning until a ray hits the wall, then go to it
        ms = (step_search_trace(pos, heading, direction, rot_speed, speed, move=search_moves(pos, True))
              if search is None else search)
        t_s = ms['first'][wall]
        q, h = ms['state'](t_s)
        return t_s + time_from_sensed(q, h, wall, rot_speed, close_speed, speed)
    if move_out:
        step_out = (0.0, pos)
    step, p = (step_to_sense(pos, wall, sensing=sensing) if step_out is None else step_out)
    if step > 0 and merge_step:
        # step out and search together until a ray hits the wall, then go to it
        ms = step_search_trace(pos, heading, direction, rot_speed, speed) if search is None else search
        t_s = ms['first'][wall]
        q, h = ms['state'](t_s)
        return t_s + time_from_sensed(q, h, wall, rot_speed, close_speed, speed)
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


def high_trace(pos, heading, wall, direction, rot_speed, close_speed, speed=SPEED, sensing=None, step_out=None,
               merge_step=True, search=None, move_out=True):
    ''' Return the peripheral agent's route to a wall under "High, always", as a function of time.
        Takes one position, its heading (degrees), the wall row index, the search direction, the
        rotation and closing speeds, the speed, and optionally the precomputed heading_sensing at pos
        and step_to_sense result.

        Returns a dict: arrive (time at the trigger, equal to target_time), sensed (time the wall is
        first sensed), state (a function of time giving position and heading), path (the shortest
        path's corner points after any step) and bend_times (when each later stretch of the path
        starts).

        Phases: step out (heading fixed), search turn in place, turn to centre the first leg while
        closing along the path, then follow the path. With merge_step, the step out and the search
        happen together (step_search_trace), and the turn to the first leg goes the short way. With
        move_out (default) the agent moves out while turning whenever the wall is not sensed at onset,
        as in target_time. Turning and moving are uncoupled, so at each
        bend the agent keeps moving at full speed and turns the short way to face the new stretch
        at the rotation speed. '''

    pos = np.asarray(pos, float)
    h0 = float(heading)
    if _in_trigger(pos, wall):
        return dict(arrive=0.0, sensed=0.0, state=lambda t: (pos, h0), path=pos[None], bend_times=[])
    S = heading_sensing(pos) if sensing is None else sensing
    if move_out:
        # decide only on whether the rays sense the wall at onset
        step, p1 = (1.0, pos) if sweep_angle(S[:, wall], h0, direction) > 0 else (0.0, pos)
        merged = step > 0
    else:
        step, p1 = step_to_sense(pos, wall, sensing=S) if step_out is None else step_out
        merged = step > 0 and merge_step
    if merged:
        # step out and search together, then start the path from where the wall is first sensed
        ms = (step_search_trace(pos, h0, direction, rot_speed, speed, move=search_moves(pos, move_out))
              if search is None else search)
        t_s = ms['first'][wall]
        p1, h_s = ms['state'](t_s)
    u = -pos / np.linalg.norm(pos) if step > 0 else np.zeros(2)
    sens1 = (S if step == 0 else heading_sensing(p1))[:, wall] if not merged else None
    pts = path_points(p1, wall)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(seg.sum())
    bear = np.array([float(_bearing(pts[k + 1] - pts[k])) for k in range(len(seg))])
    leg = bear[0]

    # time the step, the search turn and the turn to the first leg (as in target_time)
    t_a = step / speed
    theta_s = sweep_angle(sens1, h0, direction) if not merged else None
    if merged:
        rel = float(_wrap(leg - h_s))
        h_b, t_b, view, dir_c = h_s, t_s, abs(rel), (1.0 if rel >= 0 else -1.0)
    elif theta_s == 0:
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
        if merged and t <= t_b:
            return ms['state'](t)
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


def known_wall_times(positions, headings, rot_speed, close_speed, speed=SPEED):
    ''' Return the shortest path to every wall and the time to it for an agent that knows where all
        walls are at onset (turn the short way to the first leg, closing meanwhile, then the path).
        Takes positions (n x 2), headings (degrees), the rotation and closing speeds and the speed.
        Returns path lengths (n x 8) and times (n x 8). '''

    positions = np.atleast_2d(np.asarray(positions, float))
    D = path_lengths(positions)
    TK = np.zeros_like(D)
    for i, (p, h) in enumerate(zip(positions, headings)):
        legs = first_leg_bearing(p)
        for w in range(NUM_WALLS):
            if D[i, w] == 0 or np.isnan(legs[w]):
                continue
            t_view = abs(float(_wrap(legs[w] - h))) / rot_speed
            closed = close_speed * t_view
            TK[i, w] = D[i, w] / close_speed if closed >= D[i, w] else t_view + (D[i, w] - closed) / speed
    return D, TK


def blind_trace(pos, heading, wall, speed=SPEED):
    ''' Return the route to a wall followed at full speed without turning, as a high_trace result. Used
        for backpedalling into the agent's own trigger (Control scheme: 10 units/s in every direction).
        Takes one position, its heading (degrees), the wall row index and the speed. '''

    pos = np.asarray(pos, float)
    h0 = float(heading)
    if _in_trigger(pos, wall):
        return dict(arrive=0.0, sensed=0.0, state=lambda t: (pos, h0), path=pos[None], bend_times=[])
    pts = path_points(pos, wall)
    L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def state(t):
        return _along(pts, speed * t)[0], h0

    return dict(arrive=L / speed, sensed=0.0, state=state, path=pts, bend_times=[])


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


def known_trace(pos, heading, wall, rot_speed, close_speed, speed=SPEED):
    ''' Return the route to a wall whose position the agent knows, as a high_trace result: turn the
        short way onto the first leg while closing along the path, then follow the path, turning at
        bends. Takes one position, its heading (degrees), the wall row index, the rotation and closing
        speeds and the speed. '''

    known = np.ones((len(heading_grid()), NUM_WALLS), bool)
    return high_trace(pos, heading, wall, 1, rot_speed, close_speed, speed, sensing=known, move_out=True)


def _delta_events(trace, first, high, low, rot_speed, close_speed, speed=SPEED):
    ''' Return the delta rule's decision points along the "High, always" route (MAX_EV x 6, fields as
        EV_FIELDS). Takes the High trace, the first time each wall is sensed along it, the High and Low
        row indices, the rotation and closing speeds and the speed.

        A decision point is onset or the first sensing of any wall, before the agent reaches High. If Low
        is known there, the gap is (path to High - path to Low) / speed from the agent's position, using
        the actual path to High if High is known and its expected path (high_posterior_given, from the
        walls sensed so far) if not. For a positive gap the route to Low from there is traced: its arrival
        time, and, if High was unknown, when High is first sensed on the way, the gap then and the time to
        High after switching back. Points with no Low knowledge have gap -inf. '''

    out = np.full((MAX_EV, len(EV_FIELDS)), np.nan)
    out[:, 1] = -np.inf
    t_known = trace['sensed']
    times = np.unique(np.concatenate([first[np.isfinite(first) & (first < trace['arrive'])],
                                      [t_known] if t_known < trace['arrive'] else []]))
    for k, t in enumerate(times[:MAX_EV]):
        out[k, 0] = t
        seen = first <= t + 1e-12
        high_known = t >= t_known - 1e-9
        if not seen[low]:
            continue
        q, hd = trace['state'](t)
        D = path_lengths(q[None])[0]
        if high_known:
            d_high = D[high]
        else:
            mask = seen.copy()
            mask[high] = False
            post = high_posterior_given(mask, low)
            d_high = float((post * D).sum())
        g = (d_high - D[low]) / speed
        out[k, 1] = g
        if g <= 0:
            continue
        # trace the switch to Low, and whether High comes into view on the way
        lt = known_trace(q, hd, low, rot_speed, close_speed, speed)
        out[k, 2] = t + lt['arrive']
        if not high_known:
            fh = trace_first_sensed(lt)[high]
            if fh < lt['arrive']:
                q2, hd2 = lt['state'](fh)
                D2 = path_lengths(q2[None])[0]
                out[k, 3] = t + fh
                out[k, 4] = (D2[high] - D2[low]) / speed
                out[k, 5] = t + fh + time_from_sensed(q2, hd2, high, rot_speed, close_speed, speed)
    return out


def delta_outcome(races, delta):
    ''' Return whether the delta rule ends at Low, and the time to the wall reached, for each race row.
        Takes rows from build_races_rays and the threshold delta (s).

        At the first decision point whose gap exceeds delta, the agent switches to Low. If High then comes
        into view on the way and its gap there is at most delta, it switches back to High. With no such
        point it reaches High on the "High, always" route. '''

    G = np.column_stack([races[f'ev_g_{k}'].to_numpy(float) for k in range(MAX_EV)])
    hit = G > delta
    any_hit = hit.any(axis=1)
    k = hit.argmax(axis=1)
    pick = lambda f: np.column_stack([races[f'ev_{f}_{j}'].to_numpy(float) for j in range(MAX_EV)])[np.arange(len(races)), k]
    tl, tb, gb, Tb = pick('tl'), pick('tb'), pick('gb'), pick('Tb')
    back = any_hit & np.isfinite(tb) & (gb <= delta)
    go_low = any_hit & ~back
    t = np.where(go_low, tl, np.where(back, Tb, races['t_high'].to_numpy(float)))
    return go_low, t


def delta_direct(pos, heading, high, low, direction, delta, rot_speed, close_speed, speed=SPEED):
    ''' Return whether the delta rule ends at Low and the time to the wall reached, simulated directly for
        one agent and layout (for checking delta_outcome). Takes the position, heading, High and Low row
        indices, search direction, delta, the rotation and closing speeds and the speed. '''

    pos = np.asarray(pos, float)
    tr = high_trace(pos, heading, high, direction, rot_speed, close_speed, speed)
    first = trace_first_sensed(tr)
    times = sorted({float(x) for x in first if np.isfinite(x) and x < tr['arrive']}
                   | ({tr['sensed']} if tr['sensed'] < tr['arrive'] else set()))
    for t in times:
        seen = first <= t + 1e-12
        if not seen[low]:
            continue
        q, hd = tr['state'](t)
        D = path_lengths(q[None])[0]
        known = t >= tr['sensed'] - 1e-9
        if known:
            d_high = D[high]
        else:
            mask = seen.copy()
            mask[high] = False
            d_high = float((high_posterior_given(mask, low) * D).sum())
        if (d_high - D[low]) / speed > delta:
            lt = known_trace(q, hd, low, rot_speed, close_speed, speed)
            if not known:
                fh = trace_first_sensed(lt)[high]
                if fh < lt['arrive']:
                    q2, hd2 = lt['state'](fh)
                    D2 = path_lengths(q2[None])[0]
                    if (D2[high] - D2[low]) / speed <= delta:
                        return False, t + fh + time_from_sensed(q2, hd2, high, rot_speed, close_speed, speed)
            return True, t + lt['arrive']
    return False, tr['arrive']


def joint_posterior(sensed, high_seen, low_seen):
    ''' Return the probability of each (High, Low) layout (8 x 8, High rows) given what the rays have
        sensed. Takes which walls have been hit (8 bools), and the High and Low row indices if they have
        been hit, else None.

        A wall hit that is neither High nor Low is unlit, so neither wall can be there. A sensed wall is
        fixed where it was seen; an unsensed one can only be at a wall not yet hit. '''

    sensed = np.asarray(sensed, bool)
    h_ok = np.zeros(NUM_WALLS, bool)
    l_ok = np.zeros(NUM_WALLS, bool)
    if high_seen is None:
        h_ok = ~sensed
    else:
        h_ok[high_seen] = True
    if low_seen is None:
        l_ok = ~sensed
    else:
        l_ok[low_seen] = True
    J = pair_prior() * h_ok[:, None] * l_ok[None, :]
    return J / J.sum()


def _own_wall_p(q, sensed, target, high, low):
    ''' Return the agent's own wall (the alcove it stands in, or None outside an alcove or on its own
        trigger) and the probability, from what has been sensed, that its target is that wall. Takes the
        position, the sensed walls, the target row index and the true High and Low row indices (used only
        to read whether each is sensed). '''

    q = np.asarray(q, float)
    if not in_alcove(q):
        return None, 0.0
    own = int(nearest_wall(q[None])[0])
    if _in_trigger(q, own):
        return None, 0.0
    if sensed[target]:
        return own, float(target == own)
    hs, ls = bool(sensed[high]), bool(sensed[low])
    J = joint_posterior(sensed, high if hs else None, low if ls else None)
    p_t = J.sum(axis=1) if target == high else J.sum(axis=0)
    return own, float(p_t[own])


def onset_known(pos, sensed):
    ''' Return the walls known at onset: those the rays hit, plus the agent's own wall if it stands on that
        wall's trigger (the trial did not end at once, so that wall is unlit). Takes the position and the
        walls hit at onset (8 bools). '''

    known = np.asarray(sensed, bool).copy()
    if not OWN_TRIGGER_KNOWN:
        return known
    pos = np.asarray(pos, float)
    own = int(nearest_wall(pos[None])[0])
    if _in_trigger(pos, own):
        known[own] = True
    return known


def _motion_kind(q, sensed, target, high, low):
    ''' Return the kind of motion toward a target: 'backpedal' (in its own alcove, off its trigger, and
        the target is its own wall with probability >= BACKPEDAL_P), 'route' (target sensed) or 'search'.
        Takes the position, the sensed walls, the target row index and the true High and Low row indices. '''

    own, p_own = _own_wall_p(q, sensed, target, high, low) if BACKPEDAL else (None, 0.0)
    if own is not None and p_own >= BACKPEDAL_P - 1e-12:
        return 'backpedal'
    return 'route' if sensed[target] else 'search'


def _gap_both(q, sensed, high, low, speed=SPEED):
    ''' Return (distance to High - distance to Low) / speed for an agent at q, using actual paths for
        sensed walls and expected paths (joint_posterior) for unsensed ones. Takes the position, the
        sensed walls, the true High and Low row indices (used only to read whether each is sensed) and
        the speed. '''

    D = path_lengths(np.asarray(q, float)[None])[0]
    hs, ls = bool(sensed[high]), bool(sensed[low])
    J = joint_posterior(sensed, high if hs else None, low if ls else None)
    d_high = D[high] if hs else float((J.sum(axis=1) * D).sum())
    d_low = D[low] if ls else float((J.sum(axis=0) * D).sum())
    return (d_high - d_low) / speed


_TRACE_CACHE = {}
TRACE_CACHE_MAX = 20000                        # traces kept before the cache is cleared


def _cached_trace(kind, q, hd, arg, rot_speed, close_speed, speed, fn):
    ''' Return a trace (dict with state, first and arrive) from the cache, computing it with fn if new.
        Takes the trace kind, position, heading, its wall or direction, the speeds and the function.
        The same motion recurs across layouts that share a start, so caching saves recasting rays. '''

    key = (kind, round(float(q[0]), 9), round(float(q[1]), 9), round(float(hd), 9), arg, rot_speed, close_speed, speed)
    if key not in _TRACE_CACHE:
        if len(_TRACE_CACHE) >= TRACE_CACHE_MAX:
            _TRACE_CACHE.clear()
        _TRACE_CACHE[key] = fn()
    return _TRACE_CACHE[key]


def _motion_to(q, hd, target, sensed, high, low, d_tie, rot_speed, close_speed, speed=SPEED):
    ''' Return the agent's motion toward a target wall from its current position and heading (_motion_kind):
        a backpedal into its own trigger without turning, a route if the target is sensed, otherwise a search
        (move out of an alcove while turning, or turn in place) in the direction that should sense it sooner
        given where it could be. Takes the position, heading, target row index, sensed walls, true High and
        Low row indices, the tie-break direction, the rotation and closing speeds and the speed. Returns a
        dict with kind, state, first (relative first sensing time of each wall), arrive (inf if the motion
        does not end at a lit wall) and wall (the wall reached on arrival, None if it does not).

        A backpedal ends at the own trigger. If the own wall is lit (as the target, or as the other lit wall)
        the trial ends there. If not, the agent has now experienced that its own wall is unlit, which is recorded as a first
        sensing of it on arrival, so the rule decides again from there. '''

    def route(tr, wall):
        return dict(kind='route', state=tr['state'], first=trace_first_sensed(tr), arrive=tr['arrive'], wall=wall)

    q = np.asarray(q, float)
    kind = _motion_kind(q, sensed, target, high, low)
    if kind == 'backpedal':
        own = int(nearest_wall(q[None])[0])
        lit = own in (high, low)

        def back():
            tr = blind_trace(q, hd, own, speed)
            first = trace_first_sensed(tr)
            if lit:
                return dict(kind='backpedal', state=tr['state'], first=first, arrive=tr['arrive'], wall=own)
            # stop on the unlit trigger, which tells the agent its own wall is not lit
            first[own] = min(first[own], tr['arrive'])
            t_end = tr['arrive']
            return dict(kind='backpedal', state=lambda t: tr['state'](min(t, t_end)), first=first, arrive=np.inf, wall=None)

        return _cached_trace('back', q, hd, (own, lit), rot_speed, close_speed, speed, back)
    if kind == 'route':
        return _cached_trace('known', q, hd, int(target), rot_speed, close_speed, speed,
                             lambda: route(known_trace(q, hd, target, rot_speed, close_speed, speed), int(target)))
    hs, ls = bool(sensed[high]), bool(sensed[low])
    J = joint_posterior(sensed, high if hs else None, low if ls else None)
    p_t = J.sum(axis=1) if target == high else J.sum(axis=0)
    move = in_alcove(q)
    searches = {d: _cached_trace('search', q, hd, (d, move), rot_speed, close_speed, speed,
                                 lambda d=d: step_search_trace(q, hd, d, rot_speed, speed, move=move)) for d in (1, -1)}
    keep = p_t > 0
    e = {d: float((p_t[keep] * searches[d]['first'][keep]).sum()) for d in (1, -1)}
    d = d_tie if abs(e[1] - e[-1]) <= 1e-9 else (1 if e[1] < e[-1] else -1)
    return dict(kind='search', state=searches[d]['state'], first=searches[d]['first'], arrive=np.inf, wall=None)


def _delta2_pieces(p0, h0, high, low, d_tie, sensed0, rot_speed, close_speed, speed=SPEED):
    ''' Return the delta2 outcome over every delta >= 0, as pieces (upper delta bound, time, ends at Low),
        sorted. Takes the start position and heading, the High and Low row indices, the tie-break search
        direction, the walls sensed at onset, the rotation and closing speeds and the speed.

        At onset and at every new wall sensed, the agent goes for Low when the gap (_gap_both) exceeds delta,
        otherwise High; this splits the delta range at the gap. Each part then follows its own motion
        (_motion_to), keeping it while the target and the kind of motion (_motion_kind) are unchanged. '''

    p0 = np.asarray(p0, float)
    if _in_trigger(p0, high):
        return [(np.inf, 0.0, False)]
    if _in_trigger(p0, low):
        return [(np.inf, 0.0, True)]
    pieces = []
    sensed0 = onset_known(p0, sensed0)
    g = _gap_both(p0, sensed0, high, low, speed)
    stack = []
    for lo, hi, tgt in ((0.0, g, low), (max(0.0, g), np.inf, high)):
        if lo < hi:
            m = _motion_to(p0, h0, tgt, sensed0, high, low, d_tie, rot_speed, close_speed, speed)
            stack.append(dict(lo=lo, hi=hi, tgt=tgt, m=m, t0=0.0, rel=0.0, sensed=np.asarray(sensed0, bool).copy()))
    while stack:
        b = stack.pop()
        m, sensed = b['m'], b['sensed']
        f = m['first']
        cand = [f[w] for w in range(NUM_WALLS) if not sensed[w] and np.isfinite(f[w]) and f[w] >= b['rel'] - 1e-12]
        nxt = min(cand) if cand else np.inf
        if np.isfinite(m['arrive']) and m['arrive'] <= nxt:
            pieces.append((b['lo'], b['hi'], b['t0'] + m['arrive'], m['wall'] == low))
            continue
        if not np.isfinite(nxt):
            raise RuntimeError('search never senses its target')
        # take the new walls sensed at this moment, and decide again
        new = np.array([not sensed[w] and np.isfinite(f[w]) and f[w] <= nxt + 1e-12 for w in range(NUM_WALLS)])
        s2 = sensed | new
        q, hd = m['state'](nxt)
        te = b['t0'] + nxt
        g = _gap_both(q, s2, high, low, speed)
        for lo, hi, tgt in ((b['lo'], min(b['hi'], g), low), (max(b['lo'], g), b['hi'], high)):
            if lo >= hi:
                continue
            if tgt == b['tgt'] and _motion_kind(q, s2, tgt, high, low) == m['kind']:
                # keep the same motion
                stack.append(dict(lo=lo, hi=hi, tgt=tgt, m=m, t0=b['t0'], rel=nxt, sensed=s2.copy()))
            else:
                m2 = _motion_to(q, hd, tgt, s2, high, low, d_tie, rot_speed, close_speed, speed)
                stack.append(dict(lo=lo, hi=hi, tgt=tgt, m=m2, t0=te, rel=0.0, sensed=s2.copy()))
    pieces.sort()
    return [(hi, t, lw) for lo, hi, t, lw in pieces]


def delta2_direct(pos, heading, high, low, direction, delta, rot_speed, close_speed, speed=SPEED):
    ''' Return whether delta2 ends at Low and the time to the wall reached, simulated at one delta (for
        checking delta2_outcome). Takes the position, heading, High and Low row indices, tie-break search
        direction, delta, the rotation and closing speeds and the speed. '''

    pos = np.asarray(pos, float)
    if _in_trigger(pos, high):
        return False, 0.0
    if _in_trigger(pos, low):
        return True, 0.0
    sensed = onset_known(pos, onset_sensed(pos[None], [heading])[0])
    tgt = low if _gap_both(pos, sensed, high, low, speed) > delta + GAP_TIE else high
    m = _motion_to(pos, heading, tgt, sensed, high, low, direction, rot_speed, close_speed, speed)
    t0, rel = 0.0, 0.0
    while True:
        f = m['first']
        cand = [f[w] for w in range(NUM_WALLS) if not sensed[w] and np.isfinite(f[w]) and f[w] >= rel - 1e-12]
        nxt = min(cand) if cand else np.inf
        if np.isfinite(m['arrive']) and m['arrive'] <= nxt:
            return m['wall'] == low, t0 + m['arrive']
        sensed = sensed | np.array([not sensed[w] and np.isfinite(f[w]) and f[w] <= nxt + 1e-12 for w in range(NUM_WALLS)])
        q, hd = m['state'](nxt)
        new_tgt = low if _gap_both(q, sensed, high, low, speed) > delta + GAP_TIE else high
        if new_tgt == tgt and _motion_kind(q, sensed, tgt, high, low) == m['kind']:
            rel = nxt
        else:
            tgt = new_tgt
            t0 = t0 + nxt
            m = _motion_to(q, hd, tgt, sensed, high, low, direction, rot_speed, close_speed, speed)
            rel = 0.0


def delta2_route(pos, heading, high, low, direction, delta, rot_speed, close_speed, speed=SPEED):
    ''' Return the delta2 route at one delta, for drawing. Takes the same inputs as delta2_direct.

        Returns a dict: go_low, time (as delta2_direct), segments (list of (state function, start time,
        end time) in the motion's own clock, with the segment's start time on the trial clock and whether
        it is a search), kinds (each segment's motion kind: 'route', 'search' or 'backpedal') and decisions
        (list of (trial time, position, target is Low)). '''

    pos = np.asarray(pos, float)
    if _in_trigger(pos, high) or _in_trigger(pos, low):
        return dict(go_low=_in_trigger(pos, low) and not _in_trigger(pos, high), time=0.0, segments=[],
                    decisions=[(0.0, pos, _in_trigger(pos, low) and not _in_trigger(pos, high))], kinds=[])
    sensed = onset_known(pos, onset_sensed(pos[None], [heading])[0])
    tgt = low if _gap_both(pos, sensed, high, low, speed) > delta + GAP_TIE else high
    m = _motion_to(pos, heading, tgt, sensed, high, low, direction, rot_speed, close_speed, speed)
    t0, rel = 0.0, 0.0
    segs, decisions, kinds = [], [(0.0, pos, tgt == low)], []
    seg_start = 0.0
    while True:
        f = m['first']
        cand = [f[w] for w in range(NUM_WALLS) if not sensed[w] and np.isfinite(f[w]) and f[w] >= rel - 1e-12]
        nxt = min(cand) if cand else np.inf
        if np.isfinite(m['arrive']) and m['arrive'] <= nxt:
            segs.append((m['state'], seg_start, m['arrive'], t0, False))
            kinds.append(m['kind'])
            return dict(go_low=m['wall'] == low, time=t0 + m['arrive'], segments=segs, decisions=decisions, kinds=kinds)
        sensed = sensed | np.array([not sensed[w] and np.isfinite(f[w]) and f[w] <= nxt + 1e-12 for w in range(NUM_WALLS)])
        q, hd = m['state'](nxt)
        new_tgt = low if _gap_both(q, sensed, high, low, speed) > delta + GAP_TIE else high
        if new_tgt == tgt and _motion_kind(q, sensed, tgt, high, low) == m['kind']:
            rel = nxt
        else:
            segs.append((m['state'], seg_start, nxt, t0, m['kind'] == 'search'))
            kinds.append(m['kind'])
            decisions.append((t0 + nxt, q, new_tgt == low))
            tgt, t0 = new_tgt, t0 + nxt
            m = _motion_to(q, hd, tgt, sensed, high, low, direction, rot_speed, close_speed, speed)
            rel, seg_start = 0.0, 0.0


def delta2_outcome(races, delta):
    ''' Return whether delta2 ends at Low, and the time to the wall reached, for each race row.
        Takes rows from build_races_rays and delta (s). Reads the stored pieces (d2_* columns). '''

    HI = np.column_stack([races[f'd2_hi_{k}'].to_numpy(float) for k in range(MAX_PIECES)])
    T = np.column_stack([races[f'd2_t_{k}'].to_numpy(float) for k in range(MAX_PIECES)])
    L = np.column_stack([races[f'd2_low_{k}'].to_numpy(float) for k in range(MAX_PIECES)])
    # treat a gap within GAP_TIE of delta as a tie, which goes High (equal distances differ only by rounding)
    k = (delta + GAP_TIE < np.nan_to_num(HI, nan=-np.inf)).argmax(axis=1)
    idx = np.arange(len(races))
    return L[idx, k] > 0.5, T[idx, k]


def _position_arrays(args):
    ''' Return one peripheral position's times, sensing and "nearest" decisions for every wall,
        layout and search direction. Takes a tuple (position, heading, rotation speed, closing speed,
        speed, merge_step). Runs in a worker process when build_races_rays is parallel.

        Returns T (8 x 2), STEP (8), THETA (8 x 2), TSENSE (8 x 2), S0 (8), NEAR_LOW (40 x 2),
        T_NEAR (40 x 2) and EV (40 x 2 x MAX_EV x 6, the delta rule's decision points), as described in
        build_races_rays. '''

    p, h, rot_speed, close_speed, speed, merge_step, move_out = args
    lay = layout_table()
    T, THETA, TSENSE = np.zeros((NUM_WALLS, 2)), np.zeros((NUM_WALLS, 2)), np.zeros((NUM_WALLS, 2))
    STEP = np.zeros(NUM_WALLS)
    NEAR_LOW, T_NEAR = np.zeros((len(lay), 2), bool), np.zeros((len(lay), 2))
    EV = np.full((len(lay), 2, MAX_EV, len(EV_FIELDS)), np.nan)
    PIECES = np.full((len(lay), 2, MAX_PIECES, 3), np.nan)
    S = heading_sensing(p)
    S0 = S[int(round((h % 360) / HEAD_STEP)) % len(S)]
    legs, lengths = first_leg_bearing(p), path_lengths(p[None])[0]
    # trace the combined step-and-search motion once per direction (it does not depend on the wall)
    searches = ([step_search_trace(p, h, d, rot_speed, speed, move=search_moves(p, move_out)) for d in (1, -1)]
                if (merge_step or move_out) else [None, None])
    for w in range(NUM_WALLS):
        st = step_to_sense(p, w, sensing=S) if not _in_trigger(p, w) else (0.0, p)
        # keep the step and the search angle at which the wall is first sensed
        STEP[w] = st[0]
        sens_w = (S if st[0] == 0 else heading_sensing(st[1]))[:, w]
        for j, d in enumerate((1, -1)):
            THETA[w, j] = sweep_angle(sens_w, h, d)
            # take when the wall is first sensed: at onset (from the start position, not a stepped one),
            # during the move out (move_out), or as before
            if move_out:
                TSENSE[w, j] = 0.0 if S0[w] else searches[j]['first'][w]
            else:
                TSENSE[w, j] = (THETA[w, j] / rot_speed if st[0] == 0
                                else searches[j]['first'][w] if merge_step
                                else st[0] / speed + THETA[w, j] / rot_speed)
            T[w, j] = target_time(p, h, w, d, rot_speed, close_speed, speed, sensing=S, step_out=st,
                                  legs=legs, lengths=lengths, merge_step=merge_step, search=searches[j],
                                  move_out=move_out)
        # trace this wall as High under "High, always", then decide "nearest" for every Low
        for j, d in enumerate((1, -1)):
            tr = high_trace(p, h, w, d, rot_speed, close_speed, speed, sensing=S, step_out=st,
                            merge_step=merge_step, search=searches[j], move_out=move_out)
            first = trace_first_sensed(tr)
            for li in np.flatnonzero(lay['high'].to_numpy() == w):
                NEAR_LOW[li, j], T_NEAR[li, j] = _decide_nearest(tr, first, w, lay['low'].iat[li], T[w, j],
                                                                 rot_speed, close_speed, speed)
                EV[li, j] = _delta_events(tr, first, w, lay['low'].iat[li], rot_speed, close_speed, speed)
                # get the delta2 outcome over every delta (inference about both walls)
                pcs = _delta2_pieces(p, h, w, lay['low'].iat[li], d, S0, rot_speed, close_speed, speed)
                if len(pcs) > MAX_PIECES:
                    raise RuntimeError(f'{len(pcs)} delta2 pieces, more than MAX_PIECES')
                PIECES[li, j, :len(pcs)] = pcs
    return T, STEP, THETA, TSENSE, S0, NEAR_LOW, T_NEAR, EV, PIECES


def build_races_rays(periph_pos, periph_heading, rot_speed, close_speed, groups=None, speed=SPEED, verbose=False,
                     merge_step=True, n_jobs=N_JOBS, move_out=True):
    ''' Return one row per peripheral position x layout x search direction, with the peripheral
        agent's ray-based times to High and Low (target_time) and whether its rays sense each at onset.
        Takes peripheral start points (n x 2), their onset headings (degrees), the rotation and
        closing speeds, an optional group label per point, the speed, a progress flag, whether a step
        out happens with the search, and the number of worker processes.

        Each search direction carries half the row's probability. prob sums to 1 within each group.
        merge_step chooses whether a needed step out happens together with the search (default, as
        turning and moving are separate controls), or before it (the earlier version). move_out
        (default) makes the agent move out while turning whenever its target is not sensed at onset,
        so its behaviour depends only on what it has sensed; with move_out=False the earlier versions
        apply. Positions are
        independent, so with n_jobs > 1 they are split across worker processes (spawned, so the
        workers import this module afresh); n_jobs = 1 runs in this process. Results are the same
        either way. '''

    periph_pos = np.atleast_2d(np.asarray(periph_pos, float))
    heading = np.asarray(periph_heading, float)
    groups = np.zeros(len(periph_pos), int) if groups is None else np.asarray(groups)
    args = [(p, float(h), rot_speed, close_speed, speed, merge_step, move_out) for p, h in zip(periph_pos, heading)]
    out = []
    if n_jobs > 1 and len(args) > 1:
        # split the positions across worker processes, keeping their order
        chunk = max(1, len(args) // (n_jobs * 8))
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs,
                                                    mp_context=multiprocessing.get_context('spawn')) as ex:
            for k, res in enumerate(ex.map(_position_arrays, args, chunksize=chunk)):
                out.append(res)
                if verbose and k % 200 == 0:
                    print(k, end=' ', flush=True)
    else:
        for k, a in enumerate(args):
            out.append(_position_arrays(a))
            if verbose and k % 200 == 0:
                print(k, end=' ', flush=True)
    T, STEP, THETA, TSENSE, S0, NEAR_LOW, T_NEAR, EV, PIECES = (np.array(x) for x in zip(*out))

    lay = layout_table()
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
        high_tsense=TSENSE[pi, high, di], low_tsense=TSENSE[pi, low, di],
        nearest_low=NEAR_LOW[pi, li, di], t_nearest=T_NEAR[pi, li, di],
        t_central=TRIGGER_REACH / speed, prob=lay['prob'].to_numpy()[li] / n_in_group / 2))
    races['low_first'] = low_sensed_first(races, merge_step, move_out)
    # add the delta rule's decision points, one column per point and field
    ev = EV[pi, li, di]
    pc = PIECES[pi, li, di]
    races = pd.concat([races, pd.DataFrame({f'ev_{f}_{k}': ev[:, k, m] for k in range(MAX_EV)
                                            for m, f in enumerate(EV_FIELDS)}, index=races.index),
                       pd.DataFrame({f'd2_{f}_{k}': pc[:, k, m] for k in range(MAX_PIECES)
                                     for m, f in enumerate(('hi', 't', 'low'))}, index=races.index)], axis=1)
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


def low_sensed_first(races, merge_step=True, move_out=True):
    ''' Return whether the peripheral agent's rays hit Low strictly before High on each race row.
        Takes rows from build_races_rays, whether the step out happens together with the search, and
        whether the agent moves out while searching (move_out, default).

        With move_out: both sensed at onset is a tie (High). Otherwise the agent moves out while
        turning, and the lit wall its rays hit first wins (a wall sensed at onset counts as time 0).
        Ties go to High. The description below is for move_out=False.

        Both sensed at onset counts as a tie (High). Otherwise the agent searches without stepping if
        any heading from where it stands senses either lit wall, and the first wall its search finds
        wins. Only if neither can be sensed from there does it step out. With merge_step it steps and
        turns together, and the wall its rays hit first along that motion wins. Otherwise the wall that
        needs the shorter step wins. Ties go to High. '''

    both0 = races['high_sensed0'].to_numpy() & races['low_sensed0'].to_numpy()
    if move_out:
        return (races['low_tsense'].to_numpy() < races['high_tsense'].to_numpy()) & ~both0
    hs, ls = races['high_step'].to_numpy(), races['low_step'].to_numpy()
    ht, lt = races['high_theta'].to_numpy(), races['low_theta'].to_numpy()
    no_step = (hs == 0) | (ls == 0)
    # without a step, a wall that would need one cannot be found by the search
    ht_ns, lt_ns = np.where(hs == 0, ht, np.inf), np.where(ls == 0, lt, np.inf)
    both_step = (races['low_tsense'].to_numpy() < races['high_tsense'].to_numpy()) if merge_step else (ls < hs)
    first = np.where(no_step, lt_ns < ht_ns, both_step)
    return first & ~(races['high_sensed0'].to_numpy() & races['low_sensed0'].to_numpy())


# inference of where High is

def pair_prior():
    ''' Return the layout prior as a matrix (High row x Low column, 8 x 8), from layout_table.
        Takes no inputs. '''

    pr = np.zeros((NUM_WALLS, NUM_WALLS))
    lay = layout_table()
    np.add.at(pr, (lay['high'].to_numpy(), lay['low'].to_numpy()), lay['prob'].to_numpy())
    return pr


def low_offset_weights():
    ''' Return the probability of each Low offset from High ((low - high) mod 8). Takes no inputs. '''

    lay = layout_table()
    return lay.groupby((lay['low'] - lay['high']) % NUM_WALLS)['prob'].sum().to_dict()


def high_posterior_given(sensed, low_seen):
    ''' Return the probability that High is at each wall (8), given what the rays sensed at onset.
        Takes which walls were hit by a ray (8 bools, High not among them) and the Low wall's row index
        if Low was hit, else None.

        Every wall hit and not Low is unlit, so neither High nor Low can be there. If Low was hit, it is
        at that wall; if not, it is at a wall not hit. Matches agent_raycast.high_posterior. '''

    sensed = np.asarray(sensed, bool)
    pr = pair_prior()
    if low_seen is None:
        post = (pr * ~sensed[None, :]).sum(axis=1) * ~sensed
    else:
        post = pr[:, low_seen] * ~sensed
    return post / post.sum()


def _choose_direction(post, t_dir):
    ''' Return the search direction (1, -1, or 0 for a tie) with the lower expected time to sense High.
        Takes the posterior over walls (8) and the time to first sense each wall in each direction (8 x 2). '''

    keep = post > 0
    e = [(post[keep] * t_dir[keep, j]).sum() for j in (0, 1)]
    if abs(e[0] - e[1]) <= 1e-9:
        return 0
    return 1 if e[0] < e[1] else -1


def central_time_one(heading, low_rel, rot_speed, close_speed, speed=SPEED):
    ''' Return the inferring central agent's time to High and its chosen search direction (1, -1, or 0
        for a tie, when the time is the mean of both directions). Takes its heading (degrees, with High
        at bearing 0), Low's offset from High in walls, the rotation and closing speeds and the speed.

        From the centre a wall is sensed when its bearing is within central_visible_half_angle of the
        heading. If High is sensed the agent turns the short way to it. Otherwise it infers where High
        could be from the walls in view, turns in the direction with the lower expected search time
        (a tie is 50/50), and once High is sensed keeps turning to centre it, closing at close_speed. '''

    V = central_visible_half_angle()
    bearings = 45.0 * np.arange(NUM_WALLS)
    rel = _wrap(bearings - heading)
    sensed = np.abs(rel) <= V

    def time_for(turn):
        return turn / rot_speed + (TRIGGER_REACH - close_speed * min(turn, V) / rot_speed) / speed

    if sensed[0]:
        return time_for(abs(float(rel[0]))), 0
    low = low_rel % NUM_WALLS
    post = high_posterior_given(sensed, low if sensed[low] else None)
    # angle to turn in each direction before each wall is sensed
    t_dir = np.stack([np.maximum((d * rel) % 360 - V, 0.0) for d in (1, -1)], axis=1)
    d = _choose_direction(post, t_dir)
    turns = {1: float(rel[0] % 360), -1: float((-rel[0]) % 360)}
    if d == 0:
        return 0.5 * (time_for(turns[1]) + time_for(turns[-1])), 0
    return time_for(turns[d]), d


def central_times_inferred(rot_speed, close_speed, speed=SPEED, n=N_HEADINGS):
    ''' Return the inferring central agent's times to High for each Low offset from High (dict offset ->
        2n equally likely times), over a uniform grid of n headings. Takes the rotation and closing speeds,
        the speed and the grid size.

        Each heading appears twice: once per search direction when the directions tie, otherwise twice
        with the chosen direction. Use with evaluate, which matches each race row to its Low offset. '''

    heads = (np.arange(n) + 0.5) * 360.0 / n
    V = central_visible_half_angle()

    def time_for(turn):
        return turn / rot_speed + (TRIGGER_REACH - close_speed * min(turn, V) / rot_speed) / speed

    out = {}
    for k in low_offset_weights():
        times = []
        for h in heads:
            _, d = central_time_one(h, k, rot_speed, close_speed, speed)
            rel0 = float(_wrap(-h))
            if abs(rel0) <= V:
                times += [time_for(abs(rel0))] * 2
            elif d == 0:
                times += [time_for(rel0 % 360), time_for((-rel0) % 360)]
            else:
                times += [time_for((d * rel0) % 360)] * 2
        out[k] = np.array(times)
    return out


def onset_sensed(positions, headings):
    ''' Return which walls the rays hit at onset for each peripheral agent (n x 8), at the heading
        rounded to the search grid as in build_races_rays. Takes positions (n x 2) and headings (degrees). '''

    positions = np.atleast_2d(np.asarray(positions, float))
    h = np.round(np.asarray(headings, float) % 360 / HEAD_STEP) * HEAD_STEP
    ids, _, _ = agent_raycast.cast_rays(positions, np.radians(h))
    return (ids[..., None] == np.arange(1, NUM_WALLS + 1)).any(axis=1)


def infer_direction(races, sensed):
    ''' Return the race rows with each search direction's probability set by inference.
        Takes rows from build_races_rays and onset_sensed for its positions.

        For each position and layout where High is not sensed at onset, the agent works out where High
        could be (high_posterior_given) and takes the direction with the lower expected time until a ray
        first hits High (high_tsense, the first-sense time along that direction's search, for every
        candidate wall). That direction gets the row's whole probability (a tie keeps 50/50). Rows where
        High is sensed at onset keep 50/50, as the direction does not matter. '''

    races = races.copy()
    # take every wall's first-sense time per position and direction from the rows where it is High
    first = races.pivot_table(index=['pos', 'high'], columns='direction', values='high_tsense', aggfunc='first')
    n_pos = int(races['pos'].max()) + 1
    T = np.full((n_pos, NUM_WALLS, 2), np.inf)
    for (i, w), row in first.iterrows():
        T[i, w] = [row[1], row[-1]]
    keys = races[['pos', 'high', 'low']].drop_duplicates()
    choice = {}
    for i, hgh, lw in keys.itertuples(index=False):
        S = np.asarray(sensed[i], bool)
        if S[hgh]:
            choice[(i, hgh, lw)] = 0
            continue
        post = high_posterior_given(S, lw if S[lw] else None)
        choice[(i, hgh, lw)] = _choose_direction(post, T[i])
    d = np.array([choice[k] for k in zip(races['pos'], races['high'], races['low'])])
    w = np.where(d == 0, 0.5, (races['direction'].to_numpy() == d).astype(float))
    races['prob'] = races['prob'] * 2 * w
    races['direction_chosen'] = d
    return races


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
        nearer (build_races_rays rows only). 'delta' is the threshold rule with inference (delta_outcome;
        build_races_rays rows only). 'delta2' is the threshold rule with inference about both walls,
        which can also search for a probably near, unseen Low (delta2_outcome; build_races_rays rows only). '''

    gap = (races['t_high'] - races['t_low']).to_numpy()
    if policy == 'high':
        return np.zeros(len(races), bool)
    if policy == 'closest':
        return gap > TIE
    if policy == 'low_first':
        return races['low_first'].to_numpy(bool)
    if policy == 'nearest':
        return races['nearest_low'].to_numpy(bool)
    if policy == 'delta':
        return delta_outcome(races, delta)[0]
    if policy == 'delta2':
        return delta2_outcome(races, delta)[0]
    if policy == 'threshold':
        if delta is None:
            raise ValueError("threshold needs delta")
        return gap > delta + TIE
    raise ValueError(f"policy must be one of {POLICIES}")


def _against_central(t_p, central_t):
    ''' Return the peripheral win probability and expected race time against a set of equally likely
        central times. Takes the peripheral times and the central times. '''

    # score each row against every central time, using the sorted times and their running sums
    ct = np.sort(np.asarray(central_t, float))
    n = len(ct)
    csum = np.concatenate([[0.0], np.cumsum(ct)])
    lo, hi = np.searchsorted(ct, t_p - TIE, 'left'), np.searchsorted(ct, t_p + TIE, 'right')
    win = ((n - hi) + 0.5 * (hi - lo)) / n
    race = (csum[lo] + t_p * (n - lo)) / n
    return win, race


def evaluate(races, policy='high', delta=None, central_t=None):
    ''' Return each race row's outcome under a peripheral wall-choice policy.
        Takes the race rows, the policy, its delta and the central agent's times (an array from
        central_times, each equally likely, or a dict by Low offset from central_times_inferred).
        Without central_t, the central agent takes its travel time to High.

        A peripheral agent already overlapping a lit trigger wins that wall at time 0, whatever
        the policy. Otherwise the faster agent wins, and equal times split the win. '''

    go_low = choose_low(races, policy, delta)
    tH, tL, tC = races['t_high'].to_numpy(), races['t_low'].to_numpy(), races['t_central'].to_numpy()
    # take the wall the peripheral agent reaches, with an instant win overriding the choice
    t_p = np.where(go_low, tL, tH)
    if policy == 'nearest':
        # take the time to the wall reached after the switch decision
        t_p = races['t_nearest'].to_numpy()
    if policy == 'delta':
        t_p = delta_outcome(races, delta)[1]
    if policy == 'delta2':
        t_p = delta2_outcome(races, delta)[1]
    val = np.where(go_low, LOW_SCORE, HIGH_SCORE)
    t_p = np.where(tH == 0, 0.0, np.where(tL == 0, 0.0, t_p))
    val = np.where(tH == 0, HIGH_SCORE, np.where(tL == 0, LOW_SCORE, val))
    if central_t is None:
        win = np.where(np.abs(t_p - tC) <= TIE, 0.5, (t_p < tC).astype(float))
        race = np.minimum(t_p, tC)
    elif isinstance(central_t, dict):
        # match each row to the central times for its Low offset from High
        off = ((races['low'] - races['high']) % NUM_WALLS).to_numpy()
        win, race = np.zeros(len(races)), np.zeros(len(races))
        for k, ct in central_t.items():
            m = off == k
            win[m], race[m] = _against_central(t_p[m], ct)
    else:
        win, race = _against_central(t_p, central_t)
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
    elif isinstance(central_t, dict):
        # average the race time over Low offsets, both agents drawing from the same layout's times
        w = low_offset_weights()
        race_s = sum(w[k] * central_reference(overhead, speed, central_t[k])['race_s'] for k in central_t)
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


DELAY_PARTS = ('trial start to slice onset', 'trigger to trial end', 'trial end to next trial')
HUMAN_DELAYS_CACHE = 'human_trial_delays'


def trial_delays(trial):
    ''' Return a trial's time outside the race in three parts (s), keyed by DELAY_PARTS, or None if the
        trial has no slice onset, trigger or trial end. Takes a trial dataframe.

        A trial's log runs from its trial start to just before the next trial start, so the ITI is at
        its end (trial end to the last frame). '''

    ev, tt = trial['eventDescription'], trial['timeApplication']
    so, trig, te = (tt[ev == e] for e in (globals.SLICE_ONSET, globals.SELECTED_TRIGGER_ACTIVATION, globals.TRIAL_END))
    if len(so) == 0 or len(trig) == 0 or len(te) == 0:
        return None
    return dict(zip(DELAY_PARTS, ((so.iloc[0] - tt.iloc[0]).total_seconds(), (te.iloc[0] - trig.iloc[0]).total_seconds(),
                                  (tt.iloc[-1] - te.iloc[0]).total_seconds())))


def human_trial_delays(recompute=False):
    ''' Return one row per human trial with its session type and time outside the race in three parts
        (trial_delays), from cache when possible. Takes a recompute flag. Drops each session's first
        trial. Recomputing loads the social and solo trial pickles (several GB each). '''

    import gc
    import pickle
    import data_strings
    import data_extraction.data_saving as data_saving

    def compute():
        rows = []
        for kind, name in (('social', data_strings.FILENAME_SOCIAL), ('solo', data_strings.FILENAME_SOLO)):
            with open(os.path.join(data_strings.DATAFRAME_PATH, name + '.pkl'), 'rb') as f:
                sessions = pickle.load(f)
            for trial_list in sessions:
                for trial in trial_list[1:]:
                    d = trial_delays(trial)
                    if d is not None:
                        rows.append({'session type': kind, **d})
            # free each pickle before loading the next
            del sessions
            gc.collect()
        return pd.DataFrame(rows)

    return data_saving.cached(HUMAN_DELAYS_CACHE, compute, recompute=recompute, stamp='v1')


def human_overhead(session_type='social', recompute=False):
    ''' Return the human mean time outside the race per trial (s), the sum of the three delay parts.
        Takes the session type ('social' or 'solo') and a recompute flag. '''

    d = human_trial_delays(recompute)
    return float(d.loc[d['session type'] == session_type, list(DELAY_PARTS)].sum(axis=1).mean())


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
