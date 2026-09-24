# Plot two real frames that show why the angular wall-visibility method differs from the new method
# when nothing blocks sight.
#   Left:  the corner moves when walls use physical positions (reachable + player radius)
#   Right: close to a wall, the angular method tests the wrong corner
#
# Frames are from 260723 model 23, player 1 (trial_list index 59 frame 141, and index 29 frame 89,
# frames counted from slice onset). Found by diagnose_method_difference.py.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python sandbox/visibility_occlusion/plot_method_difference_examples.py

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import globals
from plotting import plot_octagon
from trajectory_analysis import visibility_occlusion as vo
import diagnose_method_difference as D

OUT = os.path.dirname(os.path.abspath(__file__))
FOV = 110

# colours (categorical slots 1, 2, 7) and ink
BLUE, ORANGE, VIOLET = '#2a78d6', '#eb6834', '#4a3aa7'
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#c9c8c2'

EX1 = dict(eye=np.array([-15.797136306762695, -3.611041307449341]), heading=253.86127167765017, wall=5)
EX2 = dict(eye=np.array([-9.5713529586792, 14.04570484161377]), heading=259.62455279448955, wall=7)


def unit(deg):
    ''' Return the unit vector for an angle clockwise from +z. Takes degrees. '''

    return np.array([math.sin(math.radians(deg)), math.cos(math.radians(deg))])


def angle_between(eye, heading_deg, point):
    ''' Return the angle (degrees) between the heading and the direction to a point. Takes eye, heading and point. '''

    d = point - eye
    return math.degrees(math.acos(np.clip(d @ unit(heading_deg) / np.linalg.norm(d), -1, 1)))


def draw_arena(ax, eye, heading_deg, reach=True):
    ''' Draw the physical outline, optionally the reachable outline, the field of view and the heading.
        Takes axes, eye position, heading and whether to draw the reachable outline. '''

    # plot the field of view as a shaded wedge (matplotlib angles run counterclockwise from +x)
    mpl_heading = 90 - heading_deg
    ax.add_patch(Wedge(eye, 40, mpl_heading - FOV / 2, mpl_heading + FOV / 2, color=GRID, alpha=0.35, lw=0, zorder=0))
    for side in (-1, 1):
        edge = eye + 40 * unit(heading_deg + side * FOV / 2)
        ax.plot(*zip(eye, edge), color=INK_2, lw=0.8, ls=':', zorder=1)

    # plot the outlines
    segments = vo.get_wall_segments()
    phys = np.vstack([segments[:, 0], segments[:1, 0]])
    ax.plot(phys[:, 0], phys[:, 1], color=INK, lw=2, zorder=2, label='physical walls')
    if reach:
        outline = np.array(plot_octagon.return_octagon_path_points())
        ax.plot(outline[:, 0], outline[:, 1], color=INK_2, lw=1, ls='--', zorder=2, label='reachable outline')

    # plot the eye and heading
    ax.plot(*eye, 'o', color=INK, ms=8, zorder=5)
    ax.annotate('', xy=eye + 2.2 * unit(heading_deg), xytext=eye,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=2), zorder=5)


def sight_line(ax, eye, point, colour, label):
    ''' Draw a line from the eye to a point with a ringed marker. Takes axes, eye, point, colour and legend label. '''

    ax.plot(*zip(eye, point), color=colour, lw=2, zorder=3)
    ax.plot(*point, 'o', color=colour, ms=10, mec='white', mew=2, zorder=6, label=label)


fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw=dict(width_ratios=[1, 1.45]))

# ---- example 1: physical corner sits in a different direction ----
ax = axes[0]
eye, h, wall = EX1['eye'], EX1['heading'], EX1['wall']
reach_corner = D.wall_corners(globals.OCTAGON_WALLFACE_RADIUS, globals.ALCOVE_HALF_WIDTH)[wall, 4]
phys_corner = D.wall_corners(D.W_PHYS, globals.ALCOVE_HALF_WIDTH + D.R)[wall, 4]
a_reach, a_phys = angle_between(eye, h, reach_corner), angle_between(eye, h, phys_corner)

draw_arena(ax, eye, h)
sight_line(ax, eye, reach_corner, BLUE, f'reachable corner: {a_reach:.1f}°, hidden')
sight_line(ax, eye, phys_corner, ORANGE, f'physical corner: {a_phys:.1f}°, visible')
ax.text(*(reach_corner + [0.25, 0.05]), f'{a_reach:.1f}°', color=INK, fontsize=11, va='center')
ax.text(*(phys_corner + [-0.25, -0.35]), f'{a_phys:.1f}°', color=INK, fontsize=11, ha='right', va='center')
ax.text(-17.75, -3.0, 'west wall', color=INK_2, fontsize=10, rotation=90, ha='right', va='center')
ax.text(-15.2, -10.0, 'south-west wall', color=INK_2, fontsize=10, rotation=-45, ha='center', va='center')
ax.set(xlim=(-20, -11), ylim=(-11, -0.5))
ax.set_title('1. Physical walls move the corner', loc='left', fontsize=13, color=INK, pad=10)
ax.legend(loc='upper right', fontsize=9, frameon=False)
ax.text(0.0, -0.13,
        'Both geometries test the same corner of the south-west wall.\n'
        f'The player is {D.W_PHYS - abs(eye[0]):.1f} from the west wall, so the corner is close.\n'
        'Moving the corner 0.61 outward turns its direction\n'
        f'by {a_reach - a_phys:.1f}°, across the FoV edge (±{FOV // 2}°).',
        transform=ax.transAxes, fontsize=10.5, color=INK, va='top')

# ---- example 2: the wrong corner is tested ----
ax = axes[1]
eye, h, wall = EX2['eye'], EX2['heading'], EX2['wall']
corners = D.wall_corners(D.W_PHYS, globals.ALCOVE_HALF_WIDTH + D.R)[wall]
ccw_corner, cw_corner = corners[0], corners[4]
alcove_centre = unit(45 * wall) * (D.W_PHYS + globals.ALCOVE_DEPTH / 2)
a_cw, a_ccw = angle_between(eye, h, cw_corner), angle_between(eye, h, ccw_corner)

draw_arena(ax, eye, h, reach=False)
ax.plot(*zip(eye, alcove_centre), color=VIOLET, lw=1.5, ls='--', zorder=3)
ax.plot(*alcove_centre, 's', color=VIOLET, ms=9, mec='white', mew=2, zorder=6,
        label='alcove centre: counterclockwise of heading')
sight_line(ax, eye, cw_corner, BLUE, f'corner the angular method tests: {a_cw:.1f}°, hidden')
sight_line(ax, eye, ccw_corner, ORANGE, f'other corner: {a_ccw:.1f}°, visible')
ax.text(*(cw_corner + [0.35, -0.9]), f'{a_cw:.1f}°', color=INK, fontsize=11)
ax.text(*(ccw_corner + [0.35, -0.6]), f'{a_ccw:.1f}°', color=INK, fontsize=11)
ax.text(-10.1, 16.6, 'north-west wall', color=INK_2, fontsize=10, rotation=45, ha='center', va='center')
ax.set(xlim=(-19.5, 9), ylim=(4.5, 20.5))
ax.set_title('2. Close to a wall, the wrong corner is tested', loc='left', fontsize=13, color=INK, pad=10)
ax.legend(loc='lower right', fontsize=9, frameon=False)
ax.text(0.0, -0.13,
        'The angular method checks which side of the heading the alcove centre is on (here counterclockwise).\n'
        "It assumes the whole wall is on that side, and tests the wall's corner nearest the heading from that side.\n"
        f'The player is {D.W_PHYS - (eye @ unit(45 * wall)):.2f} from this wall, so the wall wraps behind them.\n'
        f'The tested corner is {a_cw:.0f}° away, but the other corner is only {a_ccw:.0f}° away.',
        transform=ax.transAxes, fontsize=10.5, color=INK, va='top')

for ax in axes:
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)

fig.text(0.01, 0.97, 'Why the angular method differs when nothing blocks sight  (heading arrow, grey wedge = ±55° field of view)',
         fontsize=12, color=INK_2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.2, wspace=0.12)
fig.savefig(os.path.join(OUT, 'method_difference_examples.png'), dpi=130, facecolor='white')
