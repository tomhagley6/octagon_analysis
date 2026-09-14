import sys, os
sys.path.insert(0, '/home/tom/repos/octagon_analysis')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from plotting import plot_octagon

OUT = os.path.dirname(os.path.abspath(__file__))

# measured reach of the prefab centre (mean over 8 walls, 260723 models 1, 23, 33, 51)
W = 16.7468          # wall face
B = 19.7466          # alcove back
H = 1.9001           # alcove half-width
V = W / np.cos(np.radians(22.5))   # vertex, from the wall-face lines

def measured_outline():
    ''' Return x, z of the measured reachable outline. Takes no inputs. '''
    t = np.tan(np.radians(22.5))
    # one wall in its own frame (q along the wall, p toward the wall)
    qp = [(-W*t, W), (-H, W), (-H, B), (H, B), (H, W), (W*t, W)]
    xs, zs = [], []
    for k in range(8):
        a = np.radians(45 * k)
        for q, p in qp:
            # rotate from the wall frame back to arena x, z
            xs.append(p*np.sin(a) + q*np.cos(a))
            zs.append(p*np.cos(a) - q*np.sin(a))
    return np.r_[xs, xs[0]], np.r_[zs, zs[0]]

# get a subsample of logged positions for context (pickle path can be passed as the first argument)
pkl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, 'positions_260723_peripheral.pkl')
pos = pd.read_pickle(pkl).sample(40000, random_state=0)
mx, mz = measured_outline()

fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
for ax in axes:
    ax.scatter(pos.x, pos.z, s=0.3, c='0.75', alpha=0.4, rasterized=True)
    # plot the measured outline thick, then plot_octagon thin and dashed on top so both stay visible
    ax.plot(mx, mz, c='C3', lw=3, alpha=0.6)
    plot_octagon.plot_octagon(ax)
    ax.get_lines()[-1].set(lw=1, ls='--')
    # plot circles at the measured reach distances
    for r, c, lab in [(W, 'C0', f'wall face {W:.3f}'), (V, 'C2', f'vertex {V:.3f}'), (B, 'C1', f'alcove back {B:.3f}')]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=':', lw=1, color=c, label=lab))
axes[0].plot([], [], c='k', lw=1, ls='--', label=f'plot_octagon (diameter {plot_octagon.diameter:.3f})')
axes[0].plot([], [], c='C3', lw=3, alpha=0.6, label='measured reach')
axes[0].legend(loc='center', fontsize=8, frameon=False)
axes[0].set(xlim=(-21.5, 21.5), ylim=(-21.5, 21.5), title='Measured reach over plot_octagon')
# zoom on the +z alcove and the vertex to its right
axes[1].set(xlim=(-3.5, 8.5), ylim=(15.0, 20.5), title='Zoom: +z alcove and adjacent vertex')
# zoom on the +x+z diagonal alcove
axes[2].set(xlim=(9.5, 16.5), ylim=(9.5, 16.5), title='Zoom: +x+z diagonal alcove')
for ax in axes:
    ax.set_xlabel('x'); ax.set_ylabel('z')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'measured_outline.png'), dpi=130)
