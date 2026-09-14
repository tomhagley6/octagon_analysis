import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.dirname(os.path.abspath(__file__))
pos = pd.read_pickle(os.path.join(OUT, 'positions_260723_peripheral.pkl'))
x, z = pos.x.to_numpy(), pos.z.to_numpy()
r = np.hypot(x, z)
# angle clockwise from +z, as in plot_octagon (x = sin, y = cos)
ang = np.degrees(np.arctan2(x, z)) % 360
b = np.floor(ang).astype(int)
prof = pd.DataFrame(dict(b=b, r=r)).groupby('b').r.agg(['max', 'size'])
# fold onto one 45-degree sector (0 = wall-centre direction in plot_octagon)
fold = (ang % 45)
fb = np.floor(fold * 2) / 2
fprof = pd.DataFrame(dict(fb=fb, r=r)).groupby('fb').r.agg(['max', 'size'])
print(fprof.to_string())

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(prof.index, prof['max'], lw=1)
for a in range(0, 360, 45): ax[0].axvline(a, color='C1', lw=.5)
for a in range(0, 360, 45): ax[0].axvline(a + 22.5, color='C2', lw=.5, ls='--')
ax[0].set(xlabel='angle from +z (deg)', ylabel='max distance from centre', title='orange = wall centres, green dashed = vertices (plot_octagon)')
ax[1].scatter(x[::20], z[::20], s=.2, alpha=.3); ax[1].set_aspect('equal')
fig.savefig(os.path.join(OUT, 'profile.png'), dpi=110, bbox_inches='tight')
