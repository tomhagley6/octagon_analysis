import os, numpy as np, pandas as pd
from scipy.optimize import least_squares
OUT = os.path.dirname(os.path.abspath(__file__))
pos = pd.read_pickle(os.path.join(OUT, 'positions_260723_peripheral.pkl'))
x, z = pos.x.to_numpy(), pos.z.to_numpy()
W, H = 16.7468, 1.9001   # reachable wall face and alcove half-width (mean over walls)

qs, ps = [], []
for k in range(8):
    a = np.radians(45 * k)
    p = x*np.sin(a) + z*np.cos(a); q = x*np.cos(a) - z*np.sin(a)
    for s in (1, -1):
        qq = s*q
        sel = (qq > H) & (qq < H + 1.0) & (p > W - 0.5) & (p < W + 1.0)
        qs.append(qq[sel]); ps.append(p[sel])
q = np.concatenate(qs); p = np.concatenate(ps)
# upper envelope in 0.02 lateral bins, pooled over 16 mouth corners
bins = np.floor((q - H) / 0.02).astype(int)
env = pd.DataFrame(dict(b=bins, q=q, p=p)).groupby('b').agg(q=('q', 'mean'), p=('p', 'max'))

def model(c, qv):
    # reachable p near a reflex corner at (H + c, W + c): arc of radius c, then the flat wall face
    d = qv - (H + c)
    return np.where(d < 0, W + c - np.sqrt(np.clip(c**2 - d**2, 0, None)), W)

env = env.iloc[2:]
fit = least_squares(lambda c: model(c[0], env.q.to_numpy()) - env.p.to_numpy(), x0=[0.4], bounds=(0.05, 1.5))
c = fit.x[0]
print(f"fitted agent radius c = {c:.3f}  (rms resid {np.sqrt(np.mean(fit.fun**2)):.3f})")
for cc in (0.3, 0.4, 0.5, 0.6):
    print(f"  c={cc}: rms {np.sqrt(np.mean((model(cc, env.q.to_numpy()) - env.p.to_numpy())**2)):.3f}")
print(env.assign(model=model(c, env.q.to_numpy())).iloc[::3].round(3).to_string())
