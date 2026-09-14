import os, numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
pos = pd.read_pickle(os.path.join(OUT, 'positions_260723_peripheral.pkl'))
x, z = pos.x.to_numpy(), pos.z.to_numpy()
t = np.radians(22.5)

rec = []
for k in range(8):
    # wall-centre direction k*45 deg clockwise from +z
    a = np.radians(45 * k)
    n = np.array([np.sin(a), np.cos(a)]); u = np.array([np.cos(a), -np.sin(a)])
    p = x * n[0] + z * n[1]           # distance along the wall normal
    q = x * u[0] + z * u[1]           # lateral offset along the wall
    # wall face: laterally between alcove and vertex, away from both
    wf = (np.abs(q) > 3.5) & (np.abs(q) < 5.5)
    # alcove: beyond the wall face line
    al = p > 17.2
    # vertex: within 1 deg of the vertex direction, on the + side
    av = np.radians(45 * k + 22.5)
    ang = np.arctan2(x, z)
    dv = np.abs((ang - av + np.pi) % (2 * np.pi) - np.pi) < np.radians(0.5)
    r = np.hypot(x, z)
    rec.append(dict(wall=k,
                    wall_face=p[wf].max(), wall_face_p9999=np.quantile(p[wf], .9999),
                    alcove_back=p[al].max(), alcove_back_p9999=np.quantile(p[al], .9999),
                    alcove_halfwidth=np.abs(q[al & (p > 18)]).max(),
                    vertex=r[dv].max(), n_wf=wf.sum(), n_al=al.sum(), n_v=dv.sum()))
df = pd.DataFrame(rec)
pd.set_option('display.width', 200)
print(df.round(4).to_string(index=False))
m = df.mean(numeric_only=True)
print('\nmean over walls:\n', m.round(4).to_string())

W, B, H, V = m.wall_face, m.alcove_back, m.alcove_halfwidth, m.vertex
print(f"\nimplied vertex from wall face / cos(22.5) = {W/np.cos(t):.4f}  (observed {V:.4f})")
print(f"alcove depth (back - wall face)            = {B-W:.4f}")
print(f"reachable alcove width                     = {2*H:.4f};  reachable wall side 2*W*tan(22.5) = {2*W*np.tan(t):.4f}, /3 = {2*W*np.tan(t)/3:.4f}")

# corner rounding at the alcove mouth gives the agent radius c
# (reachable boundary near a reflex corner is an arc of radius c about that corner)
k = 0; a = 0.0
n = np.array([0, 1.0]); u = np.array([1.0, 0])
p = z; q = x
sel = (p > W - 1.5) & (p < W + 1.5) & (q > H - 1.5) & (q < H + 1.5)
# upper envelope: for each lateral bin, max p
qb = np.round(q[sel], 2); env = pd.Series(p[sel]).groupby(qb).max()
print('\nenvelope near alcove mouth corner (wall 0, +q side): lateral q -> max normal p')
print(env.iloc[::10].round(3).to_string())
env.to_csv(os.path.join(OUT, 'mouth_envelope.csv'))
