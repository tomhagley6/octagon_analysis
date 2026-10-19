''' Replay logged episodes through a trained ML-Agents policy and test which inputs it uses.

The exported ONNX policy (`OctagonAgentSocial.onnx`) is re-implemented in numpy, so it runs in the
analysis environment without onnxruntime. Graph, as exported by ML-Agents for these runs:
  concat(obs_0 [42 ray values], obs_1 [1 value, always 0])
  -> Gemm 512 + swish -> Gemm 512 + swish -> LSTM (hidden 128; memory 256 = h and c)
  -> three Gemm heads (move, strafe, rotate) -> softmax.
ONNX orders the LSTM gates i, o, f, c. The sampling step (Multinomial) is left out, as the action
probabilities are what the tests compare.

Observations are rebuilt from the logs with `analysis.agent_raycast`: 7 rays, each giving
[distance / 60, HighWall, LowWall, HighWallTrigger, LowWallTrigger, opponent]. The opponent value is
d / 60 before the 261014 builds and 1 - d / 60 from them on, 0 meaning no opponent (OBS_* below).
  - Memory resets at each trigger (`EndEpisode` in OctagonWallTrigger.cs), so an episode runs from the
    previous trial's trigger to this trial's trigger.
  - Walls are washed and triggers switched off at the trigger, and lit again at slice onset, so tags
    and trigger boxes exist only from slice onset.
  - The opponent channel follows the seat rules in agent_raycast (`opponent_channel_seats`).

`export_weights` needs the `onnx` package (the `mlagents` conda env has it). Everything else runs in the
analysis env.
'''

import numpy as np
import pandas as pd

import globals
import analysis.agent_raycast as ar

RAY_LENGTH = 60.0
N_RAYS = 7
PER_RAY = 6                         # distance, HighWall, LowWall, HighWallTrigger, LowWallTrigger, opponent
OPPONENT = 5                        # index of the opponent channel within a ray

# observation rules of each build generation. Before 261014 only seat 0 has the opponent channel, the
# opponent is seen through walls, and the channel holds distance (d / 60). From 261014 both seats have it,
# walls hide the opponent, and it holds proximity (1 - d / 60, about 0.34-0.99 in the arena). 0 means no
# opponent in both
OBS_PRE_261014 = dict(ar.BUILD_PRE_261014, opponent_encoding='distance')
OBS_261014 = dict(ar.BUILD_261014, opponent_encoding='proximity')


def export_weights(onnx_path, out_path):
    ''' Save the policy weights of an exported ML-Agents model to a .npz file. Takes the .onnx path and
        the output path. Run in an environment with `onnx` installed. '''

    import onnx
    from onnx import numpy_helper

    m = onnx.load(onnx_path)
    w = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    lstm = [n for n in m.graph.node if n.op_type == 'LSTM'][0]
    body = 'network_body._body_endoder.seq_layers'
    heads = 'action_model._discrete_distribution.branches'
    np.savez(out_path, W0=w[f'{body}.0.weight'], b0=w[f'{body}.0.bias'], W2=w[f'{body}.2.weight'], b2=w[f'{body}.2.bias'],
             LW=w[lstm.input[1]], LR=w[lstm.input[2]], LB=w[lstm.input[3]],
             H0=w[f'{heads}.0.weight'], h0=w[f'{heads}.0.bias'], H1=w[f'{heads}.1.weight'], h1=w[f'{heads}.1.bias'],
             H2=w[f'{heads}.2.weight'], h2=w[f'{heads}.2.bias'])


class Policy:
    ''' Numpy forward pass of the exported policy. Takes the .npz from export_weights, or a dict of arrays. '''

    def __init__(self, weights):
        w = np.load(weights) if isinstance(weights, str) else weights
        self.W0, self.b0, self.W2, self.b2 = w['W0'], w['b0'], w['W2'], w['b2']
        self.LW, self.LR, self.LB = w['LW'][0], w['LR'][0], w['LB'][0]
        self.heads = [(w['H0'], w['h0']), (w['H1'], w['h1']), (w['H2'], w['h2'])]
        self.H = self.LR.shape[1]

    def run(self, obs):
        ''' Return action probabilities (timepoints*3 branches*3 actions) for one episode of observations
            (timepoints*42), starting from zero memory. '''

        H = self.H
        h, c = np.zeros(H), np.zeros(H)
        x = np.concatenate([obs, np.zeros((len(obs), 1))], axis=1)
        z = x @ self.W0.T + self.b0
        z = z / (1 + np.exp(-z))
        z = z @ self.W2.T + self.b2
        z = z / (1 + np.exp(-z))
        pre = z @ self.LW.T + self.LB[:4 * H] + self.LB[4 * H:]
        sig = lambda v: 1 / (1 + np.exp(-v))
        out = np.zeros((len(obs), 3, 3))
        for t in range(len(obs)):
            g = pre[t] + self.LR @ h
            i, o, f, cc = sig(g[:H]), sig(g[H:2 * H]), sig(g[2 * H:3 * H]), np.tanh(g[3 * H:])
            c = f * c + i * cc
            h = o * np.tanh(c)
            for b, (Wh, bh) in enumerate(self.heads):
                lg = Wh @ h + bh
                e = np.exp(lg - lg.max())
                out[t, b] = e / e.sum()
        return out


def build_obs(eyes, yaw, opp, seat, high, low, lit_from, opponent_channel_seats=(0,), occlude=False,
              opponent_encoding='distance'):
    ''' Return the 42 ray observations of one agent per frame. Takes its positions and yaw, the opponent's
        positions (all timepoints*2 / timepoints), its logged seat, the lit walls, the frame from which they
        are lit, and the seats that fill the opponent channel.

        In builds before 261014 only seat 0 (PlayerAgent) passes through its opponent, and seat 1's rays
        stop at seat 0's body, and the channel holds d / 60. From 261014 both seats pass through each other,
        walls hide the opponent, and the channel holds 1 - d / 60. Pass **OBS_PRE_261014 or **OBS_261014
        (the default is the older builds). '''

    T = len(eyes)
    obs = np.zeros((T, N_RAYS, PER_RAY))
    blocked = seat not in opponent_channel_seats
    for lit, sl in ((False, slice(0, lit_from)), (True, slice(lit_from, T))):
        if sl.stop - sl.start <= 0:
            continue
        ids, kinds, dist = ar.cast_rays(eyes[sl], yaw[sl], blocker=opp[sl] if blocked else None, triggers=lit)
        o = obs[sl]
        o[..., 0] = dist / RAY_LENGTH
        if lit:
            o[..., 1] = (kinds == ar.KIND_WALL) & (ids == high)
            o[..., 2] = (kinds == ar.KIND_WALL) & (ids == low)
            o[..., 3] = (kinds == ar.KIND_TRIGGER) & (ids == high)
            o[..., 4] = (kinds == ar.KIND_TRIGGER) & (ids == low)
        if not blocked:
            t_opp = ar.opponent_ray_distance(eyes[sl], yaw[sl], opp[sl], occlude=occlude)
            value = t_opp / RAY_LENGTH if opponent_encoding == 'distance' else 1 - t_opp / RAY_LENGTH
            o[..., OPPONENT] = np.where(np.isfinite(t_opp), value, 0.0)
        obs[sl] = o
    return obs.reshape(T, N_RAYS * PER_RAY)


def episodes(trial_list):
    ''' Yield one dict per episode (previous trigger -> this trigger) with both seats' positions and yaw,
        the lit walls and the frame they light (slice onset). Takes a trial list. '''

    for prev, cur in zip(trial_list[:-1], trial_list[1:]):
        tp = prev.index[prev['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
        tc = cur.index[cur['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
        so = cur.index[cur['eventDescription'] == globals.SLICE_ONSET]
        if not len(tp) or not len(tc) or not len(so):
            continue
        ep = pd.concat([prev.loc[tp[0]:], cur.loc[:tc[0]]])
        ep = ep[ep['eventDescription'] == 'time']
        lit_from = int((ep.index < so[0]).sum())
        d = dict(high=int(cur[globals.WALL_1].dropna().iloc[0]), low=int(cur[globals.WALL_2].dropna().iloc[0]),
                 lit_from=lit_from)
        for p in (0, 1):
            loc, rot = globals.PLAYER_LOC_DICT[p], globals.PLAYER_ROT_DICT[p]
            d[f'xy{p}'] = ep[[loc['xloc'], loc['yloc']]].to_numpy(float)
            d[f'yaw{p}'] = np.deg2rad(ep[rot['yrot']].to_numpy(float))
        if np.isfinite(d['xy0']).all() and np.isfinite(d['xy1']).all() and len(ep) > lit_from + 5:
            yield d


def observed_actions(xy, yaw, lag=1):
    ''' Return the move, strafe and rotate action each frame's movement implies (0 none, 1 forward / right /
        clockwise, 2 back / left / anticlockwise), using the step `lag` frames later. NaN where unclear.
        Takes positions and yaw. '''

    T = len(xy)
    a = np.full((T, 3), np.nan)
    for t in range(T - 1 - lag):
        d = xy[t + 1 + lag] - xy[t + lag]
        fw = d @ np.array([np.sin(yaw[t]), np.cos(yaw[t])])
        st = d @ np.array([np.cos(yaw[t]), -np.sin(yaw[t])])
        dy = (np.degrees(yaw[t + 1 + lag] - yaw[t + lag]) + 180) % 360 - 180
        a[t, 0] = 1 if fw > 0.08 else 2 if fw < -0.08 else (0 if abs(fw) < 0.02 else np.nan)
        a[t, 1] = 1 if st > 0.08 else 2 if st < -0.08 else (0 if abs(st) < 0.02 else np.nan)
        a[t, 2] = 1 if dy > 2 else 2 if dy < -2 else (0 if abs(dy) < 0.3 else np.nan)
    return a


def total_variation(p, q):
    ''' Return the total variation distance between two sets of action probabilities per frame, averaged
        over the three branches. Takes two timepoints*3*3 arrays. '''

    return 0.5 * np.abs(p - q).sum(axis=2).mean(axis=1)


def opponent_input_check(policy, eps, seat, opponent_channel_seats=(0,), occlude=False, opponent_encoding='distance'):
    ''' Return one row per episode comparing the policy's actions with and without opponent information,
        and with High and Low tags swapped (positive control) and the ray order reversed (sanity control).
        Takes a Policy, a list of episodes, the seat to replay and the build's observation rules
        (pass **OBS_PRE_261014 or **OBS_261014).

        For a seat with the opponent channel, the channel is zeroed. For a seat without it, the opponent's
        body is made see-through instead. The opponent comparisons use frames from the first frame the
        opponent changes the input. '''

    rows = []
    for e in eps:
        xy, yaw, opp = e[f'xy{seat}'], e[f'yaw{seat}'], e[f'xy{1 - seat}']
        base = build_obs(xy, yaw, opp, seat, e['high'], e['low'], e['lit_from'], opponent_channel_seats, occlude, opponent_encoding)
        if seat in opponent_channel_seats:
            pert = base.copy()
            pert[:, OPPONENT::PER_RAY] = 0
        else:
            pert = build_obs(xy, yaw, opp, seat, e['high'], e['low'], e['lit_from'], opponent_channel_seats=(seat,), occlude=occlude,
                             opponent_encoding=opponent_encoding)
            pert[:, OPPONENT::PER_RAY] = 0
        changed = np.abs(pert - base).sum(axis=1) > 1e-9
        swapped = base.reshape(-1, N_RAYS, PER_RAY).copy()
        swapped[:, :, [1, 2, 3, 4]] = swapped[:, :, [2, 1, 4, 3]]
        reversed_ = base.reshape(-1, N_RAYS, PER_RAY)[:, ::-1, :].reshape(-1, N_RAYS * PER_RAY)

        Pb, Po = policy.run(base), policy.run(pert)
        Ps, Pr = policy.run(swapped.reshape(-1, N_RAYS * PER_RAY)), policy.run(reversed_)
        trial = np.arange(len(xy)) >= e['lit_from']
        A = observed_actions(xy, yaw)

        def p_observed(P):
            vals = [P[~np.isnan(A[:, b]), b, A[~np.isnan(A[:, b]), b].astype(int)] for b in range(3)]
            return float(np.concatenate(vals).mean())

        row = dict(frames_with_opponent_input=float(changed.mean()),
                   swap_tv=float(total_variation(Pb, Ps)[trial].mean()),
                   swap_action_changed=float((Pb.argmax(2) != Ps.argmax(2))[trial].any(axis=1).mean()),
                   p_observed=p_observed(Pb), p_observed_without_opponent=p_observed(Po),
                   p_observed_rays_reversed=p_observed(Pr), opponent_tv=np.nan, opponent_action_changed=np.nan)
        if changed.any():
            first = int(np.argmax(changed))
            row['opponent_tv'] = float(total_variation(Pb, Po)[first:].mean())
            row['opponent_action_changed'] = float((Pb.argmax(2) != Po.argmax(2))[first:].any(axis=1).mean())
        rows.append(row)
    return pd.DataFrame(rows)
