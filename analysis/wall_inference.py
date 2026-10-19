''' Test whether agents infer where an unseen High wall is from the walls they can see.

The evidence is what an agent's 7 rays sense at slice onset (frame 0), before it moves: which walls are
hit, and whether each is lit High, lit Low or unlit. An ideal observer combines that with the trial
generator's wall separations (45, 90 or 180 degrees) to give a probability for where High is
(`analysis.agent_raycast.high_posterior`). Seeing a wall unlit rules it out, so seeing Low with every
wall within 90 degrees of it unlit leaves only the opposite wall, a 180 degree trial.

The comparison keeps to trials where heading for High is the better move whatever the agent values:
Low is sensed, High is not, and High is closer than Low. High is then both nearer and worth more.
'''

import numpy as np
import pandas as pd

import analysis.agent_raycast as ar
from analysis.race_estimate import ALCOVE_CENTRES

SEPARATION_DEG = {1: 45, 2: 90, 4: 180}


def onset_walls(row):
    ''' Return the walls (1-8) any ray hits at slice onset. Takes a row of a sensing table. '''

    return [w for w in range(1, 9) if row[f'ray_first_{w}'] == 0]


def onset_inference(seen, high, low, xy0, sep_prior=ar.SEP_PRIOR):
    ''' Return the ideal observer's reading at slice onset. Takes the walls sensed, the lit walls, the
        agent's onset position and the separation weights.

        Returns a dict: whether High and Low were sensed, the posterior on High's true wall, the number
        of candidate walls left, and whether every candidate is closer to the agent than the Low alcove. '''

    high_seen, low_seen = high in seen, low in seen          # matches wall_visibility_player_slice_onset on the same array
    post = ar.high_posterior(seen_high=high if high_seen else None, seen_low=low if low_seen else None,
                             seen_unlit=[w for w in seen if w not in (high, low)], sep_prior=sep_prior)
    d_low = np.linalg.norm(ALCOVE_CENTRES[low - 1] - xy0)
    closer = [np.linalg.norm(ALCOVE_CENTRES[w - 1] - xy0) < d_low for w in post]
    return dict(high_seen0=high_seen, low_seen0=low_seen, p_high_true=post.get(high, 0.0),
                n_candidates=len(post), all_candidates_closer=bool(post) and all(closer))


def inference_state(r):
    ''' Return the evidence state of one agent-trial at slice onset. Takes a row with the onset_inference
        fields. '''

    if r['high_seen0']:
        return 'High sensed'
    if not r['low_seen0']:
        return 'Low not sensed'
    if r['p_high_true'] >= 1 - 1e-9:
        return 'High position certain'
    if r['all_candidates_closer']:
        return 'High certainly closer, side uncertain'
    return 'uncertain'


def add_onset_inference(table, sep_prior=ar.SEP_PRIOR):
    ''' Return the trial table with the ideal observer's onset reading and evidence state added. Takes a
        table from opponent_response.build_trial_table (it keeps the ray_first columns). '''

    t = table.copy()
    rows = [onset_inference(onset_walls(r), int(r.high_wall), int(r.low_wall), np.array([r.x0, r.z0]), sep_prior)
            for _, r in t.iterrows()]
    for k in rows[0]:
        t[k] = [r[k] for r in rows]
    t['state'] = [inference_state(r) for _, r in t.iterrows()]
    t['high_closer'] = t['d_high0'] < t['d_low0']
    # separation comes from get_indices.get_wall_difference, stored by the sensing table
    t['separation_deg'] = t['wall_sep'].map(SEPARATION_DEG)
    return t
