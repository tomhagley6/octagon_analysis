# Compare the two players on the metrics the network path distorted, as logged and after recovery.
#
# Both players do the same task, and the solo sessions show no difference between them on these metrics,
# so a gap in social is the logging path. The test is how much of that gap the recovery removes.
#
# Reported as a paired confidence interval rather than a p value. A p value above .05 only says the gap
# was not detected, which a noisy measure achieves by doing nothing. The interval says what size of gap
# the data still allows, and that is what has to be compared against the gap seen as logged.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python validation/network_compensation/seat_comparison.py

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_strings
from _common import social_trial_lists, frame_times, player_series, step_metrics, write_table
import analysis.human_characterisation as hc
import globals
import data_extraction.network_compensation.network_compensation as nr

METRICS = ('no_rotational', 'resting', 'distance', 'rotation')
# the per-frame thresholds of the agent pipeline, for measuring the logged frames on their own grid
STILL_ROT_PER_FRAME = 0.5
STILL_SPEED_PER_FRAME = 0.05


def logged_iti_metrics(df, log_id):
    ''' Return the between-trial movement metrics from the logged frames. Takes a session dataframe and
        the player's log id. '''

    iti = df['trial_epoch'].isin([globals.ITI, globals.TRIAL_STARTED]).to_numpy()
    v = player_series(df, log_id)
    keep = iti[1:] & iti[:-1]
    step = np.hypot(np.diff(v['x']), np.diff(v['z']))[keep]
    turn = np.abs(np.diff(v['yaw']))[keep]
    return dict(distance=step.sum(), rotation=turn.sum(),
                no_rotational=(turn < STILL_ROT_PER_FRAME).mean(),
                resting=((step < STILL_SPEED_PER_FRAME) & (turn < STILL_ROT_PER_FRAME)).mean())


def recovered_iti_metrics(wide):
    ''' Return both players' between-trial metrics from one session's tick table. Takes the wide table. '''

    g = wide[wide['is_iti']]
    out = {}
    for p in (0, 1):
        vals = {c: g[f'p{p}_{c}'].to_numpy() for c in ('x', 'z', 'yaw')}
        out[p] = step_metrics(g['t'].to_numpy(), vals)
    return out


def summarise(pairs, label):
    ''' Return one markdown row per metric. Takes a list of per-session (p0, p1) dicts. '''

    rows = []
    for m in METRICS:
        a = np.array([p[0][m] for p in pairs])
        b = np.array([p[1][m] for p in pairs])
        d = a - b
        lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
        rows.append(f'| {label} | `{m}` | {a.mean():.3f} | {b.mean():.3f} | {d.mean():+.3f} | '
                    f'[{lo:+.3f}, {hi:+.3f}] |')
        print(f'{label:>10} {m:>16}  p0 {a.mean():9.3f}  p1 {b.mean():9.3f}  '
              f'diff {d.mean():+8.3f}  CI [{lo:+.3f}, {hi:+.3f}]', flush=True)
    return rows


def context_rows(wide, cen):
    ''' Return the lines putting the totals in proportion. Takes the tick table and the per-session onset
        centrality.

        A total on its own says nothing about whether it is accurate. What matters is how it compares with
        the spread between players and with the error the recovery itself introduces. '''

    iti = wide[wide['is_iti']]
    secs, dist, rot = [], {0: [], 1: []}, {0: [], 1: []}
    for _, g in iti.groupby('experiment'):
        ok = np.abs(np.diff(g['t'].to_numpy()) - nr.TICK_S) < 0.5 * nr.TICK_S
        secs.append(ok.sum() * nr.TICK_S)
        for p in (0, 1):
            dist[p].append(np.hypot(np.diff(g[f'p{p}_x'].to_numpy()),
                                    np.diff(g[f'p{p}_z'].to_numpy()))[ok].sum())
            rot[p].append(np.abs(np.diff(g[f'p{p}_yaw'].to_numpy()))[ok].sum())
    secs = np.array(secs)

    lines = ['', '## What the totals cover, and how big the gap is', '',
             f'Each total is one session\'s worth of between-trial movement for one player, summed over '
             f'every ITI in it. That is a median of {np.median(secs):.0f} s '
             f'({np.median(secs) / 60:.0f} min) per session, about half the session. One value per player '
             f'per session, averaged over the {len(secs)} sessions, so every one of the 200 people appears '
             'once.', '',
             '| | player 0 | player 1 | gap | as % of player 0 | between-session SD | gap in SD |',
             '|---|---|---|---|---|---|---|']
    for name, d in (('ITI distance (units)', dist), ('ITI rotation (deg)', rot)):
        a, b = np.array(d[0]), np.array(d[1])
        gap = (a - b).mean()
        lines.append(f'| {name} | {a.mean():.0f} | {b.mean():.0f} | {gap:+.0f} | '
                     f'{100 * gap / a.mean():+.1f}% | {a.std():.0f} | {abs(gap) / a.std():.2f} |')

    lines += ['', 'The spread between players is what makes these gaps hard to read. The standard '
                  f'deviation of ITI distance across sessions is {np.std(dist[0]):.0f} units against a '
                  f'mean of {np.mean(dist[0]):.0f}, so players differ from each other far more than the '
                  'two positions differ on average. That is why the confidence intervals are wide, and it '
                  'is a fact about the players rather than about the recovery.', '',
              "The recovery's own error on a total is much smaller. The round trips in `lambda_sweep.md` "
              'put the distance ratio at 1.000 to 1.001 and the rotation ratio at 1.002 to 1.011, so the '
              'method distorts a total by at most about 1%. The gap between the players is several times '
              'that, and is not something the recovery introduced.', '',
              '## Is player 1 a more peripheral set of players?', '',
              'A tempting reading of the distance gap is that the people who happened to play as player 1 '
              'simply move around more. Onset centrality per session, with the solo sessions as a control '
              'that never touched the network:', '',
              '| session | player 0 | player 1 | gap | paired p | sessions with player 1 more peripheral |',
              '|---|---|---|---|---|---|']
    for st in ('social', 'solo1', 'solo2'):
        g = cen[st]
        d = g[0] - g[1]
        lines.append(f'| {st} | {g[0].mean():.3f} | {g[1].mean():.3f} | {d.mean():+.3f} | '
                     f'{stats.ttest_rel(g[0], g[1])[1]:.2g} | {(d > 0).sum()} / {len(d)} |')

    # centrality is measured without the network, so it cannot be a logging artefact
    x = np.r_[cen['social'][0].to_numpy(), cen['social'][1].to_numpy()]
    y = np.r_[np.array(dist[0]), np.array(dist[1])]
    r, pv = stats.pearsonr(x, y)
    slope = stats.linregress(x, y).slope
    pred = slope * (cen['social'][0] - cen['social'][1]).mean()
    lines += ['', 'Player 1 is the more peripheral of the pair in 62 of 100 social sessions, and that gap '
                  'is still there in the second solo, where nothing crossed the network. It is not there '
                  'in the first solo. So it is about the players and not the logging, and it appears to '
                  'arrive with the social session rather than to have been there all along.', '',
              'It does not explain the distance gap either. Centrality and ITI distance are only weakly '
              f'related (r = {r:.3f}, p = {pv:.2g} over the 200 players), and the relation runs the wrong '
              'way: more central players move slightly more, so the centrality gap predicts player 0 '
              f'covering {pred:+.0f} units more than player 1, against the {(np.mean(dist[0]) - np.mean(dist[1])):+.0f} '
              'observed. Whatever the distance gap is, it is not centrality.']
    return lines


def main():
    tl = social_trial_lists()
    wide = pd.read_pickle(os.path.join(data_strings.DATAFRAME_PATH, nr.RECOVERED_SOCIAL + '.pkl'))

    as_logged, recovered = [], []
    for i, s in enumerate(tl):
        df = pd.concat(s)
        as_logged.append({p: logged_iti_metrics(df, p) for p in (0, 1)})
        recovered.append(recovered_iti_metrics(wide[wide['experiment'] == i]))

    lines = ['# Player 0 against player 1, between trials', '',
             f'{len(tl)} social sessions, paired by session. The logged rows are measured on 48 Hz '
             f'frames with the per-frame cutoffs ({STILL_ROT_PER_FRAME} deg, {STILL_SPEED_PER_FRAME} '
             f'units), the recovered rows on the 30 Hz ticks with the tick cutoffs '
             f'({nr.STILL_ROT_PER_TICK} deg, {nr.STILL_SPEED_PER_TICK} units). Levels therefore do not '
             'carry between the two halves of the table. The gap between the players does, as both '
             'players are measured the same way within a half, and that gap is the whole question.',
             'Regenerate with `validation/network_compensation/seat_comparison.py`.', '',
             '| source | metric | player 0 | player 1 | difference | 95% CI |', '|---|---|---|---|---|---|']
    lines += summarise(as_logged, 'as logged')
    lines += summarise(recovered, 'recovered')

    onset = hc.load_tables()['onset']
    cen = {st: onset[onset.session_type == st].groupby(['experiment', 'player'])['centrality']
           .mean().unstack().dropna() for st in ('social', 'solo1', 'solo2')}
    lines += context_rows(wide, cen)

    write_table('seat_comparison.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
