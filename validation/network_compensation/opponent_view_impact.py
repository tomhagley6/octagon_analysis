# How much does the network change whether a player could see their opponent?
#
# Each player sees itself directly and the other player through the network, so the two screens differ
# from the truth and from each other. This reads the saved opponent view table and reports how far each
# screen sits from where the players really were at slice onset.
#
# The marginal visible rate is not the thing to look at. The flag is used as a condition in
# opponent_visibility, and a condition that is wrong on some trials attenuates whatever is conditioned on
# it, whichever way the errors point. Worse, the errors here are not spread evenly: they land on the
# trials where the opponent was already moving, and those trials have a different outcome. The last two
# sections measure both of these.
#
# Rebuild the table first with human_characterisation.collect_opponent_views, which takes a few minutes
# and is not automatic. Then run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python validation/network_compensation/opponent_view_impact.py

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import social_trial_lists, write_table
import globals
import analysis.human_characterisation as hc
import data_extraction.network_compensation.network_compensation as nr

VIEWER_NAME = {0: 'player 0 (host, renders 1 tick behind)', 1: 'player 1 (client, renders 2 ticks behind)'}


def onset_speeds():
    ''' Return each player's speed at slice onset, per trial, from the tick grid. Takes no inputs. '''

    ticks = hc.load_social_ticks()
    rows = []
    for i, trial_list in enumerate(social_trial_lists()):
        g = ticks[ticks.experiment == i]
        t = g['t'].to_numpy()
        speed = {p: np.r_[0.0, np.hypot(np.diff(g[f'p{p}_x'].to_numpy()),
                                        np.diff(g[f'p{p}_z'].to_numpy())) / nr.TICK_S] for p in (0, 1)}
        base = pd.concat(trial_list)['timeLocal'].iloc[0]
        for trial in trial_list:
            num = trial[globals.TRIAL_NUM].dropna()
            onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
            if not len(num) or not len(onset):
                continue
            at = (onset['timeLocal'].iloc[0] - base).total_seconds()
            k = int(np.clip(np.searchsorted(t, at), 0, len(t) - 1))
            rows.append(dict(experiment=i, trial_num=float(num.iloc[0]),
                             p0_speed=speed[0][k], p1_speed=speed[1][k]))
    return pd.DataFrame(rows)


def trial_winners():
    ''' Return the winning player for each trial. Takes no inputs. '''

    rows = []
    for i, trial_list in enumerate(social_trial_lists()):
        for trial in trial_list:
            num = trial[globals.TRIAL_NUM].dropna()
            got = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
            if len(num) and len(got):
                rows.append(dict(experiment=i, trial_num=float(num.iloc[0]),
                                 winner=got[globals.TRIGGER_CLIENT].iloc[0]))
    return pd.DataFrame(rows)


def main():
    d = hc.load_opponent_views().dropna(subset=['true_angle', 'seen_angle'])
    d = d.merge(trial_winners(), on=['experiment', 'trial_num'], how='left')
    d['flip'] = d.true_visible != d.seen_visible

    lines = ['# What each player saw of the other', '',
             f'{d.experiment.nunique()} social sessions, measured at slice onset. `true` is where both '
             'players really were, `seen` is what that viewer\'s screen showed. Regenerate with',
             '`validation/network_compensation/opponent_view_impact.py`.', '',
             '| viewer | trials | visible, true | visible, on screen | angle error, median | angle error, p90 | screen disagrees |',
             '|---|---|---|---|---|---|---|']
    for v in (0, 1):
        g = d[d.viewer == v]
        err = (g.true_angle - g.seen_angle).abs()
        flip = g.true_visible != g.seen_visible
        lines.append(f'| {VIEWER_NAME[v]} | {len(g)} | {g.true_visible.mean():.3f} | '
                     f'{g.seen_visible.mean():.3f} | {err.median():.2f} deg | {err.quantile(.9):.2f} deg | '
                     f'{100 * flip.mean():.2f}% |')
        print(f'viewer {v}: visible true {g.true_visible.mean():.3f} seen {g.seen_visible.mean():.3f}  '
              f'disagree {100 * flip.mean():.2f}%', flush=True)

    # split the disagreements, as a screen can hide an opponent or show one who had already moved away
    lines += ['', 'Where the screen and the truth disagree, they disagree both ways in about equal '
                  'numbers, so the network does not bias how often an opponent appears visible:', '',
              '| viewer | on screen but not really visible | really visible but not on screen |',
              '|---|---|---|']
    for v in (0, 1):
        g = d[d.viewer == v]
        flip = g.true_visible != g.seen_visible
        lines.append(f'| {VIEWER_NAME[v]} | {int((flip & g.seen_visible).sum())} | '
                     f'{int((flip & ~g.seen_visible).sum())} |')

    # paired over sessions, so the two players are compared on the same trials
    per = d.groupby(['experiment', 'viewer']).apply(
        lambda g: pd.Series({'flip': (g.true_visible != g.seen_visible).mean(),
                             'err': (g.true_angle - g.seen_angle).abs().median()}),
        include_groups=False).unstack()
    lines += ['', 'Paired over sessions, the extra tick the client carries does not make its screen '
                  'measurably worse than the host\'s:', '',
              '| measure | player 0 | player 1 | paired p |', '|---|---|---|---|']
    for key, name in (('flip', 'disagreement rate'), ('err', 'median angle error, deg')):
        a, b = per[(key, 0)], per[(key, 1)]
        lines.append(f'| {name} | {a.mean():.4f} | {b.mean():.4f} | {stats.ttest_rel(a, b)[1]:.2g} |')

    lines += ['', '## What the error does to an analysis that conditions on the flag', '',
              'The visible flag is a condition, so what matters is how often it is wrong, not whether the '
              'errors cancel in the marginal rate. A condition that is wrong on some trials pulls whatever '
              'is conditioned on it toward no effect, and the two error directions add to that rather than '
              'cancelling.', '',
              '| viewer | sensitivity | specificity | attenuation factor | conditional effects shrink by |',
              '|---|---|---|---|---|']
    for v in (0, 1):
        g = d[d.viewer == v]
        tv, sv = g.true_visible.to_numpy(bool), g.seen_visible.to_numpy(bool)
        sens = (sv & tv).sum() / tv.sum()
        spec = (~sv & ~tv).sum() / (~tv).sum()
        lines.append(f'| {VIEWER_NAME[v]} | {sens:.4f} | {spec:.4f} | {sens + spec - 1:.4f} | '
                     f'{100 * (2 - sens - spec):.1f}% |')

    lines += ['', '## The errors are not spread evenly', '',
              'A disagreement needs the opponent to have moved between when their position was sent and '
              'when it was drawn, so every one of them sits on a trial where the opponent was already '
              'running at slice onset. Those are not ordinary trials.', '',
              '| viewer | opponent speed at onset, disagreeing trials | agreeing trials | p |',
              '|---|---|---|---|']
    speeds = onset_speeds()
    m = d.merge(speeds, on=['experiment', 'trial_num'], how='inner')
    for v in (0, 1):
        g = m[m.viewer == v]
        other = g['p1_speed'] if v == 0 else g['p0_speed']
        a, b = other[g.flip], other[~g.flip]
        lines.append(f'| {VIEWER_NAME[v]} | {a.median():.2f} units/s | {b.median():.2f} units/s | '
                     f'{stats.mannwhitneyu(a, b)[1]:.1g} |')

    lines += ['', 'And they carry a different outcome with them, so the error is tied to the thing an '
                  'analysis would be trying to explain:', '',
              '| viewer | own-win rate, disagreeing trials | agreeing trials | p |', '|---|---|---|---|']
    for v in (0, 1):
        g = m[(m.viewer == v)].dropna(subset=['winner'])
        own = g.winner == g.viewer
        p = stats.chi2_contingency(pd.crosstab(g.flip, own))[1]
        lines.append(f'| {VIEWER_NAME[v]} | {own[g.flip].mean():.3f} | {own[~g.flip].mean():.3f} | '
                     f'{p:.2g} |')

    lines += ['', 'Slice onset is the most forgiving frame in the trial for this. Pooling both players, '
                  f'{100 * (speeds[["p0_speed", "p1_speed"]].to_numpy().ravel() < 0.5).mean():.0f}% of '
                  'players are within half a unit per second of standing still at onset, and a filter '
                  'applied to someone who is not moving changes nothing. That is why the median angle '
                  'error is zero. The same measure taken later in the trial would not be as safe.', '',
              'One asymmetry is worth keeping in mind when reading the table. For player 0 the `seen` '
                  'side is the log itself, as the logged player 1 values already are the host\'s rendered '
                  'copy, and the `true` side is the recovery. For player 1 it is the other way round: '
                  '`true` is player 0 as logged and `seen` is simulated. The modelled side is not the '
                  'same one for the two players.']

    write_table('opponent_view_impact.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
