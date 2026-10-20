#!/usr/bin/env python
# coding: utf-8

# Measure how a human player's waiting position changes over a session, to label mixed players.
#
# Every measure uses onset centrality in play order, with no boundary. trend_r2 is the share of the
# player's variation that a straight line over the session explains. runs_r1 is the correlation between
# consecutive trials once that line is removed, so steady drift does not count as runs. local_jump is the
# largest change between the trials just before and just after any point. Each is tested against the
# player's own trials in shuffled order.
#
# The notebook demos/strategy_characterisation/humans/mixed_strategy_characterisation.ipynb (section 5)
# holds the plots and the written-up results. Everything that computes a number lives here.

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

import analysis.human_characterisation as hc


JUMP_WINDOW = 10            # trials either side of a point for the local jump
MIN_SEQ_TRIALS = 20         # fewest trials a session needs for the sequence measures (two jump windows)
N_SHUFFLE = 1000            # shuffled orders per player for the tests
SEQUENCE_KINDS = ('mixer', 'one switch', 'noisy one switch', 'long runs', 'drifter', 'hoverer')
ALPHA = 0.05                # shuffle-test p below which order structure counts as present
RUNS_MIN = 0.46             # runs_r1 for runs of about 5+ trials, allowing for real players' spread within a position
TREND_MIN = 0.10            # trend_r2 for a drifter (above simulated mixers' and hoverers' 90th percentile)
DRIFT_MIN = 0.135           # robust (Theil-Sen) change over the session a drifter needs (about half the shift cut)
JUMP_MIN = 0.30             # local jump size for a single switch (below simulated one-switch players, above mixers)
MIN_SECTION = 5             # fewest trials in each stable section of a one-switch player
MAX_TRANSITION = 0.2        # longest transition between the two sections, as a share of the session
N_SHUFFLE_SECTIONS = 200    # shuffled orders for the two-section test (each refits every split)
PURITY_MIN = 0.8            # share of a section's trials that must sit nearer its own level than the other
EXCURSION_MAX = 5           # longest unbroken run of trials nearer the other level allowed inside a section
SIM_EXCURSION_RATE = 0.05   # simulated noisy one switch: chance per trial of starting a trip to the other position
SIM_EXCURSION_LEN = (1, 3)  # its trips last 1 to 3 trials
SIM_CLOSENESS = 0.2         # its correlation between consecutive trials' scatter within a position
FLAT_SPLIT_RATIO = hc.ONE_FLAT_SPLIT_RATIO  # gap_vs_sd above this counted as two positions in the first rules
SHIFT_Q = 0.95              # a shift is a trial-to-trial difference bigger than this share of central players' within-centre ones
MIDDLE_Q = 0.05             # near the middle = above peripheral players' 95th and below central players' 5th percentile mean
LABELS = ('one switch', 'long runs', 'drifter', 'hoverer', 'mixer', 'short runs', 'unclear')


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------

def _measures(Y, window):
    ''' Return trend_r2, trend_change, resid_sd, runs_r1, local_jump and jump index for each row of Y.
        Takes an m x n array of series (one per row, in play order) and the jump window. '''

    m, n = Y.shape
    t = np.arange(n) - (n - 1) / 2
    # fit a straight line to each series
    ybar = Y.mean(axis=1, keepdims=True)
    slope = (Y - ybar) @ t / (t @ t)
    resid = Y - ybar - slope[:, None] * t
    ss_total = ((Y - ybar) ** 2).sum(axis=1)
    ss_resid = (resid ** 2).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        trend_r2 = np.where(ss_total > 0, 1 - ss_resid / ss_total, 0.0)
        # correlate each residual with the next one
        runs_r1 = np.where(ss_resid > 0, (resid[:, :-1] * resid[:, 1:]).sum(axis=1) / ss_resid, 0.0)
    # compare the mean of the window after each point with the window before it
    c = np.concatenate([np.zeros((m, 1)), np.cumsum(Y, axis=1)], axis=1)
    k = np.arange(window, n - window + 1)
    jumps = (c[:, k + window] - c[:, k]) / window - (c[:, k] - c[:, k - window]) / window
    at = np.argmax(np.abs(jumps), axis=1)
    return dict(trend_r2=trend_r2, trend_change=slope * (n - 1), resid_sd=np.sqrt(ss_resid / n), runs_r1=runs_r1,
                local_jump=jumps[np.arange(m), at], jump_index=k[at])


def series_measures(y, window=JUMP_WINDOW):
    ''' Return the sequence measures of one series. Takes the centrality series in play order and the
        jump window.

        trend_r2 is the share of variation a straight line explains (0 to 1). trend_change is the line's
        rise from first to last trial. resid_sd is the spread of trials around the line. runs_r1 is the
        lag-1 autocorrelation of the residuals (0 = no runs, positive = runs). local_jump is the
        largest signed change between the window after a point and the window before it, and jump_at
        is where it falls (0 to 1 through the session). '''

    y = np.asarray(y, dtype=float)
    out = {key: float(val[0]) for key, val in _measures(y[None, :], window).items()}
    out['jump_at'] = out.pop('jump_index') / len(y)
    return out


def _two_section_core(Y, min_section, max_transition):
    ''' Return the best two-section fit of each row of Y: its R2, section A end, section B start and the two
        levels. Takes an m x n array of series in play order, the fewest trials per section and the longest
        transition (share of the session).

        Section A runs from the first trial up to (not including) a, section B from b to the last trial.
        Trials a to b - 1 are the transition, fitted as a straight move from A's level to B's. Every
        allowed (a, b) is scored from running sums, so no trial loop is needed. '''

    m, n = Y.shape
    longest = int(np.floor(max_transition * n))
    pairs = [(a, b) for a in range(min_section, n - min_section + 1)
             for b in range(a, min(a + longest, n - min_section) + 1)]
    a, b = (np.array(v) for v in zip(*pairs))
    t = np.arange(n)
    # build running sums of y, t * y and y^2 (with a leading zero)
    pad = lambda x: np.concatenate([np.zeros((m, 1)), np.cumsum(x, axis=1)], axis=1)
    c0, c1, c2 = pad(Y), pad(Y * t), pad(Y ** 2)
    # get each section's level and spread
    sum_a, sum_b = c0[:, a], c0[:, [n]] - c0[:, b]
    level_a, level_b = sum_a / a, sum_b / (n - b)
    sse = (c2[:, a] - sum_a * level_a) + ((c2[:, [n]] - c2[:, b]) - sum_b * level_b)
    # add the misfit of the transition trials, predicted on a straight line between the two levels
    L = (b - a).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        s_y, s_ty, s_yy = c0[:, b] - c0[:, a], c1[:, b] - c1[:, a], c2[:, b] - c2[:, a]
        s_yw = (s_ty - (a - 1) * s_y) / (L + 1)
        s_w, s_ww = L / 2, L * (2 * L + 1) / (6 * (L + 1))
        d = level_b - level_a
        ramp = s_yy - 2 * (level_a * s_y + d * s_yw) + L * level_a ** 2 + 2 * level_a * d * s_w + d ** 2 * s_ww
    sse = sse + np.where(L > 0, ramp, 0.0)
    ss_total = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        r2 = np.where(ss_total > 0, 1 - sse / ss_total, 0.0)
    best = np.argmax(r2, axis=1)
    rows = np.arange(m)
    return r2[rows, best], a[best], b[best], level_a[rows, best], level_b[rows, best]


def two_section_fit(y, min_section=MIN_SECTION, max_transition=MAX_TRANSITION):
    ''' Return the best two-section fit of one series. Takes the centrality series in play order, the fewest
        trials per section and the longest transition (share of the session).

        two_section_r2 is the share of variation the fit explains. section_a_end and section_b_start
        bound the transition (equal for an instant switch), level_a and level_b are the section means,
        section_gap is level_b - level_a, and switch_at is the transition's midpoint (0 to 1). '''

    y = np.asarray(y, dtype=float)
    r2, a, b, la, lb = (v[0] for v in _two_section_core(y[None, :], min_section, max_transition))
    return dict(two_section_r2=float(r2), section_a_end=int(a), section_b_start=int(b), transition_len=int(b - a),
                level_a=float(la), level_b=float(lb), section_gap=float(lb - la), switch_at=(a + b) / 2 / len(y))


def _two_section_residuals(y, fit):
    ''' Return each trial's distance from its two-section fit. Takes the series and its fit. '''

    n, a, b = len(y), fit['section_a_end'], fit['section_b_start']
    pred = np.where(np.arange(n) < a, fit['level_a'], fit['level_b']).astype(float)
    if b > a:
        pred[a:b] = fit['level_a'] + fit['section_gap'] * np.arange(1, b - a + 1) / (b - a + 1)
    return y - pred


def section_purity(y, fit):
    ''' Return how much of each section sits at its own level, and the longest trip to the other level.
        Takes the series and its two-section fit.

        A trial is nearer its own section's level if it sits on that level's side of the midpoint
        between the two levels. purity_a and purity_b are the shares of each section's trials nearer
        their own level, and purity_min the smaller. longest_excursion is the longest unbroken run of
        trials, inside either section, nearer the other level. Transition trials are not counted.
        Consecutive trials sitting close together do not lower purity, only trials that cross to the
        other level do. '''

    a, b = fit['section_a_end'], fit['section_b_start']
    mid = (fit['level_a'] + fit['level_b']) / 2
    out = {}
    longest = 0
    for name, part, level in (('a', y[:a], fit['level_a']), ('b', y[b:], fit['level_b'])):
        # flag trials on the other level's side of the midpoint
        away = np.sign(part - mid) != np.sign(level - mid)
        out[f'purity_{name}'] = float(1 - away.mean())
        run = 0
        for flag in away:
            run = run + 1 if flag else 0
            longest = max(longest, run)
    out['purity_min'] = min(out['purity_a'], out['purity_b'])
    out['longest_excursion'] = int(longest)
    return out


def two_section_table(onset, n_shuffle=N_SHUFFLE_SECTIONS, min_section=MIN_SECTION, max_transition=MAX_TRANSITION,
                      min_trials=MIN_SEQ_TRIALS, random_state=0):
    ''' Return each player's best two-section fit, with two shuffle tests. Takes the onset table, the shuffle
        count, the fewest trials per section, the longest transition, the fewest trials and a seed.

        p_two_section asks whether two sections fit better than they would in a random order of the same
        trials. Every shuffle is refitted over every split, as searching always finds some fit.
        resid_runs_r1 is the lag-1 autocorrelation of the trials' distances from the fit, and
        p_resid_runs its shuffle test (the distances in random order). No runs left means each section
        is stable. Runs left mean more than one switch. purity_a, purity_b, purity_min and
        longest_excursion come from section_purity. '''

    rng = np.random.default_rng(random_state)
    rows = []
    for key, grp in onset.sort_values('trial_pos').groupby(list(hc.KEYS)):
        y = grp['centrality'].to_numpy(dtype=float)
        if len(y) < max(min_trials, 2 * min_section):
            continue
        fit = two_section_fit(y, min_section, max_transition)
        # refit two sections on shuffled orders of the same trials
        null_r2 = _two_section_core(rng.permuted(np.tile(y, (n_shuffle, 1)), axis=1), min_section, max_transition)[0]
        # test the leftover distances for runs, against the same distances in random order
        e = _two_section_residuals(y, fit)
        r1 = lambda E: (E[..., :-1] * E[..., 1:]).sum(axis=-1) / (E ** 2).sum(axis=-1)
        real_r1 = float(r1(e))
        null_r1 = r1(rng.permuted(np.tile(e, (n_shuffle, 1)), axis=1))
        rows.append(dict(zip(hc.KEYS, key), **fit, **section_purity(y, fit),
                         p_two_section=(1 + np.sum(null_r2 >= fit['two_section_r2'] - 1e-12)) / (1 + n_shuffle),
                         resid_runs_r1=real_r1, p_resid_runs=(1 + np.sum(null_r1 >= real_r1 - 1e-12)) / (1 + n_shuffle)))
    return pd.DataFrame(rows)


def run_length_guide(r1):
    ''' Return the average run length that a lag-1 autocorrelation r1 means for a player with two exact
        positions, 2 / (1 - r1). Takes r1. Spread within each position lowers r1, so this is a rough guide. '''

    return 2 / (1 - np.asarray(r1, dtype=float))


def sequence_table(onset, n_shuffle=N_SHUFFLE, window=JUMP_WINDOW, min_trials=MIN_SEQ_TRIALS, random_state=0):
    ''' Return the sequence measures per player per session type, each with a shuffle p value.
        Takes the onset table, the shuffle count, the jump window, the fewest trials and a seed.

        Each shuffle puts the player's own trials in a random order, keeping their values. p_trend,
        p_runs and p_jump are the share of shuffles at least as extreme as the real order (trend_r2,
        runs_r1 and the size of local_jump), counting the real order once. The jump is maximised over
        points in every shuffle too, as picking the largest of many differences always finds one. The p
        values are per player and uncorrected. Sessions with fewer than min_trials are dropped. '''

    rng = np.random.default_rng(random_state)
    rows = []
    for key, grp in onset.sort_values('trial_pos').groupby(list(hc.KEYS)):
        y = grp['centrality'].to_numpy(dtype=float)
        if len(y) < min_trials:
            continue
        real = series_measures(y, window)
        # recompute every measure on shuffled orders of the same trials
        shuffled = _measures(rng.permuted(np.tile(y, (n_shuffle, 1)), axis=1), window)
        row = dict(zip(hc.KEYS, key), n_trials=len(y), mean_centrality=y.mean(), **real)
        for name, real_val, null in (('trend', real['trend_r2'], shuffled['trend_r2']),
                                     ('runs', real['runs_r1'], shuffled['runs_r1']),
                                     ('jump', abs(real['local_jump']), np.abs(shuffled['local_jump']))):
            row[f'p_{name}'] = (1 + np.sum(null >= real_val - 1e-12)) / (1 + n_shuffle)
        rows.append(row)
    return pd.DataFrame(rows)


def robust_trend_table(onset, min_trials=MIN_SEQ_TRIALS):
    ''' Return each player's robust change over the session, per session type. Takes the onset table and
        the fewest trials.

        drift_change is the Theil-Sen slope (median of the slopes between every pair of trials) times the
        session length, i.e. how far the usual position moves from the first trial to the last (negative =
        toward the wall). Brief trips barely move it, so trips bunched early or late do not count as drift. '''

    rows = []
    for key, grp in onset.sort_values('trial_pos').groupby(list(hc.KEYS)):
        y = grp['centrality'].to_numpy(dtype=float)
        if len(y) < min_trials:
            continue
        slope = theilslopes(y, np.arange(len(y)))[0]
        rows.append(dict(zip(hc.KEYS, key), drift_change=slope * (len(y) - 1)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def shift_noise_cut(onset, labels, boundary, q=SHIFT_Q, label_col='fixed'):
    ''' Return, per session type, the trial-to-trial difference in centrality that a one-position player
        rarely exceeds. Takes the onset table, a player table with labels, the trial boundary, the quantile
        and the label column.

        Every pair of consecutive trials gives a difference, as a player never waits at exactly the same
        centrality twice. The cut is the q quantile of central players' differences between consecutive
        trials that both sit at or above the boundary, so trips to the wall do not count. A difference
        bigger than this is an intentional shift rather than noise around one position. '''

    central = labels.loc[labels[label_col] == 'central', list(hc.KEYS)]
    sub = onset.merge(central, on=list(hc.KEYS)).sort_values('trial_pos')
    cuts = {}
    for st, grp in sub.groupby('session_type'):
        changes = []
        for _, g in grp.groupby(['experiment', 'player']):
            y = g['centrality'].to_numpy()
            # keep differences where both trials sit on the centre side
            keep = (y[1:] >= boundary) & (y[:-1] >= boundary)
            changes.append(np.abs(np.diff(y))[keep])
        cuts[st] = float(np.quantile(np.concatenate(changes), q))
    return pd.Series(cuts)


def shift_table(onset, shift_cut):
    ''' Return, per player per session type, how often their trial-to-trial difference is bigger than the
        shift cut. Takes the onset table and the cut per session type.

        n_changes is the number of differences (one per pair of consecutive trials). shift_share is the
        share of them bigger than the cut. '''

    shift_cut = pd.Series(shift_cut)
    rows = []
    for key, grp in onset.sort_values('trial_pos').groupby(list(hc.KEYS)):
        change = np.abs(np.diff(grp['centrality'].to_numpy()))
        n, k = len(change), int(np.sum(change > shift_cut[key[2]]))
        rows.append(dict(zip(hc.KEYS, key), n_changes=n, n_shifts=k, shift_share=k / n if n else np.nan))
    return pd.DataFrame(rows)


def shift_reference(shifts, labels, q=SHIFT_Q, label_col='fixed'):
    ''' Return, per session type, the shift share that one-position players rarely exceed.
        Takes the shift table, a player table with labels, the quantile and the label column.

        It is the q quantile of central players' own shift shares, measured the same way as everyone
        else's. A player above it shifts beyond noise more often than nearly all one-position players do. '''

    central = labels.loc[labels[label_col] == 'central', list(hc.KEYS)]
    return shifts.merge(central, on=list(hc.KEYS)).groupby('session_type')['shift_share'].quantile(q)


def middle_range(seq, q=MIDDLE_Q, label_col='fixed'):
    ''' Return, per session type, the (low, high) range of mean centrality that counts as near the middle.
        Takes a table with session_type, mean_centrality and labels, the quantile and the label column.

        low is the 1 - q quantile of peripheral players' means, high the q quantile of central players'
        means. So only about q of each group sits inside the range, and a hoverer waits between the two
        groups. q = 0 gives the group extremes. '''

    means = seq.groupby(['session_type', label_col])['mean_centrality']
    lo = means.quantile(1 - q).xs('peripheral', level=label_col)
    hi = means.quantile(q).xs('central', level=label_col)
    return {st: (float(lo[st]), float(hi[st])) for st in lo.index}


def assign_labels(seq, narrow_cut, middle, shift_ref=None, gap_min=None, runs_min=RUNS_MIN, trend_min=TREND_MIN,
                  drift_min=None,
                  jump_min=None, one_switch_rule='sections', stable_rule='purity', purity_min=PURITY_MIN,
                  excursion_max=EXCURSION_MAX, short_runs=True, allow_short_runs=True, mixer_rule='shifts',
                  alpha=ALPHA, flat_ratio=FLAT_SPLIT_RATIO):
    ''' Return the sequence table with a flag for each rule and one label per player.
        Takes the sequence table (with the two_section_table and shift_table columns, or gap_vs_sd for the
        first rules), the narrow cut, the (low, high) middle range, the shift reference and the smallest
        section gap per session type (dicts or Series), the thresholds, and the rule choices.

        - one switch (one_switch_rule='sections'): the two-section fit passes its shuffle test, the two
          section levels differ by at least gap_min, each section is stable, and two sections explain
          more of the variation than a straight line (two_section_r2 > trend_r2), which keeps a slow drift
          out. With stable_rule='purity' (default) a stable section has purity_min >= purity_min and no
          excursion longer than excursion_max, so brief trips and consecutive trials sitting close together
          are allowed. With 'runs' no runs
          may be left once the fit is removed (the earlier, stricter check). one_switch_rule=None drops
          this label (the first rules).
        - long runs: runs pass the shuffle test and runs_r1 >= runs_min. With jump_min set, a significant
          local jump of at least jump_min also counts (the earlier rules). By default the jump only
          describes switchers.
        - drifter: the trend passes the shuffle test and trend_r2 >= trend_min. With drift_min set, the robust
          change (drift_change, from robust_trend_table) must also be at least drift_min in size, so trips
          bunched early or late that tilt the straight line do not count as drift.
        - hoverer: the trend test not passed, resid_sd <= the narrow cut, and the mean inside the middle range.
        - mixer: the trend test not passed, wider than the narrow cut, and (mixer_rule='shifts') shift_share
          above the shift reference, so the player shifts beyond noise more often than nearly all
          one-position players. They move back and forth beyond noise, in no particular order. With mixer_rule='point' (the first
          rules) a mixer needs gap_vs_sd > flat_ratio instead, and other wide players are 'broad'.
        - short runs: runs pass the shuffle test, but the player fits none of the labels above. With
          short_runs=False these players are unclear, as in the first rules.
        - unclear: weak or mixed signals, e.g. a trend that passes its test but is too weak for drifter.
        Labels are given in that order, so a player gets the first one they fit. By default, runs that pass
        the test do not block drifter, hoverer or mixer, as labels are descriptive. With
        allow_short_runs=False, drifter, hoverer and mixer also need the runs test not passed (the earlier
        rules). has_short_runs flags
        short runs beside any label. For a one-switch player it means runs left inside the two sections
        (p_resid_runs < alpha), as the switch itself leaves runs around a straight line. For anyone else it
        means runs that pass the test without being long runs.

        Besides the one priority label, fits_<label> records whether the player passes each label's own
        rule, ignoring the order (fits_short_runs is has_short_runs), and all_labels lists every label
        passed in priority order ('unclear' if none). '''

    out = seq.copy()
    narrow_cut, middle = pd.Series(narrow_cut), pd.Series(middle)
    runs_sig, trend_sig = out['p_runs'] < alpha, out['p_trend'] < alpha
    lo = out['session_type'].map(lambda st: middle[st][0])
    hi = out['session_type'].map(lambda st: middle[st][1])
    # flag each rule
    out['is_runs'] = runs_sig & (out['runs_r1'] >= runs_min - 1e-12)
    out['is_jump'] = ((out['p_jump'] < alpha) & (out['local_jump'].abs() >= jump_min - 1e-12)
                      if jump_min is not None else False)
    out['is_switcher'] = out['is_runs'] | out['is_jump']
    if one_switch_rule == 'sections':
        if stable_rule == 'purity':
            out['stable_sections'] = ((out['purity_min'] >= purity_min - 1e-12)
                                      & (out['longest_excursion'] <= excursion_max))
        else:
            out['stable_sections'] = out['p_resid_runs'] >= alpha
        out['is_one_switch'] = ((out['p_two_section'] < alpha) & out['stable_sections']
                                & (out['section_gap'].abs() >= out['session_type'].map(pd.Series(gap_min)) - 1e-12)
                                & (out['two_section_r2'] > out['trend_r2']))
    else:
        out['is_one_switch'] = False
    out['is_trend'] = (trend_sig & (out['trend_r2'] >= trend_min - 1e-12)
                       & (True if allow_short_runs else ~runs_sig)
                       & ((out['drift_change'].abs() >= drift_min - 1e-12) if drift_min is not None else True))
    out['no_order'] = ~runs_sig & ~trend_sig
    # let runs sit alongside hoverer and mixer, unless the earlier rules are asked for
    open_to = ~trend_sig & (True if allow_short_runs else ~runs_sig)
    out['is_narrow'] = out['resid_sd'] <= out['session_type'].map(narrow_cut) + 1e-12
    out['in_middle'] = (out['mean_centrality'] > lo) & (out['mean_centrality'] < hi)
    wide = open_to & ~out['is_narrow']
    conditions = [out['is_one_switch'], out['is_switcher'], out['is_trend'],
                  open_to & out['is_narrow'] & out['in_middle']]
    names = ['one switch', 'long runs', 'drifter', 'hoverer']
    # give wide players with no order the mixer label by shifts, or split them by the ratio (first rules)
    if mixer_rule == 'shifts':
        out['shifts_beyond_noise'] = out['shift_share'] > out['session_type'].map(pd.Series(shift_ref))
        conditions += [wide & out['shifts_beyond_noise']]
        names += ['mixer']
    else:
        conditions += [wide & (out['gap_vs_sd'] > flat_ratio), wide & (out['gap_vs_sd'] <= flat_ratio)]
        names += ['mixer', 'broad']
    conditions += [runs_sig & short_runs]
    names += ['short runs']
    out['label'] = np.select(conditions, names, 'unclear')
    # flag short runs beside any label, inside the two sections for a one-switch player
    within = (out['p_resid_runs'] < alpha) if 'p_resid_runs' in out else False
    out['has_short_runs'] = np.where(out['label'] == 'one switch', within, runs_sig & ~out['is_runs'])
    # record every label each player passes, ignoring the priority order
    fits = dict(zip(names, conditions))
    fits['short runs'] = out['has_short_runs']
    columns = []
    for name in names:
        col = 'fits_' + name.replace(' ', '_')
        out[col] = np.broadcast_to(np.asarray(fits[name], dtype=bool), len(out))
        columns.append((name, col))
    passed = out[[col for _, col in columns]].to_numpy()
    out['all_labels'] = [', '.join(name for (name, _), ok in zip(columns, row) if ok) or 'unclear' for row in passed]
    return out


# ---------------------------------------------------------------------------
# simulated players
# ---------------------------------------------------------------------------

def _add_excursions(state, rng, rate=SIM_EXCURSION_RATE, length=SIM_EXCURSION_LEN):
    ''' Return a 0/1 position series with brief trips to the other position added. Takes the series, a
        generator, the chance per trial of starting a trip and the (shortest, longest) trip length. '''

    state = state.copy()
    t = 0
    while t < len(state):
        if rng.random() < rate:
            # flip the next few trials to the other position
            n = int(rng.integers(length[0], length[1] + 1))
            state[t:t + n] = 1 - state[t:t + n]
            t += n
        t += 1
    return state


def _correlated_scatter(n, sd, rho, rng):
    ''' Return n scatter values with SD sd and correlation rho between consecutive values. Takes the count,
        the SD, the correlation and a generator. '''

    e = np.empty(n)
    e[0] = rng.normal(0, sd)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
    return e


def simulate_sequences(n_trials, n_players, params, rng, kinds=SEQUENCE_KINDS):
    ''' Return simulated onset trials for each kind of mixed player, and the kind of each player.
        Takes trials per player, players per kind, the settings and a generator.

        params holds low and high (the two positions), spread (SD around a position), hover_mean and
        hover_sd (the hoverer's position and spread) and run_length (mean trials per run).
        - mixer picks a position at random on every trial.
        - noisy one switch is one switch with brief trips (chance SIM_EXCURSION_RATE per trial of
          starting a trip of SIM_EXCURSION_LEN trials to the other position), and consecutive trials'
          scatter around a position correlated SIM_CLOSENESS.
        - one switch holds one position, then the other. The move takes 1 trial to 20% of the session
          and can start anywhere that leaves 5 trials either side. During the move, the chance of using
          the new position rises steadily.
        - long runs moves between the positions in runs, switching with chance 1 / run_length per trial.
        - drifter moves steadily from one position to the other, with hover_sd around its path (a
          narrow band that moves).
        - hoverer waits at hover_mean with hover_sd.
        Direction and starting position are random. Values are clipped to [0, 1]. '''

    lo, hi, spread = params['low'], params['high'], params['spread']
    rows, kind_rows = [], []
    t = np.arange(n_trials)
    for e, kind in enumerate(kinds):
        for p in range(n_players):
            start = rng.integers(2)
            # draw which position each trial uses, or the path a drifter or hoverer follows
            if kind == 'mixer':
                state = rng.integers(2, size=n_trials)
            elif kind in ('one switch', 'noisy one switch'):
                # move over 1 trial to 20% of the session, starting anywhere that leaves 5 trials either side
                length = int(rng.integers(1, max(1, int(MAX_TRANSITION * n_trials)) + 1))
                begin = int(rng.integers(MIN_SECTION, n_trials - MIN_SECTION - length + 1))
                share = np.clip((t - begin + 1) / (length + 1), 0, 1)
                state = np.where(rng.random(n_trials) < np.where(start, 1 - share, share), 1, 0)
                if kind == 'noisy one switch':
                    state = _add_excursions(state, rng)
            elif kind == 'long runs':
                flips = rng.random(n_trials) < 1 / params['run_length']
                state = (start + np.cumsum(flips)) % 2
            if kind in ('mixer', 'one switch', 'long runs'):
                y = np.where(state == 1, hi, lo) + rng.normal(0, spread, n_trials)
            elif kind == 'noisy one switch':
                y = np.where(state == 1, hi, lo) + _correlated_scatter(n_trials, spread, SIM_CLOSENESS, rng)
            elif kind == 'drifter':
                ends = (lo, hi) if start else (hi, lo)
                y = np.linspace(*ends, n_trials) + rng.normal(0, params['hover_sd'], n_trials)
            else:
                y = params['hover_mean'] + rng.normal(0, params['hover_sd'], n_trials)
            session_type = f'sim{n_trials}'
            rows.append(pd.DataFrame(dict(experiment=e, player=p, session_type=session_type, trial_pos=t,
                                          centrality=np.clip(y, 0, 1))))
            kind_rows.append(dict(experiment=e, player=p, session_type=session_type, kind=kind))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(kind_rows)
