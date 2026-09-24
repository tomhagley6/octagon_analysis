#!/usr/bin/env python
# coding: utf-8

# Characterise human players by their waiting strategy.
#
# One experiment is one pair on one day, holding a social session and four solo sessions. 'session'
# always means one of first solo, social or second solo, and 'experiment' the pair they belong to.
#
# Every player has a first solo, a social and a second solo session. Each is reduced to the onset
# table (one row per trial per player, measured at slice onset), then to the player table (one row per
# player per session). A third table, the session table, holds one row per session and describes the
# protocol that session ran under. The three sessions are analysed separately, because
# the social context differs between them, and compared only after.
#
# Centrality uses the same definition as the RL agent characterisation in model_characterisation.py
# (1 = arena centre, 0 = alcove back), so humans and agents share one axis.
#
# The notebook demos/strategy_characterisation/humans/strategy_characterisation.ipynb holds the plots
# and the written-up results. Everything that computes a number lives here.

import os
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.mixture import GaussianMixture

import data_strings
import globals
import data_extraction.data_saving as data_saving
import data_extraction.get_indices as get_indices
import analysis.model_characterisation as mc
import analysis.opponent_visibility as opponent_visibility
import analysis.trajectory_efficiency as trajectory_efficiency
import data_extraction.network_compensation.network_compensation as nr
from parse_data import identify_filepaths


SESSION_TYPES = ('solo1', 'social', 'solo2')        # in the order each player plays them
SOLO_TYPES = ('solo1', 'solo2')                     # solo index offset 0 is the first solo, 1 the second
SESSION_LABELS = {'solo1': 'first solo', 'social': 'social', 'solo2': 'second solo'}
FIRST_SOLO_CUT = 5          # trials cut from the start of the first solo (first exposure to the task)
FIXED_CUTS = (0.2, 0.8)     # fraction of peripheral trials: at or below = central, at or above = peripheral
TRIAL_SPLIT_REFERENCE = 'social'    # session type whose trials set the one peripheral boundary
WALL_SEPS = (1, 2, 3, 4)    # possible wall separations (in walls)
POSTERIOR_MIN = 0.8         # below this top posterior, a mixture label counts as uncertain
DRIFT_BLOCK = 10            # trials per block when reordering a session to test for drift
DRIFT_PERM = 1000           # reorderings per player
DRIFT_MIN_SEG = 5           # shortest segment either side of a step
HUMAN_FOV = 110.36          # human camera FoV (total degrees)
EXTREME_FRAC = 0.2          # share of players at each end of the centrality ranking that forms a group
HIGH_SCORE, LOW_SCORE = 1.0, 0.4    # trial score for reaching High or Low, on the agent scale
FORCED_SCORES = {'ForcedHigh': HIGH_SCORE, 'ForcedLow': LOW_SCORE}  # both walls of a forced trial pay the same
LOGGED_HIGH_SCORE = 50.0    # High in the human builds' cumulative score (Low is 20, the same ratio)
KEYS = ('experiment', 'player', 'session_type')     # one player in one session

# the host logs the social client (log id 1) through Netcode's ClientNetworkTransform (NGO 1.7.1). Its transform
# arrives each network tick, is rendered one tick late with linear interpolation, then smoothed by
# lerp(current, target, deltaTime / 0.1) (BufferedLinearInterpolator.cs:220). network_view applies the same path
# to locally logged traces, so every player's movement is measured through the same view
NET_TICK_S = 1 / 30         # NetworkManager TickRate 30 (OctagonStage.unity)
NET_LAG_TICKS = 1           # the host is the server, so it renders one tick behind
NET_CLIENT_LAG_TICKS = 2    # a client renders remote players two ticks behind (NetworkTransform.cs:2799)
NET_SMOOTH_S = 0.1          # BufferedLinearInterpolator.MaximumInterpolationTime
NETWORK_VIEW_IDS = (0,)     # log ids recorded locally: social player 0 and every solo player

# per-player fingerprint metrics, named as in model_characterisation. Latency, route and P(High) use the
# player's own wins; win_rate is social only
FP_METRICS = ['onset_centrality', 'iti_centrality', 'prop_no_translational', 'prop_no_rotational',
              'proportion_resting', 'iti_path_len', 'rot_total', 'winner_path_len',
              'path_ratio', 'frac_visible_at_onset',
              'late_latency_to_wall_visible', 'all_latency_to_wall_visible',
              'latency_to_reach_wall', 'frac_never_visible',
              'frac_trials_over_5s', 'p_high', 'mean_score', 'score_rate', 'win_rate']
GROUPS = ('central', 'peripheral')                  # the two groups compared; 'middle' players sit between

# names for mixture components, ordered from lowest to highest centrality
COMPONENT_NAMES = {1: ('all',), 2: ('peripheral', 'central'), 3: ('peripheral', 'intermediate', 'central')}

STRAT_PALETTE = {**mc.STRAT_PALETTE, 'mixed': '#9E9E9E'}
SESSION_PALETTE = {'solo1': '#1baf7a', 'social': '#4a3aa7', 'solo2': '#eda100'}

TABLES_CACHE = 'human_onset_tables'
FINGERPRINT_CACHE = 'human_fingerprint_tables'
ITI_CACHE = 'human_iti_ticks'
SESSION_FILES_CACHE = 'human_session_files'


# ---------------------------------------------------------------------------
# session files and alignment
# ---------------------------------------------------------------------------

def solo_index(experiment, player, solo):
    ''' Return the position of a solo session in the solo trial lists.
        Takes an experiment index, a player id (0 or 1) and the solo (0 first, 1 second). '''

    return 4 * experiment + 2 * player + solo


def solo_position(index):
    ''' Return (experiment, player, solo) for a position in the solo trial lists. Takes the position. '''

    return index // 4, (index // 2) % 2, index % 2


def _split_path(path):
    ''' Return the session folder and file name of a relative session path. Takes the path. '''

    folder, name = re.split(r'[\\/]', path.strip())[-2:]
    return folder, name


def check_file_alignment(social_paths, solo_paths):
    ''' Raise ValueError unless every solo file sits at the index its experiment and player predict.
        Takes the social and solo relative paths, in the order the pickles were built from. '''

    if len(solo_paths) != 4 * len(social_paths):
        raise ValueError(f"expected 4 solos per experiment, got {len(solo_paths)} for {len(social_paths)}")

    for e, path in enumerate(social_paths):
        folder, name = _split_path(path)
        # the two pseudonyms before '_Social' are players 0 and 1
        pseudonyms = re.search(r'_([^_]+)_([^_]+)_Social\.json$', name).groups()
        for p in (0, 1):
            for k, tag in enumerate(('FirstSolo', 'SecondSolo')):
                solo_folder, solo_name = _split_path(solo_paths[solo_index(e, p, k)])
                if solo_folder != folder or not solo_name.endswith(f'_{pseudonyms[p]}_{tag}.json'):
                    raise ValueError(f"solo index {solo_index(e, p, k)} is {solo_folder}/{solo_name}, "
                                     f"expected {folder} {pseudonyms[p]} {tag}")


def cached_session_files():
    ''' Return the saved (social, solo) session file lists, or None if they were never saved. Takes no inputs. '''

    try:
        return data_saving.load_data(SESSION_FILES_CACHE, data_saving.cache_dir)['result']
    except FileNotFoundError:
        return None


def session_files(recompute=False):
    ''' Return the (social, solo) relative session paths in pickle order, from cache when possible.
        Takes a recompute flag (recomputing needs the raw data folder mounted). '''

    return data_saving.cached(SESSION_FILES_CACHE, identify_filepaths.get_filenames, recompute=recompute)


def experiment_info(social_paths):
    ''' Return one row per experiment with its folder and date. Takes the social relative paths.

        The folder name dates the experiment, so it dates all five of its sessions. '''

    folders = [_split_path(p)[0] for p in social_paths]
    return pd.DataFrame(dict(experiment=range(len(folders)), folder=folders,
                             date=pd.to_datetime([f[:6] for f in folders], format='%y%m%d')))


# ---------------------------------------------------------------------------
# per-trial and per-session tables
# ---------------------------------------------------------------------------

def _event_centrality(trial, event, log_id):
    ''' Return a player's centrality at the first row of an event, or NaN if absent.
        Takes a trial, the event description and the player id in the log. '''

    rows = trial[trial['eventDescription'] == event]
    if rows.empty:
        return np.nan
    x = float(rows[globals.PLAYER_LOC_DICT[log_id]['xloc']].iloc[0])
    z = float(rows[globals.PLAYER_LOC_DICT[log_id]['yloc']].iloc[0])
    return float(1 - np.hypot(x, z) / mc.ARENA_RADIUS)


def _wall_sep(trial):
    ''' Return the separation of the two trial walls (in walls), or NaN if unreadable. Takes a trial. '''

    try:
        return get_indices.get_wall_difference(trial=trial)
    except (ValueError, TypeError, IndexError):
        return np.nan


def _session_rows(trial_list, experiment, session_type, id_map, start):
    ''' Return per-trial onset rows for one session, skipping the first `start` trials.
        Takes a trial list, the experiment index, the session type, a map from log player id to
        player, and the number of opening trials to cut.

        Previous-trial fields come from the trial before in the full list, so the first kept trial
        still has them when trials were cut. '''

    social = session_type == 'social'
    prev = {log_id: (np.nan, np.nan) for log_id in id_map}     # (won, centrality at trial end) last trial
    rows = []
    for t, trial in enumerate(trial_list):
        # get the winner (social only, as a solo player always reaches a wall)
        winner = np.nan
        if social:
            try:
                winner = get_indices.get_trigger_activator(trial)
            except (ValueError, TypeError):
                pass
        wall_sep = _wall_sep(trial)
        trial_num = trial[globals.TRIAL_NUM].dropna()
        trial_num = float(trial_num.iloc[0]) if len(trial_num) else np.nan

        for log_id, player in id_map.items():
            won = float(winner == log_id) if social and not np.isnan(winner) else np.nan
            if t >= start:
                try:
                    s = mc.slice_onset_centrality(trial, log_id)
                except (IndexError, ValueError):
                    s = None                                        # trial missing slice onset
                if s is not None:
                    rows.append(dict(experiment=experiment, player=player, session_type=session_type,
                                     trial=t, trial_pos=t - start, trial_num=trial_num,
                                     centrality=s['centrality'], x=s['x'], z=s['z'], wall_sep=wall_sep,
                                     won=won, prev_won=prev[log_id][0],
                                     prev_end_centrality=prev[log_id][1]))
            prev[log_id] = (won, _event_centrality(trial, globals.TRIAL_END, log_id))
    return rows


def collect_onset_table(trial_lists, kind, first_solo_cut=FIRST_SOLO_CUT):
    ''' Return the onset table, one row per trial per player, measured at the frame the walls appear.
        Takes the social or solo trial lists (in pickle order) and which kind they are.

        Slice onset is the frame that ends the wait and starts the trial, so a player's position there
        is their waiting position. Each row holds centrality (1 = arena centre, 0 = alcove back), the
        raw x and z, the separation of the two trial walls, whether this player won (social only), and
        the same player's win and centrality on the previous trial.

        Social sessions keep every trial. The first solo loses its first first_solo_cut trials, and the
        second solo keeps every trial. Solo players are logged as player 0, and take their social player
        id from their position in the solo list. trial counts from the start of the full list, trial_pos
        from the first kept trial. Previous-trial fields come from the full list, so a kept trial still
        has them when trials were cut. '''

    rows = []
    for i, trial_list in enumerate(trial_lists):
        if kind == 'social':
            experiment, session_type, id_map = i, 'social', {0: 0, 1: 1}
        elif kind == 'solo':
            experiment, player, k = solo_position(i)
            session_type, id_map = SOLO_TYPES[k], {0: player}
        else:
            raise ValueError(f"kind must be 'social' or 'solo', got {kind!r}")
        start = first_solo_cut if session_type == 'solo1' else 0
        rows.extend(_session_rows(trial_list, experiment, session_type, id_map, start))
    return pd.DataFrame(rows)


def collect_session_table(trial_lists, kind):
    ''' Return the session table, one row per session, with its trial count, duration and share of
        trials at each wall separation. Takes the social or solo trial lists (in pickle order) and
        which kind they are.

        This describes the protocol a session ran under, not the behaviour in it, so it counts every
        trial logged and ignores the first-solo cut. The protocol changed during data collection,
        which this table is there to show. Social rows have player NaN. '''

    rows = []
    for i, trial_list in enumerate(trial_lists):
        if kind == 'social':
            experiment, player, session_type = i, np.nan, 'social'
        else:
            experiment, player, k = solo_position(i)
            session_type = SOLO_TYPES[k]
        # get the separation of every trial, then the share of each
        seps = np.array([_wall_sep(t) for t in trial_list], dtype=float)
        duration = (trial_list[-1]['timeLocal'].iloc[-1] - trial_list[0]['timeLocal'].iloc[0]).total_seconds() / 60
        row = dict(experiment=experiment, player=player, session_type=session_type,
                   n_trials=len(trial_list), duration_min=duration)
        row.update({f'prop_sep_{w}': float(np.mean(seps == w)) for w in WALL_SEPS})
        rows.append(row)
    return pd.DataFrame(rows)


def load_tables(first_solo_cut=FIRST_SOLO_CUT, recompute=False):
    ''' Return a dict with the onset table ('onset') and the session table ('sessions') for every human
        session, from cache when possible. Takes the first-solo cut and a recompute flag.

        Loads the social pickle, then the solo pickle, and frees each before the next, as the two
        together do not fit comfortably in memory. The cache is rebuilt if the cut changes or either
        pickle is regenerated. '''

    pickles = {name: os.path.join(data_strings.DATAFRAME_PATH, name + '.pkl')
               for name in (data_strings.FILENAME_SOCIAL, data_strings.FILENAME_SOLO)}
    # schema is part of the stamp, so a table built under older column names is rebuilt
    stamp = dict(schema='experiment+session', first_solo_cut=first_solo_cut,
                 pickles={n: os.path.getmtime(p) for n, p in pickles.items()})

    def compute():
        onset, sessions = [], []
        for kind, name in (('social', data_strings.FILENAME_SOCIAL), ('solo', data_strings.FILENAME_SOLO)):
            trial_lists = data_saving.load_data(name)
            onset.append(collect_onset_table(trial_lists, kind, first_solo_cut))
            sessions.append(collect_session_table(trial_lists, kind))
            del trial_lists
        return dict(onset=pd.concat(onset, ignore_index=True),
                    sessions=pd.concat(sessions, ignore_index=True))

    return data_saving.cached(TABLES_CACHE, compute, recompute=recompute, stamp=stamp)


# ---------------------------------------------------------------------------
# trial-level split and per-player summary
# ---------------------------------------------------------------------------

def _present_types(df):
    ''' Return the session types in df, in play order. Takes a table with a session_type column. '''

    return [st for st in SESSION_TYPES if st in set(df['session_type'])]


def mixture_boundary(gmm, n_grid=2001):
    ''' Return the centrality between a 2-component Gaussian mixture's means at which the lower
        component stops being the more likely one. Takes a fitted 1-D mixture.

        Each component contributes its density times its weight, the share of trials it holds, so the
        posterior is P(value | component) x weight, normalised across components. The boundary is
        where those two products are equal.

        Searching only between the means avoids the second crossing a narrow component can create far
        out in the tail of a wide one, which would send extreme values back to the wider component. '''

    means = gmm.means_.ravel()
    grid = np.linspace(means.min(), means.max(), n_grid).reshape(-1, 1)
    p_low = gmm.predict_proba(grid)[:, int(np.argmin(means))]
    return float(grid[int(np.argmax(p_low < 0.5)), 0])


def fit_trial_split(onset, reference=TRIAL_SPLIT_REFERENCE, n_init=10, random_state=0):
    ''' Return the onset table with a peripheral flag on every trial, and the fitted split.
        Takes the onset table and the session type that sets the boundary.

        A 2-component Gaussian mixture is fitted to the onset centrality of every trial of the reference
        session type, with all players pooled, so a player with more trials pulls the fit more. Trials
        below the boundary between the components count as peripheral.

        One boundary applies to every session type, so peripheral means the same position in each.
        Social is the default reference (the longest session, played after some practice). '''

    ref = onset['session_type'] == reference
    if not ref.any():
        raise ValueError(f"no {reference!r} trials to fit the boundary on")
    # fit the reference session, then flag every trial below its boundary
    X = onset.loc[ref, 'centrality'].to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, n_init=n_init, random_state=random_state).fit(X)
    boundary = mixture_boundary(gmm)
    onset = onset.copy()
    onset['peripheral'] = onset['centrality'] < boundary
    return onset, dict(gmm=gmm, boundary=boundary, reference=reference)


def player_table(onset):
    ''' Return the player table, one row per player per session type, holding mean, SD and median onset
        centrality, the trial count, and the fraction of trials flagged peripheral.
        Takes the onset table (with the peripheral flag from fit_trial_split).

        The mean and the fraction summarise the same trials two ways. The mean says where a player
        waits on average, the fraction says how often they wait at a wall. '''

    agg = dict(mean_centrality=('centrality', 'mean'), sd_centrality=('centrality', 'std'),
               median_centrality=('centrality', 'median'), n_trials=('centrality', 'size'))
    if 'peripheral' in onset.columns:
        agg['frac_peripheral'] = ('peripheral', 'mean')
    players = onset.groupby(['experiment', 'player', 'session_type']).agg(**agg).reset_index()
    players['frac_peripheral'] = players['frac_peripheral'].astype(float) if 'frac_peripheral' in players else np.nan
    return players


def fixed_cut_labels(frac_peripheral, cuts=FIXED_CUTS):
    ''' Return central / mixed / peripheral labels by counting each player's peripheral trials.
        Takes the fractions from player_table and the (low, high) cuts.

        A player at or below the low cut is central, at or above the high cut is peripheral, and mixed
        in between. Nothing is fitted, so these labels do not depend on a choice of k. Missing fractions
        are 'unassigned'. '''

    f = pd.Series(frac_peripheral, dtype=float)
    labels = np.select([f <= cuts[0], f >= cuts[1]], ['central', 'peripheral'], 'mixed').astype(object)
    labels[f.isna().to_numpy()] = 'unassigned'
    return pd.Series(labels, index=f.index)


# ---------------------------------------------------------------------------
# mixtures on per-player means
# ---------------------------------------------------------------------------

def component_names(k):
    ''' Return names for k mixture components, lowest centrality first. Takes k. '''

    return COMPONENT_NAMES.get(k, tuple(f'component {i + 1}' for i in range(k)))


def fit_mixtures(players, ks=(1, 2, 3, 4), value='mean_centrality', n_init=10, random_state=0):
    ''' Return a Gaussian mixture fitted for each k, per session type, with the values it was fitted to.
        Takes the player table and the component counts.

        This fits one value per player (their mean onset centrality by default), not one per trial. '''

    fits = {}
    for st in _present_types(players):
        X = players.loc[players['session_type'] == st, value].dropna().to_numpy().reshape(-1, 1)
        models = {k: GaussianMixture(n_components=k, n_init=n_init, random_state=random_state).fit(X)
                  for k in ks}
        fits[st] = dict(X=X, models=models)
    return fits


def mixture_table(fits):
    ''' Return one row per session type per k with BIC, AIC and each component's mean, SD and weight,
        lowest centrality first. Takes the output of fit_mixtures.

        BIC and AIC score the fit against the parameters it spends, and lower is better. '''

    rows = []
    for st, fit in fits.items():
        for k, gmm in fit['models'].items():
            order = np.argsort(gmm.means_.ravel())
            rows.append(dict(session_type=st, k=k, bic=gmm.bic(fit['X']), aic=gmm.aic(fit['X']),
                             means=gmm.means_.ravel()[order].round(3),
                             sds=np.sqrt(gmm.covariances_.ravel()[order]).round(3),
                             weights=gmm.weights_[order].round(2)))
    return pd.DataFrame(rows)


def add_mixture_labels(players, fits, value='mean_centrality'):
    ''' Return the player table with the name of each player's most likely component, and that
        component's posterior, for every fitted k. Takes the player table and the output of fit_mixtures.

        Components are named by mean centrality, so labels do not depend on the order sklearn returns
        them in. A posterior near 1 means the fit places the player firmly in one component, and a
        posterior near 1/k means it cannot tell. Columns are gmm{k} and gmm{k}_posterior. '''

    players = players.copy()
    for st, fit in fits.items():
        rows = players.index[(players['session_type'] == st) & players[value].notna()]
        X = players.loc[rows, value].to_numpy().reshape(-1, 1)
        for k, gmm in fit['models'].items():
            # rank each component by its mean, then name it by rank
            rank = np.argsort(np.argsort(gmm.means_.ravel()))
            post = gmm.predict_proba(X)
            names = np.array(component_names(k), dtype=object)
            players.loc[rows, f'gmm{k}'] = names[rank[post.argmax(axis=1)]]
            players.loc[rows, f'gmm{k}_posterior'] = post.max(axis=1)
    return players


def bootstrap_best_k(players, ks=(1, 2, 3, 4), n_boot=200, value='mean_centrality', n_init=3, random_state=0):
    ''' Return how often each k has the lowest BIC across bootstrap resamples of players, per session type.
        Takes the player table, the component counts and the resample count.

        BIC on one sample can separate k by a margin smaller than the noise. Refitting on players drawn
        with replacement shows whether the winning k is stable. '''

    rng = np.random.default_rng(random_state)
    counts = {}
    for st in _present_types(players):
        X = players.loc[players['session_type'] == st, value].dropna().to_numpy().reshape(-1, 1)
        wins = pd.Series(0, index=list(ks))
        for _ in range(n_boot):
            # resample players with replacement and keep the k with the lowest BIC
            Xb = X[rng.integers(0, len(X), len(X))]
            bic = [GaussianMixture(n_components=k, n_init=n_init, random_state=random_state).fit(Xb).bic(Xb)
                   for k in ks]
            wins[ks[int(np.argmin(bic))]] += 1
        counts[st] = wins
    return pd.DataFrame(counts).T.rename_axis('session_type')


# ---------------------------------------------------------------------------
# partners
# ---------------------------------------------------------------------------

def partner_correlation(players, value='mean_centrality'):
    ''' Return the correlation between the two players of each experiment, per session type.
        Takes the player table.

        The first solo gives a baseline, as the pair have not played together yet. '''

    rows = []
    for st in _present_types(players):
        w = players[players['session_type'] == st].pivot(index='experiment', columns='player', values=value).dropna()
        r, p = stats.pearsonr(w[0], w[1])
        rows.append(dict(session_type=st, r=r, p=p, n_pairs=len(w)))
    return pd.DataFrame(rows)


def partner_gap_null(players, session_type, value='mean_centrality', n_perm=2000, random_state=0):
    ''' Return the mean gap between the two players of an experiment, against random pairings.
        Takes the player table (filtered to any subset of experiments) and a session type.

        The gap is |player 0 - player 1| of their session means, averaged over experiments. The null
        pairs each experiment's player 0 with another experiment's player 1. p is one-sided, the share
        of random pairings with a gap at or below the real one. '''

    w = players[players['session_type'] == session_type].pivot(
        index='experiment', columns='player', values=value).dropna()
    a, b = w[0].to_numpy(), w[1].to_numpy()
    gap = float(np.abs(a - b).mean())

    # pair each player 0 with a player 1 from another experiment
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(a))
    null = np.empty(n_perm)
    for k in range(n_perm):
        j = rng.permutation(len(a))
        while np.any(j == idx):
            j = rng.permutation(len(a))
        null[k] = np.abs(a - b[j]).mean()

    return dict(gap=gap, null=null, p=float((1 + np.sum(null <= gap)) / (n_perm + 1)), n_pairs=len(a))


def partner_agreement(players, label_col):
    ''' Return how often both players of an experiment share a label, against the rate expected by chance.
        Takes the player table and a label column.

        Chance is the sum of squared label shares, which is the agreement expected if partners were
        paired at random. '''

    rows = []
    for st in _present_types(players):
        sub = players[players['session_type'] == st]
        w = sub.pivot(index='experiment', columns='player', values=label_col).dropna()
        shares = sub[label_col].value_counts(normalize=True)
        rows.append(dict(session_type=st, agreement=float((w[0] == w[1]).mean()),
                         chance=float((shares ** 2).sum()), n_pairs=len(w)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# consistency across trials
# ---------------------------------------------------------------------------

def split_half(onset, value='centrality'):
    ''' Return split-half reliability of the player means, per session type. Takes the onset table.

        Each player's trials are halved, a mean is taken on each half, and the two means are correlated
        across players. odd_even splits into alternate trials and measures noise alone. first_second
        splits the session in two, so it also picks up drift. spearman_brown corrects r back up to the
        full trial count, as each half holds only half the trials. '''

    rows = []
    for st in _present_types(onset):
        sub = onset[onset['session_type'] == st].copy()
        halves = dict(odd_even=sub['trial_pos'] % 2 == 1,
                      first_second=sub['trial_pos'] > sub.groupby(['experiment', 'player'])['trial_pos'].transform('median'))
        for split, second in halves.items():
            m = sub.assign(half=second).groupby(['experiment', 'player', 'half'])[value].mean().unstack().dropna()
            r = stats.pearsonr(m[False], m[True])[0]
            rows.append(dict(session_type=st, split=split, r=r, spearman_brown=2 * r / (1 + r), n_players=len(m)))
    return pd.DataFrame(rows)


def run_consistency(onset, label_col='peripheral', n_shuffle=200, random_state=0):
    ''' Return, per player per session type, how often consecutive trials carry the same label, observed
        and with the player's own trials shuffled. Takes the onset table.

        Shuffling keeps the player's own mix of labels and destroys only their order, so it gives the
        rate expected if the player picked each trial independently. An observed rate above it means
        peripheral and central trials arrive in runs. '''

    rng = np.random.default_rng(random_state)
    rows = []
    for (e, p, st), grp in onset.sort_values('trial_pos').groupby(['experiment', 'player', 'session_type']):
        x = grp[label_col].to_numpy()
        if len(x) < 3:
            continue
        # shuffle each row of a tiled copy, then compare neighbours in every row at once
        shuffled = rng.permuted(np.tile(x, (n_shuffle, 1)), axis=1)
        rows.append(dict(experiment=e, player=p, session_type=st, n_trials=len(x),
                         p_same=float(np.mean(x[1:] == x[:-1])),
                         p_same_shuffled=float(np.mean(shuffled[:, 1:] == shuffled[:, :-1]))))
    return pd.DataFrame(rows)


def wsls_metrics(onset, session_type='social', min_trials=10):
    ''' Return, per player, how far their waiting position moves after a win and after a loss.
        Takes the onset table, a session type and the fewest trials of each outcome a player needs.

        Only consecutive trial pairs count, so a trial missing its slice onset breaks the pair. The
        change is this trial's onset centrality minus the previous trial's, and the previous trial's
        outcome splits the pairs. d_win and d_loss are mean signed changes, so they give the direction
        (positive = toward the centre). abs_win and abs_loss are mean absolute changes, so they give the
        size of the move whatever its direction. shift_diff is abs_loss - abs_win, above 0 when a player
        moves further after losing.

        Conditioning on the outcome could also pick out where the player was standing, as a player might
        win more often from some positions. coef_won is the same comparison with the previous onset held
        fixed, the coefficient of the previous win in onset ~ previous onset + previous win.

        The winner ends a trial at the wall it triggered, so an outcome effect can be leftover position
        rather than a choice. coef_won_ctrl adds where the previous trial ended, the coefficient of the
        previous win in onset ~ previous onset + previous trial end + previous win, and b_end is that
        trial end's own coefficient. abs_won_ctrl does the same for the size of the move, the coefficient
        of the previous win in |change| ~ |previous trial end - previous onset| + previous win. Both are
        NaN when no trial end is recorded. Solo players always reach a wall, so this needs a social
        session. '''

    sub = onset[onset['session_type'] == session_type].sort_values(['experiment', 'player', 'trial_pos'])
    grp = sub.groupby(['experiment', 'player'])
    sub = sub.assign(prev_onset=grp['centrality'].shift(1), prev_pos=grp['trial_pos'].shift(1))
    # keep consecutive pairs whose previous trial has an outcome
    sub = sub[(sub['trial_pos'] - sub['prev_pos'] == 1) & sub['prev_won'].notna()]
    sub = sub.assign(delta=sub['centrality'] - sub['prev_onset'])

    rows = []
    for (e, p), d in sub.groupby(['experiment', 'player']):
        win, loss = d[d['prev_won'] == 1], d[d['prev_won'] == 0]
        if len(win) < min_trials or len(loss) < min_trials:
            continue
        # fit onset on the previous onset and the previous win
        X = np.column_stack([np.ones(len(d)), d['prev_onset'], d['prev_won']])
        b = np.linalg.lstsq(X, d['centrality'].to_numpy(float), rcond=None)[0]

        # repeat with where the previous trial ended held fixed, for the position and for the size of the move
        ctrl = d.dropna(subset=['prev_end_centrality']) if 'prev_end_centrality' in d else d.iloc[:0]
        coef_won_ctrl = b_end = abs_won_ctrl = np.nan
        if len(ctrl) > 3:
            Xc = np.column_stack([np.ones(len(ctrl)), ctrl['prev_onset'], ctrl['prev_end_centrality'],
                                  ctrl['prev_won']])
            bc = np.linalg.lstsq(Xc, ctrl['centrality'].to_numpy(float), rcond=None)[0]
            b_end, coef_won_ctrl = float(bc[2]), float(bc[3])
            Xa = np.column_stack([np.ones(len(ctrl)),
                                  (ctrl['prev_end_centrality'] - ctrl['prev_onset']).abs(), ctrl['prev_won']])
            abs_won_ctrl = float(np.linalg.lstsq(Xa, ctrl['delta'].abs().to_numpy(float), rcond=None)[0][2])

        rows.append(dict(experiment=e, player=p, n_win=len(win), n_loss=len(loss),
                         d_win=float(win['delta'].mean()), d_loss=float(loss['delta'].mean()),
                         abs_win=float(win['delta'].abs().mean()), abs_loss=float(loss['delta'].abs().mean()),
                         shift_diff=float(loss['delta'].abs().mean() - win['delta'].abs().mean()),
                         direction_diff=float(loss['delta'].mean() - win['delta'].mean()),
                         b_prev=float(b[1]), coef_won=float(b[2]), coef_won_ctrl=coef_won_ctrl,
                         b_end=b_end, abs_won_ctrl=abs_won_ctrl))
    return pd.DataFrame(rows)


def switch_score(onset, label_col='peripheral'):
    ''' Return, per player per session type, how often their label changes between consecutive trials
        against the count expected if their trials came in random order. Takes the onset table.

        With k peripheral trials out of n, random order gives 2k(n - k) / n changes on average.
        switch_score is the observed count divided by that. A score near 0 means few changes and long
        runs (e.g. one switch). A score near 1 means the player moves between positions as if at
        random, and above 1 means they alternate more than that. Players holding one label throughout
        expect no changes and get NaN. '''

    rows = []
    for (e, p, st), grp in onset.sort_values('trial_pos').groupby(['experiment', 'player', 'session_type']):
        x = grp[label_col].to_numpy(bool)
        n, k = len(x), int(x.sum())
        # count label changes, and the random-order expectation
        changes = int(np.sum(x[1:] != x[:-1]))
        expected = 2 * k * (n - k) / n if n else 0.0
        rows.append(dict(experiment=e, player=p, session_type=st, n_trials=n, frac_peripheral=k / n if n else np.nan,
                         n_changes=changes, expected_changes=expected,
                         switch_score=changes / expected if expected > 0 else np.nan))
    return pd.DataFrame(rows)


def frac_peripheral_null(players, n_sim=1000, random_state=0):
    ''' Return simulated fractions of peripheral trials for the case where every player shares one rate,
        per session type. Takes the player table.

        Each simulated player keeps their real trial count and draws trials at the pooled peripheral
        rate. Observed fractions more spread than these mean players genuinely differ, rather than all
        sampling one common habit. Arrays are n_sim x players. '''

    rng = np.random.default_rng(random_state)
    null = {}
    for st in _present_types(players):
        sub = players[players['session_type'] == st]
        n = sub['n_trials'].to_numpy()
        rate = float((sub['frac_peripheral'] * n).sum() / n.sum())
        null[st] = rng.binomial(n, rate, size=(n_sim, len(n))) / n
    return null


def raster(onset, session_type, value='centrality'):
    ''' Return a players x trials matrix of one session type for plotting as a raster, with players
        sorted by their mean (lowest first). Takes the onset table and a session type.

        Reading along a row shows whether a player held one waiting position or moved between them.
        Rows are padded with NaN, as players differ in trial count. Also returns the row order. '''

    sub = onset[onset['session_type'] == session_type]
    order = (sub.groupby(['experiment', 'player'])[value].mean().rename('mean')
             .sort_values().reset_index())
    mat = np.full((len(order), int(sub['trial_pos'].max()) + 1), np.nan)
    row_of = {(e, p): i for i, (e, p) in enumerate(zip(order['experiment'], order['player']))}
    for (e, p), grp in sub.groupby(['experiment', 'player']):
        mat[row_of[(e, p)], grp['trial_pos'].to_numpy(int)] = grp[value].to_numpy()
    return mat, order


def previous_trial_effect(onset, min_trials=10):
    ''' Return, per social player, their mean onset centrality after a win and after a loss, and the
        within-player correlation between centrality at the previous trial end and at this onset.
        Takes the onset table.

        The winner ends a trial at the wall it triggered, so a player who does not walk back starts the
        next trial peripheral for that reason alone. A gap between wins and losses, or a strong
        correlation with the previous trial end, means onset position reflects inertia as much as a
        chosen waiting position. '''

    rows = []
    sub = onset[(onset['session_type'] == 'social') & onset['prev_won'].notna()]
    for (e, p), grp in sub.groupby(['experiment', 'player']):
        ok = grp['prev_end_centrality'].notna()
        r = np.nan
        if ok.sum() >= min_trials and grp.loc[ok, 'prev_end_centrality'].std() > 0:
            r = stats.pearsonr(grp.loc[ok, 'prev_end_centrality'], grp.loc[ok, 'centrality'])[0]
        rows.append(dict(experiment=e, player=p,
                         after_win=grp.loc[grp['prev_won'] == 1, 'centrality'].mean(),
                         after_loss=grp.loc[grp['prev_won'] == 0, 'centrality'].mean(),
                         n_after_win=int((grp['prev_won'] == 1).sum()),
                         n_after_loss=int((grp['prev_won'] == 0).sum()),
                         r_prev_end=r))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# drift within one session
# ---------------------------------------------------------------------------

def _block_shuffle(y, block, rng):
    ''' Return the values with their blocks of trials reordered, keeping the order inside each block.
        Takes an array, the block length and a generator.

        Reordering whole blocks keeps a player's short runs intact and removes only the slow trend,
        so runs of one strategy cannot pass as drift. '''

    blocks = [y[i:i + block] for i in range(0, len(y), block)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def _half_diff(y):
    ''' Return the second half's mean minus the first half's. Takes an array. '''

    return float(y[len(y) // 2:].mean() - y[:len(y) // 2].mean())


def _step_diffs(y, min_seg=DRIFT_MIN_SEG):
    ''' Return every split point and the difference it makes between the trials after and before it.
        Takes an array and the shortest segment either side. '''

    n = len(y)
    cuts = np.arange(min_seg, n - min_seg)
    total = np.cumsum(y)
    before = total[cuts - 1] / cuts
    after = (total[-1] - total[cuts - 1]) / (n - cuts)
    return cuts, after - before


def _best_step(y, min_seg=DRIFT_MIN_SEG):
    ''' Return the largest difference any split makes, and where in the session it falls (0 to 1).
        Takes an array and the shortest segment either side. '''

    cuts, diffs = _step_diffs(y, min_seg)
    i = int(np.argmax(np.abs(diffs)))
    return float(diffs[i]), float(cuts[i] / len(y))


def within_session_drift(onset, session_types=SESSION_TYPES, max_trials=None, block=DRIFT_BLOCK,
                         n_perm=DRIFT_PERM, min_trials=20, min_seg=DRIFT_MIN_SEG, random_state=0):
    ''' Return one row per player per session type measuring how much their waiting position moved
        during the session. Takes the onset table and which session types to measure.

        A player mean treats a strategy as fixed. These measures test that. half_diff is the second
        half's mean minus the first half's. step is the largest difference any split of the session
        makes, with step_at saying where it falls, which separates a single switch from a slow slide.
        slope is an ordinary least squares fit in centrality per 100 trials.

        p_half and p_step come from reordering the player's own blocks of trials, so they ask whether
        the movement is more than the player's own trial-to-trial variation would produce. Maximising
        the step over split points inflates it even in random data, which is why p_step compares
        against the same maximum in the reordered sessions.

        max_trials caps every player at the same trial count, so sessions of different length can be
        compared. Players with fewer than min_trials are dropped. '''

    rng = np.random.default_rng(random_state)
    rows = []
    for st in [t for t in session_types if t in set(onset['session_type'])]:
        sub = onset[onset['session_type'] == st]
        for (e, p), grp in sub.groupby(['experiment', 'player']):
            y = grp.sort_values('trial_pos')['centrality'].to_numpy(float)
            if max_trials is not None:
                y = y[:max_trials]
            if len(y) < min_trials:
                continue
            half, (step, step_at) = _half_diff(y), _best_step(y, min_seg)
            slope = float(np.polyfit(np.arange(len(y)), y, 1)[0] * 100)

            # reorder the player's own blocks and measure the same two things each time
            null_half = np.empty(n_perm)
            null_step = np.empty(n_perm)
            for k in range(n_perm):
                ys = _block_shuffle(y, block, rng)
                null_half[k] = _half_diff(ys)
                null_step[k] = np.abs(_step_diffs(ys, min_seg)[1]).max()

            rows.append(dict(experiment=e, player=p, session_type=st, n_trials=len(y),
                             half_diff=half, step=step, step_at=step_at, slope=slope,
                             p_half=float((1 + np.sum(np.abs(null_half) >= abs(half))) / (n_perm + 1)),
                             p_step=float((1 + np.sum(null_step >= abs(step))) / (n_perm + 1))))
    return pd.DataFrame(rows)


def drift_summary(drift, alpha=0.05, big_move=0.1):
    ''' Return, per session type, how far players moved and how many per-player tests passed, next to
        the count expected by chance. Takes the table from within_session_drift, the p cut and what
        counts as a big move.

        The per-player p values are uncorrected. With no drift at all, about alpha x players would still
        pass, so the flagged counts only mean something against expected_by_chance. A longer session
        gives each test more power, so flagged counts cannot be compared across session lengths. '''

    return (drift.assign(moved=lambda d: d['p_half'] < alpha,
                         stepped=lambda d: d['p_step'] < alpha,
                         big=lambda d: d['half_diff'].abs() > big_move)
            .groupby('session_type')
            .agg(players=('half_diff', 'size'), median_abs_half=('half_diff', lambda v: v.abs().median()),
                 median_slope=('slope', 'median'), flagged_half=('moved', 'sum'),
                 flagged_step=('stepped', 'sum'), share_big=('big', 'mean'))
            .assign(expected_by_chance=lambda d: alpha * d['players'])
            .reindex([t for t in SESSION_TYPES if t in set(drift['session_type'])]))


def group_drift(onset, session_types=SESSION_TYPES, max_trials=None, block=DRIFT_BLOCK, n_perm=DRIFT_PERM,
                min_trials=20, random_state=0):
    ''' Return, per session type, whether players drift as a group and how large that drift is.
        Takes the onset table and which session types to measure.

        Each player's trials are reordered in blocks, as in within_session_drift, and the second-half
        minus first-half difference is recomputed each time. Reordering keeps the player's own runs and
        removes only the slow trend.

        p_group compares the mean |half difference| across players with the same mean in each set of
        reorderings. It asks whether players drift at all, so no per-player cut is needed.

        drift_rms is the size of real drift, in centrality units. It is the root mean square of the
        observed half differences with the part that reordering alone produces taken away,
        sqrt(mean(observed^2) - mean(reordered^2)). It counts drift that players share as well as drift
        that differs between them, and does not depend on session length the way per-player tests do.
        mean_half gives the shared direction. '''

    rng = np.random.default_rng(random_state)
    rows = []
    for st in [t for t in session_types if t in set(onset['session_type'])]:
        sub = onset[onset['session_type'] == st]
        real, null = [], []
        for _, grp in sub.groupby(['experiment', 'player']):
            y = grp.sort_values('trial_pos')['centrality'].to_numpy(float)
            if max_trials is not None:
                y = y[:max_trials]
            if len(y) < min_trials:
                continue
            # get the real half difference and the same after each block reordering
            real.append(_half_diff(y))
            null.append([_half_diff(_block_shuffle(y, block, rng)) for _ in range(n_perm)])
        real = np.array(real)
        null = np.array(null)                               # players x reorderings

        # compare the group's mean |half difference| with each reordering's
        mean_abs = np.abs(real).mean()
        null_mean_abs = np.abs(null).mean(axis=0)
        # remove the mean square that reordering alone produces
        rms_real = np.sqrt(np.mean(real ** 2))
        rms_null = np.sqrt(np.mean(null ** 2))
        rows.append(dict(session_type=st, players=len(real), mean_half=float(real.mean()),
                         mean_abs_half=float(mean_abs), mean_abs_half_reordered=float(null_mean_abs.mean()),
                         p_group=float((1 + np.sum(null_mean_abs >= mean_abs)) / (n_perm + 1)),
                         rms_observed=float(rms_real), rms_reordered=float(rms_null),
                         drift_rms=float(np.sqrt(max(rms_real ** 2 - rms_null ** 2, 0.0)))))
    return pd.DataFrame(rows).set_index('session_type')


def partner_gap_over_session(onset, session_type='social', n_bins=10, n_perm=1000, random_state=0):
    ''' Return how far apart the two players of an experiment stand across the session, against a null
        that pairs players from different experiments. Takes the onset table and a session type.

        Each session is cut into n_bins equal stretches of its own length, so sessions of different
        length line up. The gap is the distance between the pair's mean centrality in a bin. A falling
        gap means partners converge as they play, and the null says how much of that is two players
        drifting the same way regardless of who they faced. p is one-sided on the slope. '''

    sub = onset[onset['session_type'] == session_type].copy()
    n = sub.groupby(['experiment', 'player'])['trial_pos'].transform('max') + 1
    # place each trial in a stretch of the player's own session
    sub['bin'] = np.minimum((sub['trial_pos'] / n * n_bins).astype(int), n_bins - 1)
    means = (sub.groupby(['experiment', 'player', 'bin'])['centrality'].mean()
             .unstack('player').dropna())
    real = means.unstack('bin')                       # one row per experiment, columns (player, bin)
    a = real[0].to_numpy()
    b = real[1].to_numpy()

    gap = np.abs(a - b).mean(axis=0)
    x = np.arange(n_bins)
    slope = float(np.polyfit(x, np.abs(a - b).mean(axis=0), 1)[0])

    # pair each experiment's player 0 with another experiment's player 1
    rng = np.random.default_rng(random_state)
    null_gaps = np.empty((n_perm, n_bins))
    null_slopes = np.empty(n_perm)
    idx = np.arange(len(a))
    for k in range(n_perm):
        j = rng.permutation(len(a))
        while np.any(j == idx):
            j = rng.permutation(len(a))
        g = np.abs(a - b[j]).mean(axis=0)
        null_gaps[k] = g
        null_slopes[k] = np.polyfit(x, g, 1)[0]

    return dict(bins=x, gap=gap, null_gap=null_gaps.mean(axis=0), null_gaps=null_gaps,
                slope=slope, null_slopes=null_slopes,
                p=float((1 + np.sum(null_slopes <= slope)) / (n_perm + 1)), n_pairs=len(a))


# ---------------------------------------------------------------------------
# change across sessions
# ---------------------------------------------------------------------------

def wide_table(players, value='mean_centrality'):
    ''' Return one row per player with one column per session type, for comparing a value across the
        three sessions. Takes the player table and the column to spread. '''

    w = players.pivot_table(index=['experiment', 'player'], columns='session_type', values=value, aggfunc='first')
    return w[[st for st in SESSION_TYPES if st in w.columns]]


def transition_table(players, label_col, from_type, to_type):
    ''' Return counts of players moving from each label in one session type to each label in another.
        Takes the player table, a label column and the two session types.

        Rows are the earlier session and columns the later one, so the diagonal holds players who kept
        their strategy. '''

    w = wide_table(players, label_col).dropna(subset=[from_type, to_type])
    return pd.crosstab(w[from_type], w[to_type])


def persistence_model(wide):
    ''' Return an OLS fit of second-solo centrality on first-solo and social centrality.
        Takes the wide per-player table.

        A social coefficient above zero means the social session predicts the second solo beyond
        what the first solo already does. '''

    return smf.ols('solo2 ~ solo1 + social', data=wide.dropna(subset=['solo1', 'social', 'solo2'])).fit()


def _residual(y, x):
    ''' Return y with the part a straight-line fit on x predicts removed. Takes two arrays. '''

    X = np.column_stack([np.ones_like(x), x])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def partner_influence(wide, target='social', baseline='solo1', n_perm=2000, random_state=0):
    ''' Return how much a player's later position follows their partner's starting position, with the
        player's own start held fixed. Takes the wide per-player table, the later session and the
        baseline session.

        Fits target ~ own baseline + partner baseline, with standard errors clustered by experiment (both
        players of an experiment appear as own and as partner). coef is the partner coefficient. It says
        how much more central a player sits in the target session per unit their partner started more
        central. The own baseline is a predictor rather than being subtracted, so no slope is built in
        and coef = 0 means no partner effect.

        x_resid and y_resid are the partner baseline and the target with the part the own baseline
        predicts removed. Their fitted slope equals coef exactly, so they make the plot for it.

        The null gives every player a stand-in partner, the baseline of a player from another experiment,
        and refits coef. It centres on 0. p_random is the share of draws at or above coef (one-sided). '''

    rng = np.random.default_rng(random_state)
    # get each player's partner baseline, and keep players with all three values
    frame = wide[[baseline, target]].copy()
    frame['partner'] = wide[baseline].reindex([(e, 1 - p) for e, p in wide.index]).to_numpy()
    frame = frame.dropna().reset_index()
    own = frame[baseline].to_numpy(float)
    partner = frame['partner'].to_numpy(float)
    y = frame[target].to_numpy(float)
    experiments = frame['experiment'].to_numpy()

    # fit the model with standard errors clustered by experiment
    fit = smf.ols(f'{target} ~ {baseline} + partner', frame).fit(
        cov_type='cluster', cov_kwds={'groups': frame['experiment']})
    lo, hi = fit.conf_int().loc['partner']

    # refit with stand-in partners drawn from other experiments
    X = np.column_stack([np.ones_like(own), own, np.empty_like(own)])
    null = np.empty(n_perm)
    for k in range(n_perm):
        j = rng.integers(0, len(own), len(own))
        same = experiments[j] == experiments
        while same.any():
            j[same] = rng.integers(0, len(own), same.sum())
            same = experiments[j] == experiments
        X[:, 2] = own[j]
        null[k] = np.linalg.lstsq(X, y, rcond=None)[0][2]

    coef = float(fit.params['partner'])
    return dict(coef=coef, ci_lo=float(lo), ci_hi=float(hi), p=float(fit.pvalues['partner']),
                own_coef=float(fit.params[baseline]), n=len(frame), null=null,
                p_random=float((1 + np.sum(null >= coef)) / (n_perm + 1)),
                x_resid=_residual(partner, own), y_resid=_residual(y, own))


def _block_bootstrap_means(y, block, n_boot, rng):
    ''' Return bootstrap means of one session, drawing overlapping blocks of trials with replacement.
        Takes an array, the block length, the draw count and a generator.

        Drawing blocks rather than single trials keeps a player's runs, so the noise on a session mean
        is not understated by pretending consecutive trials are independent. '''

    n = len(y)
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    return y[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)


def player_change(onset, from_type='solo1', to_type='solo2', block=DRIFT_BLOCK, n_boot=DRIFT_PERM,
                  min_trials=20, random_state=0):
    ''' Return one row per player with their change in mean centrality between two sessions, and an
        interval on it. Takes the onset table and the two session types.

        Positions are logged exactly, so the change itself carries no measurement error. The interval
        answers a different question, whether the same player would show a similar change over a
        different set of trials. Each session's trials are redrawn in overlapping blocks, which keeps a
        player's runs of one strategy together, and the change is recomputed each time.

        The interval scales with how much a player varies from trial to trial, so a steady player gets a
        narrow one and a mixed player a wide one. It is reported as a continuous range, not used to
        sort players into movers and non-movers. Players with fewer than min_trials in either session
        are dropped. '''

    rng = np.random.default_rng(random_state)
    rows = []
    by_session = {st: {k: g.sort_values('trial_pos')['centrality'].to_numpy(float)
                       for k, g in onset[onset['session_type'] == st].groupby(['experiment', 'player'])}
                  for st in (from_type, to_type)}

    for key, y_from in by_session[from_type].items():
        y_to = by_session[to_type].get(key)
        if y_to is None or len(y_from) < min_trials or len(y_to) < min_trials:
            continue
        # redraw both sessions and recompute the change
        boot = (_block_bootstrap_means(y_to, block, n_boot, rng)
                - _block_bootstrap_means(y_from, block, n_boot, rng))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(experiment=key[0], player=key[1], change=float(y_to.mean() - y_from.mean()),
                         ci_lo=float(lo), ci_hi=float(hi), n_from=len(y_from), n_to=len(y_to)))
    return pd.DataFrame(rows, columns=['experiment', 'player', 'change', 'ci_lo', 'ci_hi', 'n_from', 'n_to'])


def stability_ceiling(onset, players, from_type='solo1', to_type='solo2'):
    ''' Return how well a player's mean in one session predicts it in another, next to how consistent
        each session is with itself. Takes the onset table, the player table and the two session types.

        consistency is the odd-even split-half reliability of each session, corrected to its full
        length. It is the most a between-session correlation could reach if players did not change at
        all, given the trials each played. corrected divides the observed correlation by that ceiling.
        A corrected value well below 1 means players genuinely behave differently between the sessions,
        rather than the finite trial count hiding a stable habit. '''

    rel = split_half(onset).query("split == 'odd_even'").set_index('session_type')['spearman_brown']
    wide = wide_table(players)[[from_type, to_type]].dropna()
    r = float(stats.pearsonr(wide[from_type], wide[to_type])[0])
    ceiling = float(np.sqrt(rel[from_type] * rel[to_type]))
    return pd.Series({'between_sessions_r': r, f'consistency_{from_type}': float(rel[from_type]),
                      f'consistency_{to_type}': float(rel[to_type]), 'ceiling': ceiling,
                      'corrected_r': r / ceiling, 'n_players': len(wide)})


def paired_change(players, from_type='solo1', to_type='solo2', group_col=None):
    ''' Return the mean change in centrality between two sessions, overall and optionally per group, with
        a standard error that treats the two players of an experiment as linked.
        Takes the player table, the two session types and an optional label column for groups.

        Groups come from the label each player held in the earlier session. Partners share a social
        session, so their changes are not independent, and the standard error is clustered by
        experiment. A group drawn from a single experiment gets its mean but no standard error. '''

    wide = wide_table(players)[[from_type, to_type]].dropna()
    frame = pd.DataFrame(dict(change=wide[to_type] - wide[from_type])).reset_index()
    if group_col is not None:
        labels = wide_table(players, group_col)[from_type].rename('group').reset_index()
        frame = frame.merge(labels, on=['experiment', 'player'])
    parts = [('all', frame)]
    if group_col is not None:
        parts += [(g, sub) for g, sub in frame.groupby('group')]

    rows = []
    for name, sub in parts:
        row = dict(group=name, n=len(sub), mean_change=float(sub['change'].mean()),
                   se=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
        # a clustered standard error needs at least two experiments in the group
        if sub['experiment'].nunique() >= 2:
            # fit the mean as an intercept-only model, so the standard error can be clustered
            fit = smf.ols('change ~ 1', sub).fit(cov_type='cluster', cov_kwds={'groups': sub['experiment']})
            lo, hi = fit.conf_int().loc['Intercept']
            row.update(se=float(fit.bse['Intercept']), ci_lo=float(lo), ci_hi=float(hi),
                       p=float(fit.pvalues['Intercept']))
        rows.append(row)
    return pd.DataFrame(rows).set_index('group')


def shrinkage_model(wide, from_type='solo1', to_type='solo2'):
    ''' Return an OLS fit of the later session's centrality on the earlier one, clustered by experiment.
        Takes the wide per-player table and the two session types.

        A slope of 1 means players keep their position, give or take a shared shift. A slope below 1
        means players who started at an extreme end the later session nearer the middle. The earlier
        value is a predictor here rather than being subtracted from the outcome, which avoids the
        built-in negative link between a change and its own starting value. '''

    frame = wide[[from_type, to_type]].dropna().reset_index()
    return smf.ols(f'{to_type} ~ {from_type}', frame).fit(
        cov_type='cluster', cov_kwds={'groups': frame['experiment']})


def social_or_practice_model(wide, drift):
    ''' Return an OLS fit of second-solo centrality on first-solo centrality, social centrality, and the
        drift within the first solo, clustered by experiment. Takes the wide per-player table and the
        table from within_session_drift.

        Two explanations for a second-solo change compete here. If the social session caused it, a
        player's social position should predict their second solo once their first solo is known. If
        it is simply more time on the task, a player already drifting during their first solo should
        keep going, so that drift should predict it. Neither can be ruled out by design, as every
        player took the sessions in the same order. '''

    slope = (drift[drift['session_type'] == 'solo1'].set_index(['experiment', 'player'])['slope']
             .rename('solo1_drift'))
    frame = wide[['solo1', 'social', 'solo2']].join(slope).dropna().reset_index()
    return smf.ols('solo2 ~ solo1 + social + solo1_drift', frame).fit(
        cov_type='cluster', cov_kwds={'groups': frame['experiment']})


def largest_changes(change, players, label_col, from_type='solo1', to_type='solo2', n=15):
    ''' Return the players whose centrality changed most between two sessions, with the interval on the
        change, their three session means and labels, and their partner's means.
        Takes the table from player_change, the player table, a label column and the two session types.

        Sorted by size of change. Every player is eligible, so a large change with a wide interval
        still appears, and the interval says how much weight it can bear. '''

    means = wide_table(players)
    labels = wide_table(players, label_col)
    rows = (change.set_index(['experiment', 'player'])
            .join(means.add_suffix('_mean')).join(labels.add_suffix('_label')))
    partner = means.copy()
    partner.index = pd.MultiIndex.from_tuples([(e, 1 - p) for e, p in means.index], names=means.index.names)
    rows = rows.join(partner.add_prefix('partner_'))
    return (rows.assign(abs_change=lambda d: d['change'].abs())
            .sort_values('abs_change', ascending=False)
            .drop(columns='abs_change').head(n).reset_index())


def transition_detail(players, label_col, from_type, to_type, value='mean_centrality', margin=0.1):
    ''' Return each label transition with how many players made it and how far they actually moved.
        Takes the player table, a label column, the two session types, the value that measures movement,
        and how close to a cut counts as marginal.

        A label change can be a large move or a small one that happened to cross a cut. median_change
        gives the size behind each cell, and share_marginal the fraction of players in it whose values
        sat within margin of a cut in both sessions, where the label is a near thing. '''

    labels = wide_table(players, label_col)
    values = wide_table(players, value)
    frame = pd.DataFrame(dict(a=labels[from_type], b=labels[to_type],
                              change=values[to_type] - values[from_type])).dropna()

    # measure how near each player sat to flipping label in both sessions
    if label_col == 'fixed':
        frac = wide_table(players, 'frac_peripheral')
        near = pd.DataFrame({st: np.min(np.abs(frac[st].to_numpy()[:, None] - np.array(FIXED_CUTS)), axis=1)
                             for st in (from_type, to_type)}, index=frac.index)
        frame['marginal'] = (near[from_type] < margin) & (near[to_type] < margin)
    else:
        post = wide_table(players, f'{label_col}_posterior')
        frame['marginal'] = (post[from_type] < POSTERIOR_MIN) | (post[to_type] < POSTERIOR_MIN)

    out = frame.groupby(['a', 'b']).agg(n=('change', 'size'), median_change=('change', 'median'),
                                        median_abs_change=('change', lambda v: v.abs().median()),
                                        share_marginal=('marginal', 'mean'))
    out.index.names = [from_type, to_type]
    return out


def binned_timeline(onset, groups, bin_size=10, value='centrality'):
    ''' Return mean centrality per trial bin, per session type, per player group, for plotting the three
        sessions as one timeline. Takes the onset table, a series mapping (experiment, player) to a group
        label, and the bin size.

        Each player is averaged within a bin first, so players with more trials do not dominate a bin.
        A trend running across the bins points to practice rather than to a change of session. '''

    g = groups.rename('group').rename_axis(['experiment', 'player']).reset_index()
    sub = onset.merge(g, on=['experiment', 'player']).assign(bin=lambda d: d['trial_pos'] // bin_size)
    per_player = sub.groupby(['group', 'session_type', 'bin', 'experiment', 'player'])[value].mean()
    out = (per_player.groupby(['group', 'session_type', 'bin'])
           .agg(mean='mean', sem='sem', n='size').reset_index())
    out['session_type'] = pd.Categorical(out['session_type'], _present_types(out))
    return out.sort_values(['group', 'session_type', 'bin']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# behavioural fingerprint of the central and peripheral groups
# ---------------------------------------------------------------------------

def _seconds(delta):
    ''' Return a time difference in seconds. Takes a timedelta or a plain number of seconds. '''

    if isinstance(delta, (pd.Timedelta, np.timedelta64)):
        return pd.Timedelta(delta).total_seconds()
    return float(delta)


def _first_value(trial, event, col):
    ''' Return the first non-missing value of a column on an event's rows, or NaN if there is none.
        Takes a trial, the event description and the column. '''

    vals = trial.loc[trial['eventDescription'] == event, col].dropna()
    return vals.iloc[0] if len(vals) else np.nan


def _trial_num(trial):
    ''' Return a trial's trialNum, or NaN if it has none. Takes a trial. '''

    tn = trial[globals.TRIAL_NUM].dropna()
    return float(tn.iloc[0]) if len(tn) else np.nan


def _trial_outcome(trial, social):
    ''' Return the winner's log id, whether they reached High, the trial type and the trial time (s),
        or None if the trial has no readable trigger. Takes a trial and whether it is social.

        High is wall 1 on High/Low trials, as elsewhere in the repo. On a forced trial both walls pay the
        same, so reaching High is NaN there. Trial time runs from slice onset to the trigger the server
        selected. A solo player is always the winner. '''

    try:
        winner = get_indices.get_trigger_activator(trial) if social else 0
        high = float(_first_value(trial, globals.SELECTED_TRIGGER_ACTIVATION, globals.WALL_TRIGGERED)
                     == get_indices.get_walls(trial=trial)[0])
        rt = _seconds(_first_value(trial, globals.SELECTED_TRIGGER_ACTIVATION, 'timeApplication')
                      - _first_value(trial, globals.SLICE_ONSET, 'timeApplication'))
    except (ValueError, TypeError, IndexError):
        return None
    ttype = trial[globals.TRIAL_TYPE].replace('', np.nan).dropna()
    ttype = ttype.iloc[0] if len(ttype) else ''
    # score the winner from the trial type on a forced trial, and from the wall reached otherwise
    if ttype in FORCED_SCORES:
        high, score = np.nan, FORCED_SCORES[ttype]
    else:
        score = HIGH_SCORE if high else LOW_SCORE
    return dict(winner=winner, chose_high=high, win_score=score, rt_s=rt, trial_type=ttype)


def _win_path(trial, winner):
    ''' Return the winner's route length from slice onset to the trigger, the straight-line distance, and
        their ratio. Takes a trial and the winner's log id. '''

    actual = float(trajectory_efficiency.player_actual_distance_trial(trial, winner))
    direct = float(trajectory_efficiency.player_direct_distance_trial(trial, winner))
    return dict(winner_path_len=actual, winner_direct_len=direct,
                path_ratio=(actual / direct) if direct > 0 else np.nan)


def _smooth_toward(t, target, smooth_s=NET_SMOOTH_S, chunk=500):
    ''' Return target after the per-frame smoothing y_k = y_(k-1) + (x_k - y_(k-1)) * min(dt_k / smooth_s, 1).
        Takes frame times (s) and the target values.

        Solved in closed form over chunks of frames rather than in a frame loop. With P_k the running
        product of (1 - a), y_k = P_k * (y_0 + sum(a_j * x_j / P_j)). A chunk ends early at any frame with a
        large step (a >= 0.5), so P never reaches zero. '''

    n = len(target)
    a = np.minimum(np.diff(t, prepend=t[0]) / smooth_s, 1.0)
    out = np.empty(n)
    starts = sorted(set(range(0, n, chunk)) | set(np.flatnonzero(a >= 0.5).tolist()))
    y = target[0]
    for s, e in zip(starts, starts[1:] + [n]):
        # take the chunk's first frame directly, as its step may be large
        y = y + (target[s] - y) * a[s]
        out[s] = y
        if e - s > 1:
            aa, xx = a[s + 1:e], target[s + 1:e]
            P = np.cumprod(1 - aa)
            out[s + 1:e] = P * (y + np.cumsum(aa * xx / P))
            y = out[e - 1]
    return out


def simulate_network_view(t, v, angle=False, tick_s=NET_TICK_S, lag_ticks=NET_LAG_TICKS, smooth_s=NET_SMOOTH_S):
    ''' Return a trace as the host would log it for a networked client. Takes frame times (s), the values,
        and whether they are angles in degrees.

        The owner sends its latest value each tick. The host renders lag_ticks behind, interpolating
        linearly between ticks, then smooths the result toward that target. Missing frames stay missing.
        Angles are unwrapped first and wrapped back to 0-360 after. '''

    t = np.maximum.accumulate(np.asarray(t, float))
    v = np.asarray(v, float)
    missing = np.isnan(v)
    if missing.all():
        return v.copy()
    # fill missing frames from the last value, so they do not spread through the filter
    filled = pd.Series(v).ffill().bfill().to_numpy()
    if angle:
        filled = np.rad2deg(np.unwrap(np.deg2rad(filled)))

    # take the owner's latest value at each tick, and interpolate between ticks one lag behind
    ticks = np.arange(t[0], t[-1] + tick_s, tick_s)
    idx = np.clip(np.searchsorted(t, ticks, side='right') - 1, 0, len(t) - 1)
    target = np.interp(t - lag_ticks * tick_s, ticks, filled[idx])
    out = _smooth_toward(t, target, smooth_s)

    if angle:
        out = np.mod(out, 360.0)
    out[missing] = np.nan
    return out


def network_view(trial_list, log_ids=NETWORK_VIEW_IDS, lag_ticks=NET_LAG_TICKS):
    ''' Return a copy of a session's trials with each listed player's x, z and yaw passed through
        simulate_network_view. Takes a trial list, the log ids to convert and the render lag in ticks.

        The whole session is converted as one trace, so the smoothing carries across trial boundaries as it
        did in the game. Trial indices and boundaries are unchanged.

        lag_ticks is 1 for the host's view of a client and 2 for a client's view of anyone else. '''

    df = pd.concat(trial_list)
    t = (df['timeLocal'] - df['timeLocal'].iloc[0]).dt.total_seconds().to_numpy()
    df = df.copy()
    for log_id in log_ids:
        for col in (globals.PLAYER_LOC_DICT[log_id]['xloc'], globals.PLAYER_LOC_DICT[log_id]['yloc']):
            df[col] = simulate_network_view(t, df[col].to_numpy(float), lag_ticks=lag_ticks)
        ycol = globals.PLAYER_ROT_DICT[log_id]['yrot']
        df[ycol] = simulate_network_view(t, df[ycol].to_numpy(float), angle=True, lag_ticks=lag_ticks)
    # split back into the original trials
    ends = np.cumsum([len(tr) for tr in trial_list])
    return [df.iloc[e - len(tr):e] for tr, e in zip(trial_list, ends)]


def recovered_trial_list(trial_list, ticks, log_id=1):
    ''' Return a trial list with one player's x, z and yaw taken from their recovered trace.
        Takes the trial list, that player's recovered tick table and their log id.

        The session is converted in one piece and split back, so a trial boundary changes nothing. '''

    df = nr.frames_from_ticks(pd.concat(trial_list), log_id, ticks)
    ends = np.cumsum([len(tr) for tr in trial_list])
    return [df.iloc[e - len(tr):e] for tr, e in zip(trial_list, ends)]


def true_view_trial_list(trial_list, ticks):
    ''' Return a trial list holding where both players really were. Takes a social trial list and player
        1's recovered tick table.

        Player 0 is logged locally and is already true. Player 1 is replaced by its recovered trace, which
        undoes the delay and smoothing the host applied. Use this for questions about the players, not
        about what either of them could see. '''

    return recovered_trial_list(trial_list, ticks, log_id=1)


def screen_view_trial_list(trial_list, ticks, viewer):
    ''' Return a trial list holding what one player's screen showed. Takes a social trial list, player 1's
        recovered tick table and the viewer's log id.

        A player sees itself directly and the other player through the network, so each view mixes one true
        trace with one filtered one.

        Viewer 0 hosts. Its own values are logged as they happened, and the logged player 1 values are
        already the copy the host rendered, so the session as logged is what player 0 saw and nothing is
        done to it.

        Viewer 1 is the client. Its own values are the recovered trace, and it rendered player 0 two ticks
        behind rather than one, as a client is not the server (`NetworkTransform.cs:2799`). Player 0's
        logged trace is put through that path. '''

    if viewer == 0:
        return list(trial_list)
    seen = network_view(trial_list, log_ids=(0,), lag_ticks=NET_CLIENT_LAG_TICKS)
    return recovered_trial_list(seen, ticks, log_id=1)


def screen_view_trial_lists(trial_lists, ticks_table=None, viewers=(0, 1), verbose=True):
    ''' Return each session as each player's screen showed it, as a list of dicts from player id to
        trial list. Takes the social trial lists, optionally the saved recovered tick table.

        Feed this to conditioned_player_choice or opponent_visibility as `screen_views` to measure
        opponent visibility on each player's own screen. Viewer 0 is the session as logged and costs
        nothing. Viewer 1 is built, so holding it for every session takes about as much memory again as
        the social pickle.

        Without a tick table each session's player 1 is recovered on the spot, which takes a few seconds
        a session and works for any set of sessions. **The saved tick table is keyed by position in
        `social_trials_list.pkl`**, so pass it only when trial_lists is that exact list in that order.
        Handing it a different or shorter set pairs each session with another session's trace. '''

    if ticks_table is not None and len(trial_lists) != ticks_table['experiment'].nunique():
        raise ValueError(
            f'{len(trial_lists)} trial lists against {ticks_table["experiment"].nunique()} sessions in '
            'the tick table. The table is keyed by position in social_trials_list.pkl, so either pass '
            'that exact list or leave ticks_table out and let each session be recovered on the spot.')

    out = []
    for i, trial_list in enumerate(trial_lists):
        if ticks_table is None:
            ticks = nr.recover_player(pd.concat(trial_list), 1)
        else:
            ticks = ticks_table[ticks_table['experiment'] == i].rename(
                columns={'p1_x': 'x', 'p1_z': 'z', 'p1_yaw': 'yaw'})
        out.append({v: screen_view_trial_list(trial_list, ticks, v) for v in viewers})
        if verbose:
            print(f'{i + 1}/{len(trial_lists)}', end='  ', flush=True)
    return out


_SCREEN_VIEWS = None        # built once per python session, see load_screen_views


def load_screen_views(trial_lists=None, recompute=False, verbose=True):
    ''' Return each social session as each player's screen showed it, building it on first call and
        reusing it after. Takes the social trial lists, a recompute flag and a verbosity flag.

        Kept in memory rather than on disk, as it is about the size of the social pickle again and takes a
        few minutes to build. Conditions such as wall separation or chosen wall do not change it, so build
        it once and pass it to every call. Pass recompute to drop what is held and build again. '''

    global _SCREEN_VIEWS
    if _SCREEN_VIEWS is None or recompute:
        if trial_lists is None:
            trial_lists = pd.read_pickle(
                os.path.join(data_strings.DATAFRAME_PATH, data_strings.FILENAME_SOCIAL + '.pkl'))
        _SCREEN_VIEWS = screen_view_trial_lists(trial_lists, load_social_ticks(), verbose=verbose)
    return _SCREEN_VIEWS


def opponent_view_rows(trial_list, ticks, experiment, fov=HUMAN_FOV, start=0):
    ''' Return one row per trial per viewer, comparing where the opponent really was at slice onset with
        where that viewer's screen put them. Takes a social trial list, player 1's recovered tick table
        and the experiment index.

        `true_*` columns are measured on the true positions of both players. `seen_*` columns are measured
        on what the viewer's screen showed. The pair answers how often the network changed whether a
        player could see their opponent. '''

    true_list = true_view_trial_list(trial_list, ticks)
    screens = {v: screen_view_trial_list(trial_list, ticks, v) for v in (0, 1)}

    rows = []
    for i in range(start, len(trial_list)):
        for viewer in (0, 1):
            row = dict(experiment=experiment, trial_num=_trial_num(trial_list[i]), viewer=viewer)
            for tag, lst in (('true', true_list), ('seen', screens[viewer])):
                theta = opponent_visibility.get_angle_of_opponent_from_player_trial(viewer, trial=lst[i])
                row[f'{tag}_angle'] = float(np.rad2deg(np.asarray(theta, dtype=float)))
                row[f'{tag}_visible'] = opponent_visibility.get_other_visible_trial(
                    viewer, fov, trial=lst[i])
            rows.append(row)
    return rows


def collect_opponent_views(trial_lists, ticks_table, fov=HUMAN_FOV, start=0, verbose=True):
    ''' Return the opponent view table for every social session, stacked. Takes the social trial lists and
        the saved recovered tick table. '''

    rows = []
    for i, tl in enumerate(trial_lists):
        ticks = ticks_table[ticks_table['experiment'] == i].rename(
            columns={'p1_x': 'x', 'p1_z': 'z', 'p1_yaw': 'yaw'})
        rows += opponent_view_rows(tl, ticks, i, fov=fov, start=start)
        if verbose:
            print(f'{i + 1}/{len(trial_lists)}', end='  ', flush=True)
    return pd.DataFrame(rows)


OPPONENT_VIEWS = 'social_opponent_views'        # saved in data_strings.DATAFRAME_PATH


def load_opponent_views():
    ''' Return the saved opponent view table. Takes no inputs. Rebuild with collect_opponent_views, which
        is not automatic as it takes a few minutes. '''

    return pd.read_pickle(os.path.join(data_strings.DATAFRAME_PATH, OPPONENT_VIEWS + '.pkl'))


def _session_fingerprint_rows(trial_list, experiment, session_type, id_map, start, fov, vis_fn,
                              with_iti=True):
    ''' Return the ITI, visibility, route and outcome rows for one session, as a dict of row lists.
        Takes a trial list, the experiment index, the session type, a map from log player id to player,
        the number of opening trials to cut, the FoV and the visibility function.

        ITI rows are labelled with the trial they lead into. The run into the first trial of the list
        follows no trial end, so it is dropped. The ITI frames after the last trial lead into a trial
        that was never saved, so they are dropped too. '''

    social = session_type == 'social'
    out = dict(iti=[], vis=[], win=[], outcome=[])
    if len(trial_list) <= start:
        return out
    key = dict(experiment=experiment, session_type=session_type)
    nums = [_trial_num(t) for t in trial_list]

    # get ITI movement over the ITIs leading into kept trials, skipping the list's first trial
    last = trial_list[-1]
    df = pd.concat(trial_list[:-1] + [last[last['trial_epoch'] != globals.ITI]])
    iti_trials = set(nums[max(start, 1):])
    if with_iti:
        for log_id, player in id_map.items():
            w = mc.iti_metrics(df, log_id, keep_trials=iti_trials)
            if len(w):
                out['iti'].append(w.assign(player=player, **key))

    prev_logged = {log_id: 0.0 for log_id in id_map}       # cumulative score at the previous trial end
    for t, trial in enumerate(trial_list):
        res = _trial_outcome(trial, social)
        for log_id, player in id_map.items():
            # take this trial's score change from the cumulative logged score
            logged = float(_first_value(trial, globals.TRIAL_END, globals.PLAYER_SCORE_DICT[log_id]['score']))
            score_logged = logged - prev_logged[log_id]
            if not np.isnan(logged):
                prev_logged[log_id] = logged
            if t < start:
                continue
            row = dict(key, player=player, trial_num=nums[t])
            if res is not None:
                won = float(res['winner'] == log_id)
                score = won * res['win_score']
                out['outcome'].append(dict(row, trial_type=res['trial_type'], won=won,
                                           chose_high=res['chose_high'] if won else np.nan,
                                           score=score, score_logged=score_logged, rt_s=res['rt_s']))
            # get wall visibility from this player's view
            try:
                v = vis_fn(trial, log_id, fov=fov)
            except (IndexError, ValueError):
                v = None
            if v is not None:
                out['vis'].append(dict(v, **row))
        # measure the winner's route to the wall they triggered
        if t >= start and res is not None and res['winner'] in id_map:
            try:
                path = _win_path(trial, res['winner'])
            except (IndexError, ValueError, TypeError):
                continue
            out['win'].append(dict(key, player=id_map[res['winner']], trial_num=nums[t], **path))
    return out



def collect_fingerprint_tables(trial_lists, kind, first_solo_cut=FIRST_SOLO_CUT, fov=HUMAN_FOV,
                               vis_fn=mc.visibility_metrics, network_ids=(), recovered=None,
                               with_iti=False, verbose=True):
    ''' Return the long fingerprint tables for every session of one kind, as a dict of dataframes.
        Takes the social or solo trial lists (in pickle order) and which kind they are.

        Trials follow the onset table (first solo loses its first first_solo_cut trials, the others keep
        all). Every row carries experiment, player, session_type and trial_num. Keys are:
          iti     - one row per ITI x player: movement, scanning and resting position
          vis     - one row per trial x player: wall visibility latency and flags
          win     - one row per won trial: the winner's route from slice onset to the trigger
          outcome - one row per trial x player: won, reached High, score and trial time

        score is 1.0 (reached High), 0.4 (reached Low) or 0 (lost), the agent scale. A forced trial scores
        from its type, as both its walls pay the same. score_logged is the change in the logged cumulative
        score (50 for High and 20 for Low in the human builds), kept as a check.

        Positions and yaw of the log ids in network_ids are first passed through network_view, so locally
        logged players are measured as the networked social client is. Pass network_ids=() to use the
        positions as logged. '''

    parts = dict(iti=[], vis=[], win=[], outcome=[])
    for i, trial_list in enumerate(trial_lists):
        if kind == 'social':
            experiment, session_type, id_map = i, 'social', {0: 0, 1: 1}
        elif kind == 'solo':
            experiment, player, k = solo_position(i)
            session_type, id_map = SOLO_TYPES[k], {0: player}
        else:
            raise ValueError(f"kind must be 'social' or 'solo', got {kind!r}")
        start = first_solo_cut if session_type == 'solo1' else 0
        # take the networked player's frames from their recovered trace, so head angle and position at a
        # given frame are theirs rather than the host's delayed copy of them
        if recovered is not None and kind == 'social' and experiment in recovered:
            trial_list = recovered_trial_list(trial_list, recovered[experiment])
        if network_ids:
            trial_list = network_view(trial_list, [i for i in network_ids if i in id_map])
        rows = _session_fingerprint_rows(trial_list, experiment, session_type, id_map, start, fov, vis_fn,
                                         with_iti=with_iti)
        for name, r in rows.items():
            parts[name].extend(r)
        if verbose and i % 20 == 19:
            print(f"{kind} {i + 1}/{len(trial_lists)}", end='  ')

    # stack each table, with the keys first
    tables = {}
    for name, r in parts.items():
        df = pd.concat(r, ignore_index=True) if name == 'iti' and r else pd.DataFrame(r)
        lead = [c for c in list(KEYS) + ['trial_num'] if c in df.columns]
        tables[name] = df[lead + [c for c in df.columns if c not in lead]]
    return tables


def tick_paths():
    ''' Return the saved social and solo tick table paths. Takes no inputs. '''

    return {name: os.path.join(data_strings.DATAFRAME_PATH, name + '.pkl')
            for name in (nr.RECOVERED_SOCIAL, nr.RECOVERED_SOLO)}


def load_social_ticks():
    ''' Return the saved social tick table, one row per tick with both players side by side.
        Takes no inputs. Player 1 is recovered from the host's log, player 0 read locally. '''

    return pd.read_pickle(tick_paths()[nr.RECOVERED_SOCIAL])


def load_tick_traces(session_types=SESSION_TYPES):
    ''' Return the long tick table for every session, one row per player per tick.
        Takes the session types to include.

        Social comes from the recovery, solo from sampling the local trace onto the same grid. Rebuild
        either with network_compensation.build_recovered_social or local_tick_table; neither is automatic,
        as the social pass takes about 20 minutes. '''

    parts = []
    if 'social' in session_types:
        parts.append(nr.social_ticks_long(load_social_ticks()))
    if set(session_types) & set(SOLO_TYPES):
        solo = pd.read_pickle(tick_paths()[nr.RECOVERED_SOLO])
        parts.append(solo[solo['session_type'].isin(session_types)])
    return pd.concat(parts, ignore_index=True)


def iti_keep_map(onset, traces):
    ''' Return the trials whose ITI is analysed, per player and session. Takes the onset table and the
        tick table.

        An ITI is kept when it leads into an analysed trial, which is what the onset table holds. The run
        before a session's first logged trial follows no trial end, so it is not an ITI and goes. '''

    keys = list(KEYS)
    first = traces.groupby(keys)['trial_num'].min()
    out = {}
    for k, g in onset.groupby(keys):
        out[k] = set(g['trial_num'].dropna().astype(float)) - {float(first.get(k, np.nan))}
    return out


def load_iti_table(recompute=False):
    ''' Return one row per ITI per player, measured on the tick grid, from cache when possible.
        Takes a recompute flag. '''

    paths = tick_paths()
    stamp = dict(schema='ticks', cutoffs=(nr.STILL_SPEED_PER_TICK, nr.STILL_ROT_PER_TICK),
                 ticks={n: os.path.getmtime(p) for n, p in paths.items()})

    def compute():
        traces = load_tick_traces()
        onset = load_tables()['onset']
        return iti_metrics_ticks(traces, keep_map=iti_keep_map(onset, traces))

    return data_saving.cached(ITI_CACHE, compute, recompute=recompute, stamp=stamp)


def load_fingerprint_tables(first_solo_cut=FIRST_SOLO_CUT, fov=HUMAN_FOV, network_ids=(),
                            recovered=True, recompute=False):
    ''' Return the fingerprint tables for every human session, from cache when possible.
        Takes the first-solo cut, the FoV, any log ids to pass through network_view, whether to take
        social player 1 from their recovered trace, and a recompute flag.

        The ITI table is measured on the tick grid and comes from load_iti_table. The other tables are
        measured on frames, with social player 1's frames taken from their recovered trace when
        recovered is set, so their head angle and position are their own rather than the host's delayed
        copy. Loads the social pickle, then the solo pickle, and frees each before the next. Each needs
        several GB of memory. '''

    pickles = {name: os.path.join(data_strings.DATAFRAME_PATH, name + '.pkl')
               for name in (data_strings.FILENAME_SOCIAL, data_strings.FILENAME_SOLO)}
    stamp = dict(schema='recovered-frames-logged-times', first_solo_cut=first_solo_cut, fov=fov,
                 network_ids=tuple(network_ids), recovered=bool(recovered),
                 pickles={n: os.path.getmtime(p) for n, p in pickles.items()})
    if recovered:
        stamp['ticks'] = os.path.getmtime(tick_paths()[nr.RECOVERED_SOCIAL])

    def compute():
        # get the recovered ticks once, then hand each social session its own
        ticks = None
        if recovered:
            ticks = {e: g for e, g in load_social_ticks().groupby('experiment')}
            ticks = {e: g.rename(columns={'p1_x': 'x', 'p1_z': 'z', 'p1_yaw': 'yaw'}) for e, g in ticks.items()}
        parts = []
        for kind, name in (('social', data_strings.FILENAME_SOCIAL), ('solo', data_strings.FILENAME_SOLO)):
            trial_lists = data_saving.load_data(name)
            parts.append(collect_fingerprint_tables(trial_lists, kind, first_solo_cut, fov,
                                                    network_ids=network_ids,
                                                    recovered=ticks if kind == 'social' else None))
            del trial_lists
        tables = {name: pd.concat([p[name] for p in parts], ignore_index=True)
                  for name in parts[0] if name != 'iti'}
        tables['iti'] = load_iti_table()
        return tables

    return data_saving.cached(FINGERPRINT_CACHE, compute, recompute=recompute, stamp=stamp)


def extreme_groups(players, frac=EXTREME_FRAC, value='mean_centrality'):
    ''' Return the player table with a 'group' column, labelling the most central and most peripheral
        players of each session type. Takes the player table and the share of players in each group.

        Players are ranked by value within each session type. The top frac are central, the bottom frac
        peripheral, and the rest middle. Ties keep table order. '''

    players = players.copy()
    # rank within each session type, and count the players in each group
    by = players.groupby('session_type')[value]
    rank = by.rank(method='first').to_numpy()
    size = by.transform('size').to_numpy()
    n = np.round(frac * size)
    players['group'] = np.select([rank <= n, rank > size - n], ['peripheral', 'central'], 'middle')
    return players


def build_player_fingerprint(players, tables, slow_trial_s=mc.SLOW_TRIAL_S):
    ''' Return one row per player per session type with every fingerprint metric.
        Takes the player table (with any group columns) and the dict from load_fingerprint_tables.

        Latency to the wall, the slow-trial share and P(High) use the player's own wins, so they describe
        that player's route rather than their opponent's. P(High) counts High/Low trials only. mean_score
        and score_rate use every trial, with a lost trial scoring 0. score_rate is total score over total
        trial time. Visibility uses every trial, as the player's own view. '''

    keys = list(KEYS)
    fp = players.rename(columns={'mean_centrality': 'onset_centrality'})

    # average ITI behaviour per player
    iti = tables['iti'].groupby(keys)[['prop_no_translational', 'prop_no_rotational', 'proportion_resting',
                                       'iti_path_len', 'rot_total', 'iti_centrality']].mean()

    # summarise wall visibility per player, as mc.build_fingerprint does per model
    vis = tables['vis'].assign(visible_at_onset=lambda d: d['visible_at_onset'].astype(float),
                               ever_visible=lambda d: d['ever_visible'].astype(float))
    vis_by = vis.groupby(keys).agg(frac_visible_at_onset=('visible_at_onset', 'mean'),
                                   frac_never_visible=('ever_visible', lambda s: 1 - s.mean()))
    # late_: only trials with no wall in view at onset but one seen later. all_: every trial with a wall
    # seen, counting visible-at-onset as 0
    late = vis[(vis['visible_at_onset'] == 0) & (vis['ever_visible'] == 1)].groupby(keys)['latency_to_wall_visible']
    vis_by['late_latency_to_wall_visible'] = late.mean()
    vis_by['n_late_latency_to_wall_visible'] = late.size()
    vis_by['all_latency_to_wall_visible'] = vis[vis['ever_visible'] == 1].groupby(keys)['latency_to_wall_visible'].mean()

    # average the player's route on the trials they won
    win = tables['win'].groupby(keys)[['winner_path_len', 'winner_direct_len', 'path_ratio']].mean()

    # take latency, slow trials and P(High) from the player's own wins
    out = tables['outcome']
    wins = out[out['won'] == 1]
    won_by = wins.groupby(keys).agg(n_wins=('rt_s', 'size'), latency_to_reach_wall=('rt_s', 'median'),
                                    frac_trials_over_5s=('rt_s', lambda s: float(np.mean(s > slow_trial_s))))
    hl = wins[wins['trial_type'] == globals.HIGH_LOW]
    won_by['p_high'] = hl.groupby(keys)['chose_high'].mean()
    won_by['n_high_low_wins'] = hl.groupby(keys).size()

    # score over every trial. Sum score and trial time over the session, then divide once
    timed = out[out['rt_s'].notna()]
    all_by = out.groupby(keys).agg(n_outcome_trials=('score', 'size'), mean_score=('score', 'mean'),
                                   win_rate=('won', 'mean'))
    all_by['score_rate'] = timed.groupby(keys)['score'].sum() / timed.groupby(keys)['rt_s'].sum()
    # a solo player wins every trial, so win_rate says nothing there
    all_by.loc[all_by.index.get_level_values('session_type') != 'social', 'win_rate'] = np.nan

    for part in (iti, vis_by, win, won_by, all_by):
        fp = fp.merge(part.reset_index(), on=keys, how='left')
    return fp


def group_variance(fingerprint, session_type, metrics=FP_METRICS):
    ''' Return, per metric, the share of spread falling between the central and peripheral groups (eta^2)
        and each group's mean, for one session type. Takes the fingerprint table and the session type.

        Descriptive only. The groups come from onset centrality, so metrics tracking centrality separate
        by construction, and partners can fall in the same group, so players are not independent. '''

    sub = fingerprint[fingerprint['session_type'] == session_type].assign(strategy=lambda d: d['group'])
    return mc.between_strategy_variance(sub, [m for m in metrics if sub[m].notna().any()])


def group_medians(fingerprint, metrics=FP_METRICS):
    ''' Return the median of each metric per session type and group, one row per metric.
        Takes the fingerprint table. Middle players are left out. '''

    sub = fingerprint[fingerprint['group'].isin(GROUPS)]
    med = sub.groupby(['session_type', 'group'])[metrics].median().T
    cols = [(st, g) for st in SESSION_TYPES for g in GROUPS if (st, g) in med.columns]
    return med[cols]


def wall_visible_curve(fingerprint, vis, max_s=3.0):
    ''' Return the proportion of trials with a trial wall in view by each time after slice onset, one row
        per player, session type and timepoint, with the player's group. Takes the fingerprint table and
        the vis table. '''

    keys = list(KEYS)
    # give each player-session one id so the model curve function can group by it
    ids = fingerprint[keys + ['group']].reset_index(drop=True).assign(model=lambda d: d.index)
    v = vis.merge(ids[keys + ['model']], on=keys)
    curve = mc.wall_visible_curve(v, max_s=max_s)
    return curve.merge(ids, on='model').drop(columns='model')


# ---------------------------------------------------------------------------
# ITI movement measured on the network tick grid
# ---------------------------------------------------------------------------

def iti_metrics_ticks(traces, keep_map=None, tick_s=nr.TICK_S, speed=nr.STILL_SPEED_PER_TICK,
                      rot_thresh=nr.STILL_ROT_PER_TICK, arena_radius=mc.ARENA_RADIUS, min_pairs=2):
    ''' Return one row per ITI per player, measured on the tick grid rather than on logged frames.
        Takes the long tick table (one row per player per tick) and an optional map from player-session
        to the trials to keep.

        Social player 1 is only recoverable at the tick rate, so every player is measured over the same
        33 ms step and against cutoffs scaled to it (READMEs/network_compensation.md). An ITI is a run of consecutive
        ITI ticks, labelled with the trial it leads into. Steps spanning a block join are dropped, as the
        two ticks either side of one are further apart than a tick. '''

    keys = list(KEYS)
    df = traces.sort_values(keys + ['t']).reset_index(drop=True)
    g = df.groupby(keys, sort=False).ngroup().to_numpy()
    t = df['t'].to_numpy(float)
    is_iti = df['is_iti'].to_numpy(bool)
    x, z = df['x'].to_numpy(float), df['z'].to_numpy(float)
    yaw = df['yaw'].to_numpy(float)
    tn = df['trial_num'].to_numpy(float)

    # number the ITI runs by row order, before any timing is considered, so a block join inside one ITI
    # cannot split it into two. A run is a stretch of ITI ticks within one session
    starts = is_iti & ~np.r_[False, is_iti[:-1] & (g[1:] == g[:-1])]
    iti_run = np.cumsum(starts)

    # pair each tick with the next, within one session, one ITI run and one tick of each other
    dt = t[1:] - t[:-1]
    valid = (g[1:] == g[:-1]) & (dt > 0) & (dt <= 1.5 * tick_s) & is_iti[1:] & is_iti[:-1]

    # take the shortest turn between two logged angles, so a wrap past 360 is not a full spin
    d_yaw = np.abs((yaw[1:] - yaw[:-1] + 180) % 360 - 180)
    pairs = pd.DataFrame(dict(
        g=g[:-1][valid], run=iti_run[:-1][valid],
        step=np.hypot(x[1:] - x[:-1], z[1:] - z[:-1])[valid],
        rot=d_yaw[valid],
        radius=np.hypot(x[:-1], z[:-1])[valid],
        trial_num=np.maximum(tn[1:], tn[:-1])[valid]))

    # an ITI is named by the trial it leads into, which its closing ticks carry. A run whose ticks all
    # carry one trial has no trial after it, so it is the wait at the end of the session and not an ITI
    by_run = pd.Series(tn).groupby(iti_run)
    run_trial = by_run.max()
    pairs['trial_num'] = pairs['run'].map(run_trial)
    pairs = pairs[pairs['run'].map(run_trial - by_run.min()) > 0]
    by = pairs.groupby(['g', 'run'], sort=False)
    out = by.agg(n_pairs=('step', 'size'), iti_path_len=('step', 'sum'), rot_total=('rot', 'sum'),
                 prop_no_translational=('step', lambda v: float(np.mean(v < speed))),
                 prop_no_rotational=('rot', lambda v: float(np.mean(v < rot_thresh))),
                 mean_radius=('radius', 'mean'), trial_num=('trial_num', 'max')).reset_index()
    out['proportion_resting'] = by.apply(
        lambda d: float(np.mean((d['step'] < speed) & (d['rot'] < rot_thresh))), include_groups=False).to_numpy()
    out['iti_centrality'] = 1 - out['mean_radius'] / arena_radius
    out = out[out['n_pairs'] >= min_pairs]

    # put the session keys back, then keep only the ITIs leading into analysed trials
    lookup = df.loc[~df.duplicated(subset=keys), keys].reset_index(drop=True)
    lookup['g'] = df.groupby(keys, sort=False).ngroup().drop_duplicates().to_numpy()
    out = out.merge(lookup, on='g').drop(columns=['g', 'run', 'mean_radius'])
    if keep_map is not None:
        keep = [float(r.trial_num) in keep_map.get((r.experiment, r.player, r.session_type), set())
                for r in out.itertuples()]
        out = out[keep]
    return out[keys + ['trial_num', 'n_pairs', 'prop_no_translational', 'prop_no_rotational',
                       'proportion_resting', 'iti_path_len', 'rot_total', 'iti_centrality']].reset_index(drop=True)
