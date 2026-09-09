#!/usr/bin/env python
# coding: utf-8

# Characterise a batch of RL agent models by their waiting strategy.
#
# A batch of self-play inference runs (one subfolder per model) is reduced to a few long
# tables, then to one row per model. The characterisation axis is centrality at slice
# onset, and the models split into a broadly central and a broadly peripheral group.
# The remaining metrics form a behavioural fingerprint used to tell models apart within
# a strategy and to pick a spread of models for a tournament.
#
# The notebooks in demos/strategy_characterisation/ hold the per-batch config, the plots
# and the written-up results. Everything that computes a number lives here.

import contextlib
import io

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import globals
import data_extraction.get_indices as get_indices
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import analysis.trajectory_efficiency as trajectory_efficiency
from parse_data import identify_sim_filepaths, prepare_sim_data


DEFAULT_FOV = 110.0                 # agent camera FoV (total degrees); get_wall_visible tests theta < FoV/2
FS = globals.RECORDING_FREQUENCY    # 50 Hz logging
ARENA_RADIUS = 36.21 / 2            # octagon centred at (0, 0); centrality 1 = centre, 0 = wall
ITI_EPOCHS = [globals.ITI, globals.TRIAL_STARTED]  # "ITI" here = trial end -> next slice onset
STATIONARY_SPEED = 0.05             # units/frame; below this a frame counts as "not translating"
STATIONARY_ROT = 0.5                # deg/frame; below this a frame counts as "not rotating"
PLAYERS = (0, 1)                    # self-play: pool both agents

# per-model metrics making up the behavioural fingerprint
FP_METRICS = ['onset_centrality', 'iti_centrality', 'prop_no_translational', 'prop_no_rotational',
              'proportion_resting', 'iti_path_len', 'rot_total', 'winner_path_len',
              'path_ratio', 'frac_visible_at_onset',
              'late_latency_to_wall_visible', 'all_latency_to_wall_visible',
              'latency_to_reach_wall', 'frac_never_visible',
              'frac_trials_over_5s', 'p_high', 'mean_score', 'score_rate']

# the fingerprint without the two centrality columns, so PCA shows what else varies
PCA_METRICS = [m for m in FP_METRICS if 'centrality' not in m]

STRAT_PALETTE = {'central': '#4C72B0', 'peripheral': '#DD8452', 'intermediate': '#9E9E9E',
                 'unassigned': '#D9D9D9'}

# the two strategies that get compared. 'intermediate' models sit between the clusters, so they
# stay in the fingerprint table but are left out of the group comparisons and their figures
MAIN_STRATEGIES = ('central', 'peripheral')

SLOW_TRIAL_S = 5.0                  # a trial slower than this counts towards frac_trials_over_5s

# ---------------------------------------------------------------------------
# per-trial and per-ITI metrics
# ---------------------------------------------------------------------------

def slice_onset_centrality(trial, player_id, arena_radius=ARENA_RADIUS):
    ''' Return the centrality of one agent at its slice-onset frame (1 = centre, 0 = wall),
        plus the raw x,z for occupancy maps. Takes a trial and player id.
        Returns None if the slice onset or position is missing. '''

    x, z = get_indices.get_player_slice_onset_loc(trial, player_id)
    x, z = float(x), float(z)
    if np.isnan(x) or np.isnan(z):
        return None
    return dict(player=player_id, x=x, z=z,
                centrality=float(1 - np.hypot(x, z) / arena_radius))


def iti_metrics(df, player_id, arena_radius=ARENA_RADIUS, keep_trials=None):
    ''' Return one row per ITI (trial end -> next slice onset) with scanning (yaw rotation),
        movement (path length, resting proportions) and resting centrality for one agent.
        Takes a session dataframe and player id.

        An ITI is labelled with the trial it leads into, the highest trialNum in the run. Pass
        keep_trials to drop the runs leading into trials the analysis excludes. That removes the
        run before the session's first trial, which is only the wait after spawning and holds no
        ITI frames at all, and the run before an unfinished final trial. '''

    xcol = globals.PLAYER_LOC_DICT[player_id]['xloc']
    zcol = globals.PLAYER_LOC_DICT[player_id]['yloc']
    ycol = globals.PLAYER_ROT_DICT[player_id]['yrot']  # yaw rotation (degrees)

    is_iti = df['trial_epoch'].isin(ITI_EPOCHS)
    run_id = (is_iti != is_iti.shift()).cumsum()            # id each contiguous ITI/non-ITI run

    rows = []
    for _, grp in df[is_iti].groupby(run_id[is_iti]):
        # an ITI leads into the trial whose 'trial started' frames close the run
        upcoming = grp[globals.TRIAL_NUM].dropna()
        if keep_trials is not None and (upcoming.empty or float(upcoming.max()) not in keep_trials):
            continue
        # mask x, z and yaw together so translation and rotation align frame-for-frame
        x = grp[xcol].to_numpy(float); z = grp[zcol].to_numpy(float); yaw = grp[ycol].to_numpy(float)
        ok = ~np.isnan(x) & ~np.isnan(z) & ~np.isnan(yaw); x, z, yaw = x[ok], z[ok], yaw[ok]
        if len(x) < 3:
            continue
        unw = np.rad2deg(np.unwrap(np.deg2rad(yaw)))         # unwrap so large spins don't fold back
        step = np.hypot(np.diff(x), np.diff(z))              # per-frame translation
        rot = np.abs(np.diff(unw))                           # per-frame rotation (degrees)
        no_trans = step < STATIONARY_SPEED                   # frame with no translation
        no_rot = rot < STATIONARY_ROT                        # frame with no rotation
        rows.append(dict(
            player=player_id,
            n_frames=len(x),
            rot_range=float(unw.max() - unw.min()),
            rot_total=float(rot.sum()),
            iti_path_len=float(step.sum()),
            prop_no_translational=float(np.mean(no_trans)),        # no translation (may still rotate)
            prop_no_rotational=float(np.mean(no_rot)),             # no rotation (may still translate)
            proportion_resting=float(np.mean(no_trans & no_rot)),  # neither: the all-zero action the penalty-break exempts
            iti_centrality=float(1 - np.mean(np.hypot(x, z)) / arena_radius),
        ))
    return pd.DataFrame(rows)


def winner_path_metrics(trial):
    ''' Return the distance the winning agent covered between slice onset and the trigger it
        activated, alongside the straight-line distance over the same window. Takes a trial.

        Only the agent that reached the wall is measured, as the loser never completes a route.
        path_ratio divides the two, which separates a short route from simply starting close. '''

    winner = get_indices.get_trigger_activator(trial)
    actual = float(trajectory_efficiency.player_actual_distance_trial(trial, winner))
    direct = float(trajectory_efficiency.player_direct_distance_trial(trial, winner))
    return dict(player=winner, winner_path_len=actual, winner_direct_len=direct,
                path_ratio=(actual / direct) if direct > 0 else np.nan)


def visibility_metrics(trial, player_id, fov=DEFAULT_FOV):
    ''' Return the latency (s) from slice onset until a trial wall first enters the FoV, plus
        visible-at-onset / ever-visible flags, for one agent on one trial. Takes a trial and
        player id. Returns None if the trial is too short.

        get_wall_visible runs from slice onset to trigger activation, so frame 0 is slice onset
        and every time here is referenced to it. full_trial_window keeps trials shorter than
        25 frames, which agents parked on a trigger produce on about a quarter of their trials. '''

    with contextlib.redirect_stdout(io.StringIO()):        # get_wall_visible is chatty on debug paths
        wv = trajectory_headangle.get_wall_visible(trial=trial, player_id=player_id, current_fov=fov,
                                                   full_trial_window=True)
    if isinstance(wv, float) and np.isnan(wv):
        return None
    idxs = [w - 1 for w in get_indices.get_walls(trial=trial)]   # wall number -> row index
    any_vis = wv[idxs, :].any(axis=0)
    ever = bool(any_vis.any())
    return dict(
        player=player_id,
        latency_to_wall_visible=(int(np.argmax(any_vis)) / FS) if ever else np.nan,
        visible_at_onset=bool(any_vis[0]),
        ever_visible=ever,
        trial_s=len(any_vis) / FS,        # decision window length, slice onset -> trigger
    )


# ---------------------------------------------------------------------------
# main pass over a batch of models
# ---------------------------------------------------------------------------

def get_batch_models(sim_data_folder):
    ''' Return the sorted model numbers present in a batch of inference runs.
        Takes the batch's simulation folder. '''

    return sorted(num for num, _ in identify_sim_filepaths.get_model_folders(sim_data_folder=sim_data_folder))


def collect_model_data(sim_data_folder, models=None, fov=DEFAULT_FOV, players=PLAYERS, verbose=True):
    ''' Return the long tables for a batch of models, as a dict of dataframes.
        Takes the batch's simulation folder and an optional model list (default: all present).

        Loads one model at a time and discards its dataframe, to stay memory-safe over a
        batch of ~55 models at ~90k rows each. Keys are:
          slice  - one row per trial x agent: slice-onset centrality and x,z
          iti    - one row per ITI x agent: scanning, movement, resting
          vis    - one row per trial x agent: wall visibility latency and flags
          score  - one row per trial: terminal score of each agent
          win    - one row per trial: the winning agent's route from slice onset to the trigger
          rt     - one row per completed trial: slice onset -> trigger time
          failed - (model, error name) for runs with no usable trials
          vis_dropped - model -> (rows with no visibility row, rows attempted) '''

    if models is None:
        models = get_batch_models(sim_data_folder)

    slice_records, iti_records, vis_records, score_records, rt_records = [], [], [], [], []
    win_records = []        # the winning agent's route to the wall, one row per trial
    failed_models = []      # inference runs with no usable trials (e.g. agent stuck on trial 1)
    vis_dropped = {}        # model -> [dropped, attempted], so a lost visibility row is never silent

    for model in models:
        try:
            with contextlib.redirect_stdout(io.StringIO()):      # silence per-model load/preprocess prints
                df, trial_list = prepare_sim_data.prepare_single_model_data(model, sim_data_folder=sim_data_folder)
            if len(trial_list) == 0:
                raise ValueError("no completed trials")
        except Exception as e:
            failed_models.append((model, type(e).__name__))
            if verbose:
                print(f"model {model:2d} SKIPPED ({type(e).__name__})", end='  ')
            continue

        # apply the trial convention. Drop the session's first trial, and any final trial with no
        # trial end (the log stops on the last trigger, so that trial has a time but no score).
        # trial_list already excludes the latter, so dropping its first entry leaves the trials
        # every table is built from
        trial_list = trial_list[1:]
        keep_trials = {float(t[globals.TRIAL_NUM].dropna().iloc[0]) for t in trial_list}
        if not keep_trials:
            failed_models.append((model, 'no trials left after dropping the first'))
            continue

        # measure ITI movement, scanning and resting position, pooled over agents
        for pid in players:
            w = iti_metrics(df, pid, keep_trials=keep_trials)
            w.insert(0, 'model', model)
            iti_records.append(w)

        # per trial, take slice-onset centrality (the strategy axis) and wall visibility for both agents
        vis_dropped[model] = [0, len(trial_list) * len(players)]
        for trial in trial_list:
            ttype = trial[globals.TRIAL_TYPE].replace('', np.nan).dropna().unique()
            ttype = ttype[0] if len(ttype) else ''
            # measure the winner's route to the wall it triggered
            try:
                wp = winner_path_metrics(trial)
            except (IndexError, ValueError, TypeError):
                wp = None
            if wp is not None:
                wp.update(model=model, trial_type=ttype)
                win_records.append(wp)
            for pid in players:
                try:
                    s = slice_onset_centrality(trial, pid)
                except (IndexError, ValueError):
                    s = None                                     # trial missing slice onset
                if s is not None:
                    s.update(model=model, trial_type=ttype)
                    slice_records.append(s)
                try:
                    r = visibility_metrics(trial, pid, fov=fov)
                except (IndexError, ValueError):
                    r = None
                if r is not None:
                    r.update(model=model, trial_type=ttype)
                    vis_records.append(r)
                else:
                    vis_dropped[model][0] += 1

        # take one row per trial outcome. Terminal player score: 1.0 (reached High), 0.4 (reached Low),
        # 0 (lost the race). Analysis uses this SCORE, not agent reward — reward folds in step/time
        # penalties and the no-movement penalty-break, and isn't even applied at inference, so it is not
        # a clean performance measure. Agent 0's score stands for the model (both agents are the same
        # model); score_p1 is kept only to check the two agents score about equally.
        te = df[df['eventDescription'] == globals.TRIAL_END]
        te = te[te[globals.TRIAL_NUM].isin(keep_trials)]
        score_records.append(pd.DataFrame(dict(
            model=model,
            trial_num=te[globals.TRIAL_NUM].to_numpy(float),
            score_p0=te[globals.PLAYER_0_TRIAL_SCORE].to_numpy(float),
            score_p1=te[globals.PLAYER_1_TRIAL_SCORE].to_numpy(float))))

        # trial completion time = slice onset -> server-selected trigger activation, in seconds.
        # a long time means the trial dragged on (e.g. agents standing still until forced to move).
        # pair the two events on trialNum, so a trial missing one of them cannot shift the rest.
        # slice-onset and trigger rows carry no trialNum in the raw log, and take the one
        # preprocessing fills in
        so = df[df['eventDescription'] == globals.SLICE_ONSET]
        trig = df[df['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
        rt_df = pd.DataFrame(dict(trial_num=so[globals.TRIAL_NUM].to_numpy(float),
                                  t_onset=so['timeApplication'].to_numpy())).merge(
                pd.DataFrame(dict(trial_num=trig[globals.TRIAL_NUM].to_numpy(float),
                                  t_trigger=trig['timeApplication'].to_numpy())), on='trial_num')
        rt_df = rt_df[rt_df['trial_num'].isin(keep_trials)]
        if len(rt_df):
            rt_records.append(pd.DataFrame(dict(
                model=model, trial_num=rt_df['trial_num'].to_numpy(),
                rt_s=((rt_df['t_trigger'] - rt_df['t_onset']) / np.timedelta64(1, 's')).astype(float))))

        if verbose:
            print(f"model {model:2d} done", end='  ')

    # stack the per-model records into long tables (one row per trial, ITI, scoring event, or completed trial)
    slice_df = pd.DataFrame(slice_records)
    win_df = pd.DataFrame(win_records)
    iti_df = pd.concat(iti_records, ignore_index=True)
    vis_df = pd.DataFrame(vis_records)
    score_df = pd.concat(score_records, ignore_index=True)
    rt_df = pd.concat(rt_records, ignore_index=True)
    # the higher of the two agent scores is the winner's, so 1.0 means High was reached, 0.4 means Low
    score_df['winner_score'] = score_df[['score_p0', 'score_p1']].max(axis=1)

    return dict(slice=slice_df, iti=iti_df, vis=vis_df, score=score_df, rt=rt_df, win=win_df,
                failed=failed_models, vis_dropped=vis_dropped)


def summarise_collection(tables, models, verbose=True):
    ''' Return the model numbers that produced at least one trial, printing a load summary.
        Takes the dict from collect_model_data and the model list it was asked for. '''

    usable = sorted(tables['slice']['model'].unique())
    if verbose:
        print(f"\n\nusable models: {len(usable)} / {len(models)}")
        if tables['failed']:
            print(f"skipped (no usable trials): {tables['failed']}")
        print(f"slice-onset rows: {len(tables['slice'])} | ITIs: {len(tables['iti'])} | "
              f"visibility rows: {len(tables['vis'])} | scored trials: {len(tables['score'])}")
        tc = tables['score'].groupby('model').size()
        print(f"trials per model: min {tc.min()}, max {tc.max()}")

        # flag models whose visibility rows are incomplete, as the loss is not random across models
        dropped = {m: v for m, v in (tables.get('vis_dropped') or {}).items() if v[0]}
        if dropped:
            worst = sorted(((m, n / a) for m, (n, a) in dropped.items()), key=lambda kv: -kv[1])[:5]
            print(f"visibility rows dropped (trial too short): "
                  f"{sum(n for n, _ in dropped.values())} over {len(dropped)} models")
            print("  worst: " + ", ".join(f"model {m} {f:.0%}" for m, f in worst))
    return usable


# ---------------------------------------------------------------------------
# stage 1: the strategy axis
# ---------------------------------------------------------------------------

def onset_centrality_by_model(slice_df):
    ''' Return one row per model with mean and SD of slice-onset centrality.
        Takes the slice table from collect_model_data. '''

    return (slice_df.groupby('model')['centrality']
            .agg(onset_centrality='mean', onset_sd='std', n='size').reset_index())


def split_strategies(onset_by_model, fixed_cut=0.5, random_state=0, tag_intermediate=True):
    ''' Return the per-model table with central/peripheral/intermediate labels added, and the
        GMM boundary. Takes the table from onset_centrality_by_model.

        Labels twice: a fixed cut at centrality = fixed_cut (nearer the centre than a wall),
        and a 2-component Gaussian mixture on the per-model means that finds its own boundary.
        Agreement between the two, with a clear gap in the distribution, means the split is real.

        A third component then picks out models sitting between the two clusters, which are
        labelled 'intermediate' in the working 'strategy' column. Posterior probability cannot
        find these, as the tight peripheral component drives every model's posterior to 1.0. '''

    onset_by_model = onset_by_model.copy()
    vals = onset_by_model['onset_centrality'].to_numpy()
    X = vals.reshape(-1, 1)

    # (a) fixed absolute threshold
    onset_by_model['strategy_fixed'] = np.where(vals > fixed_cut, 'central', 'peripheral')

    # (b) 2-component Gaussian mixture on the 1-D per-model means
    gmm = GaussianMixture(n_components=2, n_init=10, random_state=random_state).fit(X)
    comp = gmm.predict(X)
    central_comp = int(np.argmax(gmm.means_.ravel()))            # higher-centrality component = central
    onset_by_model['strategy_gmm'] = np.where(comp == central_comp, 'central', 'peripheral')
    boundary = float(np.mean(gmm.means_.ravel()))                # rough visual divider between the two means

    # (c) tag the models between the clusters, but only if 3 components genuinely fit better
    onset_by_model['strategy'] = onset_by_model['strategy_gmm']
    if tag_intermediate and len(vals) >= 6:
        gmm3 = GaussianMixture(n_components=3, n_init=10, random_state=random_state).fit(X)
        if gmm3.bic(X) < gmm.bic(X):
            middle = int(np.argsort(gmm3.means_.ravel())[1])     # middle component by mean centrality
            onset_by_model.loc[gmm3.predict(X) == middle, 'strategy'] = 'intermediate'

    return onset_by_model, gmm, boundary


def report_strategy_split(onset_by_model, gmm, fixed_cut=0.5):
    ''' Print how well the fixed cut and the GMM agree on the central/peripheral split.
        Takes the labelled per-model table and the fitted mixture. '''

    agree = (onset_by_model['strategy_fixed'] == onset_by_model['strategy_gmm']).mean()
    print(f"GMM component means: {np.sort(gmm.means_.ravel()).round(3)}  weights: {gmm.weights_.round(2)}")
    print(f"fixed({fixed_cut}) vs GMM agreement: {agree:.0%}")
    print(onset_by_model.groupby(['strategy_gmm', 'strategy_fixed']).size().rename('n'))


def build_model_meta(usable_models, onset_by_model, strategy_col='strategy'):
    ''' Return one row per usable model with its centrality and working strategy label.
        Takes the usable model list and the labelled per-model table.

        Every label is carried through, and the working 'strategy' column copies strategy_col.
        Pass strategy_col='strategy_gmm' to force every model into central or peripheral. '''

    cols = ['model', 'onset_centrality', 'onset_sd', 'strategy_fixed', 'strategy_gmm']
    if 'strategy' in onset_by_model.columns:
        cols.append('strategy')
    model_meta = pd.DataFrame({'model': usable_models})
    model_meta = model_meta.merge(onset_by_model[cols], on='model')
    model_meta['strategy'] = model_meta[strategy_col]
    return model_meta


def report_strategy_membership(model_meta):
    ''' Print which models fall in each strategy. Takes the per-model table. '''

    print("Strategy membership (working label = GMM):")
    for lab in STRAT_PALETTE:
        ms = model_meta.loc[model_meta['strategy'] == lab, 'model'].tolist()
        print(f"  {lab:11s} (n={len(ms):2d}): {ms}")


# ---------------------------------------------------------------------------
# stage 2: the behavioural fingerprint
# ---------------------------------------------------------------------------

def build_fingerprint(model_meta, tables, slow_trial_s=SLOW_TRIAL_S):
    ''' Return one row per model with the full set of fingerprint metrics.
        Takes the per-model table and the dict from collect_model_data.

        Any extra columns already on model_meta (e.g. a training-length tag) are carried through. '''

    # average ITI behaviour per model, pooled over agents
    iti_by_model = (tables['iti'].groupby('model')[['prop_no_translational', 'prop_no_rotational',
                    'proportion_resting', 'iti_path_len', 'rot_total', 'iti_centrality']]
                    .mean().reset_index())

    # summarise wall visibility per model
    vis_df = tables['vis']
    vis_by_model = vis_df.groupby('model').agg(
        frac_visible_at_onset=('visible_at_onset', 'mean'),
        frac_never_visible=('ever_visible', lambda s: 1 - s.mean())).reset_index()
    # late_: average only over trials where the wall was not already in view, so this denominator
    # is what frac_visible_at_onset and frac_never_visible leave over. A model with a high
    # frac_visible_at_onset measures this on few trials, so carry the count alongside it
    late = vis_df[(~vis_df['visible_at_onset']) & vis_df['ever_visible']]
    vis_by_model = vis_by_model.merge(
        late.groupby('model')['latency_to_wall_visible'].mean()
            .rename('late_latency_to_wall_visible').reset_index(),
        on='model', how='left')
    vis_by_model = vis_by_model.merge(
        late.groupby('model').size().rename('n_late_latency_to_wall_visible').reset_index(),
        on='model', how='left')

    # all_: average over every trial where the wall was seen, scoring visible-at-onset as 0.
    # Never-visible trials have no latency to average, so they stay out
    seen = vis_df[vis_df['ever_visible']]
    vis_by_model = vis_by_model.merge(
        seen.groupby('model')['latency_to_wall_visible'].mean()
            .rename('all_latency_to_wall_visible').reset_index(),
        on='model', how='left')

    # average the winner's route to the wall per model
    win_by_model = tables['win'].groupby('model')[['winner_path_len', 'winner_direct_len',
                                                   'path_ratio']].mean().reset_index()

    # summarise trial completion time per model, from slice onset to the trigger the server selected.
    # take the median for latency_to_reach_wall, as a few trials run on for tens of seconds
    rt_by_model = tables['rt'].groupby('model').agg(
        latency_to_reach_wall=('rt_s', 'median'),
        frac_trials_over_5s=('rt_s', lambda s: float(np.mean(s > slow_trial_s))),
    ).reset_index()

    # compute per-model scoring on player SCORE (not reward): P(High) and agent 0 mean terminal score
    score_by_model = tables['score'].groupby('model').agg(
        p_high=('winner_score', lambda s: float(np.mean(np.isclose(s, 1.0)))),
        mean_score=('score_p0', 'mean')).reset_index()

    # score over time, in score per second. Sum the score and the trial time over the whole
    # session, then divide once. Averaging a per-trial score/time instead breaks down, as an agent
    # parked on a trigger wins in under one logged frame and that trial's rate runs to hundreds
    rate = tables['score'][['model', 'trial_num', 'winner_score']].merge(
        tables['rt'][['model', 'trial_num', 'rt_s']], on=['model', 'trial_num'])
    rate_by_model = rate.groupby('model').agg(
        score_rate_trials=('rt_s', 'size'),
        session_score=('winner_score', 'sum'),
        session_trial_s=('rt_s', 'sum')).reset_index()
    rate_by_model['score_rate'] = rate_by_model['session_score'] / rate_by_model['session_trial_s']

    return (model_meta
            .merge(iti_by_model, on='model')
            .merge(vis_by_model, on='model')
            .merge(win_by_model, on='model')
            .merge(rt_by_model, on='model')
            .merge(score_by_model, on='model')
            .merge(rate_by_model, on='model'))


def between_strategy_variance(fingerprint, metrics=FP_METRICS):
    ''' Return, for each metric, the share of its spread falling between the two strategies
        (eta^2) and a one-way ANOVA p-value. Takes the fingerprint table.

        NB eta^2 is descriptive only, and p_anova is not a valid test. The strategy labels come
        from the GMM fit to onset_centrality on this same data, so the comparison is circular for
        every metric that tracks centrality, and no correction is applied across the metrics.
        frac_trials_over_5s and frac_never_visible are near-constant in the central group, which
        inflates their F on a near-zero pooled variance, and late_latency_to_wall_visible is
        skewed with outliers and rests on few trials for peripheral models.
        Read the eta^2 ordering, not p. '''

    rows = []
    for m in metrics:
        # compare the two main strategies only, so intermediate models do not enter either the
        # groups or the total spread they are measured against
        groups = [fingerprint.loc[fingerprint['strategy'] == g, m].dropna().to_numpy()
                  for g in MAIN_STRATEGIES]
        vals = np.concatenate(groups) if any(len(g) for g in groups) else np.array([])
        grand = vals.mean() if len(vals) else np.nan
        ss_total = np.sum((vals - grand) ** 2)
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups if len(g))
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan
        F, p = stats.f_oneway(*groups) if all(len(g) > 1 for g in groups) else (np.nan, np.nan)
        rows.append(dict(metric=m, eta2_between_strategy=eta2, p_anova=p,
                         central_mean=groups[0].mean() if len(groups[0]) else np.nan,
                         peripheral_mean=groups[1].mean() if len(groups[1]) else np.nan))
    return pd.DataFrame(rows).sort_values('eta2_between_strategy', ascending=False)


def wall_visible_curve(vis_df, model_meta=None, max_s=3.0, step_s=1 / FS):
    ''' Return the proportion of trials with a trial wall in view by each time after slice onset,
        one row per model and timepoint. Takes the vis table and optionally the per-model table
        to attach strategy labels.

        A trial that ended before a wall came into view never counts as seen, so a model's curve
        plateaus at 1 - frac_never_visible rather than reaching 1. p_running gives the share of
        trials still going at each time, which is what makes that plateau readable. '''

    times = np.round(np.arange(0, max_s + step_s / 2, step_s), 6)
    rows = []
    for model, g in vis_df.groupby('model'):
        # never-visible trials get an infinite latency, so they never enter the cumulative count
        lat = g['latency_to_wall_visible'].to_numpy(float)
        lat = np.where(np.isnan(lat), np.inf, lat)
        row = dict(model=model, t=times, p_visible=(lat[:, None] <= times[None, :]).mean(axis=0))
        if 'trial_s' in g:
            dur = g['trial_s'].to_numpy(float)
            row['p_running'] = (dur[:, None] > times[None, :]).mean(axis=0)
        rows.append(pd.DataFrame(row))

    curve = pd.concat(rows, ignore_index=True)
    if model_meta is not None:
        curve = curve.merge(model_meta[['model', 'strategy']], on='model')
    return curve


# ---------------------------------------------------------------------------
# stage 3: multivariate fingerprint and tournament shortlist
# ---------------------------------------------------------------------------

def add_fingerprint_pca(fingerprint, metrics=PCA_METRICS, n_components=2):
    ''' Return the fingerprint table with pc1/pc2 columns added, and the fitted PCA.
        Takes the fingerprint table and the metrics to reduce.

        The centrality columns are left out by default, so PCA shows what else varies. '''

    fingerprint = fingerprint.copy()
    X = fingerprint[metrics].to_numpy(float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)          # fill the occasional missing latency
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components).fit(Xz)
    pc = pca.transform(Xz)
    fingerprint['pc1'], fingerprint['pc2'] = pc[:, 0], pc[:, 1]
    return fingerprint, pca


def report_pca(fingerprint, pca):
    ''' Print the variance explained by the first two components and how far PC1 tracks
        the strategy axis. Takes the fingerprint table with pc columns and the fitted PCA. '''

    r_pc1 = np.corrcoef(fingerprint['pc1'], fingerprint['onset_centrality'])[0, 1]
    print(f"PC1 explains {pca.explained_variance_ratio_[0]:.0%}, PC2 {pca.explained_variance_ratio_[1]:.0%} "
          f"of fingerprint variance")
    print(f"corr(PC1, slice-onset centrality) = {r_pc1:+.2f}  "
          f"(|r|~1 => centrality is the main axis; ~0 => something else is)")


def medoid_model(sub, metrics=PCA_METRICS):
    ''' Return the most typical model of a group, the one nearest the group's centre in
        standardised metric space. Takes a slice of the fingerprint table. '''

    z = StandardScaler().fit_transform(sub[metrics].fillna(sub[metrics].mean()))
    return int(sub['model'].iloc[np.argmin(np.linalg.norm(z - z.mean(0), axis=1))])


def tournament_shortlist(fingerprint, metrics=PCA_METRICS, extra_cols=()):
    ''' Return a shortlist of models spanning the strategy axis, as a role -> model table.
        Takes the fingerprint table.

        Picks the two centrality extremes, then the medoid and the best performer of each
        strategy. The performer is picked on score_rate, which is score per second of
        trial time. Passing extra_cols adds them to the output table (e.g. a training tag). '''

    picks = {}
    picks['most_central'] = int(fingerprint.loc[fingerprint['onset_centrality'].idxmax(), 'model'])
    picks['most_peripheral'] = int(fingerprint.loc[fingerprint['onset_centrality'].idxmin(), 'model'])
    for lab in MAIN_STRATEGIES:
        sub = fingerprint[fingerprint['strategy'] == lab]
        picks[f'typical_{lab}'] = medoid_model(sub, metrics=metrics)
        picks[f'best_scorer_{lab}'] = int(sub.loc[sub['score_rate'].idxmax(), 'model'])

    cols = ['model', 'strategy'] + list(extra_cols) + ['onset_centrality', 'score_rate',
                                                       'p_high', 'mean_score']
    return (pd.DataFrame({'role': list(picks), 'model': list(picks.values())})
            .drop_duplicates('model')
            .merge(fingerprint[cols], on='model'))


def rank_fingerprint(fingerprint):
    ''' Return the fingerprint indexed by model with centrality, score-rate, P(High) and score
        ranks added (1 = highest). Takes the fingerprint table. '''

    return fingerprint.assign(
        cent_rk=fingerprint['onset_centrality'].rank(ascending=False).astype(int),
        rate_rk=fingerprint['score_rate'].rank(ascending=False).astype(int),
        pHigh_rk=fingerprint['p_high'].rank(ascending=False).astype(int),
        score_rk=fingerprint['mean_score'].rank(ascending=False).astype(int),
    ).set_index('model')


def curated_table(fingerprint, curated, extra_cols=()):
    ''' Return a table of a hand-picked shortlist with each model's ranked metrics.
        Takes the fingerprint table and a list of (role, model) pairs. '''

    rk = rank_fingerprint(fingerprint)
    rows = []
    for role, m in curated:
        row = dict(role=role, model=m, strategy=rk.loc[m, 'strategy'])
        for col in extra_cols:
            row[col] = rk.loc[m, col]
        row.update(
            centrality=round(rk.loc[m, 'onset_centrality'], 2), cent_rk=int(rk.loc[m, 'cent_rk']),
            score_rate=round(rk.loc[m, 'score_rate'], 3), rate_rk=int(rk.loc[m, 'rate_rk']),
            p_high=round(rk.loc[m, 'p_high'], 2), pHigh_rk=int(rk.loc[m, 'pHigh_rk']),
            mean_score=round(rk.loc[m, 'mean_score'], 3), score_rk=int(rk.loc[m, 'score_rk']),
            pct_over_5s=round(rk.loc[m, 'frac_trials_over_5s'], 3),
            resting=round(rk.loc[m, 'proportion_resting'], 2),
            PC1=round(rk.loc[m, 'pc1'], 2), PC2=round(rk.loc[m, 'pc2'], 2),
        )
        rows.append(row)
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# stage 4: picking the best models of a strategy for a tournament
# ---------------------------------------------------------------------------

# metrics that say how well a model plays, and which direction counts as better. These rank
# models within one strategy. The work is mostly in the peripheral group, as the central models
# are near-identical and any of them stands in for the others
SELECTION_METRICS = {
    'score_rate': +1,                   # earns more score per second of trial time
    'latency_to_reach_wall': -1,        # gets to the wall sooner
    'frac_visible_at_onset': +1,        # already has a wall in view when the trial starts
    'late_latency_to_wall_visible': -1, # finds a wall faster when it has to turn for it
    # High-choice trials travel about 6 units further than Low-choice ones, so this does rank down
    # models that commit to High. path_ratio is the distance-independent alternative
    'winner_path_len': -1,              # takes a shorter route to the wall it triggers
}

# below this many trials, late_latency_to_wall_visible is too noisy to rank on. Peripheral models
# already face a wall at onset on most trials, so their late-latency denominator is small
MIN_LATE_TRIALS = 20

# if more than this share of a group is missing a metric, the metric is dropped for the whole
# group. Ranking some models on it and others without it is not a like-for-like comparison
MAX_MISSING_FRAC = 0.25

# the two ways of standardising a metric before the metrics are averaged. Both give every metric
# equal weight; they differ only in what counts as the group's centre and spread
RANKING_METHODS = ('robust', 'classic')

MAD_TO_SD = 1.4826                      # scales a MAD onto the same footing as an SD


def _centre_and_spread(X, method):
    ''' Return the centre and spread of each metric across a group of models.
        Takes the metric columns and a ranking method.

        classic uses the mean and the standard deviation, so every model including an extreme one
        sets the scale. robust uses the median and a scaled MAD, both fixed by the middle of the
        group, so one extreme model barely moves them. '''

    if method not in RANKING_METHODS:
        raise ValueError(f"method must be one of {RANKING_METHODS}, got {method!r}")
    if method == 'robust':
        centre = X.median()
        spread = (X - centre).abs().median() * MAD_TO_SD    # MAD, rescaled to read like an SD
    else:
        centre = X.mean()
        spread = X.std(ddof=0)
    return centre, spread.replace(0, np.nan)                # a flat metric cannot standardise


def _selection_frame(fingerprint, strategy, metrics, method, min_late_trials,
                     max_missing_frac=MAX_MISSING_FRAC, verbose=False):
    ''' Return the candidate rows, their signed standardised scores and the metrics kept.
        Takes the fingerprint table, a strategy, the metric directions and a ranking method.

        Runs the shared steps of both ranking methods. Only the centre and spread differ. '''

    # step 1: take the models of one strategy. The ranking is always within a group, never across
    sub = fingerprint if strategy is None else fingerprint[fingerprint['strategy'] == strategy]
    sub = sub.copy().reset_index(drop=True)

    # step 2a: blank a late latency measured on too few trials, so it neither ranks a model nor
    # moves the group's centre and spread. n_late_latency_to_wall_visible comes from build_fingerprint
    few_late = pd.Series(False, index=sub.index)
    if 'late_latency_to_wall_visible' in metrics and 'n_late_latency_to_wall_visible' in sub:
        few_late = sub['n_late_latency_to_wall_visible'].fillna(0) < min_late_trials
        sub.loc[few_late, 'late_latency_to_wall_visible'] = np.nan

    # step 2b: drop a metric too many of the group are missing, so every model is ranked on the
    # same set. late_latency_to_wall_visible usually goes this way for peripheral models
    kept = {c: d for c, d in metrics.items() if sub[c].isna().mean() <= max_missing_frac}
    if verbose:
        for c in metrics:
            if c not in kept:
                print(f"dropped {c}: missing for {sub[c].isna().mean():.0%} of the group")
    cols = list(kept)

    # step 3: find each metric's centre and spread across the group, the one step where the two
    # methods differ
    centre, spread = _centre_and_spread(sub[cols], method)

    # step 4: standardise every value against its own metric's centre and spread
    # step 5: flip the lower-is-better metrics, so a positive score always means better
    signed = ((sub[cols] - centre) / spread) * pd.Series(kept, dtype=float)
    return sub, signed, few_late, kept


def rank_models(fingerprint, strategy=None, n=None, metrics=SELECTION_METRICS, method='robust',
                min_late_trials=MIN_LATE_TRIALS, verbose=False):
    ''' Return the models of one strategy ranked best first, with the numbers behind each rank.
        Takes the fingerprint table, the strategy to rank within and a ranking method.

        This looks for the best models of a group, not the most typical ones, so it ranks on
        performance rather than distance from a group centre. Every metric carries equal weight.
        quality is the mean of the signed standardised scores, and margin is the gap down to the
        next model, which says how firm a cut is. Pass n to flag the top n.

        A metric most of the group is missing is dropped for all of them. Pass verbose to see
        which. A model missing one of the metrics that survive is scored on the rest, with
        n_metrics recording how many it had. '''

    sub, signed, few_late, kept = _selection_frame(fingerprint, strategy, metrics, method,
                                                   min_late_trials, verbose=verbose)
    cols = list(kept)

    # step 6: average the signed scores with equal weight, over the metrics a model actually has,
    # so one missing metric does not read as a bad score
    sub['quality'] = signed.mean(axis=1)
    sub['n_metrics'] = signed.notna().sum(axis=1)
    sub['few_late_trials'] = few_late.to_numpy()
    sub['method'] = method
    for c in cols:
        sub[f'z_{c}'] = signed[c]

    # step 7: sort best first and record how far clear each model is of the one below it
    sub = sub.sort_values('quality', ascending=False).reset_index(drop=True)
    sub['rank'] = sub.index + 1
    sub['margin'] = sub['quality'] - sub['quality'].shift(-1)
    if n is not None:
        sub['picked'] = sub.index < n

    out = (['model', 'strategy', 'method', 'rank', 'quality', 'margin']
           + (['picked'] if n is not None else [])
           + cols + [f'z_{c}' for c in cols] + ['n_metrics', 'few_late_trials'])
    return sub[out]


def compare_ranking_methods(fingerprint, strategy=None, n=5, metrics=SELECTION_METRICS,
                            min_late_trials=MIN_LATE_TRIALS):
    ''' Return one row per model with its rank and quality under both ranking methods.
        Takes the fingerprint table, the strategy and how many to shortlist.

        move is how far the model shifts between the two, and shortlist says whether it makes the
        top n under both methods, one of them, or neither. The two methods usually agree, so a
        model that moves is worth a look at its metric values. '''

    ranked = {m: rank_models(fingerprint, strategy=strategy, n=n, metrics=metrics, method=m,
                             min_late_trials=min_late_trials) for m in RANKING_METHODS}
    out = ranked['robust'][['model', 'strategy']].copy()
    for m in RANKING_METHODS:
        r = ranked[m].set_index('model')
        out[f'rank_{m}'] = out['model'].map(r['rank'])
        out[f'quality_{m}'] = out['model'].map(r['quality'])
        out[f'top{n}_{m}'] = out['model'].map(r['picked'])

    # a positive move means the model ranks better under robust than under classic
    out['move'] = out['rank_classic'] - out['rank_robust']
    both = out[f'top{n}_robust'] & out[f'top{n}_classic']
    either = out[f'top{n}_robust'] | out[f'top{n}_classic']
    out['shortlist'] = np.where(both, 'both', np.where(either, 'one method', ''))
    return out.sort_values('rank_robust').reset_index(drop=True)


def outlier_report(fingerprint, strategy=None, metrics=SELECTION_METRICS,
                   min_late_trials=MIN_LATE_TRIALS):
    ''' Return one row per metric showing how far outliers pull it, and what that does to the two
        ranking methods. Takes the fingerprint table and the strategy.

        sd_over_mad above 1 means the standard deviation is inflated by the tails, so classic
        standardising divides by a spread the typical model never sees and squashes everyone
        toward zero. influence is how much each standardised metric actually varies, so how much
        sway it has in the equal-weight average. Under classic it is 1 for every metric by
        construction; under robust it is not, so metrics with a tight middle count for more.
        extreme_model is the model furthest from the group centre, in robust units. '''

    sub, _, _, kept = _selection_frame(fingerprint, strategy, metrics, 'robust', min_late_trials)
    cols = list(kept)
    X = sub[cols]

    rows = []
    for c in cols:
        mean, sd = X[c].mean(), X[c].std(ddof=0)
        med = X[c].median()
        mad = (X[c] - med).abs().median() * MAD_TO_SD
        zc = (X[c] - mean) / sd
        zr = (X[c] - med) / mad
        far = zr.abs().idxmax()                             # furthest model from the centre
        ordered = X[c].sort_values()
        gaps = ordered.diff().abs()
        rows.append(dict(
            metric=c, mean=mean, sd=sd, median=med, mad_scaled=mad,
            sd_over_mad=sd / mad if mad else np.nan,        # >1 = the SD is stretched by the tails
            influence_classic=zc.std(ddof=0),               # always 1, kept for the comparison
            influence_robust=zr.std(ddof=0),
            extreme_model=int(sub['model'].iloc[far]), extreme_value=X[c].iloc[far],
            extreme_z_robust=zr.iloc[far],
            # how isolated the most extreme value is, as a share of the whole metric's range
            edge_gap_frac=float(max(gaps.iloc[1], gaps.iloc[-1]) / (X[c].max() - X[c].min()))))
    return pd.DataFrame(rows).sort_values('sd_over_mad', ascending=False).reset_index(drop=True)


def inspect_models(fingerprint, models, strategy=None, metrics=SELECTION_METRICS,
                   min_late_trials=MIN_LATE_TRIALS):
    ''' Return the ranking metrics for a chosen set of models, with each value's rank in the
        group and its standardised score under both methods. Takes the fingerprint table and a
        list of models.

        rank_in_group is 1 for the best model of the group on that metric, in the direction the
        ranking treats as better. '''

    sub, _, _, kept = _selection_frame(fingerprint, strategy, metrics, 'robust', min_late_trials)
    cols = list(kept)
    signed = {m: _selection_frame(fingerprint, strategy, metrics, m, min_late_trials)[1]
              for m in RANKING_METHODS}

    rows = []
    for c in cols:
        # rank the whole group on this metric, best first in the direction that counts as better.
        # method='min' keeps a tie on whole numbers rather than splitting the place between them
        order = sub[c].rank(ascending=(kept[c] < 0), method='min')
        for model in models:
            hits = sub.index[sub['model'] == model]
            if not len(hits):
                continue
            i = hits[0]
            rows.append(dict(model=model, metric=c, value=sub[c].iloc[i],
                             rank_in_group=order.iloc[i], better='higher' if kept[c] > 0 else 'lower',
                             z_robust=signed['robust'][c].iloc[i],
                             z_classic=signed['classic'][c].iloc[i]))
    out = pd.DataFrame(rows)
    out['rank_in_group'] = out['rank_in_group'].astype('Int64')
    return out.sort_values(['metric', 'model']).reset_index(drop=True)


def hidden_metrics(fingerprint, models, strategy=None, metrics=SELECTION_METRICS,
                   fp_metrics=FP_METRICS, min_late_trials=MIN_LATE_TRIALS):
    ''' Return the fingerprint metrics the ranking never sees, for a chosen set of models.
        Takes the fingerprint table and a list of models.

        These are the fingerprint metrics left over once the ranking metrics are removed, plus
        any ranking metric that was dropped for the group. No direction is implied, as these are
        descriptive, so rank_high_to_low is simply the group ordering by value. '''

    sub, _, _, kept = _selection_frame(fingerprint, strategy, metrics, 'robust', min_late_trials)
    unseen = [c for c in fp_metrics if c not in kept and c in sub]

    rows = []
    for c in unseen:
        order = sub[c].rank(ascending=False, method='min')   # by value only, no better direction
        for model in models:
            hits = sub.index[sub['model'] == model]
            if not len(hits):
                continue
            i = hits[0]
            rows.append(dict(model=model, metric=c, value=sub[c].iloc[i],
                             rank_high_to_low=order.iloc[i], group_median=sub[c].median(),
                             dropped_from_ranking=c in metrics))
    out = pd.DataFrame(rows)
    out['rank_high_to_low'] = out['rank_high_to_low'].astype('Int64')
    return out.sort_values(['metric', 'model']).reset_index(drop=True)


def metric_overlap(fingerprint, strategy=None, metrics=SELECTION_METRICS, method='robust',
                   min_late_trials=MIN_LATE_TRIALS, max_missing_frac=1.0):
    ''' Return the rank correlations between the ranking metrics within one strategy, already
        signed so that + means the two agree on which model is better. Takes the fingerprint
        table and the strategy.

        Two metrics that correlate strongly are one signal counted twice in the ranking. Every
        metric is kept by default, including any the ranking drops, so a dropped one can still be
        checked against the rest. '''

    _, signed, _, _ = _selection_frame(fingerprint, strategy, metrics, method, min_late_trials,
                                       max_missing_frac=max_missing_frac)
    return signed.corr(method='spearman')


def rank_stability(fingerprint, strategy=None, metrics=SELECTION_METRICS, method='robust', **kwargs):
    ''' Return each model's rank on the full metric set and on each set with one metric dropped.
        Takes the fingerprint table, the strategy and a ranking method.

        A model that stays near the top however the metrics are cut is a firm pick. One that
        moves a long way is riding on a single metric. '''

    full = rank_models(fingerprint, strategy=strategy, metrics=metrics, method=method, **kwargs)
    out = full[['model', 'quality']].copy().rename(columns={'quality': 'quality_all'})
    out['rank_all'] = full['rank'].to_numpy()

    for drop in metrics:
        left = {k: v for k, v in metrics.items() if k != drop}
        r = rank_models(fingerprint, strategy=strategy, metrics=left, method=method, **kwargs)
        out[f'no_{drop}'] = out['model'].map(dict(zip(r['model'], r['rank'])))

    cols = [c for c in out.columns if c.startswith('no_')] + ['rank_all']
    out['rank_worst'] = out[cols].max(axis=1)
    out['rank_best'] = out[cols].min(axis=1)
    out['rank_spread'] = out['rank_worst'] - out['rank_best']
    return out.sort_values('rank_all').reset_index(drop=True)


def report_selection(ranked, n=3):
    ''' Print the top n models of one ranking and what put them there. Takes the table from
        rank_models. '''

    zcols = [c for c in ranked.columns if c.startswith('z_')]
    print(f"top {n} of {len(ranked)} {ranked['strategy'].iloc[0]} models, "
          f"{ranked['method'].iloc[0]} standardising (+ = better than the group):")
    for _, r in ranked.head(n).iterrows():
        best = max(zcols, key=lambda c: (r[c] if pd.notna(r[c]) else -np.inf))
        worst = min(zcols, key=lambda c: (r[c] if pd.notna(r[c]) else np.inf))
        flag = "  [few late trials]" if r['few_late_trials'] else ""
        print(f"  {int(r['rank'])}. model {int(r['model']):2d}  quality {r['quality']:+.2f}  "
              f"margin {r['margin']:+.2f}{flag}")
        print(f"       best {best[2:]} {r[best]:+.2f} | weakest {worst[2:]} {r[worst]:+.2f}")


def report_method_comparison(fingerprint, strategy=None, n=5, metrics=SELECTION_METRICS,
                             min_late_trials=MIN_LATE_TRIALS):
    ''' Print both shortlists side by side, where they disagree, and which metrics outliers are
        pulling. Takes the fingerprint table, the strategy and how many to shortlist.

        Returns the comparison table so it can be read on afterwards. '''

    cmp = compare_ranking_methods(fingerprint, strategy=strategy, n=n, metrics=metrics,
                                  min_late_trials=min_late_trials)
    for m in RANKING_METHODS:
        picks = cmp.sort_values(f'rank_{m}').head(n)['model'].astype(int).tolist()
        print(f"  {m + ' shortlist':<20} {picks}")
    agree = cmp[cmp['shortlist'] == 'both']['model'].astype(int).tolist()
    only = cmp[cmp['shortlist'] == 'one method']['model'].astype(int).tolist()
    print(f"  {'in both':<20} {sorted(agree)}")
    print(f"  {'in one only':<20} {sorted(only)}")
    rho = cmp['rank_robust'].corr(cmp['rank_classic'], method='spearman')
    print(f"  {'rank agreement':<20} Spearman {rho:.3f} over {len(cmp)} models")

    moved = cmp.reindex(cmp['move'].abs().sort_values(ascending=False).index).head(3)
    print("\n  biggest movers (+ = better under robust):")
    for _, r in moved.iterrows():
        print(f"    model {int(r['model']):2d}  classic #{int(r['rank_classic']):2d} -> "
              f"robust #{int(r['rank_robust']):2d}  ({int(r['move']):+d})")

    out = outlier_report(fingerprint, strategy=strategy, metrics=metrics,
                         min_late_trials=min_late_trials)
    print("\n  outlier pull per metric (sd_over_mad > 1 = the SD is stretched by the tails):")
    for _, r in out.iterrows():
        print(f"    {r['metric']:<26} sd/mad {r['sd_over_mad']:.2f}  "
              f"influence classic 1.00 vs robust {r['influence_robust']:.2f}  "
              f"| furthest model {r['extreme_model']:2d} at {r['extreme_z_robust']:+.1f} robust units")
    return cmp


def middle_models(fingerprint, strategy, n=2, metrics=SELECTION_METRICS, method='robust',
                  min_late_trials=MIN_LATE_TRIALS):
    ''' Return the n models sitting at the middle of one strategy's quality ranking.
        Takes the fingerprint table, a strategy and how many to return.

        Middle of the pack on the same metrics the shortlist is ranked on, so these stand in for
        a typical model of the group rather than a good or a bad one. '''

    ranked = rank_models(fingerprint, strategy=strategy, metrics=metrics, method=method,
                         min_late_trials=min_late_trials)
    start = (len(ranked) - n) // 2                      # centre the window on the median rank
    return ranked['model'].iloc[start:start + n].astype(int).tolist()


def compare_across_strategies(fingerprint, models, metrics=SELECTION_METRICS,
                              ranked_within='peripheral', min_late_trials=MIN_LATE_TRIALS):
    ''' Return the ranking metrics for a set of models drawn from more than one strategy, with
        each value's rank across the whole batch. Takes the fingerprint table and a list of models.

        The standardised scores elsewhere are within-group, so they cannot put a central model
        next to a peripheral one. Each group is scaled against itself, and a quality of +1 means
        the best of that group however tight the group is. Raw values distort nothing, and a
        batch-wide rank gives the common frame instead.

        rank_in_batch is 1 for the best model in the batch on that metric, in the direction the
        ranking counts as better. ranked_within names the group whose kept metrics to use, so the
        table covers exactly the metrics that group was ranked on. '''

    # take the metrics the named group was actually ranked on, after any group-wide drop
    _, _, _, kept = _selection_frame(fingerprint, ranked_within, metrics, 'robust', min_late_trials)

    rows = []
    for c in kept:
        # rank over every model in the batch, not within a strategy, so the groups share a frame
        order = fingerprint[c].rank(ascending=(kept[c] < 0), method='min')
        for model in models:
            hits = fingerprint.index[fingerprint['model'] == model]
            if not len(hits):
                continue
            i = hits[0]
            rows.append(dict(metric=c, model=int(model),
                             strategy=fingerprint['strategy'].loc[i],
                             value=fingerprint[c].loc[i], rank_in_batch=order.loc[i],
                             better='higher' if kept[c] > 0 else 'lower',
                             batch_median=fingerprint[c].median()))
    out = pd.DataFrame(rows)
    out['rank_in_batch'] = out['rank_in_batch'].astype('Int64')
    # best in the batch first within each metric, so the gap between the groups reads off directly
    return out.sort_values(['metric', 'rank_in_batch']).reset_index(drop=True)
