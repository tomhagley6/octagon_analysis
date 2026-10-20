"""Compare the fingerprints of central human players and central RL agents.

The human fingerprint (`analysis/human_characterisation.py`) and the agent fingerprint
(`analysis/model_characterisation.py`) share most metric names, but three things stop the two
tables being merged directly:

1. The between-trial metrics sit on different clocks. Humans are sampled onto a 30 Hz network
   tick grid, agents on 50 Hz logged frames. Each side is left at its own rate: the resampling
   costs the agents up to two thirds of a metric while the humans pass through untouched, so
   matching grids would distort only the side that did not need it. `nc.downsampling_effect`
   measures this and the notebook's part 3a reports it.
2. `score_rate` is not the same quantity. The agent version sums `winner_score`, the model's own
   output in self-play; the human version sums the player's own score, counting a loss as 0.
3. Agents never see forced trials, which are 20% of human trials.

This module fixes all three, and states what is left uncomparable.
"""

import contextlib
import io
import os

import numpy as np
import pandas as pd

import data_strings
import globals
import analysis.agent_raycast as agent_raycast
import analysis.human_characterisation as hc
import analysis.model_characterisation as mc
import data_extraction.data_saving as data_saving
import data_extraction.get_indices as get_indices
import data_extraction.network_compensation.network_compensation as nc
import parse_data.prepare_sim_data as prepare_sim_data


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

TICK_S = nc.TICK_S                          # 30 Hz, the grid every human ITI metric is measured on
STILL_SPEED = nc.STILL_SPEED_PER_TICK       # 0.08 units/tick below which a step counts as no translation
STILL_ROT = nc.STILL_ROT_PER_TICK           # 0.8 deg/tick below which a step counts as no rotation
AGENT_TIME_COL = 'timeApplication'          # the agent's clean 50 Hz application clock

AGENT_TICK_CACHE = 'agent_iti_ticks'        # per-batch cache of the resampled agent ITI table
AGENT_TABLE_CACHE = 'agent_tables'          # per-batch cache of collect_model_data's long tables
AGENT_RAY_CACHE = 'agent_ray_visibility'    # per-batch cache of the ray-sensor visibility table

# the agent band the value-matched human group is cut to, taken from the 260914 main cluster
HIGH_SCORE, LOW_SCORE = hc.HIGH_SCORE, hc.LOW_SCORE


# Metrics grouped by what has to happen before humans and agents can be compared on them.
# 'direct'   - same definition, same clock, comparable as they stand
# 'rebuilt'  - comparable only after this module recomputes one or both sides
# 'own rate' - each side stays at its own logging rate. Downsampling the agents to match the human
#              tick grid costs them up to two thirds of a metric's value, while the humans pass
#              through it untouched (ratios 0.97-1.00), so matching grids would distort the one side
#              that did not need it. See READMEs/network_compensation.md, "Downsampling: test it
#              before you do it", and nc.downsampling_effect
# 'dropped'  - not comparable, and not repaired here
COMPARABILITY = {
    'onset_centrality': ('direct', 'position at one frame; sampling rate does not enter'),
    'iti_centrality': ('direct', 'mean radius over the ITI; a mean is insensitive to the grid'),
    'frac_visible_at_onset': ('rebuilt', 'each side on its own sensor: human 110 deg wedge, agent 7 rays'),
    'frac_never_visible': ('rebuilt', 'same; partly measures trial length on both sides'),
    'late_latency_to_wall_visible': ('rebuilt', 'same sensors, logged frame times on both sides'),
    'all_latency_to_wall_visible': ('rebuilt', 'same sensors, logged frame times on both sides'),
    'winner_path_len': ('direct', 'own wins on both sides, summed over logged frames'),
    'path_ratio': ('direct', 'own wins on both sides; per-trial ratio then averaged'),
    'latency_to_reach_wall': ('direct', 'median over own wins, from logged times'),
    'frac_trials_over_5s': ('direct', 'share of own wins over 5 s'),
    'p_high': ('direct', 'own wins, High/Low trials only on both sides'),
    'mean_score': ('direct', "own score including losses on both sides (agents use agent 0's)"),
    'score_rate': ('rebuilt', 'agents sum winner_score, humans sum own score; agents rebuilt per agent'),
    'prop_no_translational': ('own rate', 'humans on ticks, agents at 50 Hz; human ratio 0.99'),
    'prop_no_rotational': ('own rate', 'humans on ticks, agents at 50 Hz; human ratio 0.98'),
    'proportion_resting': ('own rate', 'humans on ticks, agents at 50 Hz; human ratio 0.97'),
    'iti_path_len': ('own rate', 'humans on ticks, agents at 50 Hz; human ratio 1.00'),
    'rot_total': ('own rate', 'humans on ticks, agents at 50 Hz; human ratio 1.00'),
    'win_rate': ('dropped', 'agent self-play is 0.5 by construction, so it carries no information'),
}

DIRECT_METRICS = [m for m, (k, _) in COMPARABILITY.items() if k == 'direct']
REBUILT_METRICS = [m for m, (k, _) in COMPARABILITY.items() if k == 'rebuilt']
OWN_RATE_METRICS = [m for m, (k, _) in COMPARABILITY.items() if k == 'own rate']
DROPPED_METRICS = [m for m, (k, _) in COMPARABILITY.items() if k == 'dropped']
COMPARED_METRICS = DIRECT_METRICS + REBUILT_METRICS + OWN_RATE_METRICS

# how each metric is read, which decides what a difference means
POSITION_METRICS = ['onset_centrality', 'iti_centrality']
PERCEPTION_METRICS = ['frac_visible_at_onset', 'frac_never_visible',
                      'late_latency_to_wall_visible', 'all_latency_to_wall_visible']
EXECUTION_METRICS = ['latency_to_reach_wall', 'winner_path_len', 'path_ratio',
                     'frac_trials_over_5s', 'p_high']
OUTCOME_METRICS = ['mean_score', 'score_rate']
ITI_METRICS = ['prop_no_translational', 'prop_no_rotational', 'proportion_resting',
               'iti_path_len', 'rot_total']


def comparability_table():
    """Return one row per fingerprint metric saying whether humans and agents can be compared on it.
    Takes nothing."""

    rows = [dict(metric=m, status=k, reason=why,
                 reading=next((n for n, g in (('position', POSITION_METRICS),
                                              ('perception', PERCEPTION_METRICS),
                                              ('execution', EXECUTION_METRICS),
                                              ('outcome', OUTCOME_METRICS),
                                              ('between trials', ITI_METRICS)) if m in g), ''))
            for m, (k, why) in COMPARABILITY.items()]
    order = {'direct': 0, 'rebuilt': 1, 'own rate': 2, 'dropped': 3}
    return pd.DataFrame(rows).sort_values(['status', 'reading'], key=lambda s: s.map(order).fillna(s)
                                          ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# agents on the human tick grid
# ---------------------------------------------------------------------------

def agent_tick_table(df, player_id, tick_s=TICK_S, time_col=AGENT_TIME_COL):
    """Return one row per 30 Hz tick for one agent, mirroring the human tick table.
    Takes a session dataframe, the agent id, the tick length and the time column.

    `network_compensation.local_tick_table` puts a locally logged human onto a regular 30 Hz grid so
    their movement is measured over the same interval as a networked player's. Agents are logged
    locally at 50 Hz, so the same resampling puts them on the same footing. Position and unwrapped
    yaw are linearly interpolated; the trial number and ITI flag are taken from the nearest frame
    at or after each tick, as they are labels rather than quantities."""

    t = np.maximum.accumulate(
        (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().to_numpy())
    times = np.arange(t[0], t[-1], tick_s)
    loc = nc.sample_player(df, player_id, times, time_col=time_col)

    # carry each tick the label of the frame it falls on
    frame = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
    return pd.DataFrame(dict(
        player=player_id, t=times, x=loc['x'], z=loc['z'], yaw=loc['yaw'],
        trial_num=df[globals.TRIAL_NUM].to_numpy()[frame],
        is_iti=pd.Series(df['trial_epoch'].to_numpy()[frame]).isin(mc.ITI_EPOCHS).to_numpy()))


def iti_metrics_from_ticks(ticks, keep_trials=None, arena_radius=mc.ARENA_RADIUS,
                           still_speed=STILL_SPEED, still_rot=STILL_ROT):
    """Return one row per ITI measured on the tick grid, with the human cutoffs.
    Takes a tick table, the trials to keep, the arena radius and the two stillness cutoffs.

    An ITI is a contiguous run of ITI ticks, labelled with the trial it leads into. This mirrors
    `human_characterisation.iti_metrics_ticks` so the two sides are measured identically. `yaw` is
    already unwrapped by the resampling, so a spin past 360 degrees does not fold back."""

    is_iti = ticks['is_iti'].to_numpy()
    run_id = np.cumsum(np.r_[True, is_iti[1:] != is_iti[:-1]])

    rows = []
    for _, grp in ticks[is_iti].groupby(run_id[is_iti]):
        # an ITI leads into the trial whose 'trial started' ticks close the run
        upcoming = grp['trial_num'].dropna()
        if keep_trials is not None and (upcoming.empty or float(upcoming.max()) not in keep_trials):
            continue
        x = grp['x'].to_numpy(float); z = grp['z'].to_numpy(float); yaw = grp['yaw'].to_numpy(float)
        ok = ~np.isnan(x) & ~np.isnan(z) & ~np.isnan(yaw)
        x, z, yaw = x[ok], z[ok], yaw[ok]
        if len(x) < 3:                      # a run needs at least 2 tick pairs to give a row
            continue
        step = np.hypot(np.diff(x), np.diff(z))     # translation per tick
        rot = np.abs(np.diff(yaw))                  # rotation per tick, in degrees
        no_trans = step < still_speed
        no_rot = rot < still_rot
        rows.append(dict(
            player=int(grp['player'].iloc[0]),
            trial_num=float(upcoming.max()) if len(upcoming) else np.nan,
            n_ticks=len(x),
            rot_total=float(rot.sum()),
            iti_path_len=float(step.sum()),
            prop_no_translational=float(np.mean(no_trans)),
            prop_no_rotational=float(np.mean(no_rot)),
            proportion_resting=float(np.mean(no_trans & no_rot)),
            iti_centrality=float(1 - np.mean(np.hypot(x, z)) / arena_radius)))
    return pd.DataFrame(rows)


def collect_agent_iti_ticks(sim_data_folder, models, players=mc.PLAYERS, tick_s=TICK_S,
                            verbose=True):
    """Return one row per ITI per agent per model, measured on the 30 Hz tick grid.
    Takes the batch's simulation folder, the model list, the agent ids and the tick length.

    Loads one model at a time and discards its dataframe, as `collect_model_data` does. Applies the
    same trial convention: the session's first trial goes, and a final trial with no trial end is
    already absent from the trial list."""

    records = []
    for model in models:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df, trial_list = prepare_sim_data.prepare_single_model_data(
                    model, sim_data_folder=sim_data_folder)
            if len(trial_list) == 0:
                raise ValueError("no completed trials")
        except Exception as e:
            if verbose:
                print(f"model {model:2d} SKIPPED ({type(e).__name__})", end='  ')
            continue

        # drop the session's first trial, matching collect_model_data
        trial_list = trial_list[1:]
        keep_trials = {float(t[globals.TRIAL_NUM].dropna().iloc[0]) for t in trial_list}
        if not keep_trials:
            continue

        for pid in players:
            ticks = agent_tick_table(df, pid, tick_s=tick_s)
            w = iti_metrics_from_ticks(ticks, keep_trials=keep_trials)
            if len(w):
                w.insert(0, 'model', model)
                records.append(w)
        if verbose:
            print(f"model {model:2d} done", end='  ')

    if verbose:
        print()
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def load_agent_iti_ticks(sim_data_folder, models, cache_name=AGENT_TICK_CACHE, recompute=False,
                         **kwargs):
    """Return the resampled agent ITI table, from cache when possible.
    Takes the simulation folder, the model list, a cache name and a recompute flag."""

    stamp = dict(schema='agent-ticks-30hz', folder=sim_data_folder, models=tuple(sorted(models)),
                 cutoffs=(STILL_SPEED, STILL_ROT), tick_s=TICK_S)
    return data_saving.cached(cache_name,
                              lambda: collect_agent_iti_ticks(sim_data_folder, models, **kwargs),
                              recompute=recompute, stamp=stamp)


def load_agent_tables(sim_data_folder, models, fov=mc.DEFAULT_FOV, cache_name=AGENT_TABLE_CACHE,
                      recompute=False):
    """Return the agent batch's long tables, from cache when possible.
    Takes the simulation folder, the model list, the FoV, a cache name and a recompute flag.

    Wraps `model_characterisation.collect_model_data`, which the batch notebooks call directly.
    Caching it keeps this notebook's first pass off the ~30 model reload."""

    stamp = dict(schema='agent-tables', folder=sim_data_folder, models=tuple(sorted(models)), fov=fov)
    return data_saving.cached(
        cache_name, lambda: mc.collect_model_data(sim_data_folder, models=models, fov=fov),
        recompute=recompute, stamp=stamp)


def human_native_iti(trial_lists, kind='social', players=(0,), keep_map=None, verbose=True):
    """Return one row per ITI per player measured on the logged frames, not the tick grid.
    Takes the trial lists, which kind they are, the log ids to measure and a keep map.

    Calls `model_characterisation.iti_metrics`, the same function the agent batch uses, so the
    human and agent native numbers come from one code path with one pair of cutoffs. Only locally
    logged players should be passed: social log id 1 reaches the host through Netcode's
    interpolation, so its frames are not its own.

    Also returns the session's measured frame rate, since the cutoffs are per frame and the human
    logs run near 48 Hz rather than the agents' 50."""

    rows = []
    for i, trial_list in enumerate(trial_lists):
        if kind == 'social':
            experiment, player_no, session_type = i, None, 'social'
        else:
            experiment, player_no, k = hc.solo_position(i)
            session_type = hc.SOLO_TYPES[k]

        df = pd.concat(trial_list)
        # measure the logged frame rate, so a slow session can be held out rather than silently pooled
        span = (df['timeLocal'].iloc[-1] - df['timeLocal'].iloc[0]).total_seconds()
        fs = (len(df) - 1) / span if span > 0 else np.nan

        logged = {float(t[globals.TRIAL_NUM].dropna().iloc[0]) for t in trial_list
                  if len(t[globals.TRIAL_NUM].dropna())}
        for pid in players:
            # a solo session logs its player as id 0, so the log id and the player number differ
            player = pid if kind == 'social' else player_no
            # look the keep set up per player, never carrying one player's set over to the next
            keep = logged if keep_map is None else keep_map.get(
                (experiment, player, session_type), logged)
            w = mc.iti_metrics(df, pid, keep_trials=keep)
            if not len(w):
                continue
            # an ITI is named by the trial it leads into, so two runs sharing a trial number means the
            # later one had no trial after it. That is the wait at the end of the session, which
            # iti_metrics_ticks drops by the same logic. Agent sessions stop on a trigger and have none
            w = w.drop_duplicates(subset='trial_num', keep='first')
            w['experiment'] = experiment
            w['player'] = player
            w['session_type'] = session_type
            w['fs'] = fs
            rows.append(w)
        del df
        if verbose and i % 20 == 0:
            print(f"{kind} session {i} done", end='  ')

    if verbose:
        print()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_human_native_iti(cache_name='human_iti_native', players=(0,), recompute=False):
    """Return the native-frame human ITI table, from cache when possible.
    Takes a cache name, the log ids to measure and a recompute flag.

    Social only, log id 0. That player is the host, logged locally, so their frames are their own."""

    path = os.path.join(data_strings.DATAFRAME_PATH, data_strings.FILENAME_SOCIAL + '.pkl')
    stamp = dict(schema='human-native-frames', players=tuple(players),
                 cutoffs=(mc.STATIONARY_SPEED, mc.STATIONARY_ROT),
                 pickle=os.path.getmtime(path))

    def compute():
        trial_lists = data_saving.load_data(data_strings.FILENAME_SOCIAL)
        out = human_native_iti(trial_lists, kind='social', players=players)
        del trial_lists
        return out

    return data_saving.cached(cache_name, compute, recompute=recompute, stamp=stamp)


def ray_visibility_metrics(trial, player_id):
    """Return an agent's wall-sensing latency and flags for one trial, from its ray sensor.
    Takes a trial and an agent id.

    The agent counterpart of `model_characterisation.visibility_metrics`, which applies the human
    110 degree wedge. Agents have no continuous field of view: `SelectivePassThroughRaycast` casts
    7 rays at 0, +/-18.3, +/-36.7 and +/-55 degrees, and a wall is sensed only when a ray lands on
    it, so the wedge overstates what an agent detects. A ray hitting any part of a lit wall
    identifies that wall, which is why a sensed wall is a whole wall rather than a point on it.

    Every field matches the wedge version, so the two can be read against each other. Frame 0 is
    slice onset and the window ends at the trigger."""

    sensed, *_ = agent_raycast.trial_sensing(trial, player_id)
    if sensed.size == 0 or sensed.shape[1] == 0:
        return None
    idxs = [w - 1 for w in get_indices.get_walls(trial=trial)]      # wall number -> row index
    any_vis = sensed[idxs, :].any(axis=0)
    ever = bool(any_vis.any())

    # put each frame on the clock the trial was logged on, as the wedge version does
    t = mc.window_times(trial, len(any_vis))
    if t is None:
        t = np.arange(len(any_vis)) / mc.FS
    return dict(
        player=player_id,
        latency_to_wall_visible=float(t[int(np.argmax(any_vis))]) if ever else np.nan,
        visible_at_onset=bool(any_vis[0]),
        ever_visible=ever,
        trial_s=float(t[-1]))


def collect_agent_ray_visibility(sim_data_folder, models, players=mc.PLAYERS, verbose=True):
    """Return one row per trial per agent with the ray-sensor visibility metrics.
    Takes the batch's simulation folder, the model list and the agent ids.

    Mirrors the visibility pass inside `collect_model_data`, including the trial convention, so the
    ray and wedge tables cover the same trials."""

    records = []
    for model in models:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df, trial_list = prepare_sim_data.prepare_single_model_data(
                    model, sim_data_folder=sim_data_folder)
            if len(trial_list) == 0:
                raise ValueError("no completed trials")
        except Exception as e:
            if verbose:
                print(f"model {model:2d} SKIPPED ({type(e).__name__})", end='  ')
            continue

        trial_list = trial_list[1:]                 # drop the session's first trial, as the batch does
        for trial in trial_list:
            for pid in players:
                try:
                    r = ray_visibility_metrics(trial, pid)
                except (IndexError, ValueError, TypeError):
                    r = None
                if r is not None:
                    r.update(model=model,
                             trial_num=float(trial[globals.TRIAL_NUM].dropna().iloc[0]))
                    records.append(r)
        del df, trial_list
        if verbose:
            print(f"model {model:2d} done", end='  ')

    if verbose:
        print()
    return pd.DataFrame(records)


def load_agent_ray_visibility(sim_data_folder, models, cache_name=AGENT_RAY_CACHE,
                              recompute=False, **kwargs):
    """Return the ray-sensor visibility table, from cache when possible.
    Takes the simulation folder, the model list, a cache name and a recompute flag."""

    stamp = dict(schema='agent-ray-visibility', folder=sim_data_folder,
                 models=tuple(sorted(models)), rays=tuple(agent_raycast.RAY_OFFSETS_DEG))
    return data_saving.cached(
        cache_name, lambda: collect_agent_ray_visibility(sim_data_folder, models, **kwargs),
        recompute=recompute, stamp=stamp)


def visibility_by_model(vis, slow_trial_s=mc.SLOW_TRIAL_S):
    """Return the per-model visibility summary from a trial-level visibility table.
    Takes the table and the slow-trial cutoff.

    The same four summaries `build_fingerprint` makes, so a ray table and a wedge table reduce the
    same way. `late_latency` averages over trials with no wall sensed at onset but one sensed
    later; `all_latency` averages over every trial a wall was ever sensed, scoring onset as 0."""

    out = []
    for model, g in vis.groupby('model'):
        late = g[~g['visible_at_onset'] & g['ever_visible']]['latency_to_wall_visible']
        allv = g[g['ever_visible']]['latency_to_wall_visible'].fillna(0.0)
        out.append(dict(
            model=model,
            frac_visible_at_onset=float(g['visible_at_onset'].mean()),
            frac_never_visible=float(1 - g['ever_visible'].mean()),
            late_latency_to_wall_visible=float(late.mean()) if len(late) else np.nan,
            all_latency_to_wall_visible=float(allv.mean()) if len(allv) else np.nan,
            n_late=int(len(late))))
    return pd.DataFrame(out)


def load_solo_native_iti(cache_name='solo_iti_native', recompute=False):
    """Return the solo ITI metrics measured on the logged frames, from cache when possible.
    Takes a cache name and a recompute flag.

    Solo players never touched the network, so there is nothing to recover and no reason to coarsen
    them. `load_iti_table` puts them on the 30 Hz grid only so their movement is measured over the
    same step as a social player's, which is needed when solo and social are compared and not
    otherwise. Use this for any question that stays inside the solo sessions.

    Trials follow the same rules as the tick table: the first solo drops its first trials, and the
    run before a session's first trial is not an ITI."""

    path = os.path.join(data_strings.DATAFRAME_PATH, data_strings.FILENAME_SOLO + '.pkl')
    stamp = dict(schema='solo-native-frames', cutoffs=(mc.STATIONARY_SPEED, mc.STATIONARY_ROT),
                 first_solo_cut=hc.FIRST_SOLO_CUT, pickle=os.path.getmtime(path))

    def compute():
        # take the trials the onset table keeps, so the first-solo cut and the dropped runs match
        onset = hc.load_tables()['onset']
        keep_map = {k: set(g['trial_num'].dropna().astype(float))
                    for k, g in onset[onset['session_type'].isin(hc.SOLO_TYPES)].groupby(list(hc.KEYS))}
        trial_lists = data_saving.load_data(data_strings.FILENAME_SOLO)
        out = human_native_iti(trial_lists, kind='solo', players=(0,), keep_map=keep_map)
        del trial_lists
        return out

    return data_saving.cached(cache_name, compute, recompute=recompute, stamp=stamp)


def resample_effect(native, ticks, keys, metrics=None):
    """Return what the 30 Hz resample does to each ITI metric, for one group.
    Takes the native table, the tick table, the keys identifying one ITI, and the metrics.

    A thin wrapper on `network_compensation.downsampling_effect` with the ITI metrics as the
    default. The general rule this serves, and the options it leaves open, are in
    READMEs/network_compensation.md under "Downsampling: test it before you do it"."""

    metrics = list(metrics or ITI_METRICS) + ['iti_centrality']
    out = nc.downsampling_effect(native, ticks, keys, metrics)
    return out.rename(columns={'resampled': 'tick', 'n_units': 'n_itis'})


def rebuild_agent_iti(fingerprint, iti_ticks):
    """Return the agent fingerprint with its ITI metrics replaced by the tick-grid ones.
    Takes the agent fingerprint and the resampled ITI table.

    **Not the default path.** Downsampling the agents costs them up to two thirds of a metric while
    the humans pass through untouched, so the analysis leaves each side at its own rate. This exists
    for the cross-check that shows the choice of grid does not change the conclusions, and for any
    later case where the ratio test says the agents survive resampling.

    Only the five sampling-sensitive metrics are swapped. `iti_centrality` is left as it is, as a
    mean radius does not move with the grid, which the tests check."""

    per_model = iti_ticks.groupby('model')[ITI_METRICS].mean()
    out = fingerprint.copy().set_index('model')
    out.loc[per_model.index, ITI_METRICS] = per_model
    return out.reset_index()


# ---------------------------------------------------------------------------
# score metrics on one definition
# ---------------------------------------------------------------------------

def agent_own_score_metrics(score_df, rt_df, players=mc.PLAYERS):
    """Return per-model mean score and score rate on the human definition, one row per model.
    Takes the agent score and rt tables.

    The human fingerprint scores a player's own trials, counting a loss as 0, and divides total
    score by total trial time. The agent fingerprint instead sums `winner_score`, which is the
    model's whole output in self-play and so runs about twice as high. Here each agent is scored on
    its own trials and the two are pooled, which is what a human player's number means.

    `score_rate` sums score and time over the session before dividing once. A per-trial rate lets a
    win inside one logged frame rate at hundreds of score per second."""

    cols = {0: 'score_p0', 1: 'score_p1'}
    long = pd.concat([score_df[['model', 'trial_num']].assign(
        player=p, score=score_df[cols[p]].to_numpy(float)) for p in players], ignore_index=True)

    # pair each trial's score with its own duration, so both sums cover the same trials
    merged = long.merge(rt_df[['model', 'trial_num', 'rt_s']], on=['model', 'trial_num'], how='inner')

    out = merged.groupby('model').agg(
        mean_score_own=('score', 'mean'),
        total_score=('score', 'sum'),
        total_time=('rt_s', 'sum'),
        n_scored=('score', 'size')).reset_index()
    # time is counted once per trial, but pooling both agents counts it once per agent
    out['score_rate_own'] = out['total_score'] / out['total_time']
    return out


def human_score_metrics(outcome, keys=hc.KEYS, highlow_only=True):
    """Return per-player mean score and score rate, one row per player per session.
    Takes the human outcome table, the grouping keys and whether to drop forced trials.

    Agents only ever see HighLow trials (`trialTypes = {"HighLow"}` in `GeneralGlobals.cs`), while
    20% of human trials are forced. Dropping forced trials puts the human score on the same trial
    mix as the agent one."""

    df = outcome
    if highlow_only:
        df = df[df['trial_type'] == globals.HIGH_LOW]

    out = df.groupby(list(keys)).agg(
        mean_score_own=('score', 'mean'),
        total_score=('score', 'sum'),
        total_time=('rt_s', 'sum'),
        n_scored=('score', 'size')).reset_index()
    out['score_rate_own'] = out['total_score'] / out['total_time']
    return out


# ---------------------------------------------------------------------------
# group definitions
# ---------------------------------------------------------------------------

def agent_central_models(fingerprint, group_col='group', main='main'):
    """Return the agent fingerprint rows of the central cluster. Takes the fingerprint and its
    group column. Off-cluster models are held out, as a group of one cannot support a group mean."""

    return fingerprint[fingerprint[group_col] == main].copy()


def agent_band(agent_fp, value='onset_centrality'):
    """Return the (min, max) slice-onset centrality of the agent central cluster.
    Takes the agent fingerprint rows and the column holding the axis."""

    return float(agent_fp[value].min()), float(agent_fp[value].max())


def human_central_players(players, session_type, rule='native', band=None,
                          value='mean_centrality', frac=hc.EXTREME_FRAC):
    """Return the central human players of one session, under either grouping rule.
    Takes the player table, the session type, the rule, the agent band and the ranking column.

    'native' takes the top `frac` of players ranked within the session, which is what
    `player_fingerprint.ipynb` reports. 'band' instead takes every player whose mean centrality
    falls inside the agent cluster's own range, which removes the selection mismatch but gives a
    group of whatever size the data allows."""

    sub = players[players['session_type'] == session_type]
    if rule == 'native':
        return hc.extreme_groups(players, frac=frac, value=value).query(
            "session_type == @session_type and group == 'central'").copy()
    if rule == 'band':
        if band is None:
            raise ValueError("rule='band' needs the agent band")
        lo, hi = band
        return sub[(sub[value] >= lo) & (sub[value] <= hi)].copy()
    raise ValueError(f"unknown rule {rule!r}")


# ---------------------------------------------------------------------------
# the comparison itself
# ---------------------------------------------------------------------------

def _cohen_d(a, b):
    """Return Cohen's d between two samples, pooling the two standard deviations. Takes two arrays."""

    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                 / (len(a) + len(b) - 2))
    return np.nan if sp == 0 else float((a.mean() - b.mean()) / sp)


def _boot_ci(a, b, n_boot=2000, seed=0):
    """Return a bootstrap interval on the human-minus-agent difference in means.
    Takes the two samples, the number of resamples and a seed.

    Both sides are resampled over their own units, players for humans and models for agents, so the
    interval asks whether a different sample of players and models would show the same gap."""

    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    diffs = [rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
             for _ in range(n_boot)]
    return tuple(float(v) for v in np.percentile(diffs, [2.5, 97.5]))


def compare_metrics(human_fp, agent_fp, metrics, n_boot=2000, seed=0):
    """Return one row per metric comparing central humans with central agents.
    Takes the two fingerprint tables, the metrics to compare and the bootstrap settings.

    A unit is a player on the human side and a model on the agent side, so `n` differs between the
    columns. The difference is human minus agent. `d` pools the two spreads, and the interval comes
    from resampling players and models independently."""

    rows = []
    for m in metrics:
        if m not in human_fp.columns or m not in agent_fp.columns:
            continue
        h = human_fp[m].to_numpy(float)
        a = agent_fp[m].to_numpy(float)
        lo, hi = _boot_ci(h, a, n_boot=n_boot, seed=seed)
        hm, am = np.nanmean(h), np.nanmean(a)
        rows.append(dict(
            metric=m,
            reading=next((n for n, g in (('position', POSITION_METRICS),
                                         ('perception', PERCEPTION_METRICS),
                                         ('execution', EXECUTION_METRICS),
                                         ('outcome', OUTCOME_METRICS),
                                         ('between trials', ITI_METRICS)) if m in g), ''),
            human_mean=hm, human_sd=np.nanstd(h, ddof=1), n_human=int(np.sum(~np.isnan(h))),
            agent_mean=am, agent_sd=np.nanstd(a, ddof=1), n_agent=int(np.sum(~np.isnan(a))),
            diff=hm - am, ratio=hm / am if am else np.nan,
            d=_cohen_d(h, a), ci_lo=lo, ci_hi=hi))
    out = pd.DataFrame(rows)
    # order by how far apart the two groups are, largest first
    return out.reindex(out['d'].abs().sort_values(ascending=False).index).reset_index(drop=True)
