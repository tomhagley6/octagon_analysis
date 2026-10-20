# Tests for analysis/human_agent_comparison.py. Synthetic sessions, tick tables and fingerprints
# check each step against a known answer.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_human_agent_comparison -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
import analysis.human_agent_comparison as hac
import analysis.human_characterisation as hc
import analysis.model_characterisation as mc


R = mc.ARENA_RADIUS
FS = 50.0           # agent logging rate


def make_session(epochs, x, z, yaw, trial_num, fs=FS):
    ''' Return a minimal agent session dataframe. Takes per-frame epoch, x, z, yaw and trial number
        arrays, and the logging rate. Both agents are given the same trace unless overridden. '''

    n = len(epochs)
    t = pd.to_timedelta(np.arange(n) / fs, unit='s')
    df = pd.DataFrame({'timeApplication': t, 'trial_epoch': epochs, globals.TRIAL_NUM: trial_num})
    for pid in (0, 1):
        df[globals.PLAYER_LOC_DICT[pid]['xloc']] = x
        df[globals.PLAYER_LOC_DICT[pid]['yloc']] = z
        df[globals.PLAYER_ROT_DICT[pid]['yrot']] = yaw
    return df


def make_ticks(x, z, yaw, is_iti=None, trial_num=None, player=0):
    ''' Return a tick table with the given per-tick traces. Takes x, z, yaw, the ITI flag and the
        trial number. '''

    n = len(x)
    return pd.DataFrame(dict(
        player=player, t=np.arange(n) * hac.TICK_S, x=x, z=z, yaw=yaw,
        trial_num=np.ones(n) if trial_num is None else trial_num,
        is_iti=np.ones(n, bool) if is_iti is None else is_iti))


class TestComparabilityTable(unittest.TestCase):
    ''' The table must classify every metric the two fingerprints share, and nothing else. '''

    def test_every_metric_classified(self):
        table = hac.comparability_table()
        self.assertEqual(set(table['metric']), set(hac.COMPARABILITY))

    def test_statuses_are_known(self):
        table = hac.comparability_table()
        self.assertTrue(set(table['status']) <= {'direct', 'rebuilt', 'own rate', 'dropped'})

    def test_covers_the_shared_fingerprint_metrics(self):
        # every metric the human fingerprint reports should be classified, so none is compared silently
        missing = set(hc.FP_METRICS) - set(hac.COMPARABILITY)
        self.assertEqual(missing, set())

    def test_sums_are_never_direct(self):
        # rot_total and iti_path_len are sums, so a coarser grid changes them. They must never be
        # labelled comparable as they stand, whatever is done about the two clocks
        for m in ('rot_total', 'iti_path_len'):
            self.assertNotEqual(hac.COMPARABILITY[m][0], 'direct')

    def test_win_rate_is_dropped(self):
        self.assertEqual(hac.COMPARABILITY['win_rate'][0], 'dropped')

    def test_own_rate_set_matches_the_iti_metrics(self):
        # the label says these five are left on their own clocks; if the two lists drift apart the
        # table and the pipeline stop describing the same thing, which is how this went wrong before
        self.assertEqual(set(hac.OWN_RATE_METRICS), set(hac.ITI_METRICS))

    def test_compared_metrics_is_everything_not_dropped(self):
        self.assertEqual(set(hac.COMPARED_METRICS),
                         set(hac.COMPARABILITY) - set(hac.DROPPED_METRICS))

    def test_no_metric_is_labelled_twice(self):
        groups = (hac.DIRECT_METRICS, hac.REBUILT_METRICS, hac.OWN_RATE_METRICS,
                  hac.DROPPED_METRICS)
        flat = [m for g in groups for m in g]
        self.assertEqual(len(flat), len(set(flat)))


class TestAgentTickTable(unittest.TestCase):
    ''' Resampling a 50 Hz agent session onto the 30 Hz grid. '''

    def test_tick_spacing(self):
        n = 500
        df = make_session(['ITI'] * n, np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n))
        ticks = hac.agent_tick_table(df, 0)
        dt = np.diff(ticks['t'].to_numpy())
        np.testing.assert_allclose(dt, hac.TICK_S, rtol=1e-9)

    def test_constant_velocity_is_preserved(self):
        # a straight run at 10 units/s should read 10 * TICK_S per tick after resampling
        n = 500
        x = np.arange(n) / FS * 10.0
        df = make_session(['ITI'] * n, x, np.zeros(n), np.zeros(n), np.ones(n))
        ticks = hac.agent_tick_table(df, 0)
        step = np.diff(ticks['x'].to_numpy())
        np.testing.assert_allclose(step, 10.0 * hac.TICK_S, rtol=1e-6)

    def test_yaw_is_unwrapped_past_360(self):
        # a continuous spin through 0 must not fold back
        n = 400
        yaw = (np.arange(n) * 5.0) % 360.0
        df = make_session(['ITI'] * n, np.zeros(n), np.zeros(n), yaw, np.ones(n))
        ticks = hac.agent_tick_table(df, 0)
        self.assertGreater(ticks['yaw'].max() - ticks['yaw'].min(), 360.0)
        self.assertTrue(np.all(np.diff(ticks['yaw'].to_numpy()) > 0))

    def test_iti_flag_follows_the_epoch(self):
        n = 300
        epochs = ['ITI'] * 150 + ['slices active'] * 150
        df = make_session(epochs, np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n))
        ticks = hac.agent_tick_table(df, 0)
        # the first half is ITI and the second is not
        self.assertTrue(ticks['is_iti'].iloc[:80].all())
        self.assertFalse(ticks['is_iti'].iloc[-80:].any())

    def test_both_agents_are_sampled_separately(self):
        n = 300
        df = make_session(['ITI'] * n, np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n))
        df[globals.PLAYER_LOC_DICT[1]['xloc']] = 5.0
        self.assertAlmostEqual(hac.agent_tick_table(df, 0)['x'].mean(), 0.0)
        self.assertAlmostEqual(hac.agent_tick_table(df, 1)['x'].mean(), 5.0)


class TestItiMetricsFromTicks(unittest.TestCase):
    ''' ITI metrics computed on a tick table with a known answer. '''

    def test_path_length_and_rotation_sums(self):
        # 10 ticks, each moving 1 unit in x and turning 2 degrees
        n = 11
        ticks = make_ticks(np.arange(n) * 1.0, np.zeros(n), np.arange(n) * 2.0)
        row = hac.iti_metrics_from_ticks(ticks).iloc[0]
        self.assertAlmostEqual(row['iti_path_len'], 10.0)
        self.assertAlmostEqual(row['rot_total'], 20.0)
        self.assertEqual(row['n_ticks'], n)

    def test_stillness_proportions_use_the_tick_cutoffs(self):
        # half the ticks move below the cutoff and half well above it
        n = 11
        step = np.r_[0.0, np.where(np.arange(1, n) <= 5, 0.001, 1.0)]
        x = np.cumsum(step)
        ticks = make_ticks(x, np.zeros(n), np.zeros(n))
        row = hac.iti_metrics_from_ticks(ticks).iloc[0]
        self.assertAlmostEqual(row['prop_no_translational'], 0.5)
        self.assertAlmostEqual(row['prop_no_rotational'], 1.0)    # never rotates
        self.assertAlmostEqual(row['proportion_resting'], 0.5)

    def test_centrality_matches_the_definition(self):
        # standing still at radius r gives centrality 1 - r/R
        n = 10
        r = 5.0
        ticks = make_ticks(np.full(n, r), np.zeros(n), np.zeros(n))
        row = hac.iti_metrics_from_ticks(ticks).iloc[0]
        self.assertAlmostEqual(row['iti_centrality'], 1 - r / R)

    def test_runs_are_split_and_labelled_by_the_trial_they_lead_into(self):
        # two ITI runs separated by a trial, leading into trials 1 and 2
        is_iti = np.r_[np.ones(10, bool), np.zeros(5, bool), np.ones(10, bool)]
        trial_num = np.r_[np.ones(10), np.ones(5), np.full(10, 2.0)]
        n = len(is_iti)
        ticks = make_ticks(np.zeros(n), np.zeros(n), np.zeros(n),
                           is_iti=is_iti, trial_num=trial_num)
        out = hac.iti_metrics_from_ticks(ticks)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(out['trial_num']), [1.0, 2.0])

    def test_keep_trials_drops_the_runs_it_excludes(self):
        is_iti = np.r_[np.ones(10, bool), np.zeros(5, bool), np.ones(10, bool)]
        trial_num = np.r_[np.ones(10), np.ones(5), np.full(10, 2.0)]
        n = len(is_iti)
        ticks = make_ticks(np.zeros(n), np.zeros(n), np.zeros(n),
                           is_iti=is_iti, trial_num=trial_num)
        out = hac.iti_metrics_from_ticks(ticks, keep_trials={2.0})
        self.assertEqual(len(out), 1)
        self.assertEqual(out['trial_num'].iloc[0], 2.0)

    def test_short_runs_are_dropped(self):
        # a run needs at least 2 tick pairs, so 2 ticks give no row
        ticks = make_ticks(np.zeros(2), np.zeros(2), np.zeros(2))
        self.assertEqual(len(hac.iti_metrics_from_ticks(ticks)), 0)

    def test_nan_frames_are_masked_together(self):
        # a NaN in x must drop the same tick from the rotation series
        n = 11
        x = np.arange(n) * 1.0
        yaw = np.arange(n) * 2.0
        x[5] = np.nan
        ticks = make_ticks(x, np.zeros(n), yaw)
        row = hac.iti_metrics_from_ticks(ticks).iloc[0]
        self.assertEqual(row['n_ticks'], n - 1)


class TestResamplingIsFaithful(unittest.TestCase):
    ''' The resample must preserve position, which is what makes the rebuilt metrics trustworthy. '''

    def test_centrality_survives_the_downsample(self):
        # iti_centrality is a mean radius, so it must not move when the grid changes
        n = 1000
        ang = np.linspace(0, 4 * np.pi, n)
        x, z = 8.0 * np.cos(ang), 8.0 * np.sin(ang)
        df = make_session(['ITI'] * n, x, z, np.zeros(n), np.ones(n))
        frame = mc.iti_metrics(df, 0).iloc[0]['iti_centrality']
        ticks = hac.agent_tick_table(df, 0)
        tick = hac.iti_metrics_from_ticks(ticks).iloc[0]['iti_centrality']
        self.assertAlmostEqual(frame, tick, places=3)

    def test_sums_fall_when_the_grid_coarsens(self):
        # a coarser grid chords the path, so both sums must come out lower
        n = 1000
        ang = np.linspace(0, 4 * np.pi, n)
        x, z = 8.0 * np.cos(ang), 8.0 * np.sin(ang)
        yaw = np.arange(n) * 3.0
        df = make_session(['ITI'] * n, x, z, yaw, np.ones(n))
        frame = mc.iti_metrics(df, 0).iloc[0]
        tick = hac.iti_metrics_from_ticks(hac.agent_tick_table(df, 0)).iloc[0]
        self.assertLess(tick['iti_path_len'], frame['iti_path_len'])


class TestResampleEffect(unittest.TestCase):
    ''' The native-against-tick comparison that checks the resample treats both sides alike. '''

    def setUp(self):
        self.keys = ['model', 'player', 'trial_num']
        self.native = pd.DataFrame({'model': [1, 1, 2], 'player': [0, 1, 0],
                                    'trial_num': [1.0, 1.0, 1.0],
                                    'rot_total': [100.0, 200.0, 300.0],
                                    'iti_path_len': [10.0, 20.0, 30.0],
                                    'prop_no_translational': [0.5, 0.5, 0.5],
                                    'prop_no_rotational': [0.5, 0.5, 0.5],
                                    'proportion_resting': [0.5, 0.5, 0.5],
                                    'iti_centrality': [0.8, 0.8, 0.8]})
        self.tick = self.native.copy()
        self.tick['rot_total'] = self.native['rot_total'] * 0.5      # a known halving

    def test_ratio_recovers_a_known_change(self):
        out = hac.resample_effect(self.native, self.tick, self.keys).set_index('metric')
        self.assertAlmostEqual(out.loc['rot_total', 'ratio'], 0.5)

    def test_unchanged_metric_gives_ratio_one(self):
        out = hac.resample_effect(self.native, self.tick, self.keys).set_index('metric')
        self.assertAlmostEqual(out.loc['iti_centrality', 'ratio'], 1.0)

    def test_only_shared_itis_are_compared(self):
        # an ITI present on one side only must not enter the comparison
        tick = self.tick[self.tick['model'] == 1]
        out = hac.resample_effect(self.native, tick, self.keys).set_index('metric')
        self.assertEqual(out.loc['rot_total', 'n_itis'], 2)

    def test_pairs_on_keys_not_row_order(self):
        # reversing one side must not change the answer
        out = hac.resample_effect(self.native, self.tick.iloc[::-1], self.keys).set_index('metric')
        self.assertAlmostEqual(out.loc['rot_total', 'ratio'], 0.5)

    def test_covers_every_iti_metric(self):
        out = hac.resample_effect(self.native, self.tick, self.keys)
        self.assertEqual(set(out['metric']), set(hac.ITI_METRICS) | {'iti_centrality'})


class TestHumanNativeIti(unittest.TestCase):
    ''' The native-frame human pass, which reuses the agent's own ITI function. '''

    def make_social_session(self, n_trials=3, fs=48.0):
        ''' Return a social trial list: each trial is an ITI run then a short active run. '''

        trials = []
        t0 = pd.Timestamp('1900-01-01 12:00:00')
        row = 0
        for k in range(1, n_trials + 1):
            n_iti, n_act = 40, 10
            epochs = [globals.ITI] * n_iti + ['slices active'] * n_act
            n = n_iti + n_act
            t = t0 + pd.to_timedelta((row + np.arange(n)) / fs, unit='s')
            d = pd.DataFrame({'timeLocal': t, 'trial_epoch': epochs,
                              globals.TRIAL_NUM: float(k)})
            for pid in (0, 1):
                d[globals.PLAYER_LOC_DICT[pid]['xloc']] = np.arange(n) * 0.1
                d[globals.PLAYER_LOC_DICT[pid]['yloc']] = 0.0
                d[globals.PLAYER_ROT_DICT[pid]['yrot']] = np.arange(n) * 1.0
            trials.append(d)
            row += n
        return trials

    def test_returns_one_row_per_iti(self):
        out = hac.human_native_iti([self.make_social_session()], kind='social', players=(0,),
                                   verbose=False)
        self.assertEqual(len(out), 3)
        self.assertEqual(sorted(out['trial_num']), [1.0, 2.0, 3.0])

    def test_frame_rate_is_measured(self):
        out = hac.human_native_iti([self.make_social_session(fs=48.0)], kind='social',
                                   players=(0,), verbose=False)
        self.assertAlmostEqual(out['fs'].iloc[0], 48.0, places=1)

    def test_carries_the_session_keys(self):
        out = hac.human_native_iti([self.make_social_session()], kind='social', players=(0,),
                                   verbose=False)
        self.assertEqual(out['session_type'].unique().tolist(), ['social'])
        self.assertEqual(out['player'].unique().tolist(), [0])
        self.assertEqual(out['experiment'].unique().tolist(), [0])

    def test_uses_the_agent_frame_cutoffs(self):
        # moving 0.1 per frame is above the agent cutoff of 0.05, so nothing counts as still
        out = hac.human_native_iti([self.make_social_session()], kind='social', players=(0,),
                                   verbose=False)
        self.assertAlmostEqual(out['prop_no_translational'].iloc[0], 0.0)

    def test_end_of_session_wait_is_dropped(self):
        # append a motionless run carrying the last trial's number, which is the wait after the final
        # trial. It has no trial after it, so it is not an ITI and must not give a row
        trials = self.make_social_session(n_trials=2)
        tail = trials[-1].copy()
        tail['trial_epoch'] = globals.ITI
        for pid in (0, 1):
            tail[globals.PLAYER_LOC_DICT[pid]['xloc']] = 0.0
            tail[globals.PLAYER_ROT_DICT[pid]['yrot']] = 0.0
        out = hac.human_native_iti([trials + [tail]], kind='social', players=(0,), verbose=False)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(out['trial_num']), [1.0, 2.0])

    def test_one_row_per_trial_per_player(self):
        out = hac.human_native_iti([self.make_social_session()], kind='social', players=(0, 1),
                                   verbose=False)
        self.assertEqual(out.duplicated(subset=['experiment', 'player', 'session_type',
                                                'trial_num']).sum(), 0)

    def test_keep_map_is_looked_up_per_player(self):
        # one player's keep set must never carry over to the next, which a shared variable would do
        session = self.make_social_session(n_trials=3)
        keep_map = {(0, 0, 'social'): {1.0}, (0, 1, 'social'): {3.0}}
        out = hac.human_native_iti([session], kind='social', players=(0, 1), keep_map=keep_map,
                                   verbose=False)
        self.assertEqual(sorted(out[out['player'] == 0]['trial_num']), [1.0])
        self.assertEqual(sorted(out[out['player'] == 1]['trial_num']), [3.0])

    def test_missing_keep_map_entry_falls_back_to_every_logged_trial(self):
        session = self.make_social_session(n_trials=3)
        out = hac.human_native_iti([session], kind='social', players=(0,), keep_map={},
                                   verbose=False)
        self.assertEqual(sorted(out['trial_num']), [1.0, 2.0, 3.0])

    def test_both_log_ids_can_be_measured(self):
        out = hac.human_native_iti([self.make_social_session()], kind='social', players=(0, 1),
                                   verbose=False)
        self.assertEqual(sorted(out['player'].unique()), [0, 1])


class TestVisibilityByModel(unittest.TestCase):
    ''' Reducing a trial-level visibility table, for the ray sensor and the wedge alike. '''

    def setUp(self):
        # 4 trials: seen at onset, seen late at 0.2 s, seen late at 0.4 s, never seen
        self.vis = pd.DataFrame(dict(
            model=1, player=0, trial_num=[1.0, 2.0, 3.0, 4.0],
            visible_at_onset=[True, False, False, False],
            ever_visible=[True, True, True, False],
            latency_to_wall_visible=[0.0, 0.2, 0.4, np.nan]))

    def test_frac_visible_at_onset(self):
        self.assertAlmostEqual(hac.visibility_by_model(self.vis)['frac_visible_at_onset'].iloc[0], 0.25)

    def test_frac_never_visible(self):
        self.assertAlmostEqual(hac.visibility_by_model(self.vis)['frac_never_visible'].iloc[0], 0.25)

    def test_late_latency_uses_only_late_trials(self):
        # trials 2 and 3, so (0.2 + 0.4) / 2
        out = hac.visibility_by_model(self.vis).iloc[0]
        self.assertAlmostEqual(out['late_latency_to_wall_visible'], 0.3)
        self.assertEqual(out['n_late'], 2)

    def test_all_latency_scores_onset_as_zero(self):
        # trials 1, 2 and 3, so (0 + 0.2 + 0.4) / 3
        out = hac.visibility_by_model(self.vis).iloc[0]
        self.assertAlmostEqual(out['all_latency_to_wall_visible'], 0.2)

    def test_never_seen_trials_are_excluded_from_both_latencies(self):
        # trial 4 has no latency, so it must not be counted as a zero
        out = hac.visibility_by_model(self.vis).iloc[0]
        self.assertFalse(np.isnan(out['all_latency_to_wall_visible']))
        self.assertAlmostEqual(out['all_latency_to_wall_visible'], 0.2)

    def test_late_latency_is_nan_when_no_late_trials(self):
        vis = self.vis[self.vis['visible_at_onset']]
        self.assertTrue(np.isnan(hac.visibility_by_model(vis)['late_latency_to_wall_visible'].iloc[0]))

    def test_one_row_per_model(self):
        vis = pd.concat([self.vis, self.vis.assign(model=2)])
        self.assertEqual(sorted(hac.visibility_by_model(vis)['model']), [1, 2])


class TestAgentOwnScoreMetrics(unittest.TestCase):
    ''' Rebuilding the agent score onto the human definition. '''

    def setUp(self):
        # 4 trials, agent 0 wins High twice, agent 1 wins Low twice; every trial takes 2 s
        self.score = pd.DataFrame(dict(model=1, trial_num=[1.0, 2.0, 3.0, 4.0],
                                       score_p0=[1.0, 0.0, 1.0, 0.0],
                                       score_p1=[0.0, 0.4, 0.0, 0.4]))
        self.rt = pd.DataFrame(dict(model=1, trial_num=[1.0, 2.0, 3.0, 4.0], rt_s=[2.0] * 4))

    def test_mean_own_score_pools_both_agents(self):
        out = hac.agent_own_score_metrics(self.score, self.rt).iloc[0]
        # 8 agent-trials scoring 1.0, 0.4, 1.0, 0.4 and four zeros
        self.assertAlmostEqual(out['mean_score_own'], 2.8 / 8)

    def test_score_rate_sums_before_dividing(self):
        out = hac.agent_own_score_metrics(self.score, self.rt).iloc[0]
        # total score 2.8 over 8 agent-trials of 2 s
        self.assertAlmostEqual(out['score_rate_own'], 2.8 / 16.0)

    def test_own_rate_is_half_the_winner_rate(self):
        # in self-play exactly one agent scores, so pooling both halves the per-trial rate
        out = hac.agent_own_score_metrics(self.score, self.rt).iloc[0]
        winner_rate = self.score[['score_p0', 'score_p1']].max(axis=1).sum() / self.rt['rt_s'].sum()
        self.assertAlmostEqual(out['score_rate_own'], winner_rate / 2)

    def test_trials_missing_a_time_are_dropped(self):
        rt = self.rt[self.rt['trial_num'] != 4.0]
        out = hac.agent_own_score_metrics(self.score, rt).iloc[0]
        self.assertEqual(out['n_scored'], 6)


class TestHumanScoreMetrics(unittest.TestCase):
    ''' The human side of the same rebuild. '''

    def setUp(self):
        self.outcome = pd.DataFrame({
            'experiment': 0, 'player': 0, 'session_type': 'social',
            'trial_type': [globals.HIGH_LOW, globals.HIGH_LOW, 'ForcedHigh', 'ForcedLow'],
            'score': [1.0, 0.0, 1.0, 0.4], 'rt_s': [2.0, 2.0, 2.0, 2.0]})

    def test_forced_trials_are_dropped_by_default(self):
        out = hac.human_score_metrics(self.outcome).iloc[0]
        self.assertEqual(out['n_scored'], 2)
        self.assertAlmostEqual(out['mean_score_own'], 0.5)

    def test_forced_trials_can_be_kept(self):
        out = hac.human_score_metrics(self.outcome, highlow_only=False).iloc[0]
        self.assertEqual(out['n_scored'], 4)

    def test_score_rate_sums_before_dividing(self):
        out = hac.human_score_metrics(self.outcome).iloc[0]
        self.assertAlmostEqual(out['score_rate_own'], 1.0 / 4.0)


class TestGroups(unittest.TestCase):
    ''' Both grouping rules. '''

    def setUp(self):
        # 10 players in one session, centrality 0.0 to 0.9
        self.players = pd.DataFrame(dict(
            experiment=range(10), player=0, session_type='social',
            mean_centrality=np.arange(10) / 10.0))

    def test_native_rule_takes_the_top_fraction(self):
        out = hac.human_central_players(self.players, 'social', rule='native', frac=0.2)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out['mean_centrality'].min(), 0.8)

    def test_band_rule_takes_the_agent_range(self):
        out = hac.human_central_players(self.players, 'social', rule='band', band=(0.5, 0.75))
        np.testing.assert_allclose(sorted(out['mean_centrality']), [0.5, 0.6, 0.7])

    def test_band_rule_needs_a_band(self):
        with self.assertRaises(ValueError):
            hac.human_central_players(self.players, 'social', rule='band')

    def test_unknown_rule_raises(self):
        with self.assertRaises(ValueError):
            hac.human_central_players(self.players, 'social', rule='nonsense')

    def test_agent_band_reads_the_cluster_range(self):
        fp = pd.DataFrame(dict(onset_centrality=[0.81, 0.87, 0.91]))
        self.assertEqual(hac.agent_band(fp), (0.81, 0.91))

    def test_agent_central_models_holds_out_off_cluster(self):
        fp = pd.DataFrame(dict(model=[1, 2, 3], group=['main', 'main', 'off']))
        self.assertEqual(list(hac.agent_central_models(fp)['model']), [1, 2])


class TestCompareMetrics(unittest.TestCase):
    ''' The side-by-side table. '''

    def setUp(self):
        rng = np.random.default_rng(0)
        self.human = pd.DataFrame(dict(a=rng.normal(1.0, 0.1, 40), b=rng.normal(5.0, 1.0, 40)))
        self.agent = pd.DataFrame(dict(a=rng.normal(2.0, 0.1, 27), b=rng.normal(5.0, 1.0, 27)))

    def test_difference_is_human_minus_agent(self):
        out = hac.compare_metrics(self.human, self.agent, ['a']).iloc[0]
        self.assertAlmostEqual(out['diff'], out['human_mean'] - out['agent_mean'])
        self.assertLess(out['diff'], 0)     # human mean 1.0 is below agent mean 2.0

    def test_counts_are_per_side(self):
        out = hac.compare_metrics(self.human, self.agent, ['a']).iloc[0]
        self.assertEqual(out['n_human'], 40)
        self.assertEqual(out['n_agent'], 27)

    def test_ordered_by_absolute_effect_size(self):
        out = hac.compare_metrics(self.human, self.agent, ['a', 'b'])
        self.assertEqual(out['metric'].iloc[0], 'a')    # a separates, b does not

    def test_interval_excludes_zero_for_a_real_difference(self):
        out = hac.compare_metrics(self.human, self.agent, ['a'], n_boot=500).iloc[0]
        self.assertLess(out['ci_hi'], 0)

    def test_interval_covers_zero_for_no_difference(self):
        out = hac.compare_metrics(self.human, self.agent, ['b'], n_boot=500).iloc[0]
        self.assertLessEqual(out['ci_lo'], 0)
        self.assertGreaterEqual(out['ci_hi'], 0)

    def test_missing_metrics_are_skipped(self):
        out = hac.compare_metrics(self.human, self.agent, ['a', 'not_a_column'])
        self.assertEqual(list(out['metric']), ['a'])

    def test_cohen_d_known_value(self):
        # two samples one pooled SD apart
        a = np.array([0.0, 2.0] * 20)       # mean 1, sd ~1.01
        b = a + 2.0
        self.assertAlmostEqual(hac._cohen_d(a, b), -2.0, places=1)

    def test_cohen_d_handles_zero_spread(self):
        self.assertTrue(np.isnan(hac._cohen_d(np.ones(5), np.ones(5))))


if __name__ == '__main__':
    unittest.main()
