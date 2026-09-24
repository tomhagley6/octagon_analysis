# Tests for analysis/human_characterisation.py. Synthetic trials and players check each step against
# a known answer. The last class checks the saved session file list against the real pickles, and is
# skipped when that list has not been cached.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_human_characterisation -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import globals
import analysis.human_characterisation as hc
import analysis.model_characterisation as mc
import data_extraction.network_compensation.network_compensation as nr


R = mc.ARENA_RADIUS


def make_trial(trial_num, walls=(1, 2), onset_r=(0.0, 0.0), end_r=(0.0, 0.0), winner=0,
               n_players=2, start_row=0):
    ''' Return a minimal trial dataframe with trial start, slice onset, trigger and trial end rows.
        Takes a trial number, the two walls, each player's radius at onset and at trial end, and the
        winning player. Players stand on the +x axis, so centrality is 1 - r / R. '''

    events = [globals.TRIAL_START, globals.SLICE_ONSET, globals.SELECTED_TRIGGER_ACTIVATION,
              globals.TRIAL_END]
    radius = {globals.TRIAL_START: onset_r, globals.SLICE_ONSET: onset_r,
              globals.SELECTED_TRIGGER_ACTIVATION: end_r, globals.TRIAL_END: end_r}
    t0 = pd.Timestamp('1900-01-01 12:00:00') + pd.Timedelta(seconds=10 * trial_num)
    rows = []
    for i, ev in enumerate(events):
        row = {'timeLocal': t0 + pd.Timedelta(seconds=i), 'eventDescription': ev,
               globals.TRIAL_NUM: float(trial_num), globals.WALL_1: walls[0], globals.WALL_2: walls[1],
               globals.TRIGGER_CLIENT: winner if ev == globals.SELECTED_TRIGGER_ACTIVATION else np.nan}
        for p in range(n_players):
            row[globals.PLAYER_LOC_DICT[p]['xloc']] = radius[ev][p]
            row[globals.PLAYER_LOC_DICT[p]['yloc']] = 0.0
        rows.append(row)
    return pd.DataFrame(rows, index=range(start_row, start_row + len(rows)))


def make_session(n_trials, onset_r, n_players=2, winners=None, end_r=None):
    ''' Return a list of synthetic trials. Takes the trial count and a per-trial list of onset radii. '''

    trials = []
    for t in range(n_trials):
        trials.append(make_trial(t + 1, onset_r=onset_r[t], n_players=n_players,
                                 winner=0 if winners is None else winners[t],
                                 end_r=onset_r[t] if end_r is None else end_r[t],
                                 start_row=100 * t))
    return trials


class TestSoloIndex(unittest.TestCase):

    def test_round_trip(self):
        # solo sessions sit at 4 * experiment + 2 * player + (0 first, 1 second)
        for e in range(5):
            for p in (0, 1):
                for k in (0, 1):
                    i = hc.solo_index(e, p, k)
                    self.assertEqual(hc.solo_position(i), (e, p, k))

    def test_known_values(self):
        self.assertEqual(hc.solo_index(0, 0, 0), 0)
        self.assertEqual(hc.solo_index(0, 1, 1), 3)
        self.assertEqual(hc.solo_index(49, 0, 0), 196)


class TestFileAlignment(unittest.TestCase):

    social = ['240913_1/2024-09-13_11-31-00_YW13_JL13_Social.json',
              '240927_1/2024-09-27_14-25-20_SH27_EN27_Social.json']
    solo = ['240913_1/2024-09-13_11-23-37_YW13_FirstSolo.json',
            '240913_1/2024-09-13_11-53-34_YW13_SecondSolo.json',
            '240913_1/2024-09-13_11-23-50_JL13_FirstSolo.json',
            '240913_1/2024-09-13_11-53-56_JL13_SecondSolo.json',
            '240927_1/2024-09-27_14-14-31_SH27_FirstSolo.json',
            '240927_1/2024-09-27_14-45-55_SH27_SecondSolo.json',
            '240927_1/2024-09-27_14-14-42_EN27_FirstSolo.json',
            '240927_1/2024-09-27_14-45-46_EN27_SecondSolo.json']

    def test_aligned_lists_pass(self):
        hc.check_file_alignment(self.social, self.solo)

    def test_swapped_players_fail(self):
        swapped = self.solo[2:4] + self.solo[0:2] + self.solo[4:]
        with self.assertRaises(ValueError):
            hc.check_file_alignment(self.social, swapped)

    def test_experiment_info_dates(self):
        info = hc.experiment_info(self.social)
        self.assertEqual(info['folder'].tolist(), ['240913_1', '240927_1'])
        self.assertEqual(info['date'].iloc[1], pd.Timestamp('2024-09-27'))


class TestCollectOnsetTable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # experiment 0's social session: player 0 at r = 0 (centrality 1), player 1 at r = R / 2
        social = [make_session(6, [(0.0, R / 2)] * 6, winners=[0, 1, 0, 1, 0, 1])]
        # the same pair's four solo sessions, 8 trials each, the player alone at log id 0
        solo = [make_session(8, [(R * 0.1 * (k + 1),)] * 8, n_players=1) for k in range(4)]
        cls.onset = hc.collect_onset_table(social, 'social')
        cls.onset_solo = hc.collect_onset_table(solo, 'solo', first_solo_cut=5)

    def test_social_keeps_every_trial(self):
        # social sessions never drop a trial
        self.assertEqual(len(self.onset), 12)
        self.assertEqual(sorted(self.onset['trial'].unique()), list(range(6)))

    def test_social_centrality_per_player(self):
        by_player = self.onset.groupby('player')['centrality'].mean()
        self.assertAlmostEqual(by_player[0], 1.0)
        self.assertAlmostEqual(by_player[1], 0.5)

    def test_first_solo_cut(self):
        # the first solo loses its first 5 trials, the second solo keeps all 8
        n = self.onset_solo.groupby(['player', 'session_type']).size()
        self.assertEqual(n[(0, 'solo1')], 3)
        self.assertEqual(n[(0, 'solo2')], 8)
        self.assertEqual(self.onset_solo.query("session_type == 'solo1'")['trial'].min(), 5)
        self.assertEqual(self.onset_solo.query("session_type == 'solo1'")['trial_pos'].min(), 0)

    def test_solo_player_mapping(self):
        # solo index 2 is player 1's first solo, stood at r = 0.3 R
        row = self.onset_solo.query("player == 1 and session_type == 'solo1'").iloc[0]
        self.assertAlmostEqual(row['centrality'], 0.7)
        self.assertEqual(row['experiment'], 0)

    def test_previous_trial_outcome(self):
        # player 0 wins trials 0, 2 and 4, so prev_won alternates and trial 0 has none
        p0 = self.onset.query('player == 0').sort_values('trial')
        self.assertTrue(np.isnan(p0['prev_won'].iloc[0]))
        self.assertEqual(p0['prev_won'].iloc[1:].tolist(), [1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(p0['prev_end_centrality'].iloc[1], 1.0)

    def test_wall_separation(self):
        self.assertTrue((self.onset['wall_sep'] == 1).all())

    def test_solo_has_no_outcome(self):
        self.assertTrue(self.onset_solo['won'].isna().all())

    def test_session_table(self):
        rec = hc.collect_session_table(
            [make_session(6, [(0.0, R / 2)] * 6)], 'social')
        self.assertEqual(rec['n_trials'].iloc[0], 6)
        self.assertAlmostEqual(rec['prop_sep_1'].iloc[0], 1.0)


class TestTrialSplit(unittest.TestCase):

    def test_boundary_between_modes(self):
        rng = np.random.default_rng(0)
        vals = np.r_[rng.normal(0.15, 0.05, 500), rng.normal(0.8, 0.08, 1000)]
        onset = pd.DataFrame(dict(session_type='social', centrality=vals))
        out, split = hc.fit_trial_split(onset)
        b = split['boundary']
        self.assertTrue(0.25 < b < 0.65)
        self.assertEqual(split['reference'], 'social')
        # low values are peripheral, high values central
        self.assertTrue(out.loc[vals < 0.1, 'peripheral'].all())
        self.assertFalse(out.loc[vals > 0.8, 'peripheral'].any())

    def test_reference_boundary_applied_to_every_session(self):
        rng = np.random.default_rng(1)
        # solo1 alone would split near 0.3, social alone near 0.7
        a = np.r_[rng.normal(0.1, 0.03, 300), rng.normal(0.5, 0.03, 300)]
        b = np.r_[rng.normal(0.5, 0.03, 300), rng.normal(0.9, 0.03, 300)]
        onset = pd.DataFrame(dict(session_type=['solo1'] * 600 + ['social'] * 600, centrality=np.r_[a, b]))
        out, split = hc.fit_trial_split(onset)
        self.assertGreater(split['boundary'], 0.6)
        # solo1 trials near 0.5 fall below the social boundary, so they count as peripheral
        solo1 = out[out['session_type'] == 'solo1']
        self.assertTrue(solo1['peripheral'].all())
        self.assertTrue((out['peripheral'] == (out['centrality'] < split['boundary'])).all())

    def test_other_reference(self):
        rng = np.random.default_rng(1)
        a = np.r_[rng.normal(0.1, 0.03, 300), rng.normal(0.5, 0.03, 300)]
        b = np.r_[rng.normal(0.5, 0.03, 300), rng.normal(0.9, 0.03, 300)]
        onset = pd.DataFrame(dict(session_type=['solo1'] * 600 + ['social'] * 600, centrality=np.r_[a, b]))
        _, split = hc.fit_trial_split(onset, reference='solo1')
        self.assertLess(split['boundary'], 0.4)

    def test_missing_reference_raises(self):
        onset = pd.DataFrame(dict(session_type='solo1', centrality=np.linspace(0, 1, 50)))
        with self.assertRaises(ValueError):
            hc.fit_trial_split(onset)


class TestPlayerTable(unittest.TestCase):

    def test_summary_and_fixed_cut(self):
        onset = pd.DataFrame(dict(
            experiment=[0] * 10 + [0] * 10, player=[0] * 10 + [1] * 10, session_type='social',
            trial=list(range(10)) * 2,
            centrality=[0.9] * 10 + [0.1] * 8 + [0.9] * 2,
            peripheral=[False] * 10 + [True] * 8 + [False] * 2))
        players = hc.player_table(onset)
        p0 = players.query('player == 0').iloc[0]
        p1 = players.query('player == 1').iloc[0]
        self.assertAlmostEqual(p0['frac_peripheral'], 0.0)
        self.assertAlmostEqual(p1['frac_peripheral'], 0.8)
        self.assertEqual(p0['n_trials'], 10)
        self.assertAlmostEqual(p1['mean_centrality'], 0.26)

    def test_fixed_cut_edges(self):
        labels = hc.fixed_cut_labels(pd.Series([0.0, 0.2, 0.21, 0.79, 0.8, 1.0, np.nan]))
        self.assertEqual(labels.tolist(),
                         ['central', 'central', 'mixed', 'mixed', 'peripheral', 'peripheral', 'unassigned'])


class TestPlayerMixtures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2)
        vals = np.r_[rng.normal(0.15, 0.04, 40), rng.normal(0.75, 0.06, 80)]
        cls.players = pd.DataFrame(dict(experiment=np.arange(120) // 2, player=np.arange(120) % 2,
                                        session_type='social', mean_centrality=vals))
        cls.fits = hc.fit_mixtures(cls.players, ks=(1, 2, 3))

    def test_components_ordered_by_mean(self):
        # component 0 is always the lowest centrality, whatever order sklearn returns
        for seed in range(4):
            fits = hc.fit_mixtures(self.players, ks=(2,), random_state=seed)
            labelled = hc.add_mixture_labels(self.players, fits)
            low = labelled.loc[labelled['mean_centrality'] < 0.3, 'gmm2']
            self.assertTrue((low == 'peripheral').all())

    def test_two_clusters_recovered(self):
        labelled = hc.add_mixture_labels(self.players, self.fits)
        self.assertEqual((labelled['gmm2'] == 'peripheral').sum(), 40)
        self.assertTrue((labelled['gmm2_posterior'] > 0.9).all())

    def test_mixture_table_has_each_k(self):
        tab = hc.mixture_table(self.fits)
        self.assertEqual(tab['k'].tolist(), [1, 2, 3])
        # two well-separated clusters should beat one on BIC
        bic = tab.set_index('k')['bic']
        self.assertLess(bic[2], bic[1])

    def test_bootstrap_counts_sum(self):
        counts = hc.bootstrap_best_k(self.players, ks=(1, 2), n_boot=20)
        self.assertEqual(int(counts.loc['social'].sum()), 20)


class TestPartners(unittest.TestCase):

    def test_partner_correlation(self):
        rng = np.random.default_rng(3)
        shared = rng.normal(0.5, 0.2, 200)
        vals = np.c_[shared + rng.normal(0, 0.05, 200), shared + rng.normal(0, 0.05, 200)].ravel()
        players = pd.DataFrame(dict(experiment=np.repeat(np.arange(200), 2), player=np.tile([0, 1], 200),
                                    session_type='social', mean_centrality=vals))
        res = hc.partner_correlation(players).set_index('session_type')
        self.assertGreater(res.loc['social', 'r'], 0.8)
        self.assertEqual(res.loc['social', 'n_pairs'], 200)

    def test_partner_agreement_chance(self):
        players = pd.DataFrame(dict(experiment=[0, 0, 1, 1], player=[0, 1, 0, 1], session_type='social',
                                    lab=['a', 'a', 'a', 'b']))
        res = hc.partner_agreement(players, 'lab').set_index('session_type')
        self.assertAlmostEqual(res.loc['social', 'agreement'], 0.5)
        # three 'a' and one 'b' gives chance 0.75^2 + 0.25^2
        self.assertAlmostEqual(res.loc['social', 'chance'], 0.625)

    @staticmethod
    def pair_players(n_experiments, shared_sd, own_sd, seed, session_type='solo1'):
        """ Return a player table where partners share a value with spread shared_sd, plus their own
            spread own_sd. Takes the experiment count, the two spreads and a seed. """
        rng = np.random.default_rng(seed)
        shared = rng.normal(0.5, shared_sd, n_experiments)
        vals = np.c_[shared + rng.normal(0, own_sd, n_experiments), shared + rng.normal(0, own_sd, n_experiments)]
        return pd.DataFrame(dict(experiment=np.repeat(np.arange(n_experiments), 2),
                                 player=np.tile([0, 1], n_experiments), session_type=session_type,
                                 mean_centrality=vals.ravel()))

    def test_partner_gap_null_similar_partners(self):
        players = self.pair_players(60, shared_sd=0.2, own_sd=0.03, seed=4)
        res = hc.partner_gap_null(players, 'solo1', n_perm=500)
        self.assertLess(res['gap'], res['null'].mean())
        self.assertLess(res['p'], 0.01)
        self.assertEqual(res['n_pairs'], 60)
        self.assertEqual(len(res['null']), 500)

    def test_partner_gap_null_independent_partners(self):
        players = self.pair_players(60, shared_sd=0.0, own_sd=0.2, seed=5)
        res = hc.partner_gap_null(players, 'solo1', n_perm=500)
        self.assertGreater(res['p'], 0.05)

    def test_partner_gap_null_known_gap(self):
        players = pd.DataFrame(dict(experiment=[0, 0, 1, 1, 2, 2], player=[0, 1] * 3, session_type='solo1',
                                    mean_centrality=[0.1, 0.2, 0.5, 0.5, 0.9, 0.6]))
        res = hc.partner_gap_null(players, 'solo1', n_perm=50)
        self.assertAlmostEqual(res['gap'], (0.1 + 0.0 + 0.3) / 3)
        # every random pairing sends player 0 to another experiment's player 1
        pairings = [np.mean(np.abs(np.array([0.1, 0.5, 0.9]) - np.array(b)))
                    for b in ([0.5, 0.6, 0.2], [0.6, 0.2, 0.5])]
        self.assertTrue(all(any(np.isclose(g, p) for p in pairings) for g in res['null']))


class TestConsistency(unittest.TestCase):

    def test_split_half_consistent_players(self):
        # each player has a fixed centrality plus small noise, so halves agree
        rng = np.random.default_rng(4)
        rows = []
        for i, level in enumerate(np.linspace(0.1, 0.9, 30)):
            for t in range(40):
                rows.append(dict(experiment=i, player=0, session_type='social', trial_pos=t,
                                 centrality=level + rng.normal(0, 0.02)))
        res = hc.split_half(pd.DataFrame(rows)).set_index(['session_type', 'split'])
        self.assertGreater(res.loc[('social', 'odd_even'), 'r'], 0.95)
        self.assertGreater(res.loc[('social', 'odd_even'), 'spearman_brown'],
                           res.loc[('social', 'odd_even'), 'r'])

    def test_run_consistency_blocks_beat_shuffle(self):
        # one block of peripheral trials then one block of central trials
        onset = pd.DataFrame(dict(experiment=0, player=0, session_type='social', trial_pos=range(40),
                                  peripheral=[True] * 20 + [False] * 20))
        res = hc.run_consistency(onset, n_shuffle=50).iloc[0]
        self.assertAlmostEqual(res['p_same'], 38 / 39)
        self.assertLess(res['p_same_shuffled'], 0.6)

    @staticmethod
    def label_onset(labels_by_player, session_type='social'):
        """ Return an onset table from a dict of player id to a list of peripheral flags. Takes the dict. """
        rows = []
        for p, labels in labels_by_player.items():
            # write trials in reverse to check the function sorts by trial_pos
            for t in reversed(range(len(labels))):
                rows.append(dict(experiment=0, player=p, session_type=session_type, trial_pos=t,
                                 peripheral=labels[t]))
        return pd.DataFrame(rows)

    def test_switch_score_single_switch(self):
        # 20 peripheral then 20 central gives one change against 2 * 20 * 20 / 40 = 20 expected
        res = hc.switch_score(self.label_onset({0: [True] * 20 + [False] * 20})).iloc[0]
        self.assertEqual(res['n_changes'], 1)
        self.assertAlmostEqual(res['expected_changes'], 20.0)
        self.assertAlmostEqual(res['switch_score'], 0.05)
        self.assertAlmostEqual(res['frac_peripheral'], 0.5)

    def test_switch_score_alternating(self):
        # strict alternation changes on every trial, above the random-order count
        res = hc.switch_score(self.label_onset({0: [True, False] * 20})).iloc[0]
        self.assertEqual(res['n_changes'], 39)
        self.assertGreater(res['switch_score'], 1.9)

    def test_switch_score_random_order_near_one(self):
        rng = np.random.default_rng(7)
        scores = [hc.switch_score(self.label_onset({0: list(rng.random(200) < 0.3)})).iloc[0]['switch_score']
                  for _ in range(30)]
        self.assertAlmostEqual(np.mean(scores), 1.0, delta=0.05)

    def test_switch_score_one_label_is_nan(self):
        res = hc.switch_score(self.label_onset({0: [False] * 30, 1: [True] * 29 + [False]}))
        res = res.set_index('player')
        self.assertTrue(np.isnan(res.loc[0, 'switch_score']))
        self.assertEqual(res.loc[0, 'expected_changes'], 0)
        # one peripheral-free trial at the end is a single change, as expected from 2 * 29 * 1 / 30
        self.assertAlmostEqual(res.loc[1, 'switch_score'], 1 / (58 / 30))

    @staticmethod
    def wsls_onset(rows, session_type='social'):
        """ Return a social onset table from (player, trial_pos, centrality, prev_won) rows. Takes the rows. """
        return pd.DataFrame([dict(experiment=0, player=p, session_type=session_type, trial_pos=t,
                                  centrality=c, prev_won=w) for p, t, c, w in rows])

    def test_wsls_known_values(self):
        # after a win the player returns to 0.8, after a loss they drop to 0.4
        seq = [(0.8, np.nan), (0.8, 1.0), (0.4, 0.0), (0.8, 1.0), (0.4, 0.0), (0.8, 1.0), (0.4, 0.0)]
        rows = [(0, t, c, w) for t, (c, w) in enumerate(seq)]
        res = hc.wsls_metrics(self.wsls_onset(rows), min_trials=3).iloc[0]
        # after a win the changes are 0 (already at 0.8), +0.4 and +0.4; after a loss they are -0.4 each
        self.assertAlmostEqual(res['d_win'], 0.8 / 3)
        self.assertAlmostEqual(res['d_loss'], -0.4)
        self.assertAlmostEqual(res['abs_win'], 0.8 / 3)
        self.assertAlmostEqual(res['abs_loss'], 0.4)
        self.assertAlmostEqual(res['shift_diff'], 0.4 - 0.8 / 3)
        self.assertAlmostEqual(res['direction_diff'], -0.4 - 0.8 / 3)
        self.assertEqual(res['n_win'], 3)
        self.assertEqual(res['n_loss'], 3)

    def test_wsls_lose_shift_bigger(self):
        # the player holds their position after a win and jumps after a loss
        rng = np.random.default_rng(11)
        rows, c = [], 0.6
        for t in range(120):
            won = float(t % 2)
            rows.append((0, t, c, np.nan if t == 0 else won_prev))
            won_prev = won
            c = c if won else np.clip(c + rng.choice([-0.3, 0.3]), 0, 1)
        res = hc.wsls_metrics(self.wsls_onset(rows)).iloc[0]
        self.assertAlmostEqual(res['abs_win'], 0.0, places=6)
        self.assertGreater(res['abs_loss'], 0.2)
        self.assertGreater(res['shift_diff'], 0.2)

    def test_wsls_skips_gaps_and_short_players(self):
        rows = [(0, t, 0.5 + 0.1 * (t % 2), float(t % 2)) for t in range(40)]
        rows = [(p, t, c, np.nan if t == 0 else w) for p, t, c, w in rows]
        # player 1 has too few trials, and player 0 loses the pair across the gap at trial 20
        rows += [(1, t, 0.5, np.nan if t == 0 else 1.0) for t in range(6)]
        onset = self.wsls_onset(rows)
        onset = onset[~((onset['player'] == 0) & (onset['trial_pos'] == 20))]
        res = hc.wsls_metrics(onset, min_trials=5)
        self.assertEqual(list(res['player']), [0])
        # 39 rows left, minus the first trial and the pair that would span the gap
        self.assertEqual(res.iloc[0]['n_win'] + res.iloc[0]['n_loss'], 37)

    def test_wsls_trial_end_control_recovers_coefficients(self):
        # onset = 0.2 + 0.3 x previous onset + 0.4 x previous trial end + 0.05 x previous win
        rng = np.random.default_rng(13)
        rows, prev, prev_won, prev_end = [], 0.5, np.nan, np.nan
        for t in range(600):
            rows.append((0, t, prev, prev_won, prev_end))
            won = float(rng.random() < 0.5)
            end = rng.uniform(0, 1)
            prev = 0.2 + 0.3 * prev + 0.4 * end + 0.05 * won + rng.normal(0, 0.02)
            prev_won, prev_end = won, end
        onset = pd.DataFrame([dict(experiment=0, player=p, session_type='social', trial_pos=t, centrality=c,
                                   prev_won=w, prev_end_centrality=e) for p, t, c, w, e in rows])
        res = hc.wsls_metrics(onset).iloc[0]
        self.assertAlmostEqual(res['coef_won_ctrl'], 0.05, delta=0.02)
        self.assertAlmostEqual(res['b_end'], 0.4, delta=0.03)

    def test_wsls_trial_end_control_removes_a_spurious_win_effect(self):
        # onset follows only the previous trial end, and winners end nearer the wall
        rng = np.random.default_rng(14)
        rows, prev, prev_won, prev_end = [], 0.5, np.nan, np.nan
        for t in range(600):
            rows.append((0, t, prev, prev_won, prev_end))
            won = float(rng.random() < 0.5)
            end = rng.normal(0.2 if won else 0.6, 0.1)
            prev = 0.1 + 0.6 * end + rng.normal(0, 0.05)
            prev_won, prev_end = won, end
        onset = pd.DataFrame([dict(experiment=0, player=p, session_type='social', trial_pos=t, centrality=c,
                                   prev_won=w, prev_end_centrality=e) for p, t, c, w, e in rows])
        res = hc.wsls_metrics(onset).iloc[0]
        # without the control the win looks like it moves the player toward the wall
        self.assertLess(res['coef_won'], -0.1)
        # with the trial end held fixed, nothing is left
        self.assertAlmostEqual(res['coef_won_ctrl'], 0.0, delta=0.02)

    def test_wsls_control_is_nan_without_trial_end(self):
        rows = [(0, t, 0.5 + 0.1 * (t % 3), np.nan if t == 0 else float(t % 2)) for t in range(40)]
        res = hc.wsls_metrics(self.wsls_onset(rows)).iloc[0]
        self.assertTrue(np.isnan(res['coef_won_ctrl']))
        self.assertFalse(np.isnan(res['coef_won']))

    def test_wsls_controlled_coefficient(self):
        # onset = 0.3 + 0.4 x previous onset + 0.1 x previous win
        rng = np.random.default_rng(12)
        rows, prev, prev_won = [], 0.5, np.nan
        for t in range(400):
            rows.append((0, t, prev, prev_won))
            won = float(rng.random() < 0.5)
            prev = 0.3 + 0.4 * prev + 0.1 * won + rng.normal(0, 0.02)
            prev_won = won
        res = hc.wsls_metrics(self.wsls_onset(rows)).iloc[0]
        self.assertAlmostEqual(res['coef_won'], 0.1, delta=0.02)
        self.assertAlmostEqual(res['b_prev'], 0.4, delta=0.05)

    def test_binomial_null_shape(self):
        players = pd.DataFrame(dict(session_type='social', n_trials=[10, 20, 30],
                                    frac_peripheral=[0.0, 0.5, 1.0]))
        null = hc.frac_peripheral_null(players, n_sim=100)
        self.assertEqual(null['social'].shape, (100, 3))
        # pooled rate is (0 + 10 + 30) / 60
        self.assertAlmostEqual(null['social'].mean(), 40 / 60, delta=0.05)

    def test_previous_trial_effect(self):
        # onset copies the previous trial end exactly, and is central after a win and peripheral after a loss
        rng = np.random.default_rng(8)
        prev_end = rng.uniform(0, 1, 40)
        prev_won = np.tile([1.0, 0.0], 20)
        onset = pd.DataFrame(dict(experiment=0, player=0, session_type='social', trial_pos=range(40),
                                  prev_end_centrality=prev_end, prev_won=prev_won,
                                  centrality=np.where(prev_won == 1, 0.9, 0.1)))
        res = hc.previous_trial_effect(onset).iloc[0]
        self.assertAlmostEqual(res['after_win'], 0.9)
        self.assertAlmostEqual(res['after_loss'], 0.1)
        self.assertEqual(res['n_after_win'], 20)

        onset['centrality'] = prev_end
        self.assertAlmostEqual(hc.previous_trial_effect(onset).iloc[0]['r_prev_end'], 1.0)

    def test_raster_orders_by_mean(self):
        onset = pd.DataFrame(dict(experiment=[0, 0, 1, 1], player=0, session_type='social',
                                  trial_pos=[0, 1, 0, 1], centrality=[0.9, 0.8, 0.1, 0.2]))
        mat, order = hc.raster(onset, 'social')
        self.assertEqual(mat.shape, (2, 2))
        self.assertEqual(order['experiment'].tolist(), [1, 0])


class TestSessionChange(unittest.TestCase):

    @staticmethod
    def pairs(n_experiments, partner_weight, seed, shared_sd=0.0):
        ''' Return a wide per-player table where social = 0.6 x own first solo + partner_weight x partner
            first solo + spread. Partners share a first-solo component with spread shared_sd.
            Takes the experiment count, the partner weight, a seed and the shared spread. '''
        rng = np.random.default_rng(seed)
        shared = rng.normal(0, shared_sd, (n_experiments, 1))
        solo1 = rng.uniform(0.2, 0.8, (n_experiments, 2)) + shared
        social = 0.1 + 0.6 * solo1 + partner_weight * solo1[:, ::-1] + rng.normal(0, 0.05, solo1.shape)
        return pd.DataFrame(dict(experiment=np.repeat(np.arange(n_experiments), 2), player=np.tile([0, 1], n_experiments),
                                 solo1=solo1.ravel(), social=social.ravel(),
                                 solo2=social.ravel())).set_index(['experiment', 'player'])

    def test_partner_effect_recovered(self):
        res = hc.partner_influence(self.pairs(100, partner_weight=0.3, seed=5), n_perm=300)
        self.assertAlmostEqual(res['coef'], 0.3, delta=0.05)
        self.assertAlmostEqual(res['own_coef'], 0.6, delta=0.05)
        self.assertTrue(res['ci_lo'] < 0.3 < res['ci_hi'])
        self.assertLess(res['p'], 0.01)
        self.assertLess(res['p_random'], 0.01)
        self.assertEqual(res['n'], 200)

    def test_no_effect_with_alike_partners(self):
        # partners share part of their first solo, but social depends only on the player's own first solo
        res = hc.partner_influence(self.pairs(100, partner_weight=0.0, seed=6, shared_sd=0.15), n_perm=300)
        self.assertAlmostEqual(res['coef'], 0.0, delta=0.05)
        self.assertGreater(res['p_random'], 0.05)

    def test_random_partner_null_centres_on_zero(self):
        res = hc.partner_influence(self.pairs(100, partner_weight=0.3, seed=7), n_perm=300)
        self.assertEqual(len(res['null']), 300)
        self.assertAlmostEqual(np.mean(res['null']), 0.0, delta=0.02)

    def test_residual_slope_equals_coefficient(self):
        res = hc.partner_influence(self.pairs(50, partner_weight=0.2, seed=8), n_perm=50)
        slope = np.polyfit(res['x_resid'], res['y_resid'], 1)[0]
        self.assertAlmostEqual(slope, res['coef'], places=10)
        self.assertEqual(len(res['x_resid']), res['n'])

    def test_players_without_partner_or_target_dropped(self):
        wide = self.pairs(20, partner_weight=0.2, seed=9)
        wide.loc[(0, 0), 'solo1'] = np.nan          # player (0, 0) has no baseline, so (0, 1) has no partner
        wide.loc[(1, 1), 'solo2'] = np.nan
        res = hc.partner_influence(wide, target='solo2', n_perm=20)
        self.assertEqual(res['n'], 37)

    def test_transition_table(self):
        players = pd.DataFrame(dict(experiment=[0, 0, 0, 0], player=[0, 0, 1, 1],
                                    session_type=['solo1', 'social', 'solo1', 'social'],
                                    lab=['central', 'peripheral', 'central', 'central']))
        tab = hc.transition_table(players, 'lab', 'solo1', 'social')
        self.assertEqual(tab.loc['central', 'peripheral'], 1)
        self.assertEqual(tab.loc['central', 'central'], 1)

    def test_persistence_model(self):
        wide = self.pairs(100, partner_weight=0.3, seed=7)
        fit = hc.persistence_model(wide)
        # solo2 equals social here, so social carries the prediction
        self.assertGreater(fit.params['social'], 0.9)

    def test_binned_timeline(self):
        onset = pd.DataFrame(dict(experiment=0, player=[0] * 20, session_type='social', trial_pos=range(20),
                                  centrality=[0.2] * 10 + [0.8] * 10))
        groups = pd.Series({(0, 0): 'central'})
        tl = hc.binned_timeline(onset, groups, bin_size=10)
        self.assertEqual(tl['mean'].round(3).tolist(), [0.2, 0.8])


class TestBlockShuffle(unittest.TestCase):

    def test_keeps_length_and_values(self):
        rng = np.random.default_rng(0)
        y = np.arange(37, dtype=float)
        out = hc._block_shuffle(y, 10, rng)
        self.assertEqual(len(out), len(y))
        self.assertEqual(sorted(out.tolist()), sorted(y.tolist()))

    def test_keeps_order_inside_blocks(self):
        rng = np.random.default_rng(1)
        y = np.arange(40, dtype=float)
        out = hc._block_shuffle(y, 10, rng)
        # every block of 10 survives as a run of consecutive values
        for start in range(0, 40, 10):
            block = list(range(start, start + 10))
            idx = out.tolist().index(float(start))
            self.assertEqual(out[idx:idx + 10].tolist(), [float(v) for v in block])

    def test_one_block_is_identity(self):
        rng = np.random.default_rng(2)
        y = np.arange(12, dtype=float)
        self.assertEqual(hc._block_shuffle(y, 12, rng).tolist(), y.tolist())


class TestBestStep(unittest.TestCase):

    def test_finds_known_step(self):
        y = np.r_[np.full(30, 0.2), np.full(70, 0.6)]
        step, at = hc._best_step(y)
        self.assertAlmostEqual(step, 0.4, places=6)
        self.assertAlmostEqual(at, 0.3, places=2)

    def test_sign_follows_direction(self):
        y = np.r_[np.full(50, 0.8), np.full(50, 0.3)]
        step, at = hc._best_step(y)
        self.assertLess(step, 0)


class TestWithinSessionDrift(unittest.TestCase):

    @staticmethod
    def onset_from(series):
        """ Return an onset table from a dict of (experiment, player) to a centrality array. Takes the dict. """
        rows = []
        for (e, p), y in series.items():
            for t, v in enumerate(y):
                rows.append(dict(experiment=e, player=p, session_type='social', trial_pos=t, centrality=v))
        return pd.DataFrame(rows)

    def test_drifting_player_beats_the_null(self):
        # a steady slide from 0.8 to 0.2 across the session
        y = np.linspace(0.8, 0.2, 120)
        drift = hc.within_session_drift(self.onset_from({(0, 0): y}), n_perm=200)
        row = drift.iloc[0]
        self.assertLess(row['p_half'], 0.02)
        self.assertLess(row['half_diff'], -0.3)
        self.assertLess(row['slope'], 0)

    def test_runs_without_a_trend_do_not_count_as_drift(self):
        # long runs of peripheral and central trials, alternating, with no net change.
        # a plain shuffle would call this drift; the block null should not
        y = np.tile(np.r_[np.full(10, 0.85), np.full(10, 0.15)], 6)
        drift = hc.within_session_drift(self.onset_from({(0, 0): y}), block=10, n_perm=200)
        self.assertGreater(drift.iloc[0]['p_half'], 0.05)
        self.assertGreater(drift.iloc[0]['p_step'], 0.05)

    def test_step_is_found_and_flagged(self):
        y = np.r_[np.full(60, 0.75), np.full(60, 0.25)]
        drift = hc.within_session_drift(self.onset_from({(0, 0): y}), block=10, n_perm=200)
        row = drift.iloc[0]
        self.assertAlmostEqual(row['step'], -0.5, places=6)
        self.assertAlmostEqual(row['step_at'], 0.5, places=2)
        self.assertLess(row['p_step'], 0.02)

    def test_short_players_dropped_and_trials_capped(self):
        series = {(0, 0): np.linspace(0.8, 0.2, 120), (1, 0): np.full(10, 0.5)}
        drift = hc.within_session_drift(self.onset_from(series), n_perm=50, min_trials=20)
        self.assertEqual(len(drift), 1)
        capped = hc.within_session_drift(self.onset_from(series), n_perm=50, max_trials=40)
        self.assertEqual(capped.iloc[0]['n_trials'], 40)

    def test_summary_counts_flags_against_chance(self):
        drift = pd.DataFrame(dict(session_type='social', half_diff=[0.2, -0.05, 0.0, -0.3],
                                  slope=[0.1, 0.0, 0.0, -0.2], p_half=[0.01, 0.5, 0.04, 0.2],
                                  p_step=[0.2, 0.01, 0.9, 0.9]))
        row = hc.drift_summary(drift).loc['social']
        self.assertEqual(row['players'], 4)
        self.assertEqual(row['flagged_half'], 2)
        self.assertEqual(row['flagged_step'], 1)
        self.assertAlmostEqual(row['expected_by_chance'], 0.2)
        self.assertAlmostEqual(row['share_big'], 0.5)


class TestGroupDrift(unittest.TestCase):

    @staticmethod
    def onset_from(players, session_type='social'):
        """ Return an onset table from a list of centrality arrays, one player per experiment.
            Takes the arrays. """
        rows = []
        for e, y in enumerate(players):
            for t, v in enumerate(y):
                rows.append(dict(experiment=e, player=0, session_type=session_type, trial_pos=t, centrality=v))
        return pd.DataFrame(rows)

    def test_shared_slide_detected_and_sized(self):
        # every player slides from 0.8 to 0.4, so the halves differ by about -0.2
        rng = np.random.default_rng(8)
        players = [np.linspace(0.8, 0.4, 100) + rng.normal(0, 0.05, 100) for _ in range(30)]
        row = hc.group_drift(self.onset_from(players), n_perm=200).loc['social']
        self.assertEqual(row['players'], 30)
        self.assertLess(row['p_group'], 0.01)
        self.assertAlmostEqual(row['mean_half'], -0.2, delta=0.02)
        self.assertAlmostEqual(row['drift_rms'], 0.2, delta=0.03)
        self.assertGreater(row['mean_abs_half'], row['mean_abs_half_reordered'])

    def test_runs_without_a_trend_give_no_drift(self):
        # each player holds a random level for whole 10-trial blocks, so reordering blocks changes nothing
        rng = np.random.default_rng(9)
        players = [np.repeat(rng.uniform(0.1, 0.9, 10), 10) for _ in range(30)]
        row = hc.group_drift(self.onset_from(players), block=10, n_perm=200).loc['social']
        self.assertGreater(row['p_group'], 0.05)
        self.assertLess(row['drift_rms'], 0.1)
        self.assertGreater(row['rms_reordered'], 0.1)

    def test_short_players_dropped_and_trials_capped(self):
        players = [np.linspace(0.8, 0.2, 120), np.full(10, 0.5), np.linspace(0.2, 0.8, 120)]
        res = hc.group_drift(self.onset_from(players), n_perm=50, min_trials=20)
        self.assertEqual(res.loc['social', 'players'], 2)
        # capping at 40 trials keeps only the start of each slide
        capped = hc.group_drift(self.onset_from(players), n_perm=50, max_trials=40)
        self.assertAlmostEqual(capped.loc['social', 'mean_abs_half'], 0.6 * 20 / 119, places=6)


class TestPartnerGap(unittest.TestCase):

    @staticmethod
    def pair_onset(n_experiments, shrink, seed, n_trials=100, n_bins=5):
        """ Return an onset table whose partners start apart and close by `shrink` over the session.
            Takes the experiment count, how much the gap closes, a seed and the session length. """
        rng = np.random.default_rng(seed)
        rows = []
        for e in range(n_experiments):
            a, b = rng.uniform(0.1, 0.9, 2)
            for t in range(n_trials):
                frac = t / (n_trials - 1)
                mid = (a + b) / 2
                for p, own in ((0, a), (1, b)):
                    # walk each player toward the pair's midpoint as the session runs
                    val = own + shrink * frac * (mid - own) + rng.normal(0, 0.02)
                    rows.append(dict(experiment=e, player=p, session_type='social', trial_pos=t,
                                     centrality=val))
        return pd.DataFrame(rows)

    def test_closing_pairs_detected(self):
        res = hc.partner_gap_over_session(self.pair_onset(40, shrink=0.8, seed=3), n_perm=300)
        self.assertLess(res['slope'], 0)
        self.assertLess(res['p'], 0.01)
        # real partners end closer than randomly paired players
        self.assertLess(res['gap'][-1], res['null_gap'][-1])

    def test_stable_pairs_not_flagged(self):
        res = hc.partner_gap_over_session(self.pair_onset(40, shrink=0.0, seed=4), n_perm=300)
        self.assertGreater(res['p'], 0.05)

    def test_bins_and_shapes(self):
        res = hc.partner_gap_over_session(self.pair_onset(10, shrink=0.5, seed=5, n_bins=5), n_bins=5, n_perm=50)
        self.assertEqual(len(res['gap']), 5)
        self.assertEqual(len(res['null_gap']), 5)
        self.assertEqual(len(res['null_slopes']), 50)


def solo_onset(series):
    """ Return an onset table from a dict of (experiment, player, session type) to centrality values.
        Takes the dict. """
    rows = []
    for (e, p, st), y in series.items():
        for t, v in enumerate(y):
            rows.append(dict(experiment=e, player=p, session_type=st, trial_pos=t, centrality=v))
    return pd.DataFrame(rows)


def wide_players(solo1, social, solo2):
    """ Return a player table with two players per experiment and one row per session.
        Takes arrays of each player's mean in the three sessions. """
    rows = []
    for i, vals in enumerate(zip(solo1, social, solo2)):
        for st, v in zip(('solo1', 'social', 'solo2'), vals):
            rows.append(dict(experiment=i // 2, player=i % 2, session_type=st, mean_centrality=v,
                             fixed='central' if v > 0.5 else 'peripheral'))
    return pd.DataFrame(rows)


class TestPlayerChange(unittest.TestCase):

    def test_change_and_interval(self):
        rng = np.random.default_rng(0)
        series = {(0, 0, 'solo1'): rng.normal(0.8, 0.05, 60), (0, 0, 'solo2'): rng.normal(0.3, 0.05, 60)}
        row = hc.player_change(solo_onset(series), n_boot=500).iloc[0]
        self.assertAlmostEqual(row['change'], -0.5, delta=0.05)
        self.assertTrue(row['ci_lo'] < row['change'] < row['ci_hi'])
        # no mover flag or p value, the interval is the output
        self.assertNotIn('mover', row.index)
        self.assertNotIn('p', row.index)

    def test_defaults_to_first_and_second_solo(self):
        series = {(0, 0, 'solo1'): np.full(30, 0.5), (0, 0, 'social'): np.full(30, 0.9),
                  (0, 0, 'solo2'): np.full(30, 0.2)}
        self.assertAlmostEqual(hc.player_change(solo_onset(series), n_boot=50).iloc[0]['change'], -0.3)

    def test_runs_widen_the_interval(self):
        # same values either way, but one player's trials come in long runs
        block = np.tile(np.r_[np.full(10, 0.9), np.full(10, 0.1)], 6)
        mixed = block.copy()
        np.random.default_rng(2).shuffle(mixed)
        series = {(0, 0, 'solo1'): block, (0, 0, 'solo2'): block,
                  (1, 0, 'solo1'): mixed, (1, 0, 'solo2'): mixed}
        out = hc.player_change(solo_onset(series), n_boot=500).set_index('experiment')
        width = out['ci_hi'] - out['ci_lo']
        self.assertGreater(width[0], width[1])

    def test_short_sessions_dropped(self):
        series = {(0, 0, 'solo1'): np.full(5, 0.5), (0, 0, 'solo2'): np.full(60, 0.2)}
        self.assertEqual(len(hc.player_change(solo_onset(series), n_boot=50, min_trials=20)), 0)


class TestStabilityCeiling(unittest.TestCase):

    def test_corrected_r_near_one_when_players_keep_their_level(self):
        # every player holds one level in both solos, with small trial-to-trial variation
        rng = np.random.default_rng(5)
        series = {}
        for e, level in enumerate(np.linspace(0.1, 0.9, 30)):
            for st in ('solo1', 'solo2'):
                series[(e, 0, st)] = level + rng.normal(0, 0.02, 40)
        onset = solo_onset(series)
        players = hc.player_table(onset)
        res = hc.stability_ceiling(onset, players)
        self.assertGreater(res['between_sessions_r'], 0.95)
        self.assertGreater(res['corrected_r'], 0.95)
        self.assertLessEqual(res['ceiling'], 1.0)

    def test_corrected_r_low_when_players_reshuffle(self):
        # each player is consistent within a session, but their level is redrawn between sessions
        rng = np.random.default_rng(6)
        series = {}
        for e in range(40):
            for st in ('solo1', 'solo2'):
                series[(e, 0, st)] = rng.uniform(0.1, 0.9) + rng.normal(0, 0.02, 40)
        onset = solo_onset(series)
        res = hc.stability_ceiling(onset, hc.player_table(onset))
        self.assertGreater(res['ceiling'], 0.9)
        self.assertLess(abs(res['corrected_r']), 0.5)


class TestPairedChange(unittest.TestCase):

    def test_mean_and_groups(self):
        players = wide_players(solo1=[0.8, 0.8, 0.2, 0.2], social=[0.5] * 4, solo2=[0.7, 0.7, 0.3, 0.3])
        res = hc.paired_change(players, group_col='fixed')
        self.assertAlmostEqual(res.loc['all', 'mean_change'], 0.0)
        self.assertAlmostEqual(res.loc['central', 'mean_change'], -0.1)
        self.assertAlmostEqual(res.loc['peripheral', 'mean_change'], 0.1)
        self.assertEqual(res.loc['central', 'n'], 2)
        # each group here comes from one experiment, so it has no clustered standard error
        self.assertTrue(np.isnan(res.loc['central', 'se']))
        self.assertFalse(np.isnan(res.loc['all', 'se']))

    def test_clustered_se_widens_for_linked_partners(self):
        # partners move together, so the effective sample is the experiments, not the players
        rng = np.random.default_rng(7)
        shared = np.repeat(rng.normal(0, 0.1, 30), 2)
        solo1 = rng.uniform(0.2, 0.8, 60)
        players = wide_players(solo1=solo1, social=solo1, solo2=solo1 + shared)
        clustered = hc.paired_change(players).loc['all', 'se']
        naive = (players.query("session_type == 'solo2'")['mean_centrality'].to_numpy()
                 - solo1).std(ddof=1) / np.sqrt(60)
        self.assertGreater(clustered, naive)


class TestShrinkageModel(unittest.TestCase):

    def test_slope_recovered(self):
        rng = np.random.default_rng(8)
        solo1 = rng.uniform(0.1, 0.9, 200)
        solo2 = 0.5 + 0.6 * (solo1 - 0.5) + rng.normal(0, 0.02, 200)
        wide = hc.wide_table(wide_players(solo1, solo1, solo2))
        fit = hc.shrinkage_model(wide)
        self.assertAlmostEqual(fit.params['solo1'], 0.6, delta=0.03)


class TestSocialOrPracticeModel(unittest.TestCase):

    def test_both_predictors_recovered(self):
        rng = np.random.default_rng(9)
        n = 200
        solo1 = rng.uniform(0.1, 0.9, n)
        social = rng.uniform(0.1, 0.9, n)
        drift_slope = rng.normal(0, 0.2, n)
        solo2 = 0.2 * solo1 + 0.7 * social + 0.1 * drift_slope + rng.normal(0, 0.01, n)
        players = wide_players(solo1, social, solo2)
        drift = pd.DataFrame(dict(experiment=np.arange(n) // 2, player=np.arange(n) % 2,
                                  session_type='solo1', slope=drift_slope))
        fit = hc.social_or_practice_model(hc.wide_table(players), drift)
        self.assertAlmostEqual(fit.params['social'], 0.7, delta=0.03)
        self.assertAlmostEqual(fit.params['solo1_drift'], 0.1, delta=0.03)


class TestLargestChanges(unittest.TestCase):

    def test_sorted_with_every_player_eligible(self):
        players = wide_players(solo1=[0.8, 0.8, 0.8, 0.8], social=[0.5] * 4, solo2=[0.7, 0.4, 0.75, 0.1])
        change = pd.DataFrame(dict(experiment=[0, 0, 1, 1], player=[0, 1, 0, 1],
                                   change=[-0.1, -0.4, -0.05, -0.7], ci_lo=-1.0, ci_hi=1.0))
        out = hc.largest_changes(change, players, 'fixed', n=4)
        self.assertEqual(len(out), 4)
        self.assertEqual(out['change'].tolist(), [-0.7, -0.4, -0.1, -0.05])
        # experiment 1 player 1's partner is player 0, whose second solo was 0.75
        self.assertAlmostEqual(out.iloc[0]['partner_solo2'], 0.75)
        self.assertIn('solo1_label', out.columns)


class TestTransitionDetail(unittest.TestCase):

    @staticmethod
    def players_from(rows):
        """ Return a player table from (experiment, session type, label, mean, fraction) rows. Takes the rows. """
        return pd.DataFrame([dict(experiment=e, player=0, session_type=st, fixed=lab,
                                  mean_centrality=m, frac_peripheral=f) for e, st, lab, m, f in rows])

    def test_counts_and_median_change(self):
        players = self.players_from([
            (0, 'solo1', 'central', 0.8, 0.0), (0, 'social', 'mixed', 0.5, 0.5),
            (1, 'solo1', 'central', 0.9, 0.1), (1, 'social', 'mixed', 0.4, 0.6),
            (2, 'solo1', 'central', 0.8, 0.0), (2, 'social', 'central', 0.75, 0.05)])
        tab = hc.transition_detail(players, 'fixed', 'solo1', 'social')
        self.assertEqual(tab.loc[('central', 'mixed'), 'n'], 2)
        self.assertAlmostEqual(tab.loc[('central', 'mixed'), 'median_change'], -0.4)
        self.assertAlmostEqual(tab.loc[('central', 'central'), 'median_abs_change'], 0.05)

    def test_marginal_transitions_flagged(self):
        # both fractions sit right by the 0.2 cut, so the label change rests on a hair's breadth
        players = self.players_from([
            (0, 'solo1', 'central', 0.6, 0.19), (0, 'social', 'mixed', 0.58, 0.22),
            (1, 'solo1', 'central', 0.9, 0.0), (1, 'social', 'mixed', 0.4, 0.6)])
        tab = hc.transition_detail(players, 'fixed', 'solo1', 'social')
        self.assertAlmostEqual(tab.loc[('central', 'mixed'), 'share_marginal'], 0.5)


def make_full_trial(trial_num, winner=0, high=True, scores=(0.0, 0.0), rt=2.0, n_players=2,
                    start_row=0, onset_r=(1.0, 2.0), step=0.1, n_iti=5, trial_type=globals.HIGH_LOW):
    ''' Return a synthetic trial with trial-started, trial and ITI frames, for the fingerprint collector.
        Takes a trial number, the winner's log id, whether they reached High, each player's cumulative
        score at trial end, the trial time (s), and the player count.

        Walls are 1 (High) and 2 (Low). Players walk along +x by `step` per frame from slice onset to
        the trigger (3 frames), then stand still through n_iti ITI frames. '''

    t0 = pd.Timestamp('1900-01-01 12:00:00') + pd.Timedelta(seconds=100 * trial_num)
    # (event, epoch, frame offset along the path, time since slice onset in s)
    frames = [(globals.TRIAL_START, globals.TRIAL_STARTED, 0, -1.0), (np.nan, globals.TRIAL_STARTED, 0, -0.5),
              (globals.SLICE_ONSET, 'slices active', 0, 0.0), (np.nan, 'slices active', 1, rt / 2),
              (np.nan, 'slices active', 2, rt / 2),
              (globals.SELECTED_TRIGGER_ACTIVATION, 'post choice', 3, rt),
              (globals.TRIAL_END, globals.ITI, 3, rt + 0.1)]
    frames += [(np.nan, globals.ITI, 3, rt + 0.2 + i) for i in range(n_iti)]
    rows = []
    for ev, epoch, k, t in frames:
        is_trig = ev == globals.SELECTED_TRIGGER_ACTIVATION
        row = {'timeLocal': t0 + pd.Timedelta(seconds=t), 'timeApplication': t0 + pd.Timedelta(seconds=t),
               'eventDescription': ev, 'trial_epoch': epoch, globals.TRIAL_NUM: float(trial_num),
               globals.TRIAL_TYPE: trial_type, globals.WALL_1: 1, globals.WALL_2: 2,
               globals.TRIGGER_CLIENT: winner if is_trig else np.nan,
               globals.WALL_TRIGGERED: (1 if high else 2) if is_trig else np.nan}
        for p in range(n_players):
            row[globals.PLAYER_LOC_DICT[p]['xloc']] = onset_r[p] + step * k
            row[globals.PLAYER_LOC_DICT[p]['yloc']] = 0.0
            row[globals.PLAYER_ROT_DICT[p]['yrot']] = 0.0
            row[globals.PLAYER_SCORE_DICT[p]['score']] = scores[p] if ev == globals.TRIAL_END else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=range(start_row, start_row + len(rows)))


def stub_vis(trial, player_id, fov=None):
    ''' Return a fixed visibility row, as synthetic trials carry no head angle. Takes a trial and player id. '''

    return dict(player=player_id, latency_to_wall_visible=0.5, visible_at_onset=False,
                ever_visible=True, trial_s=1.0)


class TestFingerprintCollection(unittest.TestCase):

    def social_session(self):
        # trial 1: player 0 wins High in 2 s. trial 2: player 1 wins Low in 4 s. trial 3: player 0 wins Low in 6 s
        spec = [(0, True, (1.0, 0.0), 2.0), (1, False, (1.0, 0.4), 4.0), (0, False, (1.4, 0.4), 6.0)]
        return [make_full_trial(i + 1, winner=w, high=h, scores=s, rt=rt, start_row=100 * i)
                for i, (w, h, s, rt) in enumerate(spec)]

    def collect(self, trial_lists, kind, cut=0):
        # positions as logged, so route and ITI values are exact. with_iti measures ITIs on frames, which
        # the pipeline no longer does by default, but the frame path is still used for agent sessions
        return hc.collect_fingerprint_tables(trial_lists, kind, first_solo_cut=cut, vis_fn=stub_vis,
                                             network_ids=(), with_iti=True, verbose=False)

    def test_social_outcome_scores_each_player_from_the_result(self):
        out = self.collect([self.social_session()], 'social')['outcome'].set_index(['player', 'trial_num'])
        self.assertEqual(list(out.loc[0, 'score']), [1.0, 0.0, 0.4])
        self.assertEqual(list(out.loc[1, 'score']), [0.0, 0.4, 0.0])
        self.assertEqual(list(out.loc[0, 'won']), [1.0, 0.0, 1.0])
        # the logged cumulative score gives the same per-trial values
        np.testing.assert_allclose(out.loc[0, 'score_logged'], [1.0, 0.0, 0.4])
        np.testing.assert_allclose(out.loc[1, 'score_logged'], [0.0, 0.4, 0.0])
        np.testing.assert_allclose(out.loc[0, 'rt_s'], [2.0, 4.0, 6.0])

    def test_forced_trials_score_from_the_trial_type(self):
        # both walls of a forced trial pay the same, so the wall reached does not set the score
        trials = [make_full_trial(1, winner=0, high=False, scores=(1.0, 0.0), trial_type='ForcedHigh'),
                  make_full_trial(2, winner=1, high=True, scores=(1.0, 0.4), trial_type='ForcedLow', start_row=100)]
        out = self.collect([trials], 'social')['outcome'].set_index(['player', 'trial_num'])
        self.assertEqual(out.loc[(0, 1.0), 'score'], 1.0)
        self.assertEqual(out.loc[(1, 2.0), 'score'], 0.4)
        # reaching High means nothing on a forced trial
        self.assertTrue(out['chose_high'].isna().all())

    def test_win_rows_go_to_the_winner(self):
        win = self.collect([self.social_session()], 'social')['win']
        self.assertEqual(sorted(win.loc[win['player'] == 0, 'trial_num']), [1.0, 3.0])
        self.assertEqual(list(win.loc[win['player'] == 1, 'trial_num']), [2.0])
        # the route stops the frame before the trigger, so 2 steps of 0.1 along x
        np.testing.assert_allclose(win['winner_path_len'], 0.2)
        np.testing.assert_allclose(win['path_ratio'], 1.0)

    def test_iti_skips_the_first_and_the_trailing_run(self):
        # the run into trial 1 follows no trial end, and the run after trial 3 leads into an unsaved trial
        iti = self.collect([self.social_session()], 'social')['iti']
        self.assertEqual(sorted(iti.loc[iti['player'] == 0, 'trial_num']), [2.0, 3.0])
        self.assertEqual(len(iti), 4)
        # players stand at the wall through the ITI, then return 0.3 to the next start position
        np.testing.assert_allclose(iti['iti_path_len'], 0.3)

    def test_first_solo_cut_drops_opening_trials(self):
        # a solo list holds 4 sessions per experiment. Only the first solo loses its opening trials
        solo = [make_full_trial(i + 1, winner=0, scores=(i + 1.0, 0.0), n_players=1, start_row=100 * i)
                for i in range(4)]
        tabs = self.collect([solo, solo, solo, solo], 'solo', cut=2)
        out = tabs['outcome']
        first = out[(out['player'] == 0) & (out['session_type'] == 'solo1')]
        self.assertEqual(list(first['trial_num']), [3.0, 4.0])
        # the logged score still differences against the cut trial before
        np.testing.assert_allclose(first['score_logged'], [1.0, 1.0])
        second = out[(out['player'] == 1) & (out['session_type'] == 'solo2')]
        self.assertEqual(list(second['trial_num']), [1.0, 2.0, 3.0, 4.0])
        # a solo player wins every trial, and takes their social player id
        self.assertTrue((out['won'] == 1).all())
        self.assertEqual(sorted(tabs['win']['player'].unique()), [0, 1])
        # the ITI into the first kept trial of the first solo is kept, as a cut trial precedes it
        iti = tabs['iti']
        self.assertEqual(list(iti.loc[(iti['player'] == 0) & (iti['session_type'] == 'solo1'), 'trial_num']), [3.0, 4.0])
        self.assertEqual(list(iti.loc[(iti['player'] == 0) & (iti['session_type'] == 'solo2'), 'trial_num']), [2.0, 3.0, 4.0])

    def test_vis_rows_cover_kept_trials_for_both_players(self):
        vis = self.collect([self.social_session()], 'social')['vis']
        self.assertEqual(len(vis), 6)
        self.assertEqual(set(vis.columns) >= {'experiment', 'player', 'session_type', 'trial_num'}, True)


def reference_network_view(t, v, tick=1 / 30, lag_ticks=1, smooth_s=0.1):
    ''' Return a trace passed through the client-transform path with a plain frame loop, as a reference
        for the vectorised version. Takes frame times and values. '''

    ticks = np.arange(t[0], t[-1] + tick, tick)
    idx = np.clip(np.searchsorted(t, ticks, side='right') - 1, 0, len(t) - 1)
    target = np.interp(t - lag_ticks * tick, ticks, v[idx])
    out = np.empty_like(v)
    cur, prev = target[0], t[0]
    for k in range(len(t)):
        cur = cur + (target[k] - cur) * min((t[k] - prev) / smooth_s, 1.0)
        prev = t[k]
        out[k] = cur
    return out


class TestNetworkView(unittest.TestCase):

    def test_matches_frame_loop(self):
        # irregular frame times, with repeated timestamps and one long gap
        rng = np.random.default_rng(1)
        dt = rng.uniform(0.015, 0.03, 3000)
        dt[rng.choice(3000, 50)] = 0.0
        dt[1500] = 0.4
        t = np.cumsum(dt)
        v = np.cumsum(rng.normal(0, 1, 3000))
        np.testing.assert_allclose(hc.simulate_network_view(t, v), reference_network_view(t, v), atol=1e-8)

    def test_constant_stays_constant(self):
        t = np.arange(500) * 0.0207
        np.testing.assert_allclose(hc.simulate_network_view(t, np.full(500, 3.5)), 3.5)

    def test_step_is_delayed_and_smoothed(self):
        # a step at 1 s is interpolated in over the next tick, then eases in without overshoot
        t = np.arange(200) * 0.02
        v = np.where(t >= 1.0, 10.0, 0.0)
        out = hc.simulate_network_view(t, v)
        self.assertTrue(np.all(out[t <= 1.0] == 0))
        # the smoothing keeps it well short of the step after one tick
        self.assertLess(out[np.searchsorted(t, 1.04)], 5.0)
        self.assertTrue(np.all(np.diff(out) >= 0))
        self.assertTrue(np.all(out <= 10.0))
        self.assertAlmostEqual(out[-1], 10.0, places=3)
        # the stop is spread over many frames, so few steps are exactly zero after the step
        after = np.diff(out[(t > 1.0) & (t < 1.4)])
        self.assertLess(np.mean(after == 0), 0.1)

    def test_nan_frames_kept(self):
        t = np.arange(100) * 0.02
        v = np.linspace(0, 1, 100)
        v[[10, 50]] = np.nan
        out = hc.simulate_network_view(t, v)
        self.assertTrue(np.isnan(out[[10, 50]]).all())
        self.assertEqual(np.isnan(out).sum(), 2)

    def test_yaw_wraps_the_short_way(self):
        # turning from 350 to 10 degrees passes through 0, not back round through 180
        t = np.arange(100) * 0.02
        yaw = np.where(t < 0.5, 350.0, 10.0)
        out = hc.simulate_network_view(t, yaw, angle=True)
        self.assertTrue(np.all((out >= 350 - 1e-9) | (out <= 10 + 1e-9)))
        self.assertAlmostEqual(out[-1], 10.0, places=3)

    def test_network_view_changes_the_collected_route(self):
        # the simulation is no longer the default, but asking for it still changes what is measured
        trials = [make_full_trial(i + 1, start_row=100 * i, step=0.5) for i in range(3)]
        kw = dict(first_solo_cut=0, vis_fn=stub_vis, verbose=False)
        viewed = hc.collect_fingerprint_tables([trials], 'social', network_ids=(0,), **kw)['win']
        logged = hc.collect_fingerprint_tables([trials], 'social', **kw)['win']
        self.assertFalse(np.allclose(viewed['winner_path_len'], logged['winner_path_len']))

    def test_recovered_frames_replace_player_1(self):
        # a recovered trace for player 1 feeds the frame-level tables
        trials = [make_full_trial(i + 1, winner=1, start_row=100 * i, step=0.5) for i in range(3)]
        df = pd.concat(trials)
        t = (df['timeLocal'] - df['timeLocal'].iloc[0]).dt.total_seconds().to_numpy()
        ticks = pd.DataFrame(dict(t=np.arange(t[0], t[-1], hc.nr.TICK_S)))
        for c in ('x', 'z', 'yaw'):
            ticks[c] = 3.0 if c == 'x' else 0.0
        kw = dict(first_solo_cut=0, vis_fn=stub_vis, verbose=False)
        out = hc.collect_fingerprint_tables([trials], 'social', recovered={0: ticks}, **kw)['win']
        # player 1 now stands still at x = 3, so their winning route has no length
        np.testing.assert_allclose(out['winner_path_len'], 0.0, atol=1e-6)

    def test_session_view_changes_only_local_player(self):
        trials = [make_full_trial(i + 1, start_row=100 * i, step=0.5) for i in range(3)]
        out = hc.network_view(trials, log_ids=(0,))
        x0, x1 = globals.PLAYER_LOC_DICT[0]['xloc'], globals.PLAYER_LOC_DICT[1]['xloc']
        self.assertFalse(np.allclose(pd.concat(out)[x0], pd.concat(trials)[x0]))
        pd.testing.assert_series_equal(pd.concat(out)[x1], pd.concat(trials)[x1])
        # trial boundaries and indices are unchanged
        self.assertEqual([list(a.index) for a in out], [list(b.index) for b in trials])


def tick_trace(n_iti=2, per=10, tick=1 / 30, step=0.0, rot=0.0, gap_at=None):
    ''' Return a tick trace with n_iti ITI runs separated by trial frames. Takes the run count, the
        ticks per run, and the per-tick movement and rotation inside a run. '''

    rows = []
    t = 0.0
    for k in range(n_iti):
        # a short block of trial ticks, then the ITI leading into trial k + 2
        for _ in range(3):
            rows.append(dict(t=t, x=0.0, z=0.0, yaw=0.0, trial_num=float(k + 1), is_iti=False))
            t += tick
        for i in range(per):
            rows.append(dict(t=t, x=step * i, z=0.0, yaw=(rot * i) % 360, trial_num=float(k + 1), is_iti=True))
            t += tick
        # the trial-started ticks that close the run carry the next trial's number
        for i in range(2):
            rows.append(dict(t=t, x=step * (per + i), z=0.0, yaw=(rot * (per + i)) % 360,
                             trial_num=float(k + 2), is_iti=True))
            t += tick
    df = pd.DataFrame(rows)
    if gap_at is not None:
        df.loc[gap_at:, 't'] += 5 * tick        # a block join, which must not read as one huge step
    df['experiment'] = 0
    df['player'] = 0
    df['session_type'] = 'social'
    return df


class TestItiMetricsTicks(unittest.TestCase):

    def test_runs_are_labelled_by_the_trial_they_lead_into(self):
        out = hc.iti_metrics_ticks(tick_trace())
        self.assertEqual(sorted(out['trial_num']), [2.0, 3.0])
        self.assertEqual(len(out), 2)

    def test_still_player_counts_as_resting(self):
        out = hc.iti_metrics_ticks(tick_trace())
        self.assertTrue((out['proportion_resting'] == 1).all())
        self.assertTrue((out['iti_path_len'] == 0).all())
        self.assertTrue((out['rot_total'] == 0).all())

    def test_moving_player_is_not_resting(self):
        # a step and a turn well above the cutoffs
        out = hc.iti_metrics_ticks(tick_trace(step=0.5, rot=5.0))
        self.assertTrue((out['proportion_resting'] == 0).all())
        self.assertTrue((out['prop_no_translational'] == 0).all())
        self.assertAlmostEqual(out['rot_total'].iloc[0], 5.0 * 11, places=4)

    def test_movement_just_under_the_cutoff_counts_as_still(self):
        out = hc.iti_metrics_ticks(tick_trace(step=0.9 * nr.STILL_SPEED_PER_TICK,
                                              rot=0.9 * nr.STILL_ROT_PER_TICK))
        self.assertTrue((out['proportion_resting'] == 1).all())
        # but the distance still accumulates
        self.assertGreater(out['iti_path_len'].iloc[0], 0)

    def test_block_join_is_not_counted_as_a_step(self):
        # a join splits one ITI in two and drops the step across it, rather than reading a five-tick jump
        whole = hc.iti_metrics_ticks(tick_trace(per=20, step=0.5))
        split = hc.iti_metrics_ticks(tick_trace(per=20, step=0.5, gap_at=10))
        # a join does not split the ITI in two, as an ITI is named by the trial it leads into
        self.assertEqual(len(whole), 2)
        self.assertEqual(len(split), 2)
        # the step across the join is dropped, so exactly one step of distance goes missing
        self.assertAlmostEqual(split['iti_path_len'].iloc[0], whole['iti_path_len'].iloc[0] - 0.5, places=6)
        self.assertEqual(split['n_pairs'].iloc[0], whole['n_pairs'].iloc[0] - 1)

    def test_trailing_wait_is_not_an_iti(self):
        # the ITI ticks after the last trial lead into no trial, so they must not be counted again
        trace = tick_trace(n_iti=2)
        extra = pd.concat([trace.iloc[-1:]] * 5, ignore_index=True)
        extra['t'] = trace['t'].max() + np.arange(1, 6) * (1 / 30)
        extra['trial_num'] = 3.0
        extra['is_iti'] = True
        out = hc.iti_metrics_ticks(pd.concat([trace, extra], ignore_index=True))
        self.assertEqual(sorted(out['trial_num']), [2.0, 3.0])

    def test_keep_map_filters_runs(self):
        trace = tick_trace()
        keep = {(0, 0, 'social'): {3.0}}
        out = hc.iti_metrics_ticks(trace, keep_map=keep)
        self.assertEqual(out['trial_num'].tolist(), [3.0])


class TestExtremeGroups(unittest.TestCase):

    def test_top_and_bottom_share_per_session(self):
        players = pd.DataFrame(dict(experiment=np.repeat(range(10), 2), player=[0, 1] * 10,
                                    session_type='social', mean_centrality=np.linspace(0, 1, 20)))
        players = pd.concat([players, players.assign(session_type='solo1', mean_centrality=players['mean_centrality'][::-1].to_numpy())])
        g = hc.extreme_groups(players, frac=0.2)
        soc = g[g['session_type'] == 'social'].sort_values('mean_centrality')
        self.assertEqual(list(soc['group'][:4]), ['peripheral'] * 4)
        self.assertEqual(list(soc['group'][-4:]), ['central'] * 4)
        self.assertEqual((soc['group'] == 'middle').sum(), 12)
        # each session is ranked on its own values
        solo = g[g['session_type'] == 'solo1']
        self.assertEqual(solo.loc[(solo['experiment'] == 0) & (solo['player'] == 0), 'group'].item(), 'central')


class TestPlayerFingerprint(unittest.TestCase):

    def test_outcome_metrics_use_own_wins(self):
        keys = dict(experiment=0, session_type='social')
        outcome = pd.DataFrame([
            dict(**keys, player=0, trial_num=1.0, trial_type=globals.HIGH_LOW, won=1.0, chose_high=True, score=1.0, rt_s=2.0),
            dict(**keys, player=0, trial_num=2.0, trial_type=globals.HIGH_LOW, won=0.0, chose_high=np.nan, score=0.0, rt_s=10.0),
            dict(**keys, player=0, trial_num=3.0, trial_type=globals.HIGH_LOW, won=1.0, chose_high=False, score=0.4, rt_s=6.0),
            dict(**keys, player=0, trial_num=4.0, trial_type='Forced', won=1.0, chose_high=False, score=0.4, rt_s=4.0)])
        vis = pd.DataFrame([dict(**keys, player=0, trial_num=float(t), latency_to_wall_visible=lat,
                                 visible_at_onset=on, ever_visible=ev, trial_s=1.0)
                            for t, lat, on, ev in [(1, 0.0, True, True), (2, 0.4, False, True),
                                                   (3, np.nan, False, False), (4, 0.2, False, True)]])
        iti = pd.DataFrame([dict(**keys, player=0, trial_num=2.0, prop_no_translational=0.5, prop_no_rotational=0.5,
                                 proportion_resting=0.2, iti_path_len=3.0, rot_total=90.0, iti_centrality=0.6)])
        win = pd.DataFrame([dict(**keys, player=0, trial_num=1.0, winner_path_len=4.0, winner_direct_len=2.0, path_ratio=2.0)])
        players = pd.DataFrame([dict(**keys, player=0, mean_centrality=0.7, n_trials=4, group='central')])

        fp = hc.build_player_fingerprint(players, dict(outcome=outcome, vis=vis, iti=iti, win=win)).iloc[0]
        # latency and slow-trial share come from the three won trials only
        self.assertAlmostEqual(fp['latency_to_reach_wall'], 4.0)
        self.assertAlmostEqual(fp['frac_trials_over_5s'], 1 / 3)
        # P(High) from won High/Low trials only
        self.assertAlmostEqual(fp['p_high'], 0.5)
        self.assertAlmostEqual(fp['win_rate'], 0.75)
        # score over every trial, and total score over total trial time
        self.assertAlmostEqual(fp['mean_score'], 0.45)
        self.assertAlmostEqual(fp['score_rate'], 1.8 / 22.0)
        self.assertAlmostEqual(fp['onset_centrality'], 0.7)
        self.assertAlmostEqual(fp['frac_visible_at_onset'], 0.25)
        self.assertAlmostEqual(fp['frac_never_visible'], 0.25)
        self.assertAlmostEqual(fp['late_latency_to_wall_visible'], 0.3)
        self.assertAlmostEqual(fp['all_latency_to_wall_visible'], 0.2)
        self.assertAlmostEqual(fp['winner_path_len'], 4.0)
        self.assertAlmostEqual(fp['iti_path_len'], 3.0)

    def test_solo_has_no_win_rate(self):
        keys = dict(experiment=0, player=0, session_type='solo2')
        outcome = pd.DataFrame([dict(**keys, trial_num=1.0, trial_type=globals.HIGH_LOW, won=1.0, chose_high=True,
                                     score=1.0, rt_s=2.0)])
        empty = lambda cols: pd.DataFrame(columns=list(hc.KEYS) + ['trial_num'] + cols)
        tables = dict(outcome=outcome,
                      vis=empty(['latency_to_wall_visible', 'visible_at_onset', 'ever_visible', 'trial_s']),
                      iti=empty(['prop_no_translational', 'prop_no_rotational', 'proportion_resting',
                                 'iti_path_len', 'rot_total', 'iti_centrality']),
                      win=empty(['winner_path_len', 'winner_direct_len', 'path_ratio']))
        players = pd.DataFrame([dict(**keys, mean_centrality=0.5, n_trials=1, group='middle')])
        fp = hc.build_player_fingerprint(players, tables).iloc[0]
        self.assertTrue(np.isnan(fp['win_rate']))
        self.assertAlmostEqual(fp['score_rate'], 0.5)


@unittest.skipUnless(hc.cached_session_files() is not None, 'session file list not cached')
class TestRealFileList(unittest.TestCase):

    def test_known_low_rate_sessions_at_expected_index(self):
        # sessions logged at a low frame rate on one recording PC, found by sampling rate in the pickles
        social, solo = hc.cached_session_files()
        self.assertIn('FA12_SL12_Social', social[5])
        self.assertIn('JK20_DD20_Social', social[49])
        self.assertIn('JK20_FirstSolo', solo[hc.solo_index(49, 0, 0)])
        self.assertIn('KH20_SecondSolo', solo[299])
        hc.check_file_alignment(social, solo)


if __name__ == '__main__':
    unittest.main()


def dense_session(n=2000, dt=1 / 48.0):
    ''' Return a one-trial session with a frame every dt, both players moving and turning. Takes the
        frame count. The network simulation needs a dense trace, unlike the four-row trials above. '''

    t = np.arange(n) * dt
    cols = {'timeLocal': pd.Timestamp('1900-01-01 12:00:00') + pd.to_timedelta(t, unit='s'),
            'eventDescription': [globals.SLICE_ONSET] + [''] * (n - 1),
            globals.TRIAL_NUM: 1.0}
    for p in (0, 1):
        cols[globals.PLAYER_LOC_DICT[p]['xloc']] = 2.0 * t + p
        cols[globals.PLAYER_LOC_DICT[p]['yloc']] = 0.5 * t
        cols[globals.PLAYER_ROT_DICT[p]['yrot']] = np.mod(30.0 * t, 360.0)
    return [pd.DataFrame(cols)]


def frame_seconds(df, time_col='timeLocal'):
    ''' Return frame times in seconds from the first frame, as the repo modules read them. Takes a
        session dataframe. '''

    return (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().to_numpy()


def dense_ticks(n=2000, dt=1 / 48.0, tick_s=nr.TICK_S):
    ''' Return a recovered tick table matching dense_session, on the tick grid. Takes the frame count. '''

    t = np.arange(0, (n - 1) * dt, tick_s)
    return pd.DataFrame({'t': t, 'x': -t, 'z': t, 'yaw': np.mod(90.0 * t, 360.0)})


class TestClientView(unittest.TestCase):
    ''' The client renders remote players two ticks behind rather than one, so reconstructing what player
        1 saw means applying the network path with that lag, not undoing it. '''

    def test_an_extra_lag_tick_shifts_a_ramp_by_one_tick(self):
        ''' On a trace moving at a steady speed the two lags differ by exactly one tick of travel, once
            the smoothing has settled. '''

        t = np.arange(0, 20, 1 / 48.0)
        speed = 3.0
        one = hc.simulate_network_view(t, speed * t, lag_ticks=1)
        two = hc.simulate_network_view(t, speed * t, lag_ticks=2)
        settled = slice(len(t) // 2, None)
        # the tick takes the latest frame value at or before it, which quantises the shift to the frame
        # grid, so the check allows half a frame of travel
        self.assertAlmostEqual(float(np.mean(one[settled] - two[settled])), speed * hc.NET_TICK_S,
                               delta=speed / 48.0 / 2)

    def test_network_view_passes_the_lag_through(self):
        ''' network_view took no lag argument before the client's view was needed, so this pins it. '''

        session = dense_session()
        xcol = globals.PLAYER_LOC_DICT[0]['xloc']
        one = pd.concat(hc.network_view(session, log_ids=(0,), lag_ticks=1))[xcol].to_numpy()
        two = pd.concat(hc.network_view(session, log_ids=(0,), lag_ticks=2))[xcol].to_numpy()
        self.assertFalse(np.allclose(one, two))
        # and it matches calling the trace function directly with the same lag
        t = frame_seconds(session[0])
        direct = hc.simulate_network_view(t, session[0][xcol].to_numpy(float), lag_ticks=2)
        np.testing.assert_allclose(two, direct, atol=1e-9)

    def test_network_view_leaves_the_other_player_alone(self):
        ''' Only the listed log ids are filtered. '''

        session = dense_session()
        seen = pd.concat(hc.network_view(session, log_ids=(0,), lag_ticks=2))
        for col in (globals.PLAYER_LOC_DICT[1]['xloc'], globals.PLAYER_ROT_DICT[1]['yrot']):
            np.testing.assert_allclose(seen[col].to_numpy(), session[0][col].to_numpy())

    def test_the_host_view_is_the_session_as_logged(self):
        ''' Player 0's own values are never filtered and the logged player 1 values are already the copy
            the host rendered, so nothing is done to them. '''

        session = dense_session()
        seen = hc.screen_view_trial_list(session, dense_ticks(), viewer=0)
        pd.testing.assert_frame_equal(pd.concat(seen), pd.concat(session))

    def test_the_client_view_filters_player_0_and_recovers_player_1(self):
        ''' Player 1 sees itself directly, so its own values are the recovered trace, and sees player 0
            through the network at two ticks. '''

        session, ticks = dense_session(), dense_ticks()
        seen = pd.concat(hc.screen_view_trial_list(session, ticks, viewer=1))
        # player 0 is filtered, so it no longer matches the log
        self.assertFalse(np.allclose(seen[globals.PLAYER_LOC_DICT[0]['xloc']].to_numpy(),
                                     session[0][globals.PLAYER_LOC_DICT[0]['xloc']].to_numpy()))
        # player 1 is the recovered trace read at the frame times
        want = np.interp(frame_seconds(seen), ticks['t'].to_numpy(), ticks['x'].to_numpy())
        np.testing.assert_allclose(seen[globals.PLAYER_LOC_DICT[1]['xloc']].to_numpy(), want, atol=1e-9)

    def test_the_two_views_differ(self):
        ''' The whole point of the pair: what player 0 saw and what player 1 saw are not the same data. '''

        session, ticks = dense_session(), dense_ticks()
        host = pd.concat(hc.screen_view_trial_list(session, ticks, viewer=0))
        client = pd.concat(hc.screen_view_trial_list(session, ticks, viewer=1))
        for p in (0, 1):
            col = globals.PLAYER_LOC_DICT[p]['xloc']
            self.assertFalse(np.allclose(host[col].to_numpy(), client[col].to_numpy()))
