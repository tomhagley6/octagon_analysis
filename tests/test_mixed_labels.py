# Tests for analysis/mixed_labels.py. Synthetic sessions with known trends, runs and steps check each
# measure against a known answer.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_mixed_labels -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.mixed_labels as ml


def session(y, experiment=0, player=0, session_type='social'):
    ''' Return an onset-style table for one player from a centrality series in play order. '''

    return pd.DataFrame(dict(experiment=experiment, player=player, session_type=session_type,
                             trial_pos=np.arange(len(y)), centrality=y))


class TestSeriesMeasures(unittest.TestCase):

    def test_trend_on_a_line(self):
        # a straight line is all trend, with no leftover spread
        y = np.linspace(0.8, 0.2, 100)
        m = ml.series_measures(y)
        self.assertAlmostEqual(m['trend_r2'], 1.0)
        self.assertAlmostEqual(m['trend_change'], -0.6)
        self.assertAlmostEqual(m['resid_sd'], 0.0, places=10)

    def test_drift_leaves_no_runs(self):
        # a drift with independent scatter has a strong trend but no runs once the line is removed
        rng = np.random.default_rng(0)
        y = np.linspace(0.75, 0.25, 2000) + rng.normal(0, 0.08, 2000)
        m = ml.series_measures(y)
        self.assertGreater(m['trend_r2'], 0.7)
        self.assertAlmostEqual(m['runs_r1'], 0.0, delta=0.05)
        self.assertAlmostEqual(m['resid_sd'], 0.08, delta=0.005)

    def test_runs_from_two_positions(self):
        # two exact positions with runs of mean length 10 give r1 near 1 - 2 / 10 = 0.8
        rng = np.random.default_rng(1)
        state, y = 0, []
        for _ in range(20000):
            if rng.random() < 0.1:
                state = 1 - state
            y.append([0.25, 0.75][state])
        m = ml.series_measures(np.array(y))
        self.assertAlmostEqual(m['runs_r1'], 0.8, delta=0.03)

    def test_independent_trials_have_no_runs(self):
        rng = np.random.default_rng(2)
        y = np.where(rng.random(5000) < 0.5, 0.25, 0.75) + rng.normal(0, 0.05, 5000)
        self.assertAlmostEqual(ml.series_measures(y)['runs_r1'], 0.0, delta=0.04)

    def test_local_jump_on_a_step(self):
        # a clean step of 0.5 gives a jump of 0.5 where it falls, a slow ramp only its slope x window
        y = np.r_[np.full(60, 0.25), np.full(40, 0.75)]
        m = ml.series_measures(y, window=10)
        self.assertAlmostEqual(m['local_jump'], 0.5)
        self.assertAlmostEqual(m['jump_at'], 0.6)
        ramp = ml.series_measures(np.linspace(0.75, 0.25, 101), window=10)
        self.assertAlmostEqual(ramp['local_jump'], -0.05, places=6)


class TestPlayerTable(unittest.TestCase):

    def test_shuffle_tests(self):
        # a strong trend and long runs beat shuffled order, independent trials do not
        rng = np.random.default_rng(3)
        trend = np.linspace(0.8, 0.2, 120) + rng.normal(0, 0.05, 120)
        runs = np.repeat(rng.choice([0.25, 0.75], 12), 10) + rng.normal(0, 0.05, 120)
        flat = rng.normal(0.5, 0.1, 120)
        onset = pd.concat([session(trend, player=0), session(runs, player=1), session(flat, player=2)])
        out = ml.sequence_table(onset, n_shuffle=500).set_index('player')
        self.assertLess(out.loc[0, 'p_trend'], 0.01)
        self.assertLess(out.loc[1, 'p_runs'], 0.01)
        self.assertGreater(out.loc[2, 'p_trend'], 0.01)
        self.assertGreater(out.loc[2, 'p_runs'], 0.01)
        self.assertLess(out.loc[1, 'p_jump'], 0.05)

    def test_short_sessions_dropped(self):
        onset = pd.concat([session(np.full(15, 0.5), player=0), session(np.linspace(0, 1, 40), player=1)])
        out = ml.sequence_table(onset, n_shuffle=50, min_trials=20)
        self.assertEqual(out['player'].tolist(), [1])

    def test_reproducible(self):
        rng = np.random.default_rng(4)
        onset = session(rng.uniform(0, 1, 60))
        a = ml.sequence_table(onset, n_shuffle=200, random_state=7)
        b = ml.sequence_table(onset, n_shuffle=200, random_state=7)
        pd.testing.assert_frame_equal(a, b)

    def test_run_length_guide(self):
        self.assertAlmostEqual(ml.run_length_guide(0.0), 2.0)
        self.assertAlmostEqual(ml.run_length_guide(0.8), 10.0)



class TestLabels(unittest.TestCase):

    def row(self, **kw):
        base = dict(experiment=0, player=0, session_type='social', p_trend=0.5, trend_r2=0.0, p_runs=0.5, runs_r1=0.0,
                    p_jump=0.5, local_jump=0.0, resid_sd=0.2, gap_vs_sd=3.0, shift_share=0.3, mean_centrality=0.45,
                    p_two_section=0.5, section_gap=0.0, p_resid_runs=0.5, two_section_r2=0.0, purity_min=1.0,
                    longest_excursion=0)
        base.update(kw)
        return base

    def test_each_label(self):
        rows = pd.DataFrame([
            self.row(player=0, p_runs=0.001, runs_r1=0.6),                              # long runs
            self.row(player=1, p_trend=0.001, trend_r2=0.5),                            # drifter
            self.row(player=2, resid_sd=0.12),                                          # hoverer
            self.row(player=3),                                                         # mixer
            self.row(player=4, shift_share=0.1),                                        # wide, few shifts: unclear
            self.row(player=5, p_runs=0.01, runs_r1=0.3),                               # weak runs but shifts back and forth: mixer
            self.row(player=6, p_trend=0.001, trend_r2=0.5, p_runs=0.01, runs_r1=0.3),  # trend with weak runs: drifter
            self.row(player=7, resid_sd=0.12, mean_centrality=0.9),                     # narrow but not middle: unclear
            self.row(player=8, p_trend=0.001, trend_r2=0.5, p_runs=0.01, runs_r1=0.4,
                     p_two_section=0.001, section_gap=-0.5, p_resid_runs=0.4,
                     two_section_r2=0.7),                                               # two stable sections: one switch
            self.row(player=9, p_trend=0.01, trend_r2=0.05),                            # very weak trend, no runs: unclear
            self.row(player=10, p_runs=0.01, runs_r1=0.3, shift_share=0.1),             # weak runs, few shifts: short runs
            self.row(player=11, p_runs=0.01, runs_r1=0.3, resid_sd=0.12),               # weak runs, narrow, middle: hoverer
        ])
        out = ml.assign_labels(rows, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, shift_ref={'social': 0.2},
                               gap_min={'social': 0.27})
        self.assertEqual(out['label'].tolist(),
                         ['long runs', 'drifter', 'hoverer', 'mixer', 'unclear', 'mixer', 'drifter', 'unclear',
                          'one switch', 'unclear', 'short runs', 'hoverer'])
        # short runs is also kept as a yes/no description beside the label
        self.assertEqual(out['has_short_runs'].tolist(),
                         [False, False, False, False, False, True, True, False, False, False, True, True])
        # every label a player passes is listed, whatever the priority order gave them
        self.assertEqual(out['all_labels'].tolist()[:2], ['long runs, mixer', 'drifter'])
        self.assertEqual(out.loc[6, 'all_labels'], 'drifter, short runs')
        self.assertEqual(out.loc[8, 'all_labels'], 'one switch, drifter')
        self.assertEqual(out.loc[4, 'all_labels'], 'unclear')
        self.assertTrue(out.loc[0, 'fits_long_runs'] and out.loc[0, 'fits_mixer'])
        self.assertFalse(out.loc[0, 'fits_drifter'])

    def test_one_switch_short_runs_are_within_sections(self):
        # a one-switch player's runs come partly from the switch, so their short runs are the runs left inside the sections
        base = dict(p_runs=0.001, runs_r1=0.5, p_two_section=0.001, section_gap=0.4, purity_min=0.9, longest_excursion=3,
                    p_trend=0.001, trend_r2=0.4, two_section_r2=0.6)
        rows = pd.DataFrame([self.row(player=0, **base, p_resid_runs=0.01), self.row(player=1, **base, p_resid_runs=0.4)])
        out = ml.assign_labels(rows, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, shift_ref={'social': 0.2},
                               gap_min={'social': 0.27})
        self.assertEqual(out['label'].tolist(), ['one switch', 'one switch'])
        self.assertEqual(out['has_short_runs'].tolist(), [True, False])

    def test_one_switch_needs_all_four_conditions(self):
        # a significant two-section fit, a big enough gap, stable sections, and a better fit than a line
        base = dict(p_runs=0.001, runs_r1=0.5, p_two_section=0.001, section_gap=0.4, purity_min=0.9, longest_excursion=3,
                    p_trend=0.001, trend_r2=0.4, two_section_r2=0.6)
        rows = pd.DataFrame([self.row(player=0, **base),
                             self.row(player=1, **{**base, 'p_two_section': 0.2}),
                             self.row(player=2, **{**base, 'section_gap': 0.2}),
                             self.row(player=3, **{**base, 'purity_min': 0.7}),
                             self.row(player=4, **{**base, 'longest_excursion': 6}),
                             self.row(player=5, **{**base, 'two_section_r2': 0.35})])
        out = ml.assign_labels(rows, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, shift_ref={'social': 0.2},
                               gap_min={'social': 0.27})
        self.assertEqual(out['label'].tolist(), ['one switch'] + ['long runs'] * 5)

    def test_stable_rule_runs_keeps_the_leftover_runs_check(self):
        # the earlier check: no runs left once the two sections are removed
        base = dict(p_runs=0.001, runs_r1=0.5, p_two_section=0.001, section_gap=0.4, p_trend=0.001, trend_r2=0.4,
                    two_section_r2=0.6, purity_min=0.9, longest_excursion=3)
        rows = pd.DataFrame([self.row(player=0, **base, p_resid_runs=0.4), self.row(player=1, **base, p_resid_runs=0.01)])
        out = ml.assign_labels(rows, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, shift_ref={'social': 0.2},
                               gap_min={'social': 0.27}, stable_rule='runs')
        self.assertEqual(out['label'].tolist(), ['one switch', 'long runs'])

    def test_local_jump_is_descriptive(self):
        # a big significant jump alone no longer makes a switcher
        rows = pd.DataFrame([self.row(player=0, p_jump=0.001, local_jump=0.5)])
        out = ml.assign_labels(rows, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, shift_ref={'social': 0.2},
                               gap_min={'social': 0.27})
        self.assertEqual(out['label'].tolist(), ['mixer'])

    def test_runs_do_not_block_drifter(self):
        # a significant trend of at least 0.10 makes a drifter even when short runs pass the test,
        # but the earlier rules still need the runs test not passed
        rows = pd.DataFrame([self.row(player=0, p_trend=0.001, trend_r2=0.15, p_runs=0.01, runs_r1=0.3, shift_share=0.1)])
        kw = dict(narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, shift_ref={'social': 0.2}, gap_min={'social': 0.27})
        self.assertEqual(ml.assign_labels(rows, **kw)['label'].tolist(), ['drifter'])
        self.assertEqual(ml.assign_labels(rows, **kw, allow_short_runs=False)['label'].tolist(), ['short runs'])

    def test_runs_block_mixer_in_earlier_rules(self):
        # with allow_short_runs=False, any passed runs test keeps a player out of mixer and hoverer
        rows = pd.DataFrame([self.row(player=0, p_runs=0.01, runs_r1=0.3)])
        out = ml.assign_labels(rows, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, shift_ref={'social': 0.2},
                               gap_min={'social': 0.27}, allow_short_runs=False)
        self.assertEqual(out['label'].tolist(), ['short runs'])

    def test_first_rules(self):
        # without the jump rule and the short runs label, a single big switch and weak runs are unclear
        rows = pd.DataFrame([self.row(player=0, p_trend=0.001, trend_r2=0.5, p_runs=0.01, runs_r1=0.4,
                                      p_jump=0.001, local_jump=-0.4),
                             self.row(player=1, p_runs=0.01, runs_r1=0.3)])
        out = ml.assign_labels(rows, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)},
                               short_runs=False, mixer_rule='point', one_switch_rule=None, allow_short_runs=False)
        self.assertEqual(out['label'].tolist(), ['unclear', 'unclear'])
        # the first rules split wide players with no order by the two-position ratio alone
        wide = pd.DataFrame([self.row(player=0, gap_vs_sd=4.0, shift_share=0.0), self.row(player=1, gap_vs_sd=3.0)])
        first = ml.assign_labels(wide, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, mixer_rule='point',
                                 one_switch_rule=None, allow_short_runs=False)
        self.assertEqual(first['label'].tolist(), ['mixer', 'broad'])

    def test_jump_needs_the_shuffle_test(self):
        # a large jump that does not beat shuffled order is not a switch
        rows = pd.DataFrame([self.row(player=0, p_jump=0.2, local_jump=0.5)])
        out = ml.assign_labels(rows, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, shift_ref={'social': 0.2},
                               gap_min={'social': 0.27})
        self.assertEqual(out['label'].tolist(), ['mixer'])

    def test_thresholds_are_inclusive(self):
        rows = pd.DataFrame([self.row(player=0, p_runs=0.001, runs_r1=0.46),
                             self.row(player=1, p_trend=0.001, trend_r2=0.10),
                             self.row(player=2, resid_sd=0.15)])
        out = ml.assign_labels(rows, narrow_cut={'social': 0.15}, middle={'social': (0.16, 0.76)}, shift_ref={'social': 0.2},
                               gap_min={'social': 0.27})
        self.assertEqual(out['label'].tolist(), ['long runs', 'drifter', 'hoverer'])


class TestShifts(unittest.TestCase):

    def test_noise_cut_from_central_trials(self):
        # the cut is the chosen quantile of central players' changes between consecutive centre-side trials
        rng = np.random.default_rng(6)
        y = 0.75 + rng.normal(0, 0.05, 5000)
        onset = session(np.r_[y, [0.1, 0.1]])
        labels = pd.DataFrame(dict(experiment=0, player=0, session_type='social', fixed=['central']))
        cut = ml.shift_noise_cut(onset, labels, boundary=0.485, q=0.95)
        expected = np.quantile(np.abs(np.diff(y)), 0.95)
        self.assertAlmostEqual(cut['social'], expected, delta=0.005)

    def test_shift_table(self):
        # alternating between far spots shifts on every change, a constant player never does
        onset = pd.concat([session(np.tile([0.2, 0.8], 30), player=0), session(np.full(60, 0.5), player=1)])
        out = ml.shift_table(onset, shift_cut={'social': 0.27}).set_index('player')
        self.assertAlmostEqual(out.loc[0, 'shift_share'], 1.0)
        self.assertAlmostEqual(out.loc[1, 'shift_share'], 0.0)
        self.assertEqual(out.loc[0, 'n_changes'], 59)

    def test_shift_reference_from_central_players(self):
        # the reference is the chosen quantile of central players' own shift shares, per session type
        shifts = pd.DataFrame(dict(experiment=0, player=range(5), session_type='social', shift_share=[0.0, 0.1, 0.2, 0.3, 0.9]))
        labels = pd.DataFrame(dict(experiment=0, player=range(5), session_type='social',
                                   fixed=['central', 'central', 'central', 'central', 'peripheral']))
        ref = ml.shift_reference(shifts, labels, q=0.5)
        self.assertAlmostEqual(ref['social'], 0.15)


class TestTwoSections(unittest.TestCase):

    def test_clean_step(self):
        # a clean switch at trial 40 of 100 is found exactly, with no transition and a perfect fit
        y = np.r_[np.full(40, 0.8), np.full(60, 0.3)]
        m = ml.two_section_fit(y)
        self.assertEqual((m['section_a_end'], m['section_b_start']), (40, 40))
        self.assertAlmostEqual(m['level_a'], 0.8)
        self.assertAlmostEqual(m['level_b'], 0.3)
        self.assertAlmostEqual(m['section_gap'], -0.5)
        self.assertAlmostEqual(m['two_section_r2'], 1.0)

    def test_short_transition_is_found(self):
        # a move over 6 trials between two steady levels is fitted as a 6-trial transition
        y = np.r_[np.full(50, 0.8), np.linspace(0.8, 0.3, 8)[1:-1], np.full(50, 0.3)]
        m = ml.two_section_fit(y)
        self.assertEqual(m['transition_len'], 6)
        self.assertAlmostEqual(m['two_section_r2'], 1.0, places=6)

    def test_sections_respect_the_minimum(self):
        # a switch after 3 trials cannot be fitted, as each section needs 5
        y = np.r_[np.full(3, 0.8), np.full(57, 0.3)]
        m = ml.two_section_fit(y, min_section=5)
        self.assertGreaterEqual(m['section_a_end'], 5)
        self.assertLess(m['two_section_r2'], 1.0)

    def test_transition_respects_the_maximum(self):
        # no transition longer than the allowed share of the session
        rng = np.random.default_rng(8)
        y = rng.uniform(0, 1, 100)
        m = ml.two_section_fit(y, max_transition=0.2)
        self.assertLessEqual(m['transition_len'], 20)

    def test_table_separates_one_switch_from_long_runs_and_drift(self):
        # one switch leaves no runs once the two sections are removed; more switches and a slow drift do
        rng = np.random.default_rng(9)
        noise = lambda n: rng.normal(0, 0.08, n)
        one = np.r_[np.full(15, 0.8), np.full(105, 0.3)] + noise(120)
        many = np.repeat([0.8, 0.3, 0.8, 0.3, 0.8, 0.3], 20) + noise(120)
        drift = np.linspace(0.8, 0.3, 120) + noise(120)
        onset = pd.concat([session(one, player=0), session(many, player=1), session(drift, player=2)])
        out = ml.two_section_table(onset, n_shuffle=200).set_index('player')
        self.assertLess(out.loc[0, 'p_two_section'], 0.01)
        self.assertGreater(out.loc[0, 'p_resid_runs'], 0.05)
        self.assertLess(out.loc[1, 'p_resid_runs'], 0.01)
        self.assertLess(out.loc[2, 'p_resid_runs'], 0.01)
        self.assertAlmostEqual(out.loc[0, 'switch_at'], 15 / 120, delta=0.02)

    def test_section_purity_and_excursions(self):
        # a clean switch has pure sections, a 3-trial trip back to the old level is one excursion of 3
        y = np.r_[np.full(40, 0.8), np.full(30, 0.3), np.full(3, 0.8), np.full(27, 0.3)]
        fit = ml.two_section_fit(y)
        purity = ml.section_purity(y, fit)
        self.assertEqual(purity['longest_excursion'], 3)
        self.assertAlmostEqual(purity['purity_a'], 1.0)
        self.assertAlmostEqual(purity['purity_b'], 57 / 60)
        self.assertAlmostEqual(purity['purity_min'], 57 / 60)
        clean = ml.section_purity(np.r_[np.full(40, 0.8), np.full(60, 0.3)], ml.two_section_fit(np.r_[np.full(40, 0.8), np.full(60, 0.3)]))
        self.assertEqual((clean['purity_min'], clean['longest_excursion']), (1.0, 0))

    def test_purity_ignores_closeness_of_consecutive_trials(self):
        # slow wandering around each level leaves the sections pure, though it leaves runs
        rng = np.random.default_rng(11)
        e = np.zeros(120)
        for t in range(1, 120):
            e[t] = 0.6 * e[t - 1] + rng.normal(0, 0.04)
        y = np.r_[np.full(50, 0.8), np.full(70, 0.3)] + e
        out = ml.two_section_table(session(y), n_shuffle=200).iloc[0]
        self.assertLess(out['p_resid_runs'], 0.05)
        self.assertGreater(out['purity_min'], 0.95)

    def test_no_structure_fails_the_fit_test(self):
        rng = np.random.default_rng(10)
        onset = session(rng.normal(0.5, 0.15, 120))
        out = ml.two_section_table(onset, n_shuffle=200)
        self.assertGreater(out['p_two_section'].iloc[0], 0.05)

class TestSimulation(unittest.TestCase):

    def test_kinds(self):
        rng = np.random.default_rng(5)
        params = dict(low=0.3, high=0.7, spread=0.1, hover_mean=0.5, hover_sd=0.08, run_length=10)
        onset, kinds = ml.simulate_sequences(n_trials=100, n_players=30, params=params, rng=rng)
        self.assertEqual(set(kinds['kind']), set(ml.SEQUENCE_KINDS))
        self.assertEqual(len(onset), 100 * 30 * len(ml.SEQUENCE_KINDS))
        # the drifter moves between the two positions, the hoverer stays at its mean
        merged = onset.merge(kinds, on=['experiment', 'player', 'session_type'])
        drift = merged[merged['kind'] == 'drifter']
        # each drifter moves (in a random direction), so compare start and end per player
        move = drift.groupby('player').apply(
            lambda d: d.loc[d['trial_pos'] >= 80, 'centrality'].mean() - d.loc[d['trial_pos'] < 20, 'centrality'].mean(),
            include_groups=False)
        self.assertTrue((move.abs() > 0.25).all())
        hover = merged[merged['kind'] == 'hoverer']['centrality']
        self.assertAlmostEqual(hover.mean(), 0.5, delta=0.02)
        self.assertAlmostEqual(hover.std(), 0.08, delta=0.01)


if __name__ == '__main__':
    unittest.main()
