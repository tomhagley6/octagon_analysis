# Tests for analysis/centrality_mixture.py. Synthetic trials with known mode shapes and weights check
# each step against a known answer.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_centrality_mixture -v

import os
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.centrality_mixture as cm


def two_mode_sample(n, pi_centre, rng, peripheral=(2.0, 14.0), centre=(9.0, 2.0)):
    ''' Return n trials drawn from a peripheral and a centre beta, with share pi_centre from the centre. '''

    from_centre = rng.random(n) < pi_centre
    return np.where(from_centre, rng.beta(*centre, n), rng.beta(*peripheral, n)), from_centre


def known_fit(peripheral=(2.0, 14.0), centre=(9.0, 2.0), weights=(0.4, 0.6)):
    ''' Return a 2-component fit dict with known shapes, peripheral first. '''

    return dict(k=2, weights=np.array(weights), a=np.array([peripheral[0], centre[0]]),
                b=np.array([peripheral[1], centre[1]]), names=('peripheral', 'centre'))


class TestPrepare(unittest.TestCase):

    def test_clip_unit(self):
        x = cm.clip_unit(np.array([-0.004, 0.0, 0.5, 1.0]), eps=0.005)
        np.testing.assert_allclose(x, [0.005, 0.005, 0.5, 0.995])

    def test_drop_spawn_trials(self):
        # only a first trial at the exact centre is the spawn position, a later one or an off-centre one is kept
        onset = pd.DataFrame(dict(experiment=0, player=0, session_type='social', trial_pos=[0, 1, 0, 2],
                                  centrality=[1.0, 1.0, 0.7, 0.3]))
        onset.loc[2, 'player'] = 1
        out = cm.drop_spawn_trials(onset)
        self.assertEqual(len(out), 3)
        self.assertFalse(((out['trial_pos'] == 0) & (out['centrality'] >= 0.999)).any())


class TestBetaMixture(unittest.TestCase):

    def test_recovers_two_components(self):
        rng = np.random.default_rng(0)
        x, _ = two_mode_sample(20000, 0.6, rng)
        fit = cm.fit_beta_mixture(x, k=2, random_state=0)
        # components come back ordered by mean, with the known weights and means
        np.testing.assert_allclose(fit['weights'], [0.4, 0.6], atol=0.02)
        np.testing.assert_allclose(fit['a'] / (fit['a'] + fit['b']), [2 / 16, 9 / 11], atol=0.01)
        self.assertEqual(fit['names'], ('peripheral', 'centre'))

    def test_three_components_ordered(self):
        rng = np.random.default_rng(1)
        x = np.r_[rng.beta(2, 14, 5000), rng.beta(20, 20, 3000), rng.beta(9, 2, 6000)]
        fit = cm.fit_beta_mixture(x, k=3, random_state=0)
        means = fit['a'] / (fit['a'] + fit['b'])
        self.assertTrue(np.all(np.diff(means) > 0))
        self.assertEqual(fit['names'], ('peripheral', 'middle', 'centre'))
        np.testing.assert_allclose(means, [2 / 16, 0.5, 9 / 11], atol=0.02)

    def test_pile_up_does_not_collapse_a_shape(self):
        # many trials at one exact value cannot shrink a shape below the minimum SD
        rng = np.random.default_rng(12)
        x = np.r_[rng.beta(2, 14, 3000), np.full(400, 0.0848), rng.beta(9, 2, 4000)]
        fit = cm.fit_beta_mixture(x, k=3, random_state=0)
        a, b = fit['a'], fit['b']
        sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        self.assertTrue(np.all(sd >= cm.MIN_SHAPE_SD - 1e-6))

    def test_beta_beats_gaussian_on_beta_data(self):
        # the comparison scores both families on the same trials
        rng = np.random.default_rng(2)
        x, _ = two_mode_sample(5000, 0.6, rng)
        table = cm.compare_models(x, ks=(2,), random_state=0).set_index('model')
        self.assertGreater(table.loc['beta k=2', 'loglik'], table.loc['gaussian k=2', 'loglik'])
        self.assertEqual(table.loc['beta k=2', 'n_params'], 5)


class TestPlayerWeights(unittest.TestCase):

    def test_recovers_known_weight(self):
        rng = np.random.default_rng(3)
        fit = known_fit()
        x, _ = two_mode_sample(4000, 0.3, rng)
        w, _ = cm.player_weights(cm.component_pdfs(x, fit))
        self.assertAlmostEqual(w[1], 0.3, delta=0.03)

    def test_weight_is_the_likelihood_maximum(self):
        # the iterated posterior mean lands on the same weight as a grid search over the likelihood
        rng = np.random.default_rng(4)
        fit = known_fit()
        x, _ = two_mode_sample(80, 0.55, rng)
        F = cm.component_pdfs(x, fit)
        w, _ = cm.player_weights(F)
        grid = np.linspace(0, 1, 20001)
        ll = np.log(np.outer(1 - grid, F[:, 0]) + np.outer(grid, F[:, 1])).sum(axis=1)
        self.assertAlmostEqual(w[1], grid[np.argmax(ll)], delta=1e-3)

    def test_trial_at_crossing_counts_half(self):
        # a trial where the two shapes are equally likely counts 0.5 to each when the weights are equal
        fit = known_fit()
        grid = np.linspace(0.2, 0.7, 50001)
        F = cm.component_pdfs(grid, fit)
        cross = grid[np.argmin(np.abs(F[:, 0] - F[:, 1]))]
        r = cm.posteriors(cm.component_pdfs(np.array([cross]), fit), np.array([0.5, 0.5]))
        np.testing.assert_allclose(r[0], [0.5, 0.5], atol=1e-3)

    def test_vectorised_weights_match_single(self):
        rng = np.random.default_rng(5)
        fit = known_fit()
        xs = [two_mode_sample(60, p, rng)[0] for p in (0.2, 0.5, 0.8)]
        stacked = np.stack([cm.component_pdfs(x, fit) for x in xs])
        w_all, _ = cm.player_weights(stacked)
        for i, x in enumerate(xs):
            np.testing.assert_allclose(w_all[i], cm.player_weights(cm.component_pdfs(x, fit))[0], atol=1e-6)


class TestLikelihoodInterval(unittest.TestCase):

    def test_contains_estimate_and_narrows(self):
        rng = np.random.default_rng(6)
        fit = known_fit()
        widths = []
        for n in (40, 400):
            x, _ = two_mode_sample(n, 0.5, rng)
            F = cm.component_pdfs(x, fit)
            w, _ = cm.player_weights(F)
            lo, hi = cm.likelihood_interval(F)
            self.assertTrue(lo <= w[1] <= hi)
            widths.append(hi - lo)
        self.assertLess(widths[1], widths[0])

    def test_coverage_on_independent_trials(self):
        # about 95% of intervals contain the true weight when trials are independent
        rng = np.random.default_rng(7)
        fit = known_fit()
        hits = []
        for _ in range(300):
            x, _ = two_mode_sample(100, 0.4, rng)
            lo, hi = cm.likelihood_interval(cm.component_pdfs(x, fit))
            hits.append(lo <= 0.4 <= hi)
        self.assertAlmostEqual(np.mean(hits), 0.95, delta=0.04)


class TestPlayerTable(unittest.TestCase):

    def make_onset(self, rng):
        rows = []
        for player, pi in enumerate((0.1, 0.5, 0.9)):
            x, _ = two_mode_sample(120, pi, rng)
            rows.append(pd.DataFrame(dict(experiment=0, player=player, session_type='social',
                                          trial_pos=np.arange(120), centrality=x)))
        return pd.concat(rows, ignore_index=True)

    def test_table_values(self):
        rng = np.random.default_rng(8)
        onset = self.make_onset(rng)
        table = cm.player_mixture_table(onset, known_fit(), n_boot=200)
        self.assertEqual(len(table), 3)
        np.testing.assert_allclose(table['w_centre'], [0.1, 0.5, 0.9], atol=0.1)
        # weights sum to 1, and each interval holds its estimate
        np.testing.assert_allclose(table['w_peripheral'] + table['w_centre'], 1)
        self.assertTrue(((table['pi_lo'] <= table['w_centre']) & (table['w_centre'] <= table['pi_hi'])).all())
        self.assertTrue(((table['pi_boot_lo'] <= table['w_centre']) & (table['w_centre'] <= table['pi_boot_hi'])).all())
        # soft positions sit near each shape's mean when the shapes barely overlap
        mixed = table.iloc[1]
        self.assertAlmostEqual(mixed['pos_peripheral'], 2 / 16, delta=0.03)
        self.assertAlmostEqual(mixed['pos_centre'], 9 / 11, delta=0.03)

    def test_too_few_effective_trials(self):
        # a player with almost no centre trials gets no centre position
        onset = pd.DataFrame(dict(experiment=0, player=0, session_type='social', trial_pos=np.arange(30),
                                  centrality=np.r_[np.full(29, 0.1), 0.85]))
        row = cm.player_mixture_table(onset, known_fit(), n_boot=50, min_eff=5).iloc[0]
        self.assertTrue(np.isnan(row['pos_centre']))
        self.assertFalse(np.isnan(row['pos_peripheral']))

    def test_three_components_have_no_interval(self):
        rng = np.random.default_rng(9)
        onset = self.make_onset(rng)
        fit3 = dict(k=3, weights=np.array([0.3, 0.2, 0.5]), a=np.array([2.0, 20.0, 9.0]),
                    b=np.array([14.0, 20.0, 2.0]), names=('peripheral', 'middle', 'centre'))
        table = cm.player_mixture_table(onset, fit3, n_boot=0)
        np.testing.assert_allclose(table[['w_peripheral', 'w_middle', 'w_centre']].sum(axis=1), 1)
        self.assertNotIn('pi_lo', table.columns)

    def test_reproducible(self):
        rng = np.random.default_rng(10)
        onset = self.make_onset(rng)
        a = cm.player_mixture_table(onset, known_fit(), n_boot=100, random_state=3)
        b = cm.player_mixture_table(onset, known_fit(), n_boot=100, random_state=3)
        pd.testing.assert_frame_equal(a, b)


class TestSimulation(unittest.TestCase):

    def test_simulated_kinds(self):
        rng = np.random.default_rng(11)
        onset, truth = cm.simulate_players(known_fit(), n_trials=100, n_players=20, boundary=0.485, rng=rng)
        self.assertEqual(set(truth['kind']), set(cm.SIM_KINDS))
        self.assertEqual(len(onset), 100 * 20 * len(cm.SIM_KINDS))
        # a switcher waits in the peripheral shape in the first half and the centre shape in the second
        sw = onset[onset['kind'] == 'switcher']
        self.assertLess(sw.loc[sw['trial_pos'] < 50, 'centrality'].mean(), 0.3)
        self.assertGreater(sw.loc[sw['trial_pos'] >= 50, 'centrality'].mean(), 0.7)
        # a hoverer waits around the boundary
        hv = onset[onset['kind'] == 'hoverer']
        self.assertAlmostEqual(hv['centrality'].mean(), 0.485, delta=0.02)
        self.assertTrue(truth.set_index('kind').loc['hoverer', 'true_pi'] != truth.set_index('kind').loc['hoverer', 'true_pi'])


class TestSummaries(unittest.TestCase):

    def test_simulation_summary(self):
        # coverage counts how often each interval holds the true weight, kinds without one get NaN
        table = pd.DataFrame(dict(kind=['mixer'] * 4 + ['hoverer'] * 2, w_centre=[0.5, 0.4, 0.6, 0.55, 0.0, 0.1],
                                  pi_lo=[0.4, 0.3, 0.55, 0.45, 0.0, 0.0], pi_hi=[0.6, 0.45, 0.7, 0.65, 0.1, 0.2],
                                  pi_boot_lo=[0.3, 0.2, 0.4, 0.4, 0.0, 0.0], pi_boot_hi=[0.7, 0.6, 0.8, 0.7, 0.1, 0.2]))
        truth = pd.DataFrame(dict(kind=['mixer', 'hoverer'], true_pi=[0.5, np.nan]))
        out = cm.simulation_summary(table, truth).set_index('kind')
        self.assertAlmostEqual(out.loc['mixer', 'likelihood_coverage'], 0.5)
        self.assertAlmostEqual(out.loc['mixer', 'block_coverage'], 1.0)
        self.assertAlmostEqual(out.loc['mixer', 'likelihood_width'], 0.175)
        self.assertTrue(np.isnan(out.loc['hoverer', 'block_coverage']))

    def test_ternary_xy(self):
        # each pure weight sits on its own corner, equal weights at the centre
        xy = cm.ternary_xy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]]))
        np.testing.assert_allclose(xy[:3], [[0, 0], [0.5, np.sqrt(3) / 2], [1, 0]], atol=1e-12)
        np.testing.assert_allclose(xy[3], [0.5, np.sqrt(3) / 6], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
