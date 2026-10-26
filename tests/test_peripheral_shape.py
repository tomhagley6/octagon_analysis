# Tests for analysis/peripheral_shape.py. Synthetic trials drawn from known shapes check each step
# against a known answer.
#
# Run from the repo root:
#   /home/tom/miniconda3/envs/octagon_analysis/bin/python -m unittest tests.test_peripheral_shape -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

# add the repo root so repo modules import when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.peripheral_shape as ps


EDGES = ps.BIN_EDGES


def sample_texp(n, lam, lo, rng):
    ''' Return n draws from an exponential with rate lam, truncated to [lo, 1]. '''

    u = rng.random(n)
    return lo - np.log1p(-u * (1 - np.exp(-lam * (1 - lo)))) / lam


def sample_model(n, weights, parts, rng):
    ''' Return n draws from a mixture. Takes weights and one sampler per component (n, rng) -> values. '''

    comp = rng.choice(len(weights), size=n, p=weights)
    x = np.empty(n)
    for j, draw in enumerate(parts):
        x[comp == j] = draw((comp == j).sum(), rng)
    return x


class TestComponents(unittest.TestCase):

    def test_bins_sum_to_one(self):
        # every component puts all its mass inside [0, 1]
        for p in (ps.beta_bins(2.0, 9.0, EDGES), ps.texp_bins(12.0, 0.0, EDGES), ps.texp_bins(-3.0, 0.2, EDGES),
                  ps.texp_bins(1e-9, 0.0, EDGES), ps.uniform_bins(0.0, ps.WALL_FACE, EDGES), ps.spike_bins(EDGES)):
            self.assertAlmostEqual(p.sum(), 1.0, places=9)
            self.assertTrue((p >= 0).all())

    def test_texp_matches_samples(self):
        # binned truncated exponential matches the share of draws in each bin
        rng = np.random.default_rng(0)
        x = sample_texp(400_000, 8.0, ps.WALL_FACE, rng)
        obs = ps.bin_counts(x, EDGES) / len(x)
        np.testing.assert_allclose(obs, ps.texp_bins(8.0, ps.WALL_FACE, EDGES), atol=2e-3)
        # nothing below the start of the exponential
        self.assertEqual(ps.texp_bins(8.0, ps.WALL_FACE, EDGES)[EDGES[1:] <= ps.WALL_FACE].sum(), 0)

    def test_uniform_and_spike(self):
        # the alcove block is flat up to the wall face, and the spike is the first bin only
        p = ps.uniform_bins(0.0, 0.15, EDGES)
        np.testing.assert_allclose(p[:15], 1 / 15)
        self.assertEqual(p[15:].sum(), 0)
        s = ps.spike_bins(EDGES)
        self.assertEqual(s[0], 1.0)

    def test_bin_counts_clips(self):
        # values just below 0 (alcove corners) and at 1 land in the end bins
        c = ps.bin_counts(np.array([-0.004, 0.0, 0.005, 0.5, 1.0]), EDGES)
        self.assertEqual(c[0], 3)
        self.assertEqual(c[-1], 1)
        self.assertEqual(c.sum(), 5)


class TestFit(unittest.TestCase):

    def test_recovers_texp_mixture(self):
        # a truncated exponential plus a beta centre is recovered from its own draws
        rng = np.random.default_rng(1)
        x = sample_model(60_000, [0.35, 0.65], [lambda n, r: sample_texp(n, 10.0, 0.0, r),
                                                lambda n, r: r.beta(9.0, 3.0, n)], rng)
        fit = ps.fit_model(ps.bin_counts(x, EDGES), 'texp')
        self.assertAlmostEqual(fit['weights'][0], 0.35, delta=0.01)
        self.assertAlmostEqual(fit['params'][0]['lam'], 10.0, delta=0.4)
        self.assertAlmostEqual(fit['params'][1]['a'], 9.0, delta=0.5)

    def test_recovers_spike_weight(self):
        # a spike at 0 is given its own weight when the model has one
        rng = np.random.default_rng(2)
        x = sample_model(60_000, [0.03, 0.3, 0.67], [lambda n, r: r.uniform(-0.004, 0.006, n),
                                                     lambda n, r: r.beta(2.0, 9.0, n),
                                                     lambda n, r: r.beta(9.0, 3.0, n)], rng)
        fit = ps.fit_model(ps.bin_counts(x, EDGES), 'beta+spike')
        self.assertAlmostEqual(fit['weights'][0], 0.03, delta=0.005)

    def test_loglik_matches_counts(self):
        # the fitted log likelihood equals the binned log likelihood recomputed from the fit
        rng = np.random.default_rng(3)
        counts = ps.bin_counts(rng.beta(2, 5, 5000), EDGES)
        fit = ps.fit_model(counts, 'beta')
        self.assertAlmostEqual(fit['loglik'], ps.binned_loglik(counts, ps.model_bins(fit)), places=6)

    def test_nested_model_not_worse(self):
        # adding a spike never lowers the fitted log likelihood on the same counts
        rng = np.random.default_rng(4)
        counts = ps.bin_counts(rng.beta(2, 5, 5000), EDGES)
        self.assertGreaterEqual(ps.fit_model(counts, 'beta+spike')['loglik'],
                                ps.fit_model(counts, 'beta')['loglik'] - 1e-6)

    def test_empirical_shape(self):
        # the empirical shape is a proper distribution built from the given trials
        rng = np.random.default_rng(5)
        p = ps.empirical_bins(rng.beta(2, 9, 3000), EDGES)
        self.assertAlmostEqual(p.sum(), 1.0, places=12)
        self.assertTrue((p > 0).all())


class TestCrossValidation(unittest.TestCase):

    def make_players(self, rng, peripheral_draw, n_players=120, n_trials=100):
        ''' Return a trial table of players with their own peripheral share, peripheral trials from
            peripheral_draw and centre trials from a beta. '''

        rows = []
        for i in range(n_players):
            share = rng.choice([0.05, 0.5, 0.95])
            n_p = rng.binomial(n_trials, share)
            x = np.concatenate([peripheral_draw(n_p, rng), rng.beta(9.0, 3.0, n_trials - n_p)])
            rows.append(pd.DataFrame(dict(experiment=i, player=0, centrality=x)))
        return pd.concat(rows, ignore_index=True)

    def test_picks_generating_shape(self):
        # held-out players prefer the shape that generated the data
        rng = np.random.default_rng(6)
        texp_data = self.make_players(rng, lambda n, r: sample_texp(n, 12.0, 0.0, r))
        res = ps.cross_validate(texp_data, ('beta', 'texp'), boundary=0.48, n_splits=4, random_state=0)
        means = res.groupby('model')['heldout_loglik'].sum()
        self.assertGreater(means['texp'], means['beta'])

    def test_splits_by_player(self):
        # no player has trials on both sides of a split
        rng = np.random.default_rng(7)
        data = self.make_players(rng, lambda n, r: r.beta(2, 9, n), n_players=20)
        for train, test in ps.player_splits(data, n_splits=3, random_state=0):
            self.assertFalse(set(train) & set(test))


class TestSimulate(unittest.TestCase):

    def test_simulated_trial_counts(self):
        # simulated players keep their trial counts, and peripheral shares follow the given shares
        rng = np.random.default_rng(8)
        counts = ps.bin_counts(sample_texp(20_000, 10.0, 0.0, rng), EDGES) + ps.bin_counts(rng.beta(9, 3, 20_000), EDGES)
        fit = ps.fit_model(counts, 'texp')
        players = pd.DataFrame(dict(experiment=[0, 1], player=[0, 0], n_trials=[200, 50], frac_peripheral=[1.0, 0.0]))
        sim = ps.simulate_players(fit, players, rng)
        self.assertEqual(sim.groupby('experiment').size().tolist(), [200, 50])
        # an all-peripheral player draws only from the exponential, so mostly low values
        self.assertLess(sim.loc[sim['experiment'] == 0, 'centrality'].median(), 0.15)
        self.assertGreater(sim.loc[sim['experiment'] == 1, 'centrality'].median(), 0.6)


if __name__ == '__main__':
    unittest.main()
