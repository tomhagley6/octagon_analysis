#!/usr/bin/env python
# coding: utf-8

# Compare shapes for the peripheral end of the pooled onset centrality distribution.
#
# Every model has the same beta centre shape and differs only in what describes the wall end: a beta, an
# exponential falling from the alcove back, a flat block inside the alcove plus an exponential falling from the
# wall face, or the trials of peripheral players themselves. Each can add a spike at the alcove back.
#
# Models are fitted and scored on binned counts, not raw values. Trials pile at single values (the alcove back,
# and 0.0848), and a likelihood on raw values rewards any shape narrow enough to sit on a pile. Binning gives
# every model the same data, and a spike is simply all its mass in the first bin.
#
# The notebook demos/strategy_characterisation/humans/peripheral_shape.ipynb holds the plots and the results.

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import globals
import analysis.human_characterisation as hc


BIN_WIDTH = 0.01
BIN_EDGES = np.round(np.arange(0, 1 + BIN_WIDTH / 2, BIN_WIDTH), 10)
WALL_FACE = 1 - globals.OCTAGON_WALLFACE_RADIUS / globals.OCTAGON_ALCOVE_RADIUS    # centrality of the wall face (0.152)
LAM_RANGE = (-200.0, 500.0)     # rates an exponential may take (negative rises toward 1)
EMPIRICAL_PSEUDO = 0.5          # count added to every bin of the empirical shape, so no bin has zero probability
N_PARAMS = dict(beta=2, texp0=1, texp_wall=1, alcove=0, spike=0, empirical=0)

# the components of each model, wall end first and the beta centre last
BASE_MODELS = {
    'beta': ('beta', 'beta'),
    'texp': ('texp0', 'beta'),
    'alcove+exp': ('alcove', 'texp_wall', 'beta'),
    'empirical': ('empirical', 'beta'),
    '3 beta': ('beta', 'beta', 'beta'),
}
MODELS = tuple(BASE_MODELS) + tuple(f'{m}+spike' for m in BASE_MODELS)


def model_components(name):
    ''' Return the component kinds of a model, spike first when it has one. Takes the model name. '''

    base, spike = (name[:-len('+spike')], True) if name.endswith('+spike') else (name, False)
    return (('spike',) if spike else ()) + BASE_MODELS[base]


def n_params(name):
    ''' Return the number of fitted parameters of a model (shape parameters plus k - 1 weights).
        Takes the model name. The empirical shape is built from other trials, so it counts as none. '''

    kinds = model_components(name)
    return sum(N_PARAMS[c] for c in kinds) + len(kinds) - 1


# ---------------------------------------------------------------------------
# binned component shapes
# ---------------------------------------------------------------------------

def bin_counts(x, edges=BIN_EDGES):
    ''' Return the number of trials in each bin. Takes trial centrality and the bin edges.

        Values are clipped into [0, 1] first, so alcove corners (just below 0) count in the first bin. '''

    x = np.clip(np.asarray(x, dtype=float), edges[0], edges[-1])
    idx = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1).astype(float)


def beta_bins(a, b, edges=BIN_EDGES):
    ''' Return the probability of each bin under a beta shape. Takes a, b and the bin edges. '''

    return np.diff(stats.beta.cdf(edges, a, b))


def texp_bins(lam, lo, edges=BIN_EDGES):
    ''' Return the probability of each bin under an exponential with rate lam, truncated to [lo, 1].
        Takes the rate, the start and the bin edges.

        A positive rate falls away from lo, a negative one rises toward 1, and a rate of 0 is flat. '''

    lam = float(np.clip(lam, *LAM_RANGE))
    d = np.clip(edges, lo, 1) - lo
    # get the CDF at each edge, flat when the rate is near 0
    cdf = d / (1 - lo) if abs(lam) < 1e-8 else np.expm1(-lam * d) / np.expm1(-lam * (1 - lo))
    return np.diff(cdf)


def uniform_bins(lo, hi, edges=BIN_EDGES):
    ''' Return the probability of each bin under a flat shape on [lo, hi]. Takes lo, hi and the bin edges. '''

    return np.diff(np.clip(edges, lo, hi) - lo) / (hi - lo)


def spike_bins(edges=BIN_EDGES):
    ''' Return a spike at the alcove back, with all its probability in the first bin. Takes the bin edges. '''

    p = np.zeros(len(edges) - 1)
    p[0] = 1.0
    return p


def empirical_bins(x, edges=BIN_EDGES, pseudo=EMPIRICAL_PSEUDO):
    ''' Return the share of trials in each bin, with a small count added to every bin.
        Takes trial centrality, the bin edges and the added count. '''

    c = bin_counts(x, edges) + pseudo
    return c / c.sum()


def _component_bins(kind, par, edges):
    ''' Return the bin probabilities of one component. Takes its kind, its parameters and the bin edges. '''

    if kind == 'beta':
        return beta_bins(par['a'], par['b'], edges)
    if kind in ('texp0', 'texp_wall'):
        return texp_bins(par['lam'], par['lo'], edges)
    if kind == 'alcove':
        return uniform_bins(0.0, WALL_FACE, edges)
    if kind == 'spike':
        return spike_bins(edges)
    if kind == 'empirical':
        return par['probs']
    raise ValueError(f'unknown component {kind!r}')


def component_bins(fit):
    ''' Return each component's bin probabilities (k x bins), weights not applied. Takes a fit. '''

    return np.array([_component_bins(kind, par, fit['edges']) for kind, par in zip(fit['kinds'], fit['params'])])


def model_bins(fit):
    ''' Return the bin probabilities of the whole model. Takes a fit. '''

    return fit['weights'] @ component_bins(fit)


def binned_loglik(counts, probs):
    ''' Return the log likelihood of bin counts under bin probabilities. Takes the counts and the probabilities. '''

    return float(counts @ np.log(np.maximum(probs, 1e-300)))


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def _unpack(theta, kinds, empirical):
    ''' Return the component parameters and weights from an unbounded vector.
        Takes the vector, the component kinds and the empirical shape (or None). '''

    params, i = [], 0
    for kind in kinds:
        if kind == 'beta':
            params.append(dict(a=float(np.exp(theta[i])), b=float(np.exp(theta[i + 1]))))
        elif kind in ('texp0', 'texp_wall'):
            params.append(dict(lam=float(np.clip(theta[i], *LAM_RANGE)), lo=0.0 if kind == 'texp0' else WALL_FACE))
        elif kind == 'empirical':
            params.append(dict(probs=empirical))
        else:
            params.append({})
        i += N_PARAMS[kind]
    weights = special.softmax(np.concatenate([[0.0], theta[i:]]))
    return params, weights


def _start(kinds, rng, jitter):
    ''' Return a starting vector, with the beta shapes spread from wall to centre.
        Takes the component kinds, a generator and the jitter SD. '''

    betas = [k for k in kinds if k == 'beta']
    # place the betas from the wall end to the centre, with the last at the centre peak
    means = {1: [0.78], 2: [0.2, 0.78], 3: [0.1, 0.4, 0.78]}[len(betas)]
    theta, b = [], 0
    for kind in kinds:
        if kind == 'beta':
            m, conc = means[b], 12.0
            theta += [np.log(m * conc), np.log((1 - m) * conc)]
            b += 1
        elif kind in ('texp0', 'texp_wall'):
            theta.append(8.0)
    theta += [0.0] * (len(kinds) - 1)
    theta = np.array(theta)
    return theta + rng.normal(0, jitter, len(theta)) if jitter else theta


def fit_model(counts, name, edges=BIN_EDGES, empirical=None, n_starts=6, random_state=0):
    ''' Return a model fitted to binned trial counts by maximum likelihood.
        Takes the bin counts, the model name, the bin edges, the empirical shape (empirical models only),
        the number of starts and a seed.

        The first start places the shapes by hand, and the rest jitter it. The best start is kept. '''

    kinds = model_components(name)
    if 'empirical' in kinds and empirical is None:
        raise ValueError('the empirical model needs an empirical shape')
    rng = np.random.default_rng(random_state)

    def nll(theta):
        params, weights = _unpack(theta, kinds, empirical)
        probs = weights @ np.array([_component_bins(k, p, edges) for k, p in zip(kinds, params)])
        return -binned_loglik(counts, probs)

    best = None
    for s in range(n_starts):
        res = optimize.minimize(nll, _start(kinds, rng, 0.0 if s == 0 else 0.5), method='L-BFGS-B')
        if best is None or res.fun < best.fun:
            best = res
    params, weights = _unpack(best.x, kinds, empirical)
    return dict(name=name, kinds=kinds, params=params, weights=weights, loglik=-float(best.fun),
                n=float(counts.sum()), n_params=n_params(name), edges=edges)


def describe_fit(fit):
    ''' Return one row per component with its weight and shape parameters. Takes a fit. '''

    rows = []
    for kind, par, w in zip(fit['kinds'], fit['params'], fit['weights']):
        row = dict(model=fit['name'], component=kind, weight=w)
        if kind == 'beta':
            a, b = par['a'], par['b']
            row.update(mean=a / (a + b), sd=np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))), a=a, b=b)
        elif kind in ('texp0', 'texp_wall'):
            row.update(start=par['lo'], rate=par['lam'])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# held-out players
# ---------------------------------------------------------------------------

def _player_keys(trials):
    ''' Return the (experiment, player) key of every trial. Takes a trial table. '''

    return list(zip(trials['experiment'], trials['player']))


def player_splits(trials, n_splits=5, n_repeats=1, random_state=0):
    ''' Yield (training players, held-out players) pairs, each player held out once per repeat.
        Takes a trial table, the number of folds, the number of repeats and a seed. '''

    keys = sorted(set(_player_keys(trials)))
    rng = np.random.default_rng(random_state)
    for _ in range(n_repeats):
        order = rng.permutation(len(keys))
        for fold in np.array_split(order, n_splits):
            test = [keys[i] for i in fold]
            held = set(test)
            yield [k for k in keys if k not in held], test


def peripheral_trials(trials, boundary):
    ''' Return the trials of fixed-cut peripheral players (at least 80% of trials below the boundary).
        Takes a trial table and the trial boundary. '''

    frac = (trials['centrality'] < boundary).groupby([trials['experiment'], trials['player']]).mean()
    keep = set(frac.index[hc.fixed_cut_labels(frac) == 'peripheral'])
    return trials[[k in keep for k in _player_keys(trials)]]


def cross_validate(trials, models=MODELS, boundary=None, n_splits=5, n_repeats=1, edges=BIN_EDGES, random_state=0):
    ''' Return the held-out log likelihood of each model on each fold of held-out players.
        Takes a trial table (experiment, player, centrality), the model names, the trial boundary (for the
        empirical shape), the number of folds and repeats, the bin edges and a seed.

        Each fold fits every model to the training players and scores the held-out players. The empirical
        shape is rebuilt from the training players' peripheral players, so it never sees held-out trials. '''

    keys = np.array(_player_keys(trials), dtype=object)
    key_index = pd.MultiIndex.from_arrays([trials['experiment'], trials['player']])
    rows = []
    for split, (train, test) in enumerate(player_splits(trials, n_splits, n_repeats, random_state)):
        tr, te = trials[key_index.isin(train)], trials[key_index.isin(test)]
        c_tr, c_te = bin_counts(tr['centrality'], edges), bin_counts(te['centrality'], edges)
        emp = None
        if any('empirical' in model_components(m) for m in models):
            emp = empirical_bins(peripheral_trials(tr, boundary)['centrality'], edges)
        for m in models:
            fit = fit_model(c_tr, m, edges, empirical=emp, random_state=random_state)
            rows.append(dict(split=split, model=m, n_params=fit['n_params'], train_loglik=fit['loglik'],
                             heldout_loglik=binned_loglik(c_te, model_bins(fit)), n_heldout=len(te)))
    return pd.DataFrame(rows)


def cv_summary(cv, reference='beta'):
    ''' Return each model's held-out log likelihood summed over folds, and its gain over a reference model.
        Takes the output of cross_validate and the reference model.

        The gain per fold is also summarised, so a gain that holds in every fold can be told from one carried
        by a few folds. Per-trial values divide by the held-out trials of a full repeat. '''

    ref = cv[cv['model'] == reference].set_index('split')['heldout_loglik']
    d = cv.assign(gain=cv['heldout_loglik'] - cv['split'].map(ref))
    n_trials = d[d['model'] == reference]['n_heldout'].sum()
    out = d.groupby('model').agg(n_params=('n_params', 'first'), heldout_loglik=('heldout_loglik', 'sum'),
                                 gain=('gain', 'sum'), folds_better=('gain', lambda g: int((g > 0).sum())),
                                 folds=('gain', 'size'))
    out['gain_per_1000_trials'] = 1000 * out['gain'] / n_trials
    return out.sort_values('heldout_loglik', ascending=False)


# ---------------------------------------------------------------------------
# simulated players
# ---------------------------------------------------------------------------

def _draw_bins(probs, n, edges, rng):
    ''' Return n values drawn from bin probabilities, uniform within each bin.
        Takes the bin probabilities, n, the bin edges and a generator. '''

    idx = rng.choice(len(probs), size=n, p=probs / probs.sum())
    return edges[idx] + rng.random(n) * np.diff(edges)[idx]


def simulate_players(fit, players, rng):
    ''' Return simulated trials from a fitted model, one set per player.
        Takes a fit, a player table (experiment, player, n_trials, frac_peripheral) and a generator.

        Each player keeps their trial count, and draws each trial from the wall end (every component but
        the centre) with probability frac_peripheral, otherwise from the centre shape. This keeps the real
        mix of central, peripheral and intermediate players. Trials within a player are independent. '''

    comps = component_bins(fit)
    w = fit['weights']
    wall = w[:-1] @ comps[:-1]
    rows = []
    for p in players.itertuples():
        n_wall = rng.binomial(int(p.n_trials), p.frac_peripheral)
        x = np.concatenate([_draw_bins(wall, n_wall, fit['edges'], rng),
                            _draw_bins(comps[-1], int(p.n_trials) - n_wall, fit['edges'], rng)])
        rows.append(pd.DataFrame(dict(experiment=p.experiment, player=p.player, centrality=rng.permutation(x))))
    return pd.concat(rows, ignore_index=True)
