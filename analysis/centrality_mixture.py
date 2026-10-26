#!/usr/bin/env python
# coding: utf-8

# Describe each human player's waiting strategy with shared mode shapes.
#
# A beta mixture is fitted once to the onset centrality of every social trial, pooled over players. Its
# component shapes (peripheral and centre, plus a middle one for comparison) are then held fixed. Each player
# gets their own weight on each shape, fitted by maximum likelihood, and a soft position within each
# shape. A trial between the shapes counts partly to each, rather than all to one side of a boundary.
#
# The notebook demos/strategy_characterisation/humans/intermediate_strategy_characterisation.ipynb holds the
# plots and the written-up results. Everything that computes a number lives here.

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.mixture import GaussianMixture

import analysis.human_characterisation as hc


CLIP_EPS = 0.005            # centrality is clipped to [eps, 1 - eps] for the beta shapes (alcove corners sit just below 0)
SPAWN_CENTRALITY = 0.999    # a first trial at or above this is the spawn position at the arena centre
MIN_SHAPE_SD = 0.03         # narrowest a shape may be (stops one collapsing onto trials piled at one exact value)
MIN_EFF_TRIALS = 5          # fewest effective trials (summed posteriors) a shape needs to give a position
COMPONENT_NAMES = {2: ('peripheral', 'centre'), 3: ('peripheral', 'middle', 'centre')}
LIKELIHOOD_LEVEL = 0.95     # coverage of the likelihood interval on a player's centre weight
SIM_KINDS = ('central', 'peripheral', 'mixer', 'switcher', 'hoverer', 'drifter')
SIM_SPREAD = 0.08           # SD of the hoverer's and drifter's trials around their (moving) position


# ---------------------------------------------------------------------------
# preparing trials
# ---------------------------------------------------------------------------

def clip_unit(x, eps=CLIP_EPS):
    ''' Return centrality clipped into [eps, 1 - eps], so a beta density is finite at every trial.
        Takes the values and eps. '''

    return np.clip(np.asarray(x, dtype=float), eps, 1 - eps)


def drop_spawn_trials(onset, level=SPAWN_CENTRALITY):
    ''' Return the onset table without first trials taken at the arena centre.
        Takes the onset table and the centrality that counts as the centre.

        Players spawn at the centre, and the walls of the first trial can appear before they move. That
        trial shows the spawn position, not a choice of where to wait. '''

    spawn = (onset['trial_pos'] == 0) & (onset['centrality'] >= level)
    return onset[~spawn].copy()


# ---------------------------------------------------------------------------
# beta mixture
# ---------------------------------------------------------------------------

def _moment_beta(mean, var):
    ''' Return beta (a, b) with the given mean and variance. Takes a mean and a variance. '''

    var = min(var, mean * (1 - mean) * 0.99)
    common = mean * (1 - mean) / var - 1
    return mean * common, (1 - mean) * common


def _beta_from(theta, sd_min):
    ''' Return beta (a, b) from an unbounded (mean, concentration) pair, with SD at least sd_min.
        Takes the pair and sd_min.

        The mean is the logistic of theta[0]. The concentration a + b is a logistic share of the largest
        concentration that still gives SD sd_min at that mean. '''

    mu = special.expit(theta[0])
    kappa_max = mu * (1 - mu) / sd_min ** 2 - 1
    kappa = kappa_max * special.expit(theta[1])
    return mu * kappa, (1 - mu) * kappa


def _weighted_beta_mle(lx, l1x, w, start, sd_min=MIN_SHAPE_SD):
    ''' Return the beta (a, b) that maximises the weighted log likelihood, with SD at least sd_min.
        Takes log(x), log(1 - x), the trial weights, a starting (a, b) and sd_min. '''

    sw, slx, sl1x = w.sum(), w @ lx, w @ l1x

    def nll(theta):
        a, b = _beta_from(theta, sd_min)
        return -((a - 1) * slx + (b - 1) * sl1x - sw * special.betaln(a, b))

    # start from the given shape, pulled inside the SD limit
    mu = np.clip(start[0] / (start[0] + start[1]), 1e-3, 1 - 1e-3)
    kappa_max = mu * (1 - mu) / sd_min ** 2 - 1
    share = np.clip((start[0] + start[1]) / kappa_max, 1e-6, 0.999)
    res = optimize.minimize(nll, [special.logit(mu), special.logit(share)], method='L-BFGS-B')
    return np.array(_beta_from(res.x, sd_min))


def _em_beta(x, resp, max_iter, tol):
    ''' Return a beta mixture fitted by EM from starting responsibilities, and its log likelihood.
        Takes clipped trials, an n x k responsibility matrix, the iteration cap and the tolerance. '''

    lx, l1x = np.log(x), np.log1p(-x)
    k = resp.shape[1]
    a, b = np.ones(k), np.ones(k)
    ll_old = -np.inf
    for it in range(max_iter):
        # update each shape from the trials it currently holds
        weights = resp.mean(axis=0)
        for j in range(k):
            w = resp[:, j]
            m = np.average(x, weights=w)
            start = (a[j], b[j]) if it else _moment_beta(m, np.average((x - m) ** 2, weights=w))
            a[j], b[j] = _weighted_beta_mle(lx, l1x, w, start)
        # reassign trials to the shapes in proportion to how likely each makes them
        logp = np.log(weights) + stats.beta.logpdf(x[:, None], a, b)
        ll = special.logsumexp(logp, axis=1)
        resp = np.exp(logp - ll[:, None])
        if ll.sum() - ll_old < tol * len(x):
            break
        ll_old = ll.sum()
    return dict(weights=resp.mean(axis=0), a=a.copy(), b=b.copy()), float(ll.sum())


def fit_beta_mixture(x, k=2, n_init=5, max_iter=1000, tol=1e-8, eps=CLIP_EPS, random_state=0):
    ''' Return a k-component beta mixture fitted to trial centrality, components ordered peripheral to centre.
        Takes the trials, k, the number of starts, the iteration cap, the tolerance, the clip eps and a seed.

        Each start splits the trials into k groups by value. The first uses equal-sized groups, the others
        random cut points. The start with the highest log likelihood is kept. '''

    x = clip_unit(x, eps)
    rng = np.random.default_rng(random_state)
    rank = np.argsort(np.argsort(x)) / len(x)
    cuts = [np.arange(1, k) / k] + [np.sort(rng.uniform(0.1, 0.9, k - 1)) for _ in range(n_init - 1)]
    best, best_ll = None, -np.inf
    for cut in cuts:
        # start each trial mostly in the group its value falls in
        resp = np.eye(k)[np.searchsorted(cut, rank, side='right')] * 0.9 + 0.1 / k
        fit, ll = _em_beta(x, resp, max_iter, tol)
        if ll > best_ll:
            best, best_ll = fit, ll
    # order the components by mean, lowest centrality first
    idx = np.argsort(best['a'] / (best['a'] + best['b']))
    return dict(k=k, weights=best['weights'][idx], a=best['a'][idx], b=best['b'][idx], loglik=best_ll,
                n=len(x), eps=eps, names=COMPONENT_NAMES.get(k, tuple(f'component {i + 1}' for i in range(k))))


def component_pdfs(x, fit):
    ''' Return the density of each fixed shape at each trial (n x k, or ... x n x k for stacked trials).
        Takes trial centrality and a fit. Weights are not applied. '''

    xc = clip_unit(x, fit.get('eps', CLIP_EPS))
    return stats.beta.pdf(xc[..., None], fit['a'], fit['b'])


def mixture_pdf(grid, fit):
    ''' Return the weighted density of each component and their total on a grid. Takes a grid and a fit. '''

    parts = component_pdfs(grid, fit) * fit['weights']
    return parts, parts.sum(axis=-1)


def posteriors(F, weights):
    ''' Return each trial's share in each shape, given the shape densities and weights.
        Takes an n x k density matrix (or stacked ... x n x k) and the weights (k or ... x k). '''

    p = F * np.asarray(weights)[..., None, :]
    return p / p.sum(axis=-1, keepdims=True)


def compare_models(x, ks=(2, 3), eps=CLIP_EPS, random_state=0):
    ''' Return the log likelihood and BIC of Gaussian and beta mixtures with each k, on the same trials.
        Takes the trials, the component counts, the clip eps and a seed.

        Both families have 3k - 1 parameters, so the log likelihoods compare directly. A Gaussian puts
        some density outside [0, 1], where no trial can be, which only lowers its likelihood. '''

    xc = clip_unit(x, eps)
    rows = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, n_init=5, random_state=random_state).fit(xc.reshape(-1, 1))
        beta = fit_beta_mixture(xc, k=k, eps=eps, random_state=random_state)
        for model, ll in ((f'gaussian k={k}', gmm.score(xc.reshape(-1, 1)) * len(xc)), (f'beta k={k}', beta['loglik'])):
            n_params = 3 * k - 1
            rows.append(dict(model=model, k=k, loglik=ll, n_params=n_params, bic=-2 * ll + n_params * np.log(len(xc))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# per-player weights and positions
# ---------------------------------------------------------------------------

def _centre_weight(F, n_iter=60):
    ''' Return the centre weight that maximises a 2-shape player's log likelihood.
        Takes an n x 2 density matrix, or stacked ... x n x 2.

        The log likelihood is concave in the weight, so Newton steps held inside [0, 1] reach the
        maximum. At an edge where the slope still points outward, the weight stays on the edge. '''

    diff = F[..., 1] - F[..., 0]
    pi = np.full(F.shape[:-2], 0.5)
    for _ in range(n_iter):
        mix = F[..., 0] + pi[..., None] * diff
        d1 = (diff / mix).sum(axis=-1)
        d2 = -((diff / mix) ** 2).sum(axis=-1)
        step = np.where(d2 < 0, -d1 / np.where(d2 < 0, d2, -1.0), 0.0)
        new = np.clip(pi + step, 0.0, 1.0)
        if np.max(np.abs(new - pi)) < 1e-12:
            pi = new
            break
        pi = new
    return pi


def player_weights(F, max_iter=2000, tol=1e-10):
    ''' Return a player's weight on each fixed shape by maximum likelihood, and the trial posteriors.
        Takes an n x k density matrix, or stacked ... x n x k for many players or redraws at once.

        Each step sets the weights to the mean posterior of the player's trials, computed with the
        player's own current weights. It converges on the weights that make the player's trials most
        likely. Using the population weights instead would pull every player toward the population share.
        With 2 shapes the same maximum is found directly (_centre_weight), which is much faster. '''

    F = np.asarray(F, dtype=float)
    k = F.shape[-1]
    if k == 2:
        pi = _centre_weight(F)
        w = np.stack([1 - pi, pi], axis=-1)
        return w, posteriors(F, w)
    w = np.full(F.shape[:-2] + (k,), 1 / k)
    for _ in range(max_iter):
        r = posteriors(F, w)
        new = r.mean(axis=-2)
        done = np.max(np.abs(new - w)) < tol
        w = new
        if done:
            break
    return w, posteriors(F, w)


def likelihood_interval(F, level=LIKELIHOOD_LEVEL, n_grid=4001):
    ''' Return the likelihood interval on the centre weight of a 2-shape player.
        Takes the n x 2 density matrix (peripheral, centre) and the coverage.

        The interval holds every weight whose log likelihood is within chi2(1) / 2 of the best. It
        treats trials as independent, so runs of one position make it too narrow. '''

    grid = np.linspace(0, 1, n_grid)
    with np.errstate(divide='ignore'):
        ll = np.log(np.outer(1 - grid, F[:, 0]) + np.outer(grid, F[:, 1])).sum(axis=1)
    keep = grid[ll >= ll.max() - stats.chi2.ppf(level, 1) / 2]
    return float(keep.min()), float(keep.max())


def player_mixture_table(onset, fit, n_boot=hc.DRIFT_PERM, block=hc.DRIFT_BLOCK, min_eff=MIN_EFF_TRIALS,
                         random_state=0):
    ''' Return, per player per session type, the weight on each fixed shape, the effective trials and the
        soft position in each, and for 2 shapes an interval on the centre weight two ways.
        Takes the onset table, a fit, the redraw count, the block length, the fewest effective trials
        and a seed.

        A soft position is the mean centrality of the player's trials, each weighted by its posterior for
        that shape. A shape whose summed posterior is below min_eff gets no position.

        For 2 shapes, pi_lo / pi_hi is the likelihood interval (independent trials), and pi_boot_lo /
        pi_boot_hi the 95% interval over block redraws in play order (keeps runs, as in
        hc.gap_ratio_intervals). n_boot = 0 skips the redraws. '''

    names = fit['names']
    rng = np.random.default_rng(random_state)
    rows = []
    for key, grp in onset.sort_values('trial_pos').groupby(list(hc.KEYS)):
        x = grp['centrality'].to_numpy()
        F = component_pdfs(x, fit)
        w, r = player_weights(F)
        row = dict(zip(hc.KEYS, key), n_trials=len(x))
        for j, name in enumerate(names):
            # take the weight, the effective trials and the posterior-weighted position of each shape
            eff = r[:, j].sum()
            row[f'w_{name}'] = w[j]
            row[f'eff_{name}'] = eff
            row[f'pos_{name}'] = (r[:, j] @ x) / eff if eff >= min_eff else np.nan
        if fit['k'] == 2:
            row['pi_lo'], row['pi_hi'] = likelihood_interval(F)
            if n_boot:
                # redraw the session in blocks and refit the weights on every redraw at once
                idx = hc._block_bootstrap_index(len(x), block, n_boot, rng)
                w_boot, _ = player_weights(F[idx])
                row['pi_boot_lo'], row['pi_boot_hi'] = np.percentile(w_boot[:, 1], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# simulated players
# ---------------------------------------------------------------------------

def _beta_draws(mean, sd, rng):
    ''' Return one beta draw per mean, each with the given SD. Takes the means, the SD and a generator. '''

    mean = np.clip(np.asarray(mean, dtype=float), 0.02, 0.98)
    a, b = np.transpose([_moment_beta(m, sd ** 2) for m in mean])
    return rng.beta(a, b)


def simulate_players(fit, n_trials, n_players, boundary, rng, kinds=SIM_KINDS, spread=SIM_SPREAD):
    ''' Return simulated onset trials for players with known behaviour, and the truth for each kind.
        Takes a 2-shape fit, trials per player, players per kind, the trial-level boundary, a generator,
        the kinds and the SD of the hoverer's and drifter's trials.

        central / peripheral draw 95% / 5% of trials from the fitted centre shape, the rest from the peripheral
        shape. mixer draws half from each, in random order. switcher draws the first half from the peripheral
        shape and the second half from the centre shape. hoverer waits around the boundary. drifter's
        position moves steadily from the peripheral shape's mean to the centre shape's mean. hoverer and drifter
        follow neither shape, so they have no true centre weight. '''

    peripheral_ab, centre_ab = (fit['a'][0], fit['b'][0]), (fit['a'][1], fit['b'][1])
    means = fit['a'] / (fit['a'] + fit['b'])
    true_pi = dict(central=0.95, peripheral=0.05, mixer=0.5, switcher=0.5, hoverer=np.nan, drifter=np.nan)
    rows = []
    for e, kind in enumerate(kinds):
        for p in range(n_players):
            t = np.arange(n_trials)
            # draw where each trial is waited, by kind
            if kind in ('central', 'peripheral', 'mixer'):
                centre = rng.random(n_trials) < true_pi[kind]
            elif kind == 'switcher':
                centre = t >= n_trials // 2
            if kind in ('central', 'peripheral', 'mixer', 'switcher'):
                x = np.where(centre, rng.beta(*centre_ab, n_trials), rng.beta(*peripheral_ab, n_trials))
            elif kind == 'hoverer':
                x = _beta_draws(np.full(n_trials, boundary), spread, rng)
            else:
                x = _beta_draws(np.linspace(means[0], means[1], n_trials), spread, rng)
            rows.append(pd.DataFrame(dict(experiment=e, player=p, session_type=f'sim{n_trials}', kind=kind,
                                          trial_pos=t, centrality=x)))
    truth = pd.DataFrame(dict(kind=list(kinds), true_pi=[true_pi[k] for k in kinds]))
    return pd.concat(rows, ignore_index=True), truth


def simulation_summary(table, truth):
    ''' Return, per simulated kind, the median centre weight and how well each interval holds the truth.
        Takes the player mixture table with a kind column and the truth table from simulate_players.

        Coverage is the share of players whose interval holds the true weight, which should be about
        0.95. Kinds with no true weight (hoverer, drifter) get NaN coverage. '''

    t = table.merge(truth, on='kind')
    rows = []
    for kind, g in t.groupby('kind', sort=False):
        true = g['true_pi'].iloc[0]
        row = dict(kind=kind, players=len(g), true_pi=true, w_centre=g['w_centre'].median(),
                   w_centre_q25=g['w_centre'].quantile(0.25), w_centre_q75=g['w_centre'].quantile(0.75))
        for name, lo, hi in (('likelihood', 'pi_lo', 'pi_hi'), ('block', 'pi_boot_lo', 'pi_boot_hi')):
            row[f'{name}_width'] = (g[hi] - g[lo]).median()
            row[f'{name}_coverage'] = ((g[lo] <= true) & (true <= g[hi])).mean() if true == true else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ternary_xy(w):
    ''' Return 2-D plotting positions for 3-shape weights (peripheral at bottom left, middle at the top,
        centre at bottom right). Takes an n x 3 weight array (peripheral, middle, centre). '''

    w = np.asarray(w, dtype=float)
    return np.column_stack([w[:, 2] + w[:, 1] / 2, w[:, 1] * np.sqrt(3) / 2])

