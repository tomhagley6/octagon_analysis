#!/usr/bin/env python
# coding: utf-8

# Analyse a cross-model tournament: which trained models win in competition.
#
# A tournament folder holds one subfolder per matchup, named `<runA>__vs__<runB>`, where
# model A drives agent 0 (P1) and model B drives agent 1 (P2). Each matchup contains one
# session JSON. This module reduces that to a per-trial table, then to per-model rankings.
#
# Two outcome measures are carried side by side:
#   score  - terminal outcome, winner-take-all: 1.0 for High, 0.4 for Low, loser 0.
#   reward - score minus step_penalty x episode steps, i.e. score net of movement cost.
# Whether reward is meaningful depends on the BUILD the tournament was run with: if the
# no-movement penalty break is absent, resting is penalised too and (episodes being fixed
# length and symmetric between competitors) reward is score minus a near-constant, which
# reorders nothing. Check the build before reading anything into a reward ranking.
#
# The notebooks in demos/tournament/ hold the per-batch config (which folder, which
# entrants, their strategy labels) and the written-up results. Everything that computes a
# number or draws a figure lives here, so a new batch is a new config cell, not new code.

import contextlib
import io
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

import globals
import parse_data.loading as loading
import parse_data.preprocess_sim as preprocess_sim


HIGH, LOW = 1.0, 0.4                # winner-take-all trial scores
N_WALLS = 8                         # octagon; wall indices 1..8, so separations wrap at 8

STRAT_ORDER = ['central', 'intermediate', 'peripheral']
STRAT_PALETTE = {'central': '#4C72B0', 'intermediate': '#55A868', 'peripheral': '#DD8452'}

MATCHUP_SEP = '__vs__'              # matchup folder name separator


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def model_ids(folder):
    ''' Return (model_A, model_B) as ints from a matchup folder name.
        Takes a folder like "260715_0022__vs__260715_0002". A drives agent 0 (P1),
        B drives agent 1 (P2). '''

    a, b = folder.split(MATCHUP_SEP)
    return int(a.split('_')[1]), int(b.split('_')[1])


def matchup_folders(base):
    ''' Return the sorted matchup folder names in a tournament dir. Takes the dir. '''

    return sorted(d for d in os.listdir(base)
                  if MATCHUP_SEP in d and os.path.isdir(os.path.join(base, d)))


def extract_matchup(folder, base):
    ''' Return a per-trial dataframe for one matchup. Takes a matchup folder name and the
        tournament dir. One row per completed trial, with both models, the winning client,
        each model's score and reward, and the wall separation.

        Winner and walls come from the server-selected trigger event, the authoritative
        per-agent score/reward from the trial-end event; the two event streams are merged
        on trialNum, which is robust to the usual off-by-one between them. '''

    d = os.path.join(base, folder)
    # first filename that fits: the session log, not a macOS resource fork or a config
    fn = next(f for f in os.listdir(d)
              if f.endswith('.json') and not f.startswith('._') and 'config' not in f)
    # redirect_stdout mutes the pipeline's per-file prints so a loop over 64 matchups stays readable
    with contextlib.redirect_stdout(io.StringIO()):
        df = loading.loading_pipeline(d, fn)
        df = preprocess_sim.standard_preprocessing_sim(df)

    TN = globals.TRIAL_NUM
    trig = df[df['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION][
        [TN, globals.TRIGGER_CLIENT, globals.WALL_1, globals.WALL_2]].dropna(subset=[TN])

    # reward is logged by newer builds only; take it when present, NaN when not
    reward_cols = [globals.PLAYER_0_TRIAL_REWARD, globals.PLAYER_1_TRIAL_REWARD]
    have_reward = all(c in df.columns for c in reward_cols)
    te_cols = [TN, globals.PLAYER_0_TRIAL_SCORE, globals.PLAYER_1_TRIAL_SCORE]
    te = df[df['eventDescription'] == globals.TRIAL_END][
        te_cols + (reward_cols if have_reward else [])].dropna(subset=[TN])

    m = trig.merge(te, on=TN, how='inner')                  # align the two streams by trial number

    a, b = model_ids(folder)
    w1 = m[globals.WALL_1].to_numpy(float)
    w2 = m[globals.WALL_2].to_numpy(float)
    # walls are numbered 1..8, so |w1 - w2| is already 0..7; the circular wrap (walls 1 and
    # 8 are adjacent) is closed by taking min(gap, 8 - gap) below
    dd = np.abs(w1 - w2)
    out = pd.DataFrame(dict(
        matchup=folder, model_A=a, model_B=b, trial_num=m[TN].to_numpy(float),
        winner_client=m[globals.TRIGGER_CLIENT].to_numpy(float),
        score_A=m[globals.PLAYER_0_TRIAL_SCORE].astype(float).to_numpy(),
        score_B=m[globals.PLAYER_1_TRIAL_SCORE].astype(float).to_numpy(),
        sep=np.minimum(dd, N_WALLS - dd).astype(int)))      # 1/2/4 = 45/90/180 deg
    out['reward_A'] = m[globals.PLAYER_0_TRIAL_REWARD].astype(float).to_numpy() if have_reward else np.nan
    out['reward_B'] = m[globals.PLAYER_1_TRIAL_REWARD].astype(float).to_numpy() if have_reward else np.nan
    return out


def load_tournament(base, folders=None, verbose=True):
    ''' Return one long per-trial dataframe over every matchup in a tournament dir.
        Takes the dir and an optional folder list (default: all matchups present).

        Also reports which matchups are missing DONE.txt (the run harness writes it on a
        clean finish) and which failed to parse; a matchup that fails is left out of the
        table rather than killing the pass. '''

    if folders is None:
        folders = matchup_folders(base)

    records, failed, no_done = [], [], []
    for folder in folders:
        if not os.path.exists(os.path.join(base, folder, 'DONE.txt')):
            no_done.append(folder)
        try:
            records.append(extract_matchup(folder, base))
            if verbose:
                print(f"{folder} done", end='  ')
        except Exception as e:
            failed.append((folder, f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"\n{folder} FAILED ({type(e).__name__})")

    trials_df = pd.concat(records, ignore_index=True)
    if verbose:
        print(f"\n\nmatchups loaded: {len(records)} / {len(folders)}   trials: {len(trials_df)}")
        if no_done:
            print(f"MISSING DONE.txt ({len(no_done)}): {no_done}")
        if failed:
            print(f"FAILED: {failed}")
    return trials_df


def per_matchup_summary(trials_df, verbose=True):
    ''' Return one row per matchup (ordered A=P1 vs B=P2): trial count, P1 win rate and
        each model's mean score and reward. Takes the per-trial table.

        Uses pandas named aggregation: .groupby(keys).agg(out=(col, func)) gives one row per
        group and one column per out. model_A/model_B are constant within a matchup so adding
        them does NOT change the grouping — it just carries the ids through to the output. '''

    per_matchup = trials_df.groupby(['matchup', 'model_A', 'model_B']).agg(
        n=('winner_client', 'size'),                                        # trials in the matchup
        p1_win_rate=('winner_client', lambda s: float(np.mean(s == 0))),    # fraction won by P1 (client 0)
        mean_score_A=('score_A', 'mean'),
        mean_score_B=('score_B', 'mean'),
        mean_reward_A=('reward_A', 'mean'),
        mean_reward_B=('reward_B', 'mean')).reset_index()
    if verbose:
        print(f"trials per matchup: min {per_matchup.n.min()}, max {per_matchup.n.max()}")
    return per_matchup


def perspective_table(trials_df, drop_self=True):
    ''' Return one row per (focus model, opponent) per trial, so both seat orders pool.
        Takes the per-trial table; drop_self removes self-matchups, which carry no
        competitive information (a model cannot out-rank itself).

        Every trial appears twice, once framed from each model's side, so this table must
        always be indexed by `focus` — aggregating it without grouping double-counts. '''

    perspective = pd.concat([
        trials_df.assign(focus=trials_df.model_A, opp=trials_df.model_B,
                         f_score=trials_df.score_A, f_reward=trials_df.reward_A,
                         f_win=(trials_df.winner_client == 0).astype(float)),   # bool -> float
        trials_df.assign(focus=trials_df.model_B, opp=trials_df.model_A,
                         f_score=trials_df.score_B, f_reward=trials_df.reward_B,
                         f_win=(trials_df.winner_client == 1).astype(float)),
    ], ignore_index=True)[['matchup', 'focus', 'opp', 'f_score', 'f_reward', 'f_win', 'sep']]
    return perspective[perspective.focus != perspective.opp] if drop_self else perspective


def models_in_order(strategy, order=STRAT_ORDER):
    ''' Return the model ids grouped by strategy band, for matrix and plot ordering.
        Takes a {model_id: strategy} dict. Within a band the caller's dict order is kept,
        so a hand-ordered shortlist (e.g. most central first) survives into the figures. '''

    unknown = sorted({v for v in strategy.values()} - set(order))
    if unknown:
        raise ValueError(f"strategy labels not in order={list(order)}: {unknown}. "
                         "Models with an unlisted band would be silently dropped from every "
                         "matrix and figure, so add the band to `order` (and to the palette).")
    return [m for s in order for m in strategy if strategy[m] == s]


# ---------------------------------------------------------------------------
# validation: is the tournament fair?
# ---------------------------------------------------------------------------

def validation_report(trials_df, per_matchup, models, verbose=True):
    ''' Return the fairness checks for a tournament, as a dict. Takes the per-trial table,
        the per-matchup summary and the entrant list.

        Trust no ranking until these pass:
          - self-matchups (a model against a copy of itself) sit near 0.5. Far from it means
            a seat or agent asymmetry — the failure mode of the first run, where the
            opponent never moved and P1 won every trial.
          - mirror consistency: a model's win rate against an opponent is the same whether
            it played P1 or P2. Large gaps mean the seat still matters.
          - winner-take-all: exactly one agent scores each trial.

        Score is safe to use here even when the build's reward structure is off, because
        these comparisons are self- and mirror-matchups, where score proportions cannot
        differ by construction. Mirror checks are skipped if the batch has no mirrors. '''

    self_mask = per_matchup['model_A'] == per_matchup['model_B']
    self_tbl = (per_matchup[self_mask][['model_A', 'n', 'p1_win_rate']]
                .rename(columns={'model_A': 'model'}))

    pm = per_matchup.set_index(['model_A', 'model_B'])
    rows = []
    for a, b in combinations(models, 2):
        if (a, b) not in pm.index or (b, a) not in pm.index:
            continue                                    # pair not run in both seat orders
        ab = pm.loc[(a, b), 'p1_win_rate']
        ba = pm.loc[(b, a), 'p1_win_rate']
        rows.append(dict(pair=f"{a} vs {b}", A_asP1=ab, A_asP2=1 - ba, gap=abs(ab - (1 - ba))))
    mirror = pd.DataFrame(rows, columns=['pair', 'A_asP1', 'A_asP2', 'gap'])

    one_winner = float(((trials_df.score_A > 0) ^ (trials_df.score_B > 0)).mean())

    if verbose:
        if len(self_tbl):
            print("Self-matchup P1 win rate (should be ~0.5):")
            print(self_tbl.round(3).to_string(index=False))
            print(f"  mean {self_tbl.p1_win_rate.mean():.3f}, worst deviation from 0.5: "
                  f"{(self_tbl.p1_win_rate - 0.5).abs().max():.3f}")
        else:
            print("No self-matchups in this batch.")

        if len(mirror):
            print(f"\nMirror consistency over {len(mirror)} pairs — mean gap "
                  f"{mirror.gap.mean():.3f}, max {mirror.gap.max():.3f}")
            print(mirror.sort_values('gap', ascending=False).head(5).round(3).to_string(index=False))
        else:
            print("\nNo mirror matchups in this batch — seat effects cannot be checked.")

        print(f"\nwinner-take-all (exactly one agent scores per trial): {one_winner:.3f} of trials")
        if trials_df[['reward_A', 'reward_B']].notna().any().any():
            print(f"reward range: [{trials_df[['reward_A', 'reward_B']].min().min():.3f}, "
                  f"{trials_df[['reward_A', 'reward_B']].max().max():.3f}]  "
                  f"(negative = moved and lost)")
    return dict(self_matchups=self_tbl, mirror=mirror, one_winner=one_winner)


# ---------------------------------------------------------------------------
# head-to-head
# ---------------------------------------------------------------------------

def head_to_head(selfless, models):
    ''' Return the mirror-pooled head-to-head matrices, as a dict of DataFrames keyed
        score / win / reward. Takes the self-excluded perspective table and the model
        order to use for rows and columns.

        Each matrix reads row-model against column-opponent. Pooling both seat orders
        gives a seat-balanced estimate of how each model does against each opponent. '''

    h2h = selfless.groupby(['focus', 'opp']).agg(
        score=('f_score', 'mean'), win=('f_win', 'mean'), reward=('f_reward', 'mean')).reset_index()
    return {k: h2h.pivot(index='focus', columns='opp', values=k).reindex(index=models, columns=models)
            for k in ('score', 'win', 'reward')}


def plot_head_to_head(mats, figsize=(15, 6)):
    ''' Plot the mean-score and win-rate head-to-head heatmaps side by side.
        Takes the dict from head_to_head. Returns (fig, axes). '''

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, mat, title, vmin, vmax, cmap in [
            (axes[0], mats['score'], 'Mean player score (row vs column)', 0, HIGH, 'viridis'),
            (axes[1], mats['win'], 'Win rate (row vs column)', 0, 1, 'RdBu_r')]:
        sns.heatmap(mat, annot=True, fmt='.2f', cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
                    cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='white')
        ax.set_title(title)
        ax.set_xlabel('opponent')
        ax.set_ylabel('model')
    plt.tight_layout()
    return fig, axes


def plot_reward_head_to_head(mats, figsize=(8, 6.5)):
    ''' Plot the mean-reward head-to-head heatmap. Takes the dict from head_to_head.
        Diverging around 0, since a model that moves and loses ends a trial negative.
        Returns (fig, ax). '''

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mats['reward'], annot=True, fmt='+.2f', cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='white')
    ax.set_title('Mean reward (row model vs column opponent)\n'
                 '+ = net gain, - = moved and lost more than won')
    ax.set_xlabel('opponent')
    ax.set_ylabel('model')
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# overall ranking
# ---------------------------------------------------------------------------

def fit_bradley_terry(wins, prior=0.5, n_iter=2000, tol=1e-10):
    ''' Return Bradley-Terry strengths from a wins matrix. Takes wins[i, j] = times i beat j.
        A small prior (pseudo-wins each way) keeps lopsided pairs from diverging. Strengths
        are normalised to mean 1; P(i beats j) = p_i / (p_i + p_j). '''

    # add pseudo-wins each way so a 100%/0% pair cannot push a strength to 0 or infinity
    W = wins.astype(float) + prior
    np.fill_diagonal(W, 0.0)                            # no self-games
    n = W.shape[0]
    total_wins = W.sum(axis=1)                          # each model's total wins, which BT fits to
    games = W + W.T                                     # games[i, j] = contests between i and j
    p = np.ones(n)                                      # start every model at equal strength
    # iterative MLE (Zermelo / MM update): reset each strength to the value its wins imply given
    # the others' current strengths, and repeat until the strengths stop moving
    for _ in range(n_iter):
        p_old = p.copy()
        for i in range(n):
            # denominator = i's expected wins at current strengths (games weighted by win share)
            denom = np.sum([games[i, j] / (p[i] + p[j]) for j in range(n) if j != i])
            p[i] = total_wins[i] / denom if denom > 0 else p[i]   # actual / expected -> new strength
        p /= p.mean()                                   # strengths are defined only up to scale
        if np.max(np.abs(p - p_old)) < tol:             # converged
            break
    return p


def bradley_terry(selfless, models, **kwargs):
    ''' Return {model: Bradley-Terry strength} from pooled win counts.
        Takes the self-excluded perspective table and the model order. '''

    wins_df = selfless[selfless.f_win == 1].groupby(['focus', 'opp']).size().reset_index(name='w')
    wins_mat = wins_df.pivot(index='focus', columns='opp', values='w').reindex(
        index=models, columns=models).fillna(0).to_numpy()
    return dict(zip(models, fit_bradley_terry(wins_mat, **kwargs)))


def overall_ranking(selfless, strategy, models=None, sort_by='mean_score', verbose=True):
    ''' Return one row per model: mean score, total score, mean reward, win rate, trial
        count, Bradley-Terry strength, strategy band and rank. Takes the self-excluded
        perspective table, a {model: strategy} dict, and the metric to rank on
        ('mean_score' or 'mean_reward').

        Self-matchups are already excluded, so every number is competitive performance
        against other models. Ranking on score credits winning; ranking on reward credits
        winning cheaply — see the module docstring on when that distinction is real. '''

    models = models or models_in_order(strategy)
    overall = selfless.groupby('focus').agg(
        mean_score=('f_score', 'mean'),
        total_score=('f_score', 'sum'),
        mean_reward=('f_reward', 'mean'),
        win_rate=('f_win', 'mean'),
        n_trials=('f_score', 'size')).reset_index().rename(columns={'focus': 'model'})

    overall['bt_strength'] = overall['model'].map(bradley_terry(selfless, models))
    overall['strategy'] = overall['model'].map(strategy)
    overall = overall.sort_values(sort_by, ascending=False).reset_index(drop=True)
    overall['rank'] = np.arange(1, len(overall) + 1)

    if verbose:
        cols = ['rank', 'model', 'strategy', 'mean_score', 'total_score', 'win_rate', 'bt_strength']
        if sort_by == 'mean_reward':
            cols = ['rank', 'model', 'strategy', 'mean_reward', 'mean_score', 'win_rate', 'bt_strength']
        print(f"Overall ranking by {sort_by.replace('_', ' ')}:")
        print(overall[cols].round(3).to_string(index=False))
    return overall


def plot_overall_ranking(overall, metric='mean_score', palette=STRAT_PALETTE,
                         strat_order=STRAT_ORDER, with_bt=True, figsize=None):
    ''' Plot each model's overall success as a horizontal bar coloured by strategy, and
        (with_bt) its metric against Bradley-Terry strength, to show the two broadly agree.
        Takes the overall table from overall_ranking. Returns (fig, axes). '''

    n_ax = 2 if with_bt else 1
    fig, axes = plt.subplots(1, n_ax, figsize=figsize or ((14, 4.6) if with_bt else (8, 4.6)))
    axes = np.atleast_1d(axes)

    order = overall.sort_values(metric)
    label = 'mean player score / trial' if metric == 'mean_score' else 'mean reward / trial (net of movement cost)'
    axes[0].barh(np.arange(len(order)), order[metric],
                 color=[palette[s] for s in order['strategy']], edgecolor='black', linewidth=0.4)
    if metric == 'mean_reward':
        axes[0].axvline(0, color='k', lw=0.8)           # reward crosses zero; score cannot
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels(order['model'])
    axes[0].set_xlabel(label)
    axes[0].set_title(f"Overall success ({metric.replace('_', ' ')})")
    axes[0].legend(handles=[Patch(facecolor=palette[s], label=s) for s in strat_order
                            if s in set(overall['strategy'])], fontsize=8)

    if with_bt:
        for s in strat_order:
            sub = overall[overall.strategy == s]
            axes[1].scatter(sub['bt_strength'], sub[metric], s=80, color=palette[s],
                            edgecolor='black', label=s)
        for _, r in overall.iterrows():
            axes[1].annotate(int(r['model']), (r['bt_strength'], r[metric]), fontsize=8,
                             xytext=(3, 3), textcoords='offset points')
        axes[1].set_xlabel('Bradley-Terry strength')
        axes[1].set_ylabel(label)
        axes[1].set_title(f"{metric.replace('_', ' ')} vs BT strength")
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# does the pairing matter? (interaction)
# ---------------------------------------------------------------------------

def intransitive_triads(win_mat, models):
    ''' Return the non-transitive triples (rock-paper-scissors loops) among models.
        Takes a win-rate matrix (win_mat.loc[i, j] = i's win rate vs j) and the model list.
        A triple is a loop when each member beats exactly one of the other two. '''

    loops = []
    for i, j, k in combinations(models, 3):
        # count how many of the other two each model beats; a transitive triad gives
        # counts {2, 1, 0}, a rock-paper-scissors loop gives {1, 1, 1}
        beats = {m: sum(win_mat.loc[m, o] > 0.5 for o in (i, j, k) if o != m) for m in (i, j, k)}
        if all(v == 1 for v in beats.values()):
            loops.append((i, j, k))
    return loops


def interaction(win_mat, bt_map, models, strategy=None, verbose=True):
    ''' Return the interaction residual matrix: actual minus Bradley-Terry-predicted win
        rate for every ordered pair. Takes the win-rate matrix, the BT strengths and the
        model order.

        If success were purely each model's general strength, every head-to-head would be
        predicted by the two strengths, and every residual would be ~0. Large residuals, or
        non-transitive loops, mean the specific pairing matters beyond overall strength. '''

    pred = pd.DataFrame(index=models, columns=models, dtype=float)
    for i in models:
        for j in models:
            pred.loc[i, j] = np.nan if i == j else bt_map[i] / (bt_map[i] + bt_map[j])
    resid = (win_mat - pred).astype(float)

    if verbose:
        vals = np.array([resid.loc[i, j] for i, j in combinations(models, 2)])
        print(f"Interaction (actual - BT-predicted win rate): RMS residual "
              f"{np.sqrt(np.nanmean(vals ** 2)):.3f}, max |residual| {np.nanmax(np.abs(vals)):.3f}")
        loops = intransitive_triads(win_mat, models)
        print(f"Non-transitive triples (rock-paper-scissors loops): "
              f"{len(loops)} of {len(list(combinations(models, 3)))}")
        for t in loops:
            bands = (f"  (strategies {strategy[t[0]]}/{strategy[t[1]]}/{strategy[t[2]]})"
                     if strategy else "")
            print(f"  loop: {t[0]} > {t[1]} > {t[2]} > {t[0]}{bands}")
    return resid


def plot_interaction(resid, figsize=(7.5, 6)):
    ''' Plot the interaction residual heatmap. Takes the matrix from interaction.
        Returns (fig, ax). '''

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(resid, annot=True, fmt='+.2f', cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                ax=ax, cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='white')
    ax.set_title('Interaction: actual - BT-predicted win rate\n'
                 '(row vs column; far from 0 = pairing matters)')
    ax.set_xlabel('opponent')
    ax.set_ylabel('model')
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# score vs reward: does efficiency reorder the models?
# ---------------------------------------------------------------------------

def score_vs_reward(overall, verbose=True):
    ''' Return each model's rank under score and under reward, the shift between them, and
        the movement cost it paid (score minus reward). Takes the overall table.

        Score credits winning; reward credits winning cheaply. If lower movement cost
        matters, cheap models climb under reward. A rank_shift of 0 everywhere means the
        cost is a constant across models and reward adds nothing over score. '''

    cmp = overall[['model', 'strategy', 'mean_score', 'mean_reward']].copy()
    cmp['score_rank'] = cmp['mean_score'].rank(ascending=False).astype(int)
    cmp['reward_rank'] = cmp['mean_reward'].rank(ascending=False).astype(int)
    cmp['rank_shift'] = cmp['score_rank'] - cmp['reward_rank']      # + = climbs under reward
    cmp['cost'] = cmp['mean_score'] - cmp['mean_reward']            # movement cost paid
    cmp = cmp.sort_values('reward_rank').reset_index(drop=True)
    if verbose:
        print("Score vs reward (rank_shift > 0 = model climbs when judged on reward):")
        print(cmp.round(3).to_string(index=False))
    return cmp


def plot_score_vs_reward(cmp, palette=STRAT_PALETTE, strat_order=STRAT_ORDER, figsize=(14, 5)):
    ''' Plot mean score against mean reward per model (gap below the identity line is the
        movement cost), and the mean cost paid by each strategy band.
        Takes the table from score_vs_reward. Returns (fig, axes). '''

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for s in strat_order:
        sub = cmp[cmp.strategy == s]
        axes[0].scatter(sub['mean_score'], sub['mean_reward'], s=80, color=palette[s],
                        edgecolor='black', label=s)
    for _, r in cmp.iterrows():
        axes[0].annotate(int(r['model']), (r['mean_score'], r['mean_reward']), fontsize=8,
                         xytext=(3, 3), textcoords='offset points')
    lims = [min(cmp.mean_reward.min(), cmp.mean_score.min()) - 0.02, cmp.mean_score.max() + 0.02]
    axes[0].plot(lims, lims, 'k:', lw=1, label='score = reward')
    axes[0].set_xlabel('mean score')
    axes[0].set_ylabel('mean reward')
    axes[0].set_title('Score vs reward per model (gap below line = movement cost)')
    axes[0].legend(fontsize=8)

    cost_by = cmp.groupby('strategy')['cost'].mean().reindex(
        [s for s in strat_order if s in set(cmp['strategy'])])
    axes[1].bar(range(len(cost_by)), cost_by,
                color=[palette[s] for s in cost_by.index], edgecolor='black')
    axes[1].set_xticks(range(len(cost_by)))
    axes[1].set_xticklabels(cost_by.index)
    axes[1].set_ylabel('mean movement cost (score - reward)')
    axes[1].set_title('Cost paid by strategy')

    plt.tight_layout()
    return fig, axes
