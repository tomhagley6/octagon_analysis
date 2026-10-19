''' Estimate how a model would fare racing a different model, from self-play logs alone.

In self-play both agents run the same model, so a model is only ever raced against itself.
A peripheral model in self-play races another peripheral agent, which is also waiting in an
alcove, so its win rate and score rate say little about how it would do against a central
model. This module builds a per-agent time to the wall from where the agent stands at slice
onset, then races those times against another model's self-play trial times.

Method, for a focal model F and an opponent O:
  1. For every trial and agent, take the straight-line distance from its slice-onset position
     to the centre of each lit wall's alcove.
  2. Fit each model's trial time against the distance to the wall it reached, on the trials it
     won (trial time is only known for the winner). Trial time is slice onset to trigger.
  3. Predict a time for every agent, winner and loser alike, to a target wall. The default
     ('p_high') sends each agent to High with the model's own self-play p_high and to Low
     otherwise, drawn at random per agent and trial. The alternatives are always High ('high'),
     the nearer lit wall ('nearest'), or the wall it took on won trials and the nearer wall
     otherwise ('observed'). Add a residual drawn from the model's own won trials, so the
     trial-to-trial spread in timing is kept.
  4. Pair every predicted F time with every predicted O time. The faster side wins. A won trial
     scores 1.0 at High and 0.4 at Low, a lost one 0.

Both sides are predicted the same way. Taking O's self-play trial time instead would use the
faster of O's two agents, and underrates F (by 0.10 in win share on the tournament check).

Checked against the 260715 and 260914 tournaments (76 ordered pairs), with the residual added.
Target 'p_high': win share r = 0.97, mean error 0.031; score per trial r = 0.97, mean error 0.030.
Target 'high' gets win share as well but overstates score (mean error 0.066), as it scores every
win at 1.0. Target 'nearest' overrates peripheral models against central ones by about 0.23, as
these models take High far more often than the nearer wall.

Assumptions (stated in the notebook that uses this):
  - F's time function is fitted on won trials, where F may have moved better than average.
  - With 'p_high', the wall choice is drawn independently of where the agent stands. p_high is
    the share of self-play wins at High.
  - Neither model changes its behaviour because the opponent is different.
  - The two agents never block each other, as each is timed alone.
'''

import contextlib
import io

import numpy as np
import pandas as pd

import globals
import data_extraction.get_indices as get_indices
import parse_data.prepare_sim_data as prepare_sim_data
import plotting.plot_octagon as plot_octagon

HIGH_SCORE, LOW_SCORE = 1.0, 0.4
TARGETS = ('p_high', 'high', 'nearest', 'observed')

# alcove centre of each wall, one row per wall (row w - 1 holds wall w, wall 1 faces north)
ALCOVE_CENTRES = plot_octagon.return_alcove_centre_points().T


def trial_agent_rows(trial, players=(0, 1)):
    ''' Return one dict per agent for a trial, with its slice-onset position, its distance to
        each lit wall, and the trial outcome. Takes a trial dataframe and the agent ids. '''

    # get the lit walls (wall1 = High, wall2 = Low) and the trigger the server selected
    trig = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    if len(trig) == 0 or len(onset) == 0:
        return []
    high, low = get_indices.get_walls(trial)
    winner = get_indices.get_trigger_activator(trial)
    chosen = int(get_indices.get_chosen_walls([trial])[0])
    rt_s = (trig['timeApplication'].iloc[0] - onset['timeApplication'].iloc[0]) / np.timedelta64(1, 's')
    trial_num = float(trial[globals.TRIAL_NUM].dropna().iloc[0])

    # take the winner's position at the trigger, to check the wall numbering against the arena
    k = globals.PLAYER_LOC_DICT[winner]
    end = np.array([trig[k['xloc']].iloc[0], trig[k['yloc']].iloc[0]], dtype=float)
    end_wall = int(np.argmin(np.linalg.norm(ALCOVE_CENTRES - end, axis=1)) + 1)

    rows = []
    for pid in players:
        # take the straight-line distance from the onset position to each lit alcove
        pos = np.array(get_indices.get_player_slice_onset_loc(trial, pid), dtype=float)
        rows.append(dict(trial_num=trial_num, player=pid, x=pos[0], z=pos[1],
                         high_wall=high, low_wall=low,
                         d_high=float(np.linalg.norm(pos - ALCOVE_CENTRES[high - 1])),
                         d_low=float(np.linalg.norm(pos - ALCOVE_CENTRES[low - 1])),
                         won=(pid == winner), chosen_wall=chosen, chose_high=(chosen == high),
                         rt_s=float(rt_s), winner_end_wall=end_wall))
    return rows


def collect_race_table(sim_data_folder, models, verbose=True):
    ''' Return one row per trial x agent for each model, for the race estimate.
        Takes a batch's simulation folder and the model list.

        Drops the session's first trial, as every other analysis does. '''

    records = []
    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            _, trial_list = prepare_sim_data.prepare_single_model_data(model, sim_data_folder=sim_data_folder)
        for trial in trial_list[1:]:
            for r in trial_agent_rows(trial):
                r['model'] = model
                records.append(r)
        if verbose:
            print(f"model {model} done", end='  ')
    race = pd.DataFrame(records)
    # take the distance to the wall each winner reached, and to the nearer lit wall
    race['d_chosen'] = np.where(race['chose_high'], race['d_high'], race['d_low'])
    race['d_nearest'] = race[['d_high', 'd_low']].min(axis=1)
    return race


def fit_time_distance(race_model):
    ''' Return the slope, intercept and correlation of trial time against distance to the
        reached wall, on the won trials. Takes one model's rows of the race table. '''

    won = race_model[race_model['won']]
    slope, intercept = np.polyfit(won['d_chosen'], won['rt_s'], 1)
    r = np.corrcoef(won['d_chosen'], won['rt_s'])[0, 1]
    return dict(slope=float(slope), intercept=float(intercept), r=float(r),
                min_rt=float(won['rt_s'].min()), n=len(won))


def target_distance(race_model, target='p_high', rng=None, p_high=None):
    ''' Return each agent's distance to its target wall, and whether that wall is High.
        Takes one model's rows of the race table, the target rule (see TARGETS), a numpy random
        generator, and an optional p_high.

        For 'p_high', each agent heads for High with probability p_high, drawn independently per
        agent and trial. p_high defaults to the share of the model's won trials taken at High. '''

    if target == 'p_high':
        rng = np.random.default_rng(0) if rng is None else rng
        if p_high is None:
            p_high = race_model.loc[race_model['won'], 'chose_high'].mean()
        to_high = pd.Series(rng.random(len(race_model)) < p_high, index=race_model.index)
    elif target == 'nearest':
        to_high = race_model['d_high'] <= race_model['d_low']
    elif target == 'high':
        to_high = pd.Series(True, index=race_model.index)
    elif target == 'observed':
        # take the wall actually reached when the agent won, and the nearer wall when it lost
        to_high = race_model['chose_high'].where(race_model['won'],
                                                 race_model['d_high'] <= race_model['d_low'])
    else:
        raise ValueError(f"target must be one of {TARGETS}")
    d = np.where(to_high, race_model['d_high'], race_model['d_low'])
    return d.astype(float), to_high.to_numpy(bool)


def predict_times(race_model, fit, target='p_high', noise=False, rng=None):
    ''' Return each agent's predicted time to its target wall, and whether that wall is High.
        Takes one model's rows of the race table, its time fit, the target rule, whether to add
        timing noise, and a numpy random generator.

        Without noise, times are floored at the fastest trial the model actually won, as a
        straight-line fit runs below zero for an agent already standing at the wall. With noise,
        a residual from one of the model's own won trials is added to each time, floored at 0. '''

    rng = np.random.default_rng(0) if rng is None else rng
    d, to_high = target_distance(race_model, target, rng)
    t = fit['intercept'] + fit['slope'] * d
    if not noise:
        return np.maximum(t, fit['min_rt']), to_high

    # draw residuals from the won trials, where the time and the wall reached are both known
    won = race_model[race_model['won']]
    resid = (won['rt_s'] - (fit['intercept'] + fit['slope'] * won['d_chosen'])).to_numpy()
    return np.maximum(t + rng.choice(resid, len(t)), 0.0), to_high


def wall_score(to_high):
    ''' Return the score for reaching each target wall. Takes the High flags. '''

    return np.where(np.asarray(to_high, bool), HIGH_SCORE, LOW_SCORE)


def race(t_focal, high_focal, t_opp, score_opp):
    ''' Return the focal model's estimated win share, score per trial and score per second
        against an opponent, with the opponent's matching numbers. Takes the focal predicted
        times and High flags, and the opponent's trial times and winning scores.

        Every focal time is paired with every opponent trial. The faster side wins, and a tie
        is split evenly. Score per second is total score over total trial time, where a trial
        lasts until the faster side arrives. '''

    t_f = np.asarray(t_focal, float)[:, None]
    t_o = np.asarray(t_opp, float)[None, :]
    # count a win for the faster side, splitting ties
    win_f = (t_f < t_o) + 0.5 * (t_f == t_o)
    score_f = win_f * wall_score(high_focal)[:, None]
    score_o = (1 - win_f) * np.asarray(score_opp, float)[None, :]
    trial_s = np.minimum(t_f, t_o)
    return dict(win_share=float(win_f.mean()),
                score_per_trial=float(score_f.mean()), opp_score_per_trial=float(score_o.mean()),
                score_rate=float(score_f.sum() / trial_s.sum()),
                opp_score_rate=float(score_o.sum() / trial_s.sum()))


def selfplay_winner_check(race_model, fit, target='nearest'):
    ''' Return the share of self-play trials where the agent with the shorter predicted time
        is the one that actually won. Takes one model's rows of the race table and its fit.

        This tests the time prediction on the losing agent, which the fit never saw. The default
        target is the nearer wall, as the wall a winner took is itself part of the outcome. '''

    t, _ = predict_times(race_model, fit, target)
    rm = race_model.assign(t_pred=t)
    wide = rm.pivot_table(index='trial_num', columns='player', values=['t_pred', 'won'], aggfunc='first')
    pred_p0 = wide[('t_pred', 0)] < wide[('t_pred', 1)]
    return float((pred_p0 == wide[('won', 0)].astype(bool)).mean())


def estimate_matchup(race_f, race_o, target='p_high', noise=True, seed=0):
    ''' Return the estimated race outcome of model F against model O (see race).
        Takes each model's rows of the race table, the target rule, whether to add timing
        noise, and a random seed.

        Both sides get predicted single-agent times, from their own time fits. '''

    rng = np.random.default_rng(seed)
    t_f, high_f = predict_times(race_f, fit_time_distance(race_f), target, noise, rng)
    t_o, high_o = predict_times(race_o, fit_time_distance(race_o), target, noise, rng)
    return race(t_f, high_f, t_o, wall_score(high_o))
