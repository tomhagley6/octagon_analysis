''' Infer which lit wall a losing RL agent was heading for, only where that is clear.

A trial ends when either agent reaches a lit wall, so the loser's choice is never logged. This
module calls a loser's choice from the end of its trajectory, and makes no call when the evidence
is weak. Many losers are dropped, which is intended.

The human version (`analysis/loser_inferred_choice.py`) compares the movement direction over the
final third of the trajectory with the direction to every alcove centre. It accepts the most-aligned
wall when the mean cosine is above 0.875, or when the player ends within 4 units of it. Both rules are
too loose for agents that wait in alcoves. Leaving an alcove points an agent at the far walls, and an
agent passing in front of a lit alcove comes within 4 units of it on the way to the other wall.

Two rules here, tried in order:
  1. At the alcove. The agent's final position is past a lit wall's face (inside its alcove), within
     the alcove's width, and it has closed on that alcove over the last AT_ALCOVE_FRAMES frames.
  2. Aligned. Over the last END_WINDOW frames, the movement direction points at the
     same wall on every frame, with a mean cosine of at least COS_MIN to that wall and a mean lead of
     at least MARGIN over the next-best wall. The wall must be lit.
Anything else gets no call: too short (under END_WINDOW + SMOOTH_WINDOW + 1 frames), stopped, ambiguous,
or heading for an unlit wall.

The thresholds were set against winners' trajectories cut short (`truncation_check`), where the wall
reached is known. The notebook reports that check.

Positions are in reach space (as logged). The wall face is at OCTAGON_WALLFACE_RADIUS and the alcove
half-width ALCOVE_HALF_WIDTH, both reach values from globals.
'''

import contextlib
import io

import numpy as np
import pandas as pd

import globals
import trajectory_analysis.trajectory_direction as trajectory_direction
import trajectory_analysis.visibility_occlusion as vo
from analysis.race_estimate import ALCOVE_CENTRES

END_WINDOW = 25                     # frames (0.5 s) of movement read for the aligned rule
SMOOTH_WINDOW = 11                  # Savitzky-Golay window for the movement direction (trajectory_direction default)
COS_MIN = 0.94                      # mean cosine to the called wall (about 20 degrees)
MARGIN = 0.10                       # mean cosine lead over the next-best wall
MIN_FRAME_STEP = 0.04               # smoothed units moved per frame; below this the agent counts as stopped
AT_ALCOVE_FRAMES = 10               # frames over which an agent at an alcove must be closing on it

RULES = ('at alcove', 'aligned')
NO_CALL = ('too short', 'stopped', 'ambiguous', 'aligned to unlit wall')


def at_lit_alcove(xy, lit_walls, frames=AT_ALCOVE_FRAMES):
    ''' Return the lit wall whose alcove the agent ends inside while still closing on it, or None.
        Takes the trajectory (timepoints*2) and the lit wall ids. '''

    end = xy[-1:]
    for w in lit_walls:
        p, q = vo.arena_to_wall_frame(end, w - 1)
        closing = np.linalg.norm(xy[-frames - 1] - ALCOVE_CENTRES[w - 1]) > np.linalg.norm(xy[-1] - ALCOVE_CENTRES[w - 1])
        if p[0] >= globals.OCTAGON_WALLFACE_RADIUS and abs(q[0]) <= globals.ALCOVE_HALF_WIDTH and closing:
            return w
    return None


def alignment(xy, end_window=END_WINDOW, smooth_window=SMOOTH_WINDOW):
    ''' Return the cosine between movement direction and the direction to each alcove centre over the
        final frames (end_window*8), and the smoothed distance moved per frame. Takes the trajectory
        (timepoints*2).

        Uses trajectory_direction.cosine_similarity_throughout_trajectory (Savitzky-Golay smoothed movement
        direction, as in the human loser's-choice and visibility analyses), run over the final
        end_window + smooth_window frames so the smoothing has frames either side of the window. '''

    seg = xy[-(end_window + smooth_window):].T
    cos = trajectory_direction.cosine_similarity_throughout_trajectory(seg, window_size=smooth_window)[:, -end_window:]
    moved = np.linalg.norm(trajectory_direction.get_smoothed_player_direction_vectors_for_trajectory(
        seg, window_size=smooth_window), axis=0)[-end_window:]
    return cos.T, moved


def infer_choice(xy, high, low, end_window=END_WINDOW, smooth_window=SMOOTH_WINDOW, cos_min=COS_MIN, margin=MARGIN):
    ''' Return the inferred wall (or None) and the rule that decided it. Takes the trajectory from slice
        onset to the end of the trial (timepoints*2), and the High and Low wall ids. '''

    xy = np.asarray(xy, float)
    if len(xy) < end_window + smooth_window + 1:
        return None, 'too short'
    lit = (high, low)

    # take the alcove the agent is entering, if any
    w = at_lit_alcove(xy, lit)
    if w is not None:
        return w, 'at alcove'

    # otherwise read the direction of travel over the final window
    cos, moved = alignment(xy, end_window, smooth_window)
    if (moved < MIN_FRAME_STEP).any() or np.isnan(cos).any():
        return None, 'stopped'
    top = cos.argmax(axis=1) + 1
    srt = np.sort(cos, axis=1)
    w = int(np.bincount(top).argmax())
    if not ((top == w).all() and cos[:, w - 1].mean() >= cos_min and (srt[:, -1] - srt[:, -2]).mean() >= margin):
        return None, 'ambiguous'
    return (w, 'aligned') if w in lit else (None, 'aligned to unlit wall')


def trajectory_rows(trial):
    ''' Return one dict per agent for a trial with its slice onset -> trigger trajectory and yaw, the lit
        walls and the outcome. Takes a trial dataframe. '''

    import data_extraction.get_indices as get_indices
    import trajectory_analysis.trajectory_vectors as tv

    if not (trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION).any():
        return []
    high, low = get_indices.get_walls(trial)
    winner = get_indices.get_trigger_activator(trial)
    chosen = int(get_indices.get_chosen_walls([trial])[0])
    rows = []
    for pid in (0, 1):
        xy = tv.extract_trial_player_trajectory(trial=trial, player_id=pid).T.astype(float)
        yaw = np.asarray(tv.extract_trial_player_headangles(trial=trial, player_id=pid), float)
        n = min(len(xy), len(yaw))
        rows.append(dict(trial_num=float(trial[globals.TRIAL_NUM].dropna().iloc[0]), player=pid,
                         high_wall=high, low_wall=low, won=(pid == winner), chosen_wall=chosen,
                         xy=xy[:n], yaw=yaw[:n]))
    return rows


def collect_trajectories(sim_data_folder, models, verbose=True):
    ''' Return one row per trial x agent with the slice onset -> trigger trajectory and yaw, the lit walls
        and the outcome. Takes a batch's simulation folder and the model list. Drops the first trial. '''

    import parse_data.prepare_sim_data as prepare_sim_data

    rows = []
    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            _, trial_list = prepare_sim_data.prepare_single_model_data(model, sim_data_folder=sim_data_folder)
        for trial in trial_list[1:]:
            for r in trajectory_rows(trial):
                r['model'] = model
                rows.append(r)
        if verbose:
            print(f"model {model} done", end='  ')
    return pd.DataFrame(rows)


def truncation_check(trajectories, start=END_WINDOW + SMOOTH_WINDOW + 1, every=10, **kwargs):
    ''' Return one row per cut of a winner's trajectory, with the call made on the cut and the wall the
        winner went on to reach. Takes the table from collect_trajectories, the first cut length, the
        spacing between cuts (frames) and any infer_choice settings. '''

    rows = []
    for _, t in trajectories[trajectories['won']].iterrows():
        xy, n = t['xy'], len(t['xy'])
        for k in range(start, n, every):
            call, rule = infer_choice(xy[:k], t['high_wall'], t['low_wall'], **kwargs)
            rows.append(dict(model=t['model'], cut_s=k / globals.RECORDING_FREQUENCY, frac_of_trial=k / n,
                             dist_left=float(np.linalg.norm(xy[k - 1] - ALCOVE_CENTRES[t['chosen_wall'] - 1])),
                             reached_high=t['chosen_wall'] == t['high_wall'], call=call, rule=rule,
                             call_high=(call == t['high_wall']) if call is not None else None,
                             accepted=call is not None, correct=call == t['chosen_wall']))
    return pd.DataFrame(rows)
