import scipy
import parse_data.prepare_data as prepare_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import globals
import plotting.plot_trajectory as plot_trajectory
import plotting.plot_octagon as plot_octagon
import data_extraction.extract_trial as extract_trial
import math
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import trajectory_analysis.trajectory_headangle as trajectory_headangle
from IPython.display import Image, display
import data_extraction.get_indices as get_indices
import prominent_direction_functions





def extract_headangles_before_slice_onset(trial_list=None, trial_index=0, trial=None, player_id=0, debug=False):
    ''' Returns a timepoints-sized array of head direction Euler angles for a single player's trial
        from trial end to slice onset '''

    # get trial dataframe
    if debug:
        print(f"Extracting trial with trial index {trial_index}")
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)
    assert isinstance(trial, pd.DataFrame)
    
    prev_trial_index = trial_index - 1
    prev_trial = extract_trial.extract_trial(trial, trial_list, prev_trial_index)
    assert isinstance(prev_trial, pd.DataFrame)

    
    # get slice onset index
    slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    if slice_onset.empty:
        raise ValueError(f"No slice onset event found in trial {trial_index}")

    # relative index
    slice_onset_index = slice_onset.index[0] - trial.index[0]
    if debug:
        print(f"Slice onset index: {slice_onset_index} (Absolute index: {slice_onset.index[0]}")

    # get trial end index
    trial_end = prev_trial[prev_trial['eventDescription'] == globals.TRIAL_END]
    if debug:
        print(f"trial end is {trial_end.index[0]} type {type(trial_end.index[0])}\n and trial index is {trial.index[0]}")

    # relative index
    trial_end_index = trial_end.index[0] - prev_trial.index[0]


    # find the euler angles for the rotation around the y (Unity vertical) axis
    y_rotation = prev_trial[globals.PLAYER_ROT_DICT[player_id]['yrot']]

    # convert relative (to the trial) indices to absolute indices
    trial_end_abs = trial_end.index[0]
    slice_onset_abs = slice_onset.index[0]

    # take euler angles from trial end of the previous trial to slice onset of the next 
    y_rotation.iloc[trial_end_abs:slice_onset_abs]

    # convert to radians
    head_angles = np.deg2rad(y_rotation)

    return head_angles






def get_prev_change_indices(trial_list, steps, number_of_assignments):
    '''
    Gets change indices (cw/ccw head angle directions) for current and previous trial and checks for agreement
    Args:
    - steps: number of head angles to be smoothed over to compute change indices
    - number_of_assignments: number of change indices to determine general rotation direction from
      and compare between start of current trial and end of previous
    Output:
    - consistent_rotation_direction: list of trials, the initial rotation direction of which agrees with
                                     the rotation direciton of the end of the previous trial
    - inconsistent_rotation_direction: list of trials, the initial rotation direction of which does not agree with
                                       the rotation direciton of the end of the previous trial
    '''

    consistent_rotation_direction = []
    inconsistent_rotation_direction = [] 

    for trial in trial_list:
        
        headangles_subs = np.array(trajectory_vectors.extract_trial_player_headangles(trial=trial))
        headangles_prev = np.array(extract_headangles_before_slice_onset(trial=trial))
        
        change_indices_subs, angles_subs = prominent_direction_functions.get_change_indices_smoothed_windows(headangles_subs, steps)
        change_indices_prev, angles_prev = prominent_direction_functions.get_change_indices_smoothed_windows(headangles_prev, steps)

        if change_indices_prev.size == 0 or change_indices_subs.size == 0:  # No valid indices
            print(f"Warning: No valid changes detected in current trial, skipping...")
            continue
        
        first_smoothed_angle_differences = np.array(angles_subs[:number_of_assignments])
        first_assignments = change_indices_subs[1][:number_of_assignments]
        
        last_smoothed_angle_differences = np.array(angles_prev[:-number_of_assignments])
        last_assignments = change_indices_prev[1][:-number_of_assignments]
        
        sum_abs_cw_subs = np.sum(np.abs(first_smoothed_angle_differences[first_assignments == 1]))
        sum_abs_ccw_subs = np.sum(np.abs(first_smoothed_angle_differences[first_assignments == -1]))
        
        sum_abs_cw_prev = np.sum(np.abs(last_smoothed_angle_differences[last_assignments == 1]))
        sum_abs_ccw_prev = np.sum(np.abs(last_smoothed_angle_differences[last_assignments == -1]))
        
        if (sum_abs_cw_subs > sum_abs_ccw_subs) == (sum_abs_cw_prev > sum_abs_ccw_prev) or (sum_abs_cw_subs < sum_abs_ccw_subs) == (sum_abs_cw_prev < sum_abs_ccw_prev):
            consistent_rotation_direction.append(trial)
        else: inconsistent_rotation_direction.append(trial)
            
    return consistent_rotation_direction, inconsistent_rotation_direction