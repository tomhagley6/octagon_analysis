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







def get_change_index(headangles_array):
    '''Inputting array of head angles in RADIANS for trial returns 2D array containing indices and change indices (+1 for clockwise, -1 for 
    counterclockwise'''
    
    if len(headangles_array) < 2:  # Not enough data to compute changes
        return np.zeros((2, 0), dtype=int)  # Empty change indices
        
    change_index_list = []
    index_list = []
    
    for index in range(1, len(headangles_array)):
        
        delta_yrot = headangles_array[index] - headangles_array[index - 1]
        
        #normalise delta_yrot to be in the range [-π, π] to account for wraparound
        delta_yrot = np.arctan2(np.sin(delta_yrot), np.cos(delta_yrot))
        
        change_index = -1 if delta_yrot < 0 else +1
            
        change_index_list.append(change_index)
        index_list.append(index)

    if not index_list:  # Double-checking in case all values were skipped
        return np.zeros((2, 0), dtype=int)
        
    change_indices = np.zeros((2,len(headangles_array)-1), dtype=int)
    change_indices[0] = index_list
    change_indices[1] = change_index_list
    
    return change_indices






def get_change_indices_smoothed(headangles_array, steps):
    '''
    Note: smoothed for smoothed head angles, NOT windows.
    Inputting array of head angles in RADIANS for trial computes smoothed head angles and
    returns 1D array containing  change
    indices (+1 for clockwise, -1 for counterclockwise)
    **Fixes:**
    - Suitable for angles in radians, not degrees
    '''
    
    if len(headangles_array) < 2:  #not enough data to compute changes
        return np.zeros((2, 0), dtype=int)  #empty change indices
        
    change_index_list = []
    
    for index in range(steps, len(headangles_array), steps):

        delta_yrot = headangles_array[index] - headangles_array[index - steps]
        
        delta_yrot = np.arctan2(np.sin(delta_yrot), np.cos(delta_yrot))
        
        change_index = -1 if delta_yrot < 0 else +1
            
        change_index_list.append(change_index)
    
    return change_index_list




def get_change_indices_smoothed_windows(headangles_array, steps):
    '''
    Note: smoothed for smoothed windows.
    Inputting array of head angles in RADIANS for trial computes smoothed head angles and
    returns 2D array containing indices and change
    indices (+1 for clockwise, -1 for counterclockwise), and a list of angles (the change from
    the start to the end of the window)
    **Fixes:**
    - Suitable for angles in radians, not degrees
    '''
    
    if len(headangles_array) < 2:  #not enough data to compute changes
        return np.zeros((2, 0), dtype=int)  #empty change indices
        
    change_index_list = []
    index_list = []
    angle_list = []
    
    for index in range(steps, len(headangles_array), steps):

        delta_yrot = headangles_array[index] - headangles_array[index - steps]
        
        delta_yrot = np.arctan2(np.sin(delta_yrot), np.cos(delta_yrot))
        
        change_index = -1 if delta_yrot < 0 else +1
            
        change_index_list.append(change_index)
        index_list.append(index)
        angle_list.append(delta_yrot)

    if not index_list:  #double-checking in case all values were skipped
        return np.zeros((2, 0), dtype=int), []
        
    change_indices = np.zeros((2,len(change_index_list)), dtype=int)
    change_indices[0] = index_list
    change_indices[1] = change_index_list
    
    return change_indices, angle_list






def get_headangle_change_windows(change_indices):
    '''
    Inputting change indices (+1 for clockwise, -1 for counterclockwise) returns list of
    tuples containing attributes of the window:
    start index = windows[0], length of the window = windows[1], end index = windows[2]
    **Fixes:**
    - Suitable for angles in radians, not degrees
    '''

    if len(change_indices[1]) == 0:
        return []
        
    windows = []
    start_index = 0
    current_index = int(change_indices[1][0])
    
    for index in range(1, len(change_indices[1])):
        
        if change_indices[1][index] != current_index:
            
            windows.append((start_index, index-1, current_index))
            start_index = index
            current_index = int(change_indices[1][index])
            
    windows.append((start_index, len(change_indices[1]) - 1, current_index))
    return windows





#measure degree of change in each window
def get_headangle_change_rates(windows, headangles_array):
    '''
    Inputting an array of head angles from a given trial and windows of (continuous) change
    indices, returns a list change rates, i.e., the degree to which head angle changed
    consistently in cw/ccw direction within a given window 
    **Fixes:**
    - Suitable for angles in radians, not degrees
    '''
    if not windows:  #handle empty input
        return []

    headangle_change_rates = [] 
    for start_index, end_index, _ in windows:
        
        if start_index >= len(headangles_array) or end_index > len(headangles_array):
            print(f"Warning: Skipping invalid window ({start_index}, {end_index}) - out of bounds")
            continue  # Skip invalid windows

        if start_index == end_index:
            print(f"Warning: Skipping window ({start_index}, {end_index}) because delta_time = 0")
            continue  # Skip windows where delta_time is 0
        
        delta_headangle = headangles_array[end_index] - headangles_array[start_index]
        
        delta_headangle = np.arctan2(np.sin(delta_headangle), np.cos(delta_headangle))
            
        delta_time = end_index - start_index
        if delta_time != 0:
            headangle_change_rate = delta_headangle / delta_time
            headangle_change_rates.append(headangle_change_rate)

    return headangle_change_rates




def compare_change_rates(headangle_change_rates):
    '''Inputting a list of head angle change rates returns:
    a) a list of sorted indices corresponding to b), 
    b) a list of sorted head angle change rates in descending order, 
    c) the highest rate (rate for the biggest change in head angle compared to other windows),
    d) the second-highest rate,
    e) the difference between highest and second-highest rate, 
    f) the average difference between rates, 
    g) whether the difference between highest and second-highest is higher than average.
    
    **Fixes:**
    - Uses absolute values instead of squaring to avoid distortion.
    '''
    
    headangle_change_rates_arr = np.abs(np.array(headangle_change_rates))
    
    sorted_indices = np.argsort(headangle_change_rates_arr)[::-1]
    sorted_rates = sorted(headangle_change_rates_arr, reverse=True)
    
    differences = np.diff(sorted_rates)
    average_diff = np.mean(differences)

    highest_rate = sorted_rates[0]
    second_highest_rate = sorted_rates[1]
    first_second_diff = highest_rate - second_highest_rate

    is_high_or_low = "high" if first_second_diff > average_diff else "low"

    
    return sorted_indices, sorted_rates, highest_rate, second_highest_rate, first_second_diff, average_diff, is_high_or_low


def check_highest_change_position(headangle_change_rates, headangles_array):
    '''
    Checks whether the highest change rate happens in the first half of the trial. Returns
    bool and highest rate index.
    '''
    # Determine the number of data points (bins) in the trial

    #num_points = len(headangle_change_rates)
    
    headangle_change_rates_arr = np.abs(np.array(headangle_change_rates))
    
    # Find the index of the highest degree of change
    highest_change_idx = np.argmax(headangle_change_rates_arr)
    
    # Check if this index is in the first half of the trial
    is_in_first_half = highest_change_idx < len(headangle_change_rates) // 2 #integer result
    #is_in_first_half = highest_change_idx < num_points / 2 #integer result
    
    return is_in_first_half, highest_change_idx


def sort_trials_by_CW_and_CCW(trial_list, player_id):
    '''
    Sorts trial into clockwise and counterclockwise lists based on logic from previous
    functions, i.e., assign cw/ccw based on the change index (+1/-1) of the window with the
    greatest change in head angle direction compared to the other windows in the trial, if
    and only if said window's rate is distinct from the second-highest rate and the window is in
    the first half of the trial
    '''

    CCW_trials = []
    CW_trials = []
    
    
    for trial in trial_list:
        
        headangles = trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=player_id)
        headangles_array = np.array(headangles)

        change_indices = get_change_index(headangles_array)

        if change_indices.shape[1] == 0:  # No valid indices
            print("Warning: No valid changes detected, skipping...")
            continue

        
        windows = get_headangle_change_windows(change_indices)
        
        headangle_change_rates = get_headangle_change_rates(windows, headangles_array)

        if len(headangle_change_rates) < 2:
            print(f"Skipping trial due to insufficient meaningful data: {headangle_change_rates}")
            continue
            
        sorted_indices, sorted_rates, highest_rate, second_highest_rate, first_second_diff, average_diff, is_high_or_low = compare_change_rates(headangle_change_rates)
        is_first_half, highest_change_idx = check_highest_change_position(headangle_change_rates, headangles_array)

        window_direction_at_highest_rate = windows[highest_change_idx][2]

        if (is_high_or_low == "high") and is_first_half:
            prominent_direction = window_direction_at_highest_rate
            if prominent_direction == -1:
                CCW_trials.append(trial)
            else: CW_trials.append(trial)
                
        #elif (is_high_or_low == "low") and is_first_half:
            #print(highest_rate, second_highest_rate, first_second_diff, average_diff)
            
        else: print(sorted_indices, sorted_rates)
            
    return CCW_trials, CW_trials



def sort_trials_by_CW_and_CCW_smoothed(trial_list, player_id):
    '''Same as above but smooths the headangles.
    '''

    CCW_trials = []
    CW_trials = []
    
    
    for trial in trial_list:
        
        headangles = trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=player_id)
        timepoints = len(headangles)
        
        headangles_smoothed = []
        
        for i in range(timepoints - 5):
            smoothed_headangles = np.mean(headangles[i:i+5])
            headangles_smoothed.append(smoothed_headangles)
            
        headangles_array = np.array(headangles_smoothed)

        change_indices = get_change_index(headangles_array)

        if change_indices.shape[1] == 0:  # No valid indices
            print("Warning: No valid changes detected, skipping...")
            continue

        
        windows = get_headangle_change_windows(change_indices)
        
        headangle_change_rates = get_headangle_change_rates(windows, headangles_array)

        if len(headangle_change_rates) < 2:
            print(f"Skipping trial due to insufficient meaningful data: {headangle_change_rates}")
            continue
            
        sorted_indices, sorted_rates, highest_rate, second_highest_rate, first_second_diff, average_diff, is_high_or_low = compare_change_rates(headangle_change_rates)
        is_first_half, highest_change_idx = check_highest_change_position(headangle_change_rates, headangles_array)

        window_direction_at_highest_rate = windows[highest_change_idx][2]


        if (is_high_or_low == "high") and is_first_half:
            prominent_direction = window_direction_at_highest_rate
            if prominent_direction == -1:
                CCW_trials.append(trial)
            else: CW_trials.append(trial)
                
        #elif (is_high_or_low == "low") and is_first_half:
            #print(highest_rate, second_highest_rate, first_second_diff, average_diff)
            
        else: print(sorted_indices, sorted_rates)
            
    return CCW_trials, CW_trials





from collections import Counter




def sort_trials_by_CW_and_CCW_smoothed_window(trial_list, player_id, steps=10):
    ''' 
    Smooths change indices over ten time points and determines assignment (cw/ccw) based on the
    most prominent direction in the first 50 10-time-point windows (approx. 1 sec). Note: if the
    trial head angles are less than 10 a smaller window is taken, however, first_50_changes
    assumes change indices are smoothed over 10 time points
    '''

    CCW_trials = []
    CW_trials = []
    
    
    for trial_index, trial in enumerate(trial_list):
        
        headangles_array = np.array(trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=player_id))

        if len(headangles_array) <= 1:
            print(f"Trial number {trial_index} is empty")
            continue
            
        elif len(headangles_array) < steps:
            #print(f"Taking less steps for {trial_index}, headangles length: {len(headangles_array)}, steps: {len(headangles_array)//2}")
            #steps = len(headangles_array) // 2
            print(f"Trial number {trial_index} is ")
            continue    
        else:
            change_indices = get_change_indices_smoothed_windows(headangles_array, steps)

        first_5_assignments = change_indices[0][:5]

        if len(first_5_assignments)==0:
            print(f"Skipping trial {trial_index}, no valid change indices found.")
            continue

        count = Counter(first_5_assignments)
        prominent_change = max(count, key=count.get)
        
        if prominent_change == -1:
            CCW_trials.append(trial)
        elif prominent_change == +1:
            CW_trials.append(trial)
        else:
            print(f"Unexpected values in first 50 change indices: {first_50_changes}")
            
    return CCW_trials, CW_trials

                


def sort_trials_angle_and_smoothed_window(trial_list, player_id, steps=10):
    ''' 
    Smooths change indices over ten time points. Computes the angle travelled for each
    change indice and the total angle travelled within the first 50 time points. Sums angles
    travelled for cw and ccw windows, respectively, and assigns trial cw/ccw accordingly. Note:
    if the trial head angles are less than 10 a smaller window is taken, however,
    first_50_changes assumes change indices are smoothed over 10 time points.
    **Fixes**
    - Sum of squared changes to sum of absolute changes
    '''

    CCW_trials = []
    CW_trials = []
    equal_trials = []
    
    
    for trial_index, trial in enumerate(trial_list):
        
        headangles_array = np.array(trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=player_id))

        if len(headangles_array) <= 1:
            print(f"Trial number {trial_index} is empty")
            continue
            
        elif len(headangles_array) < steps:
            #print(f"Taking less steps for {trial_index}, headangles length: {len(headangles_array)}, steps: {len(headangles_array)//2}")
            #steps = len(headangles_array) // 2
            print(f"Trial number {trial_index} is ")
            continue
            
        change_indices, angle_list = get_change_indices_smoothed_windows(headangles_array, steps)

        if change_indices.size == 0:  # No valid indices
            print(f"Warning: No valid changes detected in trial {trial_index}, skipping...")
            continue

        #calculate angular displacement for first 50 timepoints (5 change windows)
        first_5_smoothed_angle_differences = np.array(angle_list[:5])
        first_5_assignments = change_indices[1][:5]

        sum_abs_cw = np.sum(np.abs(first_5_smoothed_angle_differences[first_5_assignments == 1]))
        sum_abs_ccw = np.sum(np.abs(first_5_smoothed_angle_differences[first_5_assignments == -1]))

        if sum_abs_cw > sum_abs_ccw:
            CW_trials.append(trial)
        elif sum_abs_ccw > sum_abs_cw:
            CCW_trials.append(trial)
        else: 
            print("equal angular displacement")
            equal_trials.append(trial)
   
    return CCW_trials, CW_trials, equal_trials

                


