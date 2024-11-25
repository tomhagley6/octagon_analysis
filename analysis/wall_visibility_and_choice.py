#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import trajectory_analysis.trajectory_direction as trajectory_direction
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import data_extraction.get_indices as get_indices
import analysis.loser_inferred_choice as loser_inferred_choice
import os


# ### Functions related to the analysis of player wall choice in the context of wall visibility

# In[ ]:


# full logic for identfying first visible and chosen wall for a single player
def first_visible_wall_chosen_session(trial_list, player_id, current_fov=110, debug=False):
    ''' Across a whole session, return 2 binary int arrays:
        first_visible_wall_chosen - was the first visible wall on this trial (for this player) chosen?
        first_visible_wall_high - was the first visible wall on this trial (for this player) the High wall? 
        Inferred choice is used here, not just the actual outcome of the trial
        Where inferred choice is missing, or there was not exactly one wall visible at the start of a trial, 
        array elements are np.nan '''
    
    player_id = player_id
    trial_list = trial_list

    # filter trial list for HighLow trialtype
    trial_indices = get_indices.get_trials_trialtype(trial_list, trial_type=globals.HIGH_LOW)
    trial_list = [trial_list[i] for i in trial_indices]
    
    # we want to find whether the first visible wall was chosen (or 'inferred chosen'), and whether it was the High wall
    first_visible_wall_chosen = np.ones(len(trial_list))*2
    first_visible_wall_high = np.ones(len(trial_list))*2
    
    # get the players choice, whether this is empirical, inferred, or nan
    player_wall_choice = loser_inferred_choice.player_wall_choice_win_or_loss(trial_list, player_id)
    
    for i in range(len(trial_list)):
        
        # get the walls for this trial
        trial = trial_list[i]
        walls = get_indices.get_walls(trial=trial)
        wall1 = walls[0]
        wall2 = walls[1]
    
        # wall_visible array for this trial 
        # boolean array of which walls are visible at each timepoint
        this_player_this_trial_wall_visible = trajectory_headangle.get_wall_visible(trial_list, i, player_id, current_fov=current_fov)
        if isinstance(this_player_this_trial_wall_visible, float) and np.isnan(this_player_this_trial_wall_visible): # if trial too short to analyse
            first_visible_wall_high[i] = np.nan
            first_visible_wall_chosen[i] = np.nan
            if debug:
                print(f"Setting this trial as np.nan because of short length")
            continue
        
        # check for wall1 and wall2 being visible at the start of the trial
        (this_player_this_trial_wall1_visible,
         this_player_this_trial_wall2_visible) = trajectory_headangle.wall_visibility_player_slice_onset(this_player_this_trial_wall_visible,
                                                                                    trial)
        
        # identify which wall first becomes visible in the trial (could alternatively be neither, or both visible at the start)
        this_player_this_trial_first_visible_wall = trajectory_headangle.get_first_visible_wall(this_player_this_trial_wall_visible,
                                                                                                this_player_this_trial_wall1_visible,
                                                                                                this_player_this_trial_wall2_visible,
                                                                                                trial)
        if debug:
            print(f" first vis wall of trial for player: {this_player_this_trial_first_visible_wall}")
            first_visible_wall_chosen, first_visible_wall_high = np.nan, np.nan
    
        # stop analysis if the player never sees walls or sees both at once. Set output as NaN
        if this_player_this_trial_first_visible_wall == 'neither' or this_player_this_trial_first_visible_wall == 'both':
            # set values in array to np.nan if both or neither wall are visible at the start of the trial
            if debug:
                print("neither or both")
            first_visible_wall_high[i] = np.nan
            first_visible_wall_chosen[i] = np.nan
            this_player_this_trial_first_visible_wall_chosen = np.nan
    
        # condition: one wall becomes visible before the other
        else:
            # check which wall is visible initially
            if this_player_this_trial_first_visible_wall == 'wall1':
                first_visible_wall_high[i] = 1
                this_player_this_trial_first_visible_wall_num = wall1
            elif this_player_this_trial_first_visible_wall == 'wall2':
                first_visible_wall_high[i] = 0
                this_player_this_trial_first_visible_wall_num = wall2
            else:
                raise ValueError("value must be either wall1, wall2, neither, or both")
    
            # compare player choice to the first visible wall on this trial 
            # check whether player choice can be retrieved
            if np.isnan(player_wall_choice[i]):
                # set values in array to np.nan if there is no choice available
                first_visible_wall_high[i] = np.nan
                first_visible_wall_chosen[i] = np.nan
                this_player_this_trial_first_visible_wall_chosen = np.nan
                if debug:
                    print("not confident in loser's choice")
                    print(f" first_vis_wall_chosen: {first_visible_wall_chosen[i]}, first_vis_wall_high: {first_visible_wall_high[i]}")
            
            else: # player choice is retrievable
                this_player_this_trial_first_visible_wall_chosen = True if player_wall_choice[i] == this_player_this_trial_first_visible_wall_num else False
                first_visible_wall_chosen[i] = this_player_this_trial_first_visible_wall_chosen
        if debug:
            print(f" first_vis_wall_chosen: {first_visible_wall_chosen[i]}, first_vis_wall_high: {first_visible_wall_high[i]}")
            print(f" player_wall_choice[i]: {player_wall_choice[i]}")
            print(f" this_player_this_trial_first_visible_wall_chosen: {this_player_this_trial_first_visible_wall_chosen}, high wall: {wall1}")

    return first_visible_wall_chosen, first_visible_wall_high


# In[ ]:


def probability_first_visible_wall_chosen_and_low(first_visible_wall_chosen, first_visible_wall_high, debug=False):
    ''' Returns a probability value for the first wall being chosen when the first wall is low.
        Takes two binary int arrays of len(trials_list), for the first visible wall being chosen, and for
        the first visible wall being high.
        If there is no choice or loser's inferred choice, input array values are np.nan.
        If both walls were initially visible, or never become visible, input array values are np.nan.
        Assumes data from a single player's session.'''

    if debug:
        print(f"Number of trials total is: {first_visible_wall_chosen.size}")
    
    # remove nans from the analysis
    first_visible_wall_chosen_not_nan = first_visible_wall_chosen[~np.isnan(first_visible_wall_chosen)]
    first_visible_wall_high_not_nan = first_visible_wall_high[~np.isnan(first_visible_wall_high)]
    first_visible_wall_low_not_nan = (first_visible_wall_high_not_nan -1) * -1
    if debug:
        print(f"Number of trials for this player that begin with one wall visible and end with a retrievable choice is: " +
                f"{first_visible_wall_high_not_nan.size}")

    # restrict data to the first visible wall being low, and also being chosen
    first_visible_low_and_also_chosen= np.where(
                    np.isnan(first_visible_wall_chosen) | np.isnan(first_visible_wall_high),   # If either element is nan
                    np.nan,                                           # Set to np.nan
                    np.where((first_visible_wall_chosen == 1.) & (first_visible_wall_high == 0.), 1., 0.)  # Else set to 1. or 0.
     )
    # again, clear nans
    first_visible_low_and_also_chosen_not_nan = first_visible_low_and_also_chosen[~np.isnan(first_visible_low_and_also_chosen)]
    
    if debug:
        print(f"Number of trials for this player that begin with Low wall visible and end with a retrievable choice is: " +
                f"{first_visible_wall_low_not_nan[first_visible_wall_low_not_nan ==1].size}")
        print(f"Number of trials for this player that begin with Low wall visible and end with Low wall chosen is: " +
                f"{first_visible_low_and_also_chosen_not_nan[first_visible_low_and_also_chosen_not_nan ==1].size}")
        print(f"Number of trials for this player that begin with High wall visible and end with a retrievable choice is: " +
                f"{first_visible_wall_high_not_nan[first_visible_wall_high_not_nan ==1].size}")
        
    # probability of first wall being chosen when the first wall is low
    num_walls_first_visible_low_and_also_chosen_not_nan = first_visible_low_and_also_chosen[first_visible_low_and_also_chosen ==1].size
    num_walls_first_visible_low_not_nan = first_visible_wall_low_not_nan[first_visible_wall_low_not_nan ==1].size
    probability_first_wall_chosen_when_low = num_walls_first_visible_low_and_also_chosen_not_nan/num_walls_first_visible_low_not_nan

    num_walls_first_visible_high_not_nan = first_visible_wall_low_not_nan[first_visible_wall_low_not_nan ==0].size
    num_trials_first_visible_low_chose_high = num_walls_first_visible_low_not_nan - num_walls_first_visible_low_and_also_chosen_not_nan
    
    if debug:
        print(f"num_walls_first_visible_low_and_also_chosen_not_nan = {num_walls_first_visible_low_and_also_chosen_not_nan}")
        print(f"num_walls_first_visible_low_not_nan = {num_walls_first_visible_low_not_nan}")
        print(f"Probability of first wall being chosen when the first wall is low: " + f"{probability_first_wall_chosen_when_low}")
        print(f"trials where low was seen first but high was chosen: {num_trials_first_visible_low_chose_high}")
    
    return probability_first_wall_chosen_when_low, num_trials_first_visible_low_chose_high


# In[ ]:


def probability_first_wall_chosen_and_low_multiple_sessions(data_folder, json_filenames_all):
    ''' Returns an array of probabilities for the first wall being chosen when the first wall is low
        and an array of the number of times this ocurred.
        These are of shape num_sessions*num_players.
        Takes a data folder path string, and a list of all json filenames (one for each session of data) '''
    
    num_sessions = len(json_filenames_all)
    probability_first_wall_chosen_array = np.zeros((num_sessions,2))
    probability_first_wall_chosen_when_low_array = np.zeros((num_sessions,2))
    times_first_wall_chosen_when_low_array = np.zeros((num_sessions,2))

    # for each session data file, identify probability of choosing first wall seen when that wall is Low
    for json_filenames_index in range(len(json_filenames_all)):
        json_filename = json_filenames_all[json_filenames_index]
        _, trials_list2 = prepare_data.prepare_data(data_folder, json_filename)
        for player_id in range(2): # for each player
            first_visible_wall_chosen, first_visible_wall_high = first_visible_wall_chosen_session(trials_list2, player_id=player_id)

            # quick detour to get the probability of choosing the first visible wall
            first_visible_wall_chosen_not_nan = first_visible_wall_chosen[~np.isnan(first_visible_wall_chosen)]
            num_first_visible_wall_chosen = first_visible_wall_chosen_not_nan[first_visible_wall_chosen_not_nan == 1].size
            probability_first_wall_chosen_array[json_filenames_index, player_id] = num_first_visible_wall_chosen/first_visible_wall_chosen_not_nan.size

            # calculate probability choosing low        
            probability_first_wall_chosen_when_low, times_first_wall_chosen_when_low = probability_first_visible_wall_chosen_and_low(first_visible_wall_chosen, first_visible_wall_high)
            probability_first_wall_chosen_when_low_array[json_filenames_index, player_id] = probability_first_wall_chosen_when_low
            times_first_wall_chosen_when_low_array[json_filenames_index, player_id] = times_first_wall_chosen_when_low 

    return probability_first_wall_chosen_when_low_array, times_first_wall_chosen_when_low_array, probability_first_wall_chosen_array


# In[ ]:


def probability_first_wall_chosen_and_low_multiple_sessions_df(trial_lists):
    ''' Returns an array of probabilities for the first wall being chosen when the first wall is low
        and an array of the number of times this ocurred.
        These are of shape num_sessions*num_players.
        Inferred choice is used here (first_visible_wall_chosen_session).
        Takes a data folder path string, and a list of all json filenames (one for each session of data) '''
    
    num_sessions = len(trial_lists)
    probability_first_wall_chosen_array = np.zeros((num_sessions,2))
    probability_first_wall_chosen_when_low_array = np.zeros((num_sessions,2))
    times_first_wall_chosen_when_low_array = np.zeros((num_sessions,2))

    # for each session data file, identify probability of choosing first wall seen when that wall is Low
    for trial_list_idx in range(len(trial_lists)):
        this_trial_list = trial_lists[trial_list_idx]
        for player_id in range(2): # for each player
            first_visible_wall_chosen, first_visible_wall_high = first_visible_wall_chosen_session(this_trial_list, player_id=player_id)

            # quick detour to get the probability of choosing the first visible wall
            first_visible_wall_chosen_not_nan = first_visible_wall_chosen[~np.isnan(first_visible_wall_chosen)]
            num_first_visible_wall_chosen = first_visible_wall_chosen_not_nan[first_visible_wall_chosen_not_nan == 1].size
            probability_first_wall_chosen_array[trial_list_idx, player_id] = num_first_visible_wall_chosen/first_visible_wall_chosen_not_nan.size

            # calculate probability choosing low        
            probability_first_wall_chosen_when_low, times_first_wall_chosen_when_low = probability_first_visible_wall_chosen_and_low(first_visible_wall_chosen, first_visible_wall_high)
            probability_first_wall_chosen_when_low_array[trial_list_idx, player_id] = probability_first_wall_chosen_when_low
            times_first_wall_chosen_when_low_array[trial_list_idx, player_id] = times_first_wall_chosen_when_low 

    return probability_first_wall_chosen_when_low_array, times_first_wall_chosen_when_low_array, probability_first_wall_chosen_array

