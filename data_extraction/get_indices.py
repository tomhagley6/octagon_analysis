#!/usr/bin/env python
# coding: utf-8

# In[1]:


import parse_data.preprocess as preprocess
import parse_data.loading as loading
import globals
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from plotting import plot_trajectory


# In[18]:


def get_walls(trial=None, trial_list=None, trial_index=None, num_walls=2):
    ''' Return a list with the numbers of all walls for this trial,
        in ascending order '''
    
    this_trial = plot_trajectory.extract_trial(trial, trial_list, trial_index)
    # print(f"Trial in get_walls is: {type(trial)}")

    wall_column_names = [globals.WALL_1, globals.WALL_2, globals.WALL_3, globals.WALL_4]
    
    walls = []
    for i in range(num_walls):
        # print(f"this_wall for trial {this_trial[globals.TRIAL_NUM].unique().item()}, wall {i}")
        this_wall = int(this_trial.iloc[0][wall_column_names[i]])
        walls.append(this_wall)

    return walls


# In[3]:


def get_wall_difference(trial=None, trial_list=None, trial_index=None, num_walls=2):
    ''' Get the difference between walls
        Assuming 2 walls in the trial '''
    
    max_val = globals.NUM_WALLS

    this_trial = plot_trajectory.extract_trial(trial, trial_list, trial_index)
    walls = get_walls(trial=trial, trial_list=trial_list, trial_index=trial_index)

    direct_difference = abs(walls[0] - walls[1])

    # account for circular variables
    wrap_around_difference = max_val - direct_difference

    # the smaller of the 2 is the real difference
    difference = min(direct_difference, wrap_around_difference)

    return difference
    


# In[4]:


def get_trials_with_wall_sep(trial_list, wall_sep=1):
    ''' Get the indices of trials with a specified wall separation (default 1)
        Assuming 2 walls in the trial '''
    max_val = globals.NUM_WALLS
    
    trial_indices = []
    for i in range(len(trial_list)):
        this_trial = trial_list[i]
        
        walls = get_walls(trial_list=trial_list, trial_index=i)
        difference = get_wall_difference(trial=this_trial)

        if difference == wall_sep:
            trial_indices.append(i)

    return np.asarray(trial_indices)


# In[5]:


def get_trials_trialtype(trial_list, trial_type=globals.HIGH_LOW):
    ''' Get the indices of trials with a specified trial type (default HighLow) '''
    
    trial_indices = []
    for i in range(len(trial_list)):
        this_trial = trial_list[i]
        
        this_trial_type = this_trial[globals.TRIAL_TYPE].unique()[0]

        if this_trial_type == trial_type:
            trial_indices.append(i)

    return np.asarray(trial_indices)


# In[ ]:


def get_trials_chose_wall(trial_list, chosen_wall):
    ''' Get indices of trials where the winner chose High '''

    trial_indices = []
    for i in range(len(trial_list)):

        # if the wallTriggered value aligns with the chosen_wall value, winner chose chosen_wall
        # find all non-nan values for wallTriggered
        this_trial_triggers = trial_list[i][
                        ~np.isnan(trial_list[i]['data.wallTriggered'])
                        ]
        # identify which of these was selected by the Server
        this_trial_selected_trigger = this_trial_triggers[
                                                           this_trial_triggers['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION
                                                          ]['data.wallTriggered'].item()

        # identify whether this matches the High wall
        chose_wall = this_trial_selected_trigger == trial_list[i][chosen_wall].unique().item()

        if chose_wall:
            trial_indices.append(i)

    return np.asarray(trial_indices)


# In[1]:


def get_trigger_activators(trial_list):
    ''' Return a trial_num length array of the player which activated the trigger
        on each trial (starting from player 0) '''

    trigger_activators = np.zeros(len(trial_list))
    for i, trial in enumerate(trial_list):

        trigger_event = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
        trigger_activators[i] = int(trigger_event[globals.TRIGGER_CLIENT].item())


    return np.asarray(trigger_activators)
        


# In[ ]:


def get_trigger_activator(trial):
    ''' Return the player on this trial that activated the trigger '''
    
    trigger_event = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    trigger_activator = int(
        trigger_event[globals.TRIGGER_CLIENT].item()
    )


    return trigger_activator


# In[9]:


def get_trigger_activators_slice_onset_loc(trial_list):
    trigger_activators = get_trigger_activators(trial_list)
    winner_x_location_slice_onset = []
    winner_y_location_slice_onset = []


    for i in range(len(trial_list)): 
        this_trial = trial_list[i]
        trigger_activator = trigger_activators[i]

        xloc_key = globals.PLAYER_LOC_DICT[trigger_activator]['xloc']
        yloc_key = globals.PLAYER_LOC_DICT[trigger_activator]['yloc']
        
        this_trial_slice_onset = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]

        if not this_trial_slice_onset.empty:
            
            this_trial_slice_onset_index = this_trial_slice_onset.index[0] - this_trial.index[0]
            # print(f"slice onset index is: {this_trial_slice_onset.index[0]}")
            # print(f"trial onset index is: {this_trial.index[0]}")
            # print(f"this_trial_slice_onset_index is: {this_trial_slice_onset_index}")
            
            this_trial_winner_x_location_slice_onset = this_trial[xloc_key].iloc[this_trial_slice_onset_index]
            this_trial_winner_y_location_slice_onset = this_trial[yloc_key].iloc[this_trial_slice_onset_index]
    
            winner_x_location_slice_onset.append(this_trial_winner_x_location_slice_onset)
            winner_y_location_slice_onset.append(this_trial_winner_y_location_slice_onset)

        

    return list(zip(winner_x_location_slice_onset, winner_y_location_slice_onset))


# In[10]:


def get_player_slice_onset_locs(trial_list, player_id_list=None):
    ''' Return a list of zipped x coordinate and y coordinate for player location
        at slice onset. By default, the player is the winner for the trial, but an array
        of player ids can be passed, with the same dimensions as trial_list '''
    
    player_x_location_slice_onset = []
    player_y_location_slice_onset = []

    if player_id_list is None:
        trigger_activators = get_trigger_activators(trial_list)


    for i in range(len(trial_list)): 
        this_trial = trial_list[i]

        if player_id_list is None:
            player_id = trigger_activators[i]
        else:
            player_id = player_id_list[i]

        xloc_key = globals.PLAYER_LOC_DICT[player_id]['xloc']
        yloc_key = globals.PLAYER_LOC_DICT[player_id]['yloc']
        
        this_trial_slice_onset = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]

        if not this_trial_slice_onset.empty:
            
            this_trial_slice_onset_index = this_trial_slice_onset.index[0] - this_trial.index[0]
            # print(f"slice onset index is: {this_trial_slice_onset.index[0]}")
            # print(f"trial onset index is: {this_trial.index[0]}")
            # print(f"this_trial_slice_onset_index is: {this_trial_slice_onset_index}")
            
            this_trial_player_x_location_slice_onset = this_trial[xloc_key].iloc[this_trial_slice_onset_index]
            this_trial_player_y_location_slice_onset = this_trial[yloc_key].iloc[this_trial_slice_onset_index]
    
            player_x_location_slice_onset.append(this_trial_player_x_location_slice_onset)
            player_y_location_slice_onset.append(this_trial_player_y_location_slice_onset)

        

    return list(zip(player_x_location_slice_onset, player_y_location_slice_onset))


# In[11]:


def get_player_slice_onset_loc(trial, player_id):
    ''' return the x,y location tuple of the given player for the given trial
        at slice onset '''

    xloc_key = globals.PLAYER_LOC_DICT[player_id]['xloc']
    yloc_key = globals.PLAYER_LOC_DICT[player_id]['yloc']

    this_trial_slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET] 
    this_trial_slice_onset_index = this_trial_slice_onset.index[0] - trial.index[0]

    this_trial_player_x_location_slice_onset = trial[xloc_key].iloc[this_trial_slice_onset_index]
    this_trial_player_y_location_slice_onset = trial[yloc_key].iloc[this_trial_slice_onset_index]

    return (this_trial_player_x_location_slice_onset, this_trial_player_y_location_slice_onset)


# In[12]:


def get_player_win_indices(trial_list, player_id):
    ''' Indices of a trial list where the specified player won '''
    
    trigger_activators = get_trigger_activators(trial_list)
    
    this_player_win_indices = np.where(trigger_activators == player_id)[0]

    return this_player_win_indices
        


# In[13]:


def get_chosen_walls(trial_list):

    chosen_walls = np.zeros(len(trial_list))
    for i in range(len(trial_list)): 
        this_trial = trial_list[i]
        

        # only select from the trigger event chosen by the server
        selected_trigger_activation_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
        wall_chosen = selected_trigger_activation_event[globals.WALL_TRIGGERED].unique()
        wall_chosen_filter_nans = wall_chosen[~np.isnan(wall_chosen)]
        wall_chosen_val = wall_chosen_filter_nans.item()

        chosen_walls[i] = int(wall_chosen_val)

    return chosen_walls


# In[ ]:


def was_high_wall_chosen(trial_list):
    ''' Identify whether the chosen wall on each trial was High or Low
        Returns a boolean array of length num_trials '''

    # initialise array
    high_wall_chosen = np.zeros(len(trial_list), dtype=np.bool)
    
    # get the chosen walls for each trial
    chosen_walls = get_chosen_walls(trial_list)

    # loop through trials, identify wall1, and compare it to chosen wall
    for i in range(len(trial_list)):
        this_trial = trial_list[i]
        
        walls = get_walls(this_trial)
        wall1 = walls[0]

        # print(f"chosen wall: {chosen_walls[i]}, wall1: {wall1}")
        # if chosen wall and wall1 are identical, set to True for this trial
        # False will be by default
        if wall1 == chosen_walls[i]:
            high_wall_chosen[i] = True

    return high_wall_chosen

    


# In[15]:


def get_indices_slice_onset_trigger_activation(trial):
    
    # get slice onset index, referenced to trial start index
    slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    slice_onset_index = slice_onset.index[0] - trial.index[0]

    # get trigger activation index, referenced to trial start
    selected_trigger_activation = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    selected_trigger_activation_index = selected_trigger_activation.index[0] - trial.index[0]

    return slice_onset_index, selected_trigger_activation_index

