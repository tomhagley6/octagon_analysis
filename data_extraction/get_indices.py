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


# In[2]:


def get_walls(trial=None, trial_list=None, trial_index=None, num_walls=2):
    ''' Return a list with the numbers of all walls for this trial,
        in ascending order '''
    
    this_trial = plot_trajectory.extract_trial(trial, trial_list, trial_index)
    # print(f"Trial in get_walls is: {type(trial)}")

    wall_column_names = [globals.WALL_1, globals.WALL_2, globals.WALL_3, globals.WALL_4]
    
    walls = []
    for i in range(num_walls):
        walls.append(this_trial.iloc[0][wall_column_names[i]])

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


# In[10]:


def get_trials_chose_high(trial_list):
    ''' Get indices of trials where the winner chose High '''

    trial_indices = []
    for i in range(len(trial_list)):
        this_trial = trial_list[i]

        # if the wallTriggered value aligns with the wall1 value, winner chose High
        # find all non-nan values for wallTriggered
        this_trial_triggers = trial_list[i][
                        ~np.isnan(trial_list[i]['data.wallTriggered'])
                        ]
        # identify which of these was selected by the Server
        this_trial_selected_trigger = this_trial_triggers[
                                                           this_trial_triggers['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION
                                                          ]['data.wallTriggered'].item()

        # identify whether this matches the High wall
        chose_high = this_trial_selected_trigger == trial_list[i]['data.wall1'].unique().item()

        if chose_high:
            trial_indices.append(i)

    return np.asarray(trial_indices)


# In[6]:


def get_trigger_activators(trial_list):

    trigger_activators = []
    for i in range(len(trial_list)):
        this_trial = trial_list[i]

        trigger_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
        trigger_activator = int(
            trigger_event[globals.TRIGGER_CLIENT].item()
        )

        trigger_activators.append(trigger_activator)


    return np.asarray(trigger_activators)
        


# In[7]:


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

