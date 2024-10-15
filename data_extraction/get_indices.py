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


# In[ ]:


def get_walls(trial=None, trial_list=None, trial_index=None, num_walls=2):
    ''' Return a list with the numbers of all walls for this trial,
        in ascending order '''
    
    this_trial = plot_trajectory.extract_trial(trial, trial_list, trial_index)

    wall_column_names = [globals.WALL_1, globals.WALL_2, globals.WALL_3, globals.WALL_4]
    
    walls = []
    for i in range(num_walls):
        walls.append(this_trial.iloc[0][wall_column_names[i]])

    return walls


# In[ ]:


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
    


# In[ ]:


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
            trial_indices.append(this_trial)

    return trial_indices

