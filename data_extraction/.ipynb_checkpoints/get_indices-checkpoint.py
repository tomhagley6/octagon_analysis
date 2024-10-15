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


# get walls
def get_walls(trial=None, trial_list=None, trial_index=None, num_walls=2):
    ''' Return a list with the numbers of all walls for this trial,
        in ascending order
    '''
    
    this_trial = plot_trajectory.extract_trial(trial, trial_list, trial_index)

    wall_column_names = [globals.WALL_1, globals.WALL_2, globals.WALL_3, globals.WALL_4]
    
    walls = []
    for i in range(num_walls):
        walls.append(this_trial.iloc[0][wall_column_names[i]]

    return walls


# In[ ]:


# find trial with wall_sep == 1
def get_trials_with_wall_sep(trial_list, wall_sep=1):
    ''' Get the indices of trials with a wall separation of 1
        Assuming 2 walls in the trial
    '''
    
    trial_indices = []
    for i in range(len(trial_list)):
        this_trial = trial_list[i]
        walls = get_walls(trial_list)
    
        if abs(walls[0] - walls[1]) == wall_sep:
            trial_indices.append(this_trial)

    return trial_indices

