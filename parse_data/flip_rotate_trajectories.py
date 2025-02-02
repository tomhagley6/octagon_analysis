#!/usr/bin/env python
# coding: utf-8

# In[332]:


import parse_data.preprocess as preprocess
import parse_data.loading as loading
from analysis.response_times import calculate_response_times, plot_response_times
import globals
from plotting import plot_octagon, plot_trajectory
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import data_extraction.get_indices as get_indices
import data_extraction.extract_trial as extract_trial


# In[339]:


## flip and rotate trials ##


# In[1]:


def find_rotation_angle_trial(trial):
    """ Find CCW angle of rotation for vector to 
    rotate arena s.t. high wall is at wall 1"""

    # print(f"Trial in find_rotation_angle_trial is: {type(trial)}")
    
    # identify trial walls
    wall1, wall2 = get_indices.get_walls(trial=trial, trial_list=None, trial_index=None, num_walls=2)
    
    # find CCW difference of high wall to wall 1
    difference = wall1 - 1
        
    # find CCW rotation angle 
    unitary_rotation_ang = 2*math.pi/globals.NUM_WALLS
    theta = unitary_rotation_ang * difference

    return theta
    


# In[2]:


def flip_rotate_trial(trial, theta, flip=True):
    """ Rotate x-y coordinates by theta 
        Flip x coordinates of vector if wall 1 CCW of wall 0
        Return altered vector """

    num_walls = globals.NUM_WALLS
    
    altered_coordinates = []
    num_players = preprocess.num_players(trial)
    
    trial_copy = trial.copy()

    # create rotation matrix
    rotM = np.array([
                    [math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]
                    ])

    # rotate and flip coordinates for all players
    for i in range(num_players):
        x,y = trial_copy[globals.PLAYER_LOC_DICT[i]['xloc']], trial_copy[globals.PLAYER_LOC_DICT[i]['yloc']]

        this_coordinates = np.vstack([x,y])
        this_altered_coordinates = np.matmul(rotM, this_coordinates) 

        # flip flag is a function parameter
        if flip:
            # flip coordinates around the x-axis if wall 0 is CCW of wall 1
            walls = get_indices.get_walls(trial=trial, trial_list=None, trial_index=None, num_walls=2)

            # calculate whether wall 0 is CCW of wall 1
            # calculate counterclockwise distance (moving from wall 0 to wall 1)
            if walls[1] < walls[0]:
                counterclockwise_distance = walls[0] - walls[1]
            else:
                counterclockwise_distance = (num_walls - walls[1]) + walls[0]
            
            # calculate clockwise distance
            clockwise_distance = num_walls - counterclockwise_distance

            # if wall 1 is closer counterclockwise from wall 0 than clockwise, we say it is CCW
            # of wall 1 and flip the x coordinates to correct
            # NB a wall separation of 4 is neither CW or CCW, but still does not fulfill the below condition
            if counterclockwise_distance < clockwise_distance:
                this_altered_coordinates = flip_trajectories(this_altered_coordinates)
   
        altered_coordinates.append(this_altered_coordinates)

    return altered_coordinates
    


# In[342]:


def flip_trajectories(altered_coordinates):
    ''' If wall 0 is CW of wall 1, flip the x coordinate
        of the trajectory data around. This keeps wall 0
        CCW of wall 1 '''
    
    altered_coordinates[0] = -altered_coordinates[0]

    return altered_coordinates
    


# In[344]:


def replace_with_altered_coordinates(trial, altered_coordinates):
    ''' Replace (in copy) the location coordinates for each player with the altered
        coordinates (rotated and/or flipped)
        Altered coordinates expects a list of np arrays which have a row for x coordinates
        and a row for y coordinates '''
    
    trial_copy = trial.copy()

    # overwrite the x location and y location columns in a copy of the dataframe for this trial
    for i in range(len(altered_coordinates)):
        trial_copy[globals.PLAYER_LOC_DICT[i]['xloc']] = altered_coordinates[i][0] # x coordinates
        trial_copy[globals.PLAYER_LOC_DICT[i]['yloc']] = altered_coordinates[i][1] # y coordinates

    return trial_copy
        


# In[348]:


# umbrella function
def flip_rotate_trajectories(trial=None, trial_list=None, trial_index=None):
    ''' Pipeline for flipping and rotating trajectories for a single trial
        Return a copy of that trial '''
    
    trial = extract_trial.extract_trial(trial=trial, trial_list=trial_list, trial_index=trial_index)
    
    theta = find_rotation_angle_trial(trial)
    altered_coords = flip_rotate_trial(trial, theta)
    trial_copy = replace_with_altered_coordinates(trial, altered_coords)
    
    return trial_copy

