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


# In[339]:


## flip and rotate trials ##


# In[340]:


def find_rotation_angle_trial(trial_list, trial_index):
    """ Find CCW angle of rotation for vector to 
    rotate arena s.t. high wall is at wall 1"""

    trial = trial_list[trial_index]
    
    # identify trial walls
    wall1 = trial.iloc[0]['data.wall1']
    wall2 = trial.iloc[0]['data.wall2']
    
    # find difference of high wall to wall 1
    difference = wall1 - 1
        
    # find CCW rotation angle 
    unitary_rotation_ang = 2*math.pi/globals.NUM_WALLS
    theta = unitary_rotation_ang * difference

    return theta
    


# In[343]:


def flip_rotate(trial_list, trial_index, theta, flip=True):
    """ Rotate x-y coordinates by theta 
        Flip x coordinates of vector if wall2 CCW of wall1
        Return altered vector """

    num_walls = globals.NUM_WALLS
    
    altered_coordinates = []
    trial = trial_list[trial_index]
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
            # flip coordinates around the x-axis if wall2 is CCW of wall1
            walls = get_indices.get_walls(trial=None, trial_list=None, trial_index=None, num_walls=2):

            # calculate whether wall 1 is CCW of wall 2
            # calculate counterclockwise distance (moving from wall 1 to wall 2)
            if walls[1] < walls[0]:
                counterclockwise_distance = walls[0] - walls[1]
            else:
                counterclockwise_distance = (num_walls - walls[1]) + walls[0]
            
            # calculate clockwise distance
            clockwise_distance = num_walls - counterclockwise_distance

            if counterclockwise_distance < clockwise_distance:
                this_altered_coordinates = flip_trajectories(this_altered_coordinates)
            
            # if walls[0] > walls[1]:
            #     this_altered_coordinates = flip_trajectories(this_altered_coordinates)

        altered_coordinates.append(this_altered_coordinates)

    return altered_coordinates
    


# In[342]:


def flip_trajectories(altered_coordinates):
    ''' If wall1 is CW of wall2, flip the x coordinate
        of the trajectory data around. This keeps wall1
        CCW of wall2 '''
    
    altered_coordinates[0] = -altered_coordinates[0]

    return altered_coordinates
    


# In[344]:


def replace_with_altered_coordinates(trial_list, trial_index, altered_coordinates):
    trial = trial_list[trial_index]
    trial_copy = trial.copy()

    # overwrite the x location and y location columns in a copy of the dataframe for this trial
    for i in range(len(altered_coordinates)):
        trial_copy[globals.PLAYER_LOC_DICT[i]['xloc']] = altered_coordinates[i][0] # x coordinates
        trial_copy[globals.PLAYER_LOC_DICT[i]['yloc']] = altered_coordinates[i][1] # y coordinates

    return trial_copy
        


# In[348]:


# umbrella function
def flip_rotate_trajectories(trial_list, trial_index=0):
    ''' Pipeline for flipping and rotating trajectories for a single trial
        Return a copy of that trial '''
    
    theta = find_rotation_angle_trial(trial_list, trial_index)
    altered_coords = flip_rotate(trial_list, trial_index, theta)
    trial_copy = replace_with_altered_coordinates(trial_list, trial_index, altered_coords)
    
    return trial_copy

