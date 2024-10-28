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


# In[ ]:


### A collection of functions used to extract and work on vectors related to player trajectories ### 


# In[ ]:


# These functions are shared between head angle vector analysis and direction vector analysis


# In[ ]:


def extract_trial_player_trajectory(trial_list=None, trial_index=0, trial=None, player_id=0):
    ''' return a 2xN array of the x- and y- coordinates for a single player's trial trajectory
        from slice onset to selected trigger activation '''
    
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)

    # get slice onset index, referenced to trial start index
    slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    slice_onset_index = slice_onset.index[0] - trial.index[0]

    # get trigger activation index, referenced to trial start
    selected_trigger_activation = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    selected_trigger_activation_index = selected_trigger_activation.index[0] - trial.index[0]

    
    x_coordinates = trial[globals.PLAYER_LOC_DICT[player_id]['xloc']].iloc[slice_onset_index:selected_trigger_activation_index]
    y_coordinates = trial[globals.PLAYER_LOC_DICT[player_id]['yloc']].iloc[slice_onset_index:selected_trigger_activation_index]

    coordinate_array = np.vstack([x_coordinates, y_coordinates])

    return coordinate_array


# In[ ]:


def get_player_to_alcove_direction_vectors_for_trajectory(trajectory, num_walls=8):
    ''' Calculate the direction vector between player and the centre of alcoves
        Input requires the smoothed direction vectors of the player for a full trajectory
        Returns a 3-dimensionl array of shape 2*num_walls*trajectory.shape[1] '''
  
    # get the central point of each alcove
    alcove_centre_points = plot_octagon.return_alcove_centre_points()
    
    # calculate the vector between the alcove point and current player location
    vector_to_alcoves = np.zeros([2, num_walls, trajectory.shape[1]])
    for time_index in range(0, trajectory.shape[1]): # for each timepoint in trajectory
        player_x_loc = trajectory[0,time_index]
        player_y_loc = trajectory[1,time_index]
        # print("player x/y loc for this timepoint: ", player_x_loc, player_y_loc)

    
        for wall_num in range(num_walls): # for each wall
            vector_to_alcove = alcove_centre_points[:, wall_num] - trajectory[:, time_index]
            vector_to_alcoves[:,wall_num,time_index] = vector_to_alcove

    return vector_to_alcoves


# In[ ]:


def calculate_vector_dot_products_for_trajectory(vector_to_alcoves, direction_vectors_smoothed, num_walls=8):
    ''' Find the dot product between the player vector and the player-to-alcove direction vector
        for each wall, for a full trajectory
        Returns an array of shape num_walls*trajectory_length '''

    trajectory_length = direction_vectors_smoothed.shape[1]
    dot_products_trajectory = np.zeros([num_walls, trajectory_length])
    for timepoint in range(trajectory_length):
        dot_products_timepoint = calculate_vector_dot_products_for_timepoint(vector_to_alcoves=vector_to_alcoves,
                                                                             direction_vectors_smoothed=direction_vectors_smoothed,
                                                                             timepoint=timepoint,
                                                                             num_walls=num_walls)
        dot_products_trajectory[:,timepoint] = dot_products_timepoint

    return dot_products_trajectory


# In[ ]:


def calculate_vector_dot_products_for_timepoint(vector_to_alcoves, direction_vectors_smoothed, timepoint, num_walls=8):
    ''' Find the dot product between the player vector and the player-to-alcove direction vector
        for each wall, for a single timepoint
        Returns a 1D array of size num_walls '''
    
    dot_products_timepoint = np.zeros(num_walls)
    for wall_num in range(num_walls):
        dot_product = np.dot(vector_to_alcoves[:,wall_num,timepoint], direction_vectors_smoothed[:,timepoint])
        dot_products_timepoint[wall_num] = dot_product

    return dot_products_timepoint


# In[ ]:


def calculate_vector_norms_for_trajectory(vector_to_alcoves, direction_vectors_smoothed, num_walls=8):
    ''' Find the norms for the player vector and the player-to-alcove direction vectors
        for each wall, for a full trajectory
        Returns 1*trajectory_length and num_walls*trajectory length arrays '''
    
    trajectory_length = direction_vectors_smoothed.shape[1]
    direction_vector_norms_trajectory = np.zeros([trajectory_length])
    player_to_alcove_vector_norms_trajectory = np.zeros([num_walls, trajectory_length])
    
    for timepoint in range(trajectory_length):
        (direction_vector_norm_timepoint,
        player_to_alcove_vector_norms_timepoint) = calculate_vector_norms_for_timepoint(vector_to_alcoves=vector_to_alcoves,
                                                                                        direction_vectors_smoothed=direction_vectors_smoothed,
                                                                                        timepoint=timepoint,
                                                                                        num_walls=num_walls)
                        
        direction_vector_norms_trajectory[timepoint] = direction_vector_norm_timepoint
        player_to_alcove_vector_norms_trajectory[:,timepoint] = player_to_alcove_vector_norms_timepoint

    return direction_vector_norms_trajectory, player_to_alcove_vector_norms_trajectory


# In[ ]:


def calculate_vector_norms_for_timepoint(vector_to_alcoves, direction_vectors_smoothed, timepoint, num_walls=8):
    ''' Return the norm of the player vector and the player-to-alcove direction vector each 
        wall, for a single timepoint.
        Returns a scalar and a 1D array of size num_walls '''

    # find norm of direction vector
    direction_vector_norm = np.linalg.norm(direction_vectors_smoothed[:,timepoint])
    
    # find norms of all of the player-to-alcove vectors
    player_to_alcove_vector_norms = np.zeros(num_walls)
    for wall_num in range(num_walls):
        player_to_alcove_vector_norm = np.linalg.norm(vector_to_alcoves[:,wall_num,timepoint])
        player_to_alcove_vector_norms[wall_num] = player_to_alcove_vector_norm

    return direction_vector_norm, player_to_alcove_vector_norms


# In[ ]:


def calculate_cosine_similarity_for_trajectory(dot_products, direction_vector_norms_trajectory,
                                               player_to_alcove_vector_norms_trajectory, num_walls=8):
    '''Calculate the cosine similarities between a given player vector and player-to-alcove direction vector
       for each wall, for an entire trajectory
       Returns an array of shape num_walls*timepoints '''
    
    trajectory_length = player_to_alcove_vector_norms_trajectory.shape[1]
    cosine_similarities = np.zeros([num_walls, trajectory_length])
    for timepoint in range(trajectory_length):
        cosine_similarities[:,timepoint] = calculate_cosine_similarity_for_timepoint(dot_products[:,timepoint],
                                                                                    direction_vector_norms_trajectory[timepoint],
                                                                                    player_to_alcove_vector_norms_trajectory[:,timepoint],
                                                                                    num_walls=8)

    return cosine_similarities
        


# In[ ]:


def calculate_cosine_similarity_for_timepoint(dot_product, direction_vector_norm, player_to_alcove_vector_norms, num_walls=8):
    ''' Find the cosine similarity between a given player vector and player-to-alcove direction vector
        for each wall '''
    
    cosine_similarities = np.zeros(num_walls)
    for wall_num in range(num_walls):
        cosine_similarity_this_wall = dot_product[wall_num]/(player_to_alcove_vector_norms[wall_num] * direction_vector_norm)
        cosine_similarities[wall_num] = cosine_similarity_this_wall

    return cosine_similarities


# In[ ]:


def calculate_thetas_for_trajectory(cosine_similarities_for_trajectory, num_walls=8):
    ''' Find the angles between a given player vector and player-to-alcove direction vector
        for each wall, from calculated cosine similarity. This is done for each time point in a trajectory
        Return an array of shape num_walls*timepoints'''

    trajectory_length = cosine_similarities_for_trajectory.shape[1]
    trajectory_thetas = np.zeros([num_walls, trajectory_length])
    for timepoint in range(trajectory_length):
        cosine_similarities_timepoint = cosine_similarities_for_trajectory[:,timepoint]
        trajectory_thetas[:,timepoint] = calculate_thetas_for_timepoint(cosine_similarities_timepoint)
    
    return trajectory_thetas


# In[ ]:


def calculate_thetas_for_timepoint(cosine_similarities):
    ''' Find the angles between a given player vector and player-to-alcove direction vector
        for each wall, from calculated cosine similarity '''

    return [math.acos(val) for val in cosine_similarities]

