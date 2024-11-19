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
from scipy import signal


# In[ ]:


def get_player_direction_vectors_for_trajectory(trajectory):
    ''' Calculate player direction vectors for a whole trajectory
        Takes a 2*timepoints array of vstacked x_coords and y_coords
        Returns an array of shape 2*(timepoints-1) '''
    
    # calculate direction vector between two points
    timepoints = trajectory.shape[1]
    direction_vectors = np.zeros([2,timepoints-1]) 
    for i in range(timepoints - 1):
        direction_vector = trajectory[:,i+1] - trajectory[:,i] # direction vector between 2 consecutive points
        direction_vectors[:,i] = direction_vector

    return direction_vectors


# In[ ]:


## Mean average window
# def get_smoothed_player_direction_vectors_for_trajectory(trajectory, window_size=10):
#     ''' Calculate smoothed player direction vectors for a whole trajectory
#         Return an array of shape 2*timepoints-window_size
#         Default window size = 10 '''
    
#     direction_vectors = get_player_direction_vectors_for_trajectory(trajectory)

#     try:
#         # mean average the rolling window of window_size direction vectors
#         timepoints = trajectory.shape[1]
#         direction_vectors_smoothed = np.zeros([2,timepoints-window_size])
#         for i in range(timepoints - window_size):
#             smoothed_direction_vector = np.mean(direction_vectors[:,i:i+window_size], axis=1) # take the mean across columns
#             direction_vectors_smoothed[:,i] = smoothed_direction_vector
#     except ValueError:
#         print("Direction vector too short to smooth, taking raw direction vector instead")
#         direction_vectors_smoothed = direction_vectors

#     return direction_vectors_smoothed


# In[ ]:


#Savitzky-Golay 
def get_smoothed_player_direction_vectors_for_trajectory(trajectory, window_size=5, debug=False):
    ''' Calculate smoothed player direction vectors for a whole trajectory
        Return an array of shape 2*timepoints-window_size
        Default window size = 10 '''
    
    direction_vectors = get_player_direction_vectors_for_trajectory(trajectory)

    try:
        # apply savgol filter to the full trajectory
        timepoints = trajectory.shape[1]
        direction_vectors_smoothed = np.zeros([2,timepoints-window_size])
        direction_vectors_smoothed = signal.savgol_filter(direction_vectors, window_length=5, polyorder=3, axis=1)
        for i in range(timepoints - window_size):
            smoothed_direction_vector = np.mean(direction_vectors[:,i:i+window_size], axis=1) # take the mean across columns
            direction_vectors_smoothed[:,i] = smoothed_direction_vector
    except ValueError:
        if debug:
            print("Direction vector too short to smooth, taking raw direction vector instead")
        direction_vectors_smoothed = direction_vectors

    return direction_vectors_smoothed


# In[ ]:


# Umbrella function for getting cosine similarities for player direction vector to player-to-alcove vectors
# for an entire trial
# stored in a num_walls*timepoints shaped array

def cosine_similarity_throughout_trajectory(trajectory, window_size=10, num_walls=8, calculate_thetas=False, debug=False):
    ''' From a trajectory, calculate the cosine similarity between the player direction vector and 
        the player-to-alcove vectors for an entire trial
        Takes a 2*timepoints array of vstacked x_coords and y_coords
        Returns an array of shape num_walls*timepoints 
        if calculate_thetas, also returns an angles (rad) array of shape num_walls*timepoints'''

    # 1. find the direction vectors for a player at each timepoint, smoothed with a rolling window
    smoothed_player_vectors = get_smoothed_player_direction_vectors_for_trajectory(trajectory,
                                                                                   window_size=window_size)
    if debug:
        print("smoothed_player_vectors.shape: ", smoothed_player_vectors.shape)

    
    # 2. find the player-to-alcove vectors for each wall, for each timepoint
    player_to_alcove_vectors = trajectory_vectors.get_player_to_alcove_direction_vectors_for_trajectory(smoothed_player_vectors,
                                                                                                         num_walls=num_walls)
    if debug:
        print("player_to_alcove_vectors.shape: ", player_to_alcove_vectors.shape)


    
    # 3. calculate the dot products between the two sets of vectors 
    dot_products_trajectory = trajectory_vectors.calculate_vector_dot_products_for_trajectory(player_to_alcove_vectors,
                                                                           smoothed_player_vectors,
                                                                           num_walls=num_walls)
    if debug:
        print("dot_products_trajectory.shape: ", dot_products_trajectory.shape)
        print("dot_products_trajectory\n", dot_products_trajectory[:,:10])
    
    
    # 4. calculate the norms for the two sets of vectors
    (direction_vector_norms_trajectory,
     player_to_alcove_vector_norms_trajectory) = trajectory_vectors.calculate_vector_norms_for_trajectory(player_to_alcove_vectors,
                                                                                       smoothed_player_vectors,
                                                                                       num_walls=8)
    if debug:
        print("direction_vector_norms_trajectory.shape: ", direction_vector_norms_trajectory.shape)
        print("player_to_alcove_vector_norms_trajectory.shape: ", player_to_alcove_vector_norms_trajectory.shape)
        
        print("direction_vector_norms_trajectory\n", direction_vector_norms_trajectory[:10])
        print("player_to_alcove_vector_norms_trajectory\n", player_to_alcove_vector_norms_trajectory[:,:10])
    
    
    # 5. calculate cosine similarity for the direction vector as compared to the vector from the player to each wall
    # this is done for all timepoints in a trajectory
    cosine_similarities_trajectory = trajectory_vectors.calculate_cosine_similarity_for_trajectory(dot_products_trajectory,
                                                                                 direction_vector_norms_trajectory,
                                                                                 player_to_alcove_vector_norms_trajectory,
                                                                                 num_walls=8)
    
    if debug:
        print("cosine_similairities_trajectory.shape: ", cosine_similarities_trajectory.shape)


    # return a num_walls*timepoints shaped array of cosine similarities
    # additionally return a num_walls*timepoints shapped array of angles between the direction vectors and the player-to-wall 
    # vectors if specified
    if calculate_thetas:
        thetas_trajectory = trajectory_vectors.calculate_thetas_for_trajectory(cosine_similarities_trajectory)
        return cosine_similarities_trajectory, thetas_trajectory
    else:
        return cosine_similarities_trajectory
    
    

