#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[ ]:


def get_player_direction_vectors_for_trajectory(trajectory):
    ''' Calculate player direction vectors for a whole trajectory
        Returns an array of shape 2*trajectory.shape[1]-1 '''
    
    # calculate direction vector between two points
    direction_vectors = np.zeros([2,trajectory.shape[1]-1]) 
    for i in range(trajectory.shape[1] - 1):
        direction_vector = trajectory[:,i+1] - trajectory[:,i] # direction vector between 2 consecutive points
        direction_vectors[:,i] = direction_vector

    return direction_vectors


# In[2]:


def get_smoothed_player_direction_vectors_for_trajectory(trajectory, window_size=10):
    ''' Calculate smoothed player direction vectors for a whole trajectory
        Return an array of shape 2*direction_vectors.shape[1]-window_size '''
    
    # calculate direction vector between two points
    direction_vectors = np.zeros([2,trajectory.shape[1]-1])
    for i in range(trajectory.shape[1] - 1):
        direction_vector = trajectory[:,i+1] - trajectory[:,i] # direction vector between 2 consecutive points
        direction_vectors[:,i] = direction_vector

    try:
        # mean average the rolling window of window_size direction vectors
        direction_vectors_smoothed = np.zeros([2,direction_vectors.shape[1]-window_size])
        for i in range(direction_vectors.shape[1] - window_size):
            smoothed_direction_vector = np.mean(direction_vectors[:,i:i+window_size], axis=1) # take the mean across columns
            direction_vectors_smoothed[:,i] = smoothed_direction_vector
    except ValueError:
        print("Direction vector too short to smooth, taking raw direction vector instead")
        direction_vectors_smoothed = direction_vectors

    return direction_vectors_smoothed


# In[2]:


# Umbrella function for getting cosine similarities for player direction vector to player-to-alcove vectors
# for an entire trial
# stored in a num_walls*timepoints shaped array

def cosine_similarity_throughout_trajectory(trajectory, window_size=10, num_walls=8, calculate_thetas=False):
    ''' From a trajectory, calculate the cosine similarity between the player direction vector and 
        the player-to-alcove vectors for an entire trial
        Returns an array of shape num_walls*timepoints '''

    # 1. find the direction vectors for a player at each timepoint, smoothed with a rolling window
    smoothed_player_vectors = get_smoothed_player_direction_vectors_for_trajectory(trajectory,
                                                                                   window_size=10)
    # print("smoothed_player_vectors.shape: ", smoothed_player_vectors.shape)

    
    # 2. find the player-to-alcove vectors for each wall, for each timepoint
    player_to_alcove_vectors = trajectory_vectors.get_player_to_alcove_direction_vectors_for_trajectory(smoothed_player_vectors,
                                                                                                         num_walls=num_walls)
    # print("player_to_alcove_vectors.shape: ", player_to_alcove_vectors.shape)


    
    # 3. calculate the dot products between the two sets of vectors 
    dot_products_trajectory = trajectory_vectors.calculate_vector_dot_products_for_trajectory(player_to_alcove_vectors,
                                                                           smoothed_player_vectors,
                                                                           num_walls=num_walls)
    # print("dot_products_trajectory.shape: ", dot_products_trajectory.shape)
    # print("dot_products_trajectory\n", dot_products_trajectory[:,:10])
    
    # 4. calculate the norms for the two sets of vectors
    (direction_vector_norms_trajectory,
     player_to_alcove_vector_norms_trajectory) = trajectory_vectors.calculate_vector_norms_for_trajectory(player_to_alcove_vectors,
                                                                                       smoothed_player_vectors,
                                                                                       num_walls=8)
    
    # print("direction_vector_norms_trajectory.shape: ", direction_vector_norms_trajectory.shape)
    # print("player_to_alcove_vector_norms_trajectory.shape: ", player_to_alcove_vector_norms_trajectory.shape)
    
    # print("direction_vector_norms_trajectory\n", direction_vector_norms_trajectory[:10])
    # print("player_to_alcove_vector_norms_trajectory\n", player_to_alcove_vector_norms_trajectory[:,:10])
    
    
    # 5. calculate cosine similarity for the direction vector as compared to the vector from the player to each wall
    # this is done for all timepoints in a trajectory
    cosine_similarities_trajectory = trajectory_vectors.calculate_cosine_similarity_for_trajectory(dot_products_trajectory,
                                                                                 direction_vector_norms_trajectory,
                                                                                 player_to_alcove_vector_norms_trajectory,
                                                                                 num_walls=8)

    # print("cosine_similairities_trajectory.shape: ", cosine_similairities_trajectory.shape)


    # return a num_walls*timepoints shaped array of cosine similarities
    # additionally return a num_walls*timepoints shapped array of angles between the direction vectors and the player-to-wall 
    # vectors if specified
    if calculate_thetas:
        thetas_trajectory = trajectory_vectors.calculate_thetas_for_trajectory(cosine_similarities_trajectory)
        return cosine_similarities_trajectory, thetas_trajectory
    else:
        return cosine_similarities_trajectory
    
    

