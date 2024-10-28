#!/usr/bin/env python
# coding: utf-8

# In[9]:


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


# In[6]:


def extract_trial_player_headangles(trial_list=None, trial_index=0, trial=None, player_id=0):
    ''' return a 2xN array of the x- and y- coordinates for a single player's trial trajectory
        from slice onset to selected trigger activation '''

    # get trial dataframe
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)

    # get slice onset index, referenced to trial start index
    slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    slice_onset_index = slice_onset.index[0] - trial.index[0]

    # get trigger activation index, referenced to trial start
    selected_trigger_activation = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    selected_trigger_activation_index = selected_trigger_activation.index[0] - trial.index[0]

    # find the euler angles for the rotation around the y (Unity vertical) axis
    y_rotation = trial[globals.PLAYER_ROT_DICT[player_id]['yrot']].iloc[slice_onset_index:selected_trigger_activation_index]
    y_rotation = np.deg2rad(y_rotation)

    # convert this angle into a unit vector in 2-dimensional (XY space)
    x_components = np.zeros(y_rotation.size)
    z_components = np.zeros(y_rotation.size)
    x_components[:] = np.sin(y_rotation)
    z_components[:] = np.cos(y_rotation)
    head_angle_vector_array = np.vstack([x_components, z_components])

    return head_angle_vector_array


# In[7]:


def get_smoothed_player_head_angle_vectors_for_trajectory(head_angle_vector_array, window_size=10):
    ''' Calculate smoothed player head angle vectors for a whole trajectory '''

    # head angle vectors with a mean average rolling window of window_size 
    window_size =10
    head_angle_vector_array_smoothed = np.zeros([2,head_angle_vector_array.shape[1]-window_size])
    for i in range(head_angle_vector_array.shape[1] - window_size):
        smoothed_head_angle_vector = np.mean(head_angle_vector_array[:,i:i+window_size], axis=1)
        head_angle_vector_array_smoothed[:,i] = smoothed_head_angle_vector

    return head_angle_vector_array_smoothed


# In[8]:


# Umbrella function for getting angle difference between FoV centre and walls for a player
# for an entire trial
# stored in a num_walls*timepoints shaped array

def head_angle_to_walls_throughout_trajectory(trajectory, head_angle_vector_array_trajectory, window_size=10, num_walls=8):
    ''' From a trajectory, calculate the angles between the player head angle vector and 
        the player-to-alcove vectors for an entire trial
        Returns an array of shape num_walls*timepoints '''

    # 1. find head angle unit vectors for a player at each timepoint, smoothed with a rolling window
    smoothed_player_head_angles = get_smoothed_player_head_angle_vectors_for_trajectory(head_angle_vector_array_trajectory,
                                                                                        window_size=10)
    print("smoothed_player_head_angles.shape: ", smoothed_player_head_angles.shape)
    print("smoothed_player_head_angles\n", smoothed_player_head_angles[:,110:120])

    # 2. find the player-to-alcove vectors for each wall, for each timepoint
    player_to_alcove_vectors = trajectory_vectors.get_player_to_alcove_direction_vectors_for_trajectory(trajectory,
                                                                                                     num_walls=num_walls)
    
    print("player_to_alcove_vectors.shape: ", player_to_alcove_vectors.shape)
    print("player_to_alcove_vectors\n", player_to_alcove_vectors[:,1,110:120])
    
    # 3. calculate the dot products between the two sets of vectors 
    dot_products_trajectory = trajectory_vectors.calculate_vector_dot_products_for_trajectory(player_to_alcove_vectors,
                                                                                   smoothed_player_head_angles,
                                                                                   num_walls=num_walls)

    print("dot_products_trajectory.shape: ", dot_products_trajectory.shape)
    print("dot_products_trajectory\n", dot_products_trajectory[:,110:120]) 


    
    # 4. calculate the norms for the two sets of vectors
    (head_angle_vector_norms_trajectory,
     player_to_alcove_vector_norms_trajectory) = trajectory_vectors.calculate_vector_norms_for_trajectory(player_to_alcove_vectors,
                                                                                                   smoothed_player_head_angles,
                                                                                                   num_walls=8)

    print("head_angle_vector_norms_trajectory\n", head_angle_vector_norms_trajectory[110:120])
    print("player_to_alcove_vector_norms_trajectory\n", player_to_alcove_vector_norms_trajectory[:,110:120])
    
    print("head_angle_vector_norms_trajectory.shape: ", head_angle_vector_norms_trajectory.shape)
    print("player_to_alcove_vector_norms_trajectory.shape: ", player_to_alcove_vector_norms_trajectory.shape)

    # 5. calculate cosine similarity for the head angle vector as compared to the vector from the player to each wall
    # this is done for all timepoints in a trajectory
    cosine_similairities_trajectory = trajectory_vectors.calculate_cosine_similarity_for_trajectory(dot_products_trajectory,
                                                                                             head_angle_vector_norms_trajectory,
                                                                                             player_to_alcove_vector_norms_trajectory,
                                                                                             num_walls=8)

    print("cosine_similairities_trajectory.shape: ", cosine_similairities_trajectory.shape)

    # 6. calculate angles between player head direction and player-to-alcove vectors for each wall
    thetas = trajectory_vectors.calculate_thetas_for_trajectory(cosine_similairities_trajectory, num_walls=8)

    return thetas
    

