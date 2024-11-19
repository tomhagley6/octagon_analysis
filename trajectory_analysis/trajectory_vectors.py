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


# These functions are shared between head angle vector analysis and direction vector analysis #


# In[ ]:


def extract_trial_player_trajectory(trial_list=None, trial_index=0, trial=None, player_id=0, debug=False):
    ''' Returns a 2xtimepoints array of vstacked x_coords and y_coords for a single player's trial trajectory
        from slice onset to selected trigger activation '''
    
    if debug:
        print(f"Extracting trial with trial index {trial_index}")
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)
    assert isinstance(trial, pd.DataFrame)

    if debug:
        print(f"Trial is of type {type(trial)}")
        print(f"events are: {trial['eventDescription'].unique()}")
        print(f"Globals.sliceonset returns as {globals.SLICE_ONSET}")

    # get slice onset index, referenced to trial start index
    if debug:
        print(f"Index for slice onset is: {trial['eventDescription'] == globals.SLICE_ONSET}")
        print(f"Number of True elements in this is {np.sum(trial['eventDescription'] == globals.SLICE_ONSET)} ")
    slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    if debug:
        print(f"slice_onset is {slice_onset.index[0]} type {type(slice_onset.index[0])}\n and trial index is {trial.index[0]}")

    if not isinstance(slice_onset, int):

        slice_onset_index = slice_onset.index[0] - trial.index[0]
    else:
        slice_onset_index = slice_onset.index - trial.index[0]

    # get trigger activation index, referenced to trial start
    selected_trigger_activation = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    if debug:
        print(f"selected trigger activation is {selected_trigger_activation.index[0]} type {type(selected_trigger_activation.index[0])}\n and trial index is {trial.index[0]}")
    selected_trigger_activation_index = selected_trigger_activation.index[0] - trial.index[0]
    
    # access the x and y locations stored in the player location dictionary indexed at the current player id
    x_coordinates = trial[globals.PLAYER_LOC_DICT[player_id]['xloc']].iloc[slice_onset_index:selected_trigger_activation_index]
    y_coordinates = trial[globals.PLAYER_LOC_DICT[player_id]['yloc']].iloc[slice_onset_index:selected_trigger_activation_index]

    coordinate_array = np.vstack([x_coordinates, y_coordinates])

    return coordinate_array


# In[ ]:


def extract_trial_player_headangles(trial_list=None, trial_index=0, trial=None, player_id=0, debug=False):
    ''' Returns a timepoints-sized array of head direction Euler angles for a single player's trial
        from slice onset to server-selected trigger activation '''

    # get trial dataframe
    if debug:
        print(f"Extracting trial with trial index {trial_index}")
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)
    assert isinstance(trial, pd.DataFrame)


    # get slice onset index, referenced to trial start index
    if debug:
        print(f"trial variable is of type {type(trial)}")
        print(f"events are: {trial['eventDescription'].unique()}")
    slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    if debug:
        print(f"slice_onset is {slice_onset.index[0]} type {type(slice_onset.index[0])}\n and trial index is {trial.index[0]}")
    slice_onset_index = slice_onset.index[0] - trial.index[0]


    # get trigger activation index, referenced to trial start
    selected_trigger_activation = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    if debug:
        print(f"selected trigger activation is {selected_trigger_activation.index[0]}, type {type(selected_trigger_activation.index[0])}\n and trial index is {trial.index[0]}")
    selected_trigger_activation_index = selected_trigger_activation.index[0] - trial.index[0]


    # find the euler angles for the rotation around the y (Unity vertical) axis
    y_rotation = trial[globals.PLAYER_ROT_DICT[player_id]['yrot']].iloc[slice_onset_index:selected_trigger_activation_index]
    head_angles = np.deg2rad(y_rotation)

    return head_angles


# In[ ]:


def extract_trial_player_trajectory_full(trial_list=None, trial_index=0, trial=None, player_id=0):
    ''' Returns a 2xtimepoints array of vstacked x_coords and y_coords for a single player's trial trajectory
        from trial start to trial end '''
    
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)

    # get trial index, referenced to trial start index
    trial_start = trial[trial['eventDescription'] == globals.TRIAL_START]
    trial_start_index = trial_start.index[0] - trial.index[0]

    # get trial end index, referenced to trial start
    trial_end = trial[trial['eventDescription'] == globals.TRIAL_END]
    trial_end_index = trial_end.index[0] - trial.index[0]

    # access the x and y locations stored in the player location dictionary indexed at the current player id
    x_coordinates = trial[globals.PLAYER_LOC_DICT[player_id]['xloc']].iloc[trial_start_index:trial_end_index]
    y_coordinates = trial[globals.PLAYER_LOC_DICT[player_id]['yloc']].iloc[trial_start_index:trial_end_index]

    coordinate_array = np.vstack([x_coordinates, y_coordinates])

    return coordinate_array


# In[ ]:


def extract_trial_player_headangles_full(trial_list=None, trial_index=0, trial=None, player_id=0):
    ''' Returns a timepoints-sized array of head direction Euler angles for a single player's trial
        from trial start to trial end  '''

    # get trial dataframe
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)

    # get trial start index, referenced to trial start index
    trial_start = trial[trial['eventDescription'] == globals.TRIAL_START]
    trial_start_index = trial_start.index[0] - trial.index[0]

    # get trial end index, referenced to trial start
    trial_end = trial[trial['eventDescription'] == globals.TRIAL_END]
    trial_end_index = trial_end.index[0] - trial.index[0]

    # find the Euler angles for the rotation around the y (Unity vertical) axis
    y_rotation = trial[globals.PLAYER_ROT_DICT[player_id]['yrot']].iloc[trial_start_index:trial_end_index]
    y_rotation = np.deg2rad(y_rotation)
    head_angles = y_rotation

    return head_angles


# In[ ]:


def get_player_to_alcove_direction_vectors_for_trajectory(trajectory, num_walls=8, debug=False):
    ''' Calculate the direction vector between player and the centre of alcoves
        Input requires the 2xtimepoints trajectory of vstacked x_coord, y_coord
        Returns a 3D array of shape 2*num_walls*timepoints '''
  
    # get the central point of each alcove
    alcove_centre_points = plot_octagon.return_alcove_centre_points()
    
    # calculate the vector between the alcove point and current player location
    timepoints = trajectory.shape[1]
    vector_to_alcoves = np.zeros((2, num_walls, timepoints))
    for time_index in range(0, trajectory.shape[1]): # for each timepoint in trajectory
        player_x_loc = trajectory[0,time_index]
        player_y_loc = trajectory[1,time_index]
        if debug:
            print("player x/y loc for this timepoint: ", player_x_loc, player_y_loc)

    
        for wall_num in range(num_walls): # for each wall
            # euclidean vector from point B (trajectory location) to point A (alcove centre location) 
            # for this wall
            vector_to_alcove = alcove_centre_points[:, wall_num] - trajectory[:, time_index]
            vector_to_alcoves[:,wall_num,time_index] = vector_to_alcove # append

    return vector_to_alcoves


# In[ ]:


def calculate_vector_dot_products_for_trajectory(vector_to_alcoves, player_vectors_smoothed, num_walls=8):
    ''' Find the dot product between the player vector and the player-to-alcove direction vector
        for each wall, for a full trajectory
        Takes a 2*num_walls*timepoints vector_to_alcoves array, and a 2*timepoints player_vectors_smoothed array
        Returns an array of shape num_walls*timepoints '''

    timepoints = player_vectors_smoothed.shape[1]
    dot_products_trajectory = np.zeros([num_walls, timepoints])
    for timepoint in range(timepoints):
        dot_products_timepoint = calculate_vector_dot_products_for_timepoint(vector_to_alcoves=vector_to_alcoves,
                                                                             player_vectors_smoothed=player_vectors_smoothed,
                                                                             timepoint=timepoint,
                                                                             num_walls=num_walls)
        dot_products_trajectory[:,timepoint] = dot_products_timepoint

    return dot_products_trajectory


# In[ ]:


def calculate_vector_dot_products_for_timepoint(vector_to_alcoves, player_vectors_smoothed, timepoint, num_walls=8):
    ''' Find the dot product between the player vector and the player-to-alcove direction vector
        for each wall, for a single timepoint
        Takes 2*num_walls*timepoints vector_to_alcoves array, a 2*timepoints player_vectors_smoothed array,
        and scalar timepoint
        Returns a 1D array of size num_walls '''
    
    dot_products_timepoint = np.zeros(num_walls)
    for wall_num in range(num_walls):
        dot_product = np.dot(vector_to_alcoves[:,wall_num,timepoint], player_vectors_smoothed[:,timepoint])
        dot_products_timepoint[wall_num] = dot_product

    return dot_products_timepoint


# In[ ]:


def calculate_vector_norms_for_trajectory(vector_to_alcoves, player_vectors_smoothed, num_walls=8):
    ''' Find the norms for the player vector and the player-to-alcove direction vectors
        for each wall, for a full trajectory
        Takes 2*num_walls*timepoints vector_to_alcoves array, a 2*timepoints player_vectors_smoothed array,
        and scalar timepoint
        Returns 1*trajectory_length player_vector_norms_trajectory array
        and num_walls*trajectory player_to_alcove_vector_norms_trajectory '''
    
    timepoints = player_vectors_smoothed.shape[1]
    player_vector_norms_trajectory = np.zeros([timepoints])
    player_to_alcove_vector_norms_trajectory = np.zeros([num_walls, timepoints])
    
    for timepoint in range(timepoints):
        (direction_vector_norm_timepoint,
        player_to_alcove_vector_norms_timepoint) = calculate_vector_norms_for_timepoint(vector_to_alcoves=vector_to_alcoves,
                                                                                        player_vectors_smoothed=player_vectors_smoothed,
                                                                                        timepoint=timepoint,
                                                                                        num_walls=num_walls)
                        
        player_vector_norms_trajectory[timepoint] = direction_vector_norm_timepoint
        player_to_alcove_vector_norms_trajectory[:,timepoint] = player_to_alcove_vector_norms_timepoint

    return player_vector_norms_trajectory, player_to_alcove_vector_norms_trajectory


# In[ ]:


def calculate_vector_norms_for_timepoint(vector_to_alcoves, player_vectors_smoothed, timepoint, num_walls=8):
    ''' Return the norm of the player vector and the player-to-alcove direction vector each 
        wall, for a single timepoint.
        Returns a scalar direction_vector_norm and a 1D player_to_alcove_vector_norms array of size num_walls '''

    # find norm of direction vector
    direction_vector_norm = np.linalg.norm(player_vectors_smoothed[:,timepoint])
    
    # find norms of all of the player-to-alcove vectors
    player_to_alcove_vector_norms = np.zeros(num_walls)
    for wall_num in range(num_walls):
        player_to_alcove_vector_norm = np.linalg.norm(vector_to_alcoves[:,wall_num,timepoint])
        player_to_alcove_vector_norms[wall_num] = player_to_alcove_vector_norm

    return direction_vector_norm , player_to_alcove_vector_norms


# In[ ]:


def calculate_cosine_similarity_for_trajectory(dot_products, player_vector_norms_trajectory,
                                               player_to_alcove_vector_norms_trajectory, num_walls=8):
    '''Calculate the cosine similarities between a given player vector and player-to-alcove direction vector
       for each wall, for an entire trajectory
       Takes a 2*wall_num*timepoints dot_products array, a timepoints-sized player_vector_norms_trajectory array,
       and a num_walls*timepoints player_to_alcove_vector_norms_trajectory array
       Returns an array of shape num_walls*timepoints '''
    
    timepoints = player_to_alcove_vector_norms_trajectory.shape[1]
    cosine_similarities = np.zeros([num_walls, timepoints])
    for timepoint in range(timepoints):
        cosine_similarities[:,timepoint] = calculate_cosine_similarity_for_timepoint(dot_products[:,timepoint],
                                                                                    player_vector_norms_trajectory[timepoint],
                                                                                    player_to_alcove_vector_norms_trajectory[:,timepoint],
                                                                                    num_walls=8)

    return cosine_similarities
        


# In[ ]:


def calculate_cosine_similarity_for_timepoint(dot_product, player_vector_norm, player_to_alcove_vector_norms, num_walls=8):
    ''' Find the cosine similarity between a given player vector and player-to-alcove direction vector
        for each wall
        Takes a 2*wall_num dot_product array, a scalar player_vector_norm array,
        and a num_walls sized player_to_alcove_vector_norms array
        Returns a 1D array of size num_walls'''
    
    cosine_similarities = np.zeros(num_walls)
    for wall_num in range(num_walls):
        # cosine_similarity = dot product/(alcove vector norm * player vector norm) for this wall
        cosine_similarity_this_wall = dot_product[wall_num]/(player_to_alcove_vector_norms[wall_num] * player_vector_norm)
        cosine_similarities[wall_num] = cosine_similarity_this_wall

    return cosine_similarities


# In[ ]:


def calculate_thetas_for_trajectory(cosine_similarities_for_trajectory, num_walls=8):
    ''' Find the angles between a given player vector and player-to-alcove direction vector
        for each wall, from calculated cosine similarity. This is done for each time point in a trajectory
        Takes a num_walls*timepoints cosine_similarities_for_trajectory array
        Return an array of shape num_walls*timepoints'''

    timepoints = cosine_similarities_for_trajectory.shape[1]
    trajectory_thetas = np.zeros([num_walls, timepoints])
    for timepoint in range(timepoints):
        cosine_similarities_timepoint = cosine_similarities_for_trajectory[:,timepoint]
        trajectory_thetas[:,timepoint] = calculate_thetas_for_timepoint(cosine_similarities_timepoint)
    
    return trajectory_thetas


# In[ ]:


def calculate_thetas_for_timepoint(cosine_similarities):
    ''' Find the angles between a given player vector and player-to-alcove direction vector
        for each wall, from calculated cosine similarity
        Takes a num_walls sized cosine_similarities array for a single timepoint
        Returns a same-sized array of angles in radians'''

    return [math.acos(val) for val in cosine_similarities]

