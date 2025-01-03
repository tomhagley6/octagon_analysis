#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import data_extraction.get_indices as get_indices
import utils.get_ordered_indices as get_ordered_indices
import time


# In[2]:


## HEADANGLES THROUGHOUT TRAJECTORY ##


# In[ ]:


def get_player_headangle_vectors_for_trial(head_angles):
    ''' Returns a 2xtimepoints array of vstacked x components and y components of head angle vector for a single player's trial trajectory
        from trial start to trial end
        This is the euclidean unit vector extracted from an Euler angle of head direction '''


    # convert this angle into a unit vector in 2-dimensional (XY space) 
    # (Y components here are named Z to separate from the y-axis above)
    x_components = np.zeros(head_angles.size)
    z_components = np.zeros(head_angles.size)
    x_components[:] = np.sin(head_angles)
    z_components[:] = np.cos(head_angles)
    head_angle_vector_array = np.vstack([x_components, z_components])

    return head_angle_vector_array


# In[ ]:


def get_smoothed_player_head_angle_vectors_for_trial(head_angles, window_size=5, debug=False):
    ''' Calculate smoothed player head angle vectors for a whole trajectory '''

    head_angle_vector_array = get_player_headangle_vectors_for_trial(head_angles)

    # mean average the rolling window of window_size head direction vectors 
    try:
        timepoints = head_angle_vector_array.shape[1]
        head_angle_vector_array_smoothed = np.zeros([2,timepoints-window_size])
        for i in range(timepoints - window_size):
            smoothed_head_angle_vector = np.mean(head_angle_vector_array[:,i:i+window_size], axis=1)
            head_angle_vector_array_smoothed[:,i] = smoothed_head_angle_vector
        
    except ValueError:
        head_angle_vector_array_smoothed = head_angle_vector_array
        if debug:
            print("head angle vector array too short to smooth, taking the raw array instead")
            print(f"Length of the unsmoothed head angle vector array is {head_angle_vector_array.shape[1]}")
        

    return head_angle_vector_array_smoothed


# In[5]:


## HEAD ANGLE COMPARED TO WALL CENTRES ##


# In[ ]:


# Umbrella function for getting angle difference between FoV centre and walls for a player
# for an entire trial

def head_angle_to_walls_throughout_trajectory(trajectory, head_angle_vector_array_trial, window_size=5, num_walls=8, debug=False):
    ''' From a trajectory, calculate the angles between the player head angle vector and 
        the player-to-alcove vectors for an entire trial
        Returns an array of angles of shape num_walls*timepoints '''

    # 1. find head angle unit vectors for a player at each timepoint, smoothed with a rolling window
    smoothed_player_head_angles = get_smoothed_player_head_angle_vectors_for_trial(head_angle_vector_array_trial,
                                                                                        window_size=window_size)
    
    # # if the trial was too short, return np.nan instead of continuing analysis
    # if smoothed_player_head_angles.shape[1] < 20:
    #     return np.nan
    
    if debug:
        print("smoothed_player_head_angles.shape: ", smoothed_player_head_angles.shape)
        print("smoothed_player_head_angles\n", smoothed_player_head_angles[:,50:60])

    # 2. find the player-to-alcove vectors for each wall, for each timepoint
    player_to_alcove_vectors = trajectory_vectors.get_player_to_alcove_direction_vectors_for_trajectory(trajectory,
                                                                                                     num_walls=num_walls)
    
    if debug:
        print("player_to_alcove_vectors.shape: ", player_to_alcove_vectors.shape)
        print("player_to_alcove_vectors\n", player_to_alcove_vectors[:,1,50:60])
    
    # 3. calculate the dot products between the two sets of vectors 
    dot_products_trajectory = trajectory_vectors.calculate_vector_dot_products_for_trajectory(player_to_alcove_vectors,
                                                                                   smoothed_player_head_angles,
                                                                                   num_walls=num_walls)
    
    if debug:
        print("dot_products_trajectory.shape: ", dot_products_trajectory.shape)
        print("dot_products_trajectory\n", dot_products_trajectory[:,50:60]) 
    
    # 4. calculate the norms for the two sets of vectors
    (head_angle_vector_norms_trajectory,
     player_to_alcove_vector_norms_trajectory) = trajectory_vectors.calculate_vector_norms_for_trajectory(player_to_alcove_vectors,
                                                                                                   smoothed_player_head_angles,
                                                                                                   num_walls=8)
    if debug:
        print("head_angle_vector_norms_trajectory\n", head_angle_vector_norms_trajectory[50:60])
        print("player_to_alcove_vector_norms_trajectory\n", player_to_alcove_vector_norms_trajectory[:,50:60])
    
        print("head_angle_vector_norms_trajectory.shape: ", head_angle_vector_norms_trajectory.shape)
        print("player_to_alcove_vector_norms_trajectory.shape: ", player_to_alcove_vector_norms_trajectory.shape)

    # 5. calculate cosine similarity for the head angle vector as compared to the vector from the player to each wall
    # this is done for all timepoints in a trajectory
    cosine_similairities_trajectory = trajectory_vectors.calculate_cosine_similarity_for_trajectory(dot_products_trajectory,
                                                                                             head_angle_vector_norms_trajectory,
                                                                                             player_to_alcove_vector_norms_trajectory,
                                                                                             num_walls=8)
    if debug:
        print("cosine_similairities_trajectory.shape: ", cosine_similairities_trajectory.shape)

    # 6. calculate angles between player head direction and player-to-alcove vectors for each wall
    thetas = trajectory_vectors.calculate_thetas_for_trajectory(cosine_similairities_trajectory, num_walls=8)

    return thetas
    


# In[7]:


## WALL VISIBILITY ##


# In[ ]:


def get_octagon_vertex_coordinates():
    ''' Return octagon vertex coordinates as a 2D array of shape 2*8
        The first point is the CCW vertex of wall 1 '''
    
    # get octagon vertex coordinates
    octagon_vertex_coords = plot_octagon.calculate_coordinates(vertex=True)
    
    # vstack as a 2*8 array
    octagon_vertex_coords = np.vstack([octagon_vertex_coords[0], octagon_vertex_coords[1]]) 
    
    # remove repeated first coordinate
    octagon_vertex_coords = octagon_vertex_coords[:,:-1]
    
    # rearrange array so that north wall is at the beginning
    octagon_vertex_coords = np.hstack([octagon_vertex_coords[:,-1:], octagon_vertex_coords[:,:-1]])

    
    return octagon_vertex_coords


# In[ ]:


def get_CW_CCW_vertex_coords(octagon_vertex_coords):
    ''' Take a 2*8 array of octagon vertex coordinates and return two arrays
        First is the 'clockwise' array, to be used when individual is CW of the wall, where the first column is the CCW vertex of wall 1
        Second is the 'counterlockwise' array, to be used when individual is CCW of the wall, where the first column is the CW vertex of wall 1
        Both returned arrays are shape 2*8 '''

    CW_octagon_vertex_coords = octagon_vertex_coords
    CCW_octagon_vertex_coords = np.hstack([octagon_vertex_coords[:,1::], octagon_vertex_coords[:,0:1:]])

    return CW_octagon_vertex_coords, CCW_octagon_vertex_coords


# In[ ]:


def calculate_cross_product(smoothed_player_headangles_trial, player_to_alcove_vectors, num_walls=8):
    ''' Calculate the cross product between the head angle vector and the alcove vectors for each time
        point in a trajectory
        Cross product is positive if the second vector is CCW of the first, and negative if the second
        vector is CW of the first
        Return a num_walls*trajectory_length-1 shaped array '''

    timepoints = smoothed_player_headangles_trial.shape[1]
    cross_products_wall_headangle = np.zeros([num_walls,timepoints])
    for timepoint in range(timepoints):
        headangle_vector_x_coord = smoothed_player_headangles_trial[0, timepoint]
        headangle_vector_y_coord = smoothed_player_headangles_trial[1, timepoint]
        
        for wall_num in range(num_walls):
            wall_vector_x_coord = player_to_alcove_vectors[0, wall_num, timepoint]
            wall_vector_y_coord = player_to_alcove_vectors[1, wall_num, timepoint]
            cross_product_this_wall = headangle_vector_x_coord*wall_vector_y_coord - headangle_vector_y_coord*wall_vector_x_coord
            cross_products_wall_headangle[wall_num,timepoint] = cross_product_this_wall

    return cross_products_wall_headangle
    


# In[ ]:


def is_wall_clockwise_of_player(cross_products_wall_headangle):
    ''' Return a boolean array of shape num_walls*timepoints
        which is True for when the wall is clockwise of the player's current headangle vector '''

    return cross_products_wall_headangle < 0


# In[ ]:


# Helper function
def get_closest_wall_section_coords_trajectory(wall_is_clockwise, CW_octagon_vertex_coords, CCW_octagon_vertex_coords, debug=False):
    ''' Takes the clockwise and counterclockwise octagon vertex coordinates (i.e., the coordinates of the
        vertices of each wall, 1-8, that would be seen first if rotating clockwise or counterclockwise)
        These are both 2*8 arrays
        Also takes a boolean array of size num_walls*timepoints which is True where the wall is clockwise of the
        current head angle at that timepoint
        Return an array of shape num_walls*timepoints*2 that records the x/y coordinates of the wall
        for all timepoints, being either CW or CCW coordinate dictated by np.where(wall_is_clockwise) '''
    
    
    wall_coords_cross_product_dependent = np.zeros((*wall_is_clockwise.shape, 2)) # add a 3rd dimension of size
                                                                                     # 2 to store x/y coordinates
    
    # reshape and broadcast the x and y coordinates of octagon_vertex_coords to fit np.where
    timepoints = wall_is_clockwise.shape[1]
    CW_octagon_vertex_coords_x = CW_octagon_vertex_coords[0].reshape(8,1)
    CW_octagon_vertex_coords_x = CW_octagon_vertex_coords_x * np.ones((8,timepoints))
    
    CCW_octagon_vertex_coords_x = CCW_octagon_vertex_coords[0].reshape(8,1)
    CCW_octagon_vertex_coords_x = CCW_octagon_vertex_coords_x * np.ones((8,timepoints))
    
    CW_octagon_vertex_coords_y = CW_octagon_vertex_coords[1].reshape(8,1)
    CW_octagon_vertex_coords_y = CW_octagon_vertex_coords_y * np.ones((8,timepoints))
    
    CCW_octagon_vertex_coords_y = CCW_octagon_vertex_coords[1].reshape(8,1)
    CCW_octagon_vertex_coords_y = CCW_octagon_vertex_coords_y * np.ones((8,timepoints))
    
    
    if debug:
        # Verify the shape of wall_angular_direction
        print("wall_is_clockwise shape:", wall_is_clockwise.shape)
    
        # Verify the shapes and contents of CW and CCW octagon vertex coordinates
        print("CW_octagon_vertex_coords_x shape:", CW_octagon_vertex_coords_x.shape)
        print("CCW_octagon_vertex_coords_x shape:", CCW_octagon_vertex_coords_x.shape)
        print("CW_octagon_vertex_coords contents:", CW_octagon_vertex_coords)
        print("CCW_octagon_vertex_coords contents:", CCW_octagon_vertex_coords)
    
    
    # where the wall is clockwise, use the clockwise coordinate, and where is counterclockwise
    # use the counterclockwise coordinate
    wall_coords_cross_product_dependent[:,:,0] = np.where(wall_is_clockwise,
                                                          CW_octagon_vertex_coords_x,
                                                          CCW_octagon_vertex_coords_x)
    wall_coords_cross_product_dependent[:,:,1] = np.where(wall_is_clockwise,
                                                          CW_octagon_vertex_coords_y,
                                                          CCW_octagon_vertex_coords_y)

    return wall_coords_cross_product_dependent


# In[ ]:


def get_player_to_closest_wall_section_direction_vectors_for_trajectory(trajectory,
                                                                        wall_coords_cross_product_dependent,
                                                                        num_walls=8,
                                                                        debug=False):
    ''' Calculate the direction vector between player location and the closest (angular) wall coordinate (of each wall)
        Input requires the trajectory of the player,
        and the wall coordinates to use, which will be dependent on the current head angle
        The first array must be shape 2*timepoints, the second array must be
        shaped wall_num*timepoints*2
        Returns a 3-dimensional array of shape 2*num_walls*trajectory.shape[1] '''
    
    timepoints = wall_coords_cross_product_dependent.shape[1]

    # calculate the vector between the closest wall section point and current player location
    vector_to_closest_wall_sections = np.zeros([2, num_walls, timepoints])
    for time_index in range(timepoints): # for each timepoint in trajectory
        for wall_num in range(num_walls): # for each wall
            vector_to_closest_wall_section = wall_coords_cross_product_dependent[wall_num, time_index, :] - trajectory[:, time_index]
            vector_to_closest_wall_sections[:,wall_num,time_index] = vector_to_closest_wall_section
            
            if debug:
                if (time_index == 10 and wall_num == 0):
                    print("at 10, wall 0")
                    print("vector_to_closest_wall_section: ", vector_to_closest_wall_section)
                    print("wall_coords_cross_product_dependent[0, 10, :] - trajectory[:, 10]: ",
                          wall_coords_cross_product_dependent[0, 10, :] - trajectory[:, 10])
                    print("vector_to_closest_wall_sections[:,0,10]: ", vector_to_closest_wall_sections[:,0,10])

    return vector_to_closest_wall_sections


# In[ ]:


# Umbrella function
def get_wall_coords_cross_product_dependent(trial_list=None, trial_index=0, trial=None, player_id=0, window_size=5):
    ''' Umbrella function
        Using the clockwise and counterclockwise octagon vertex coordinates (i.e., the coordinates of the
        vertices of each wall, 1-8, that would be seen first if rotating clockwise or counterclockwise)
        Return an array of shape num_walls*timepoints*2 that records the x/y coordinates of the wall
        for all timepoints, being either CW or CCW coordinate dictated by np.where(wall_is_clockwise)
        Where wall_is_clockwise is true when the wall is clockwise of the current headangle vector '''

    # access the dataframe for the trial
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)

    # get the trajectory for calculating direction vectors to alcoves
    trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_id)
    
    # get the vertex coordinates for the octagon, starting at CCW wall 1
    octagon_vertex_coords = get_octagon_vertex_coordinates()

    # create 2 separate coordinate arrays from above, one to use when each wall is CW of the reference, and 
    # the other assuming each wall is is counterclockwise of the reference
    CW_octagon_vertex_coords, CCW_octagon_vertex_coords = get_CW_CCW_vertex_coords(octagon_vertex_coords)

    # get the headangles for this player, for this trial
    trial_player_headangles = trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=player_id)

    # get the smoothed headangle vectors in 2D space for this player, for this trial
    smoothed_head_angle_vectors = get_smoothed_player_head_angle_vectors_for_trial(trial_player_headangles, window_size=window_size)
    
    # get vectors from player to walls to identify whether a wall is CW or CCW of player headangle
    player_to_alcove_vectors = trajectory_vectors.get_player_to_alcove_direction_vectors_for_trajectory(trajectory)
    
    # find the cross product between the headangle vector and the vector to each wall to identify whether
    # each wall is CW or CCW at each timepoint (relative to player headangle vector)
    cross_products_wall_headangle = calculate_cross_product(smoothed_head_angle_vectors, player_to_alcove_vectors)

    # boolean array to record whether each wall is CW of the player's headangle vector (True) at each timepoint
    wall_is_clockwise = is_wall_clockwise_of_player(cross_products_wall_headangle)

    # cross-product dependent wall coords for all walls and timepoints. Take the CCW wall coord if the wall is 
    # CW of the player headangle vector, and vice versa
    wall_coords_cross_product_dependent = get_closest_wall_section_coords_trajectory(wall_is_clockwise,
                                                                                     CW_octagon_vertex_coords,
                                                                                     CCW_octagon_vertex_coords)

    return wall_coords_cross_product_dependent

        


# In[ ]:


# Umbrella function for getting angle difference between FoV centre and angularly-closest section of wall for a player
# (similar to head_angle_to_walls_throughout_trajectory, see above)
def head_angle_to_closest_wall_section_throughout_trajectory(trial_list=None, trial_index=0, trial=None, player_id=0,
                                                             window_size=5, num_walls=8, debug=False):
    ''' From a trajectory, calculate the angles between the player head angle vector and 
        the player-to-closest-wall-coordinate vectors for an entire trial
        Returns an array of shape num_walls*timepoints '''

    # access the dataframe for the trial
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)

    # get the trajectory for calculating direction vectors to alcoves
    trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_id)

    # find head angle unit vectors for a player at each timepoint, smoothed with a rolling window
    trial_player_headangles = trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=player_id)
    smoothed_head_angle_vectors = get_smoothed_player_head_angle_vectors_for_trial(trial_player_headangles,
                                                                         window_size=window_size)
    
    # if smoothed head angles are too short to analyse, return np.nan
    if smoothed_head_angle_vectors.shape[1] < 20:
        return np.nan
    
    if debug:
        print("smoothed_player_head_angles.shape: ", smoothed_head_angle_vectors.shape)
        print("smoothed_player_head_angles\n", smoothed_head_angle_vectors[:,40:50])

    # find the closest (angular) wall coordinates for each wall and timepoint
    wall_coords_cross_product_dependent = get_wall_coords_cross_product_dependent(trial=trial, player_id=player_id)

    # find the player-to-closest-wall-coordinate vectors for each wall, for each timepoint
    player_to_closest_wall_section = get_player_to_closest_wall_section_direction_vectors_for_trajectory(trajectory,
                                                                                                     wall_coords_cross_product_dependent,    
                                                                                                     num_walls=num_walls)
    
    if debug:
        print("player_to_closest_wall_section.shape: ", player_to_closest_wall_section.shape)
        print("player_to_closest_wall_section\n", player_to_closest_wall_section[:,1,40:50])
        print("player_to_closest_wall_section at 10\n", player_to_closest_wall_section[:,0,10])
    
    # calculate the dot products between the closest-wall-section and player-headangle vectors (for each wall, timepoint)
    dot_products_trajectory = trajectory_vectors.calculate_vector_dot_products_for_trajectory(player_to_closest_wall_section,
                                                                                   smoothed_head_angle_vectors,
                                                                                   num_walls=num_walls)

    if debug:
        print("dot_products_trajectory.shape: ", dot_products_trajectory.shape)
        print("dot_products_trajectory\n", dot_products_trajectory[:,110:120]) 


    
    # calculate the norms for the closest-wall-section and player-headangle vectors (for each wall, timepoint)
    (head_angle_vector_norms_trajectory,
     player_to_closest_wall_section_vector_norms_trajectory) = trajectory_vectors.calculate_vector_norms_for_trajectory(player_to_closest_wall_section,
                                                                                                   smoothed_head_angle_vectors,
                                                                                                   num_walls=8)
    if debug:
        print("head_angle_vector_norms_trajectory\n", head_angle_vector_norms_trajectory[40:50])
        print("player_to_closest_wall_section_vector_norms_trajectory\n", player_to_closest_wall_section_vector_norms_trajectory[:,40:50])
        
        print("head_angle_vector_norms_trajectory.shape: ", head_angle_vector_norms_trajectory.shape)
        print("player_to_closest_wall_section_vector_norms_trajectory.shape: ", player_to_closest_wall_section_vector_norms_trajectory.shape)

    # calculate cosine similarity for the closest-wall-section and player-headangle vectors (for each wall, timepoint)
    cosine_similairities_trajectory = trajectory_vectors.calculate_cosine_similarity_for_trajectory(dot_products_trajectory,
                                                                                             head_angle_vector_norms_trajectory,
                                                                                             player_to_closest_wall_section_vector_norms_trajectory,
                                                                                             num_walls=8)

    if debug:
        print("cosine_similairities_trajectory.shape: ", cosine_similairities_trajectory.shape)

    # calculate angles between player head direction and player-to-alcove vectors for each wall
    thetas = trajectory_vectors.calculate_thetas_for_trajectory(cosine_similairities_trajectory, num_walls=8)

    return thetas
    


# In[16]:


## WALL VISIBILITY ANALYSIS


# In[ ]:


def get_wall_visible(trial_list=None, trial_index=0, trial=None, player_id=0, current_fov=110.36, debug=False):
    ''' Returns wall visibility array (boolean array of whether each wall is visible for
        the player at each timepoint, shape num_walls*timepoints), for a chosen player and 
        chosen trial '''
    
    if debug:
        start_time = time.time()
    


    # access the dataframe for the trial
    trial = extract_trial.extract_trial(trial, trial_list, trial_index)    

    
    if debug:
        print(f"get_wall_visible trial is of type {type(trial)}")
        if isinstance(trial, int):
            print(f"get_wall_visible int trials is: {trial}")
    assert(isinstance(trial, pd.DataFrame))
    
    trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_id)
    head_angle_vector_array_trial = trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=player_id)
    trial_player_headangles = get_smoothed_player_head_angle_vectors_for_trial(head_angle_vector_array_trial)

    wall_coords_cross_product_dependent = get_wall_coords_cross_product_dependent(trial=trial, player_id=player_id)

    # thetas = head_angle_to_closest_wall_section_throughout_trajectory(trajectory,
    #                                                                   trial_player_headangles,
    #                                                                   wall_coords_cross_product_dependent)

    thetas = head_angle_to_closest_wall_section_throughout_trajectory(trial=trial, player_id=player_id)
    if isinstance(thetas, float) and np.isnan(thetas):
        if debug:
            print(f"trial is too short to analyse. Returning np.nan instead of wall_visible array")
        return np.nan
    
    thetas = np.rad2deg(thetas)

    wall_visible = thetas < current_fov/2

    # output the time taken for this function
    if debug:
        end_time = time.time()
        print(f"Time taken for get_wall_visible (one trial, one player) is {end_time-start_time:.2f}")


    return wall_visible


# In[ ]:


def wall_visibility_player_slice_onset(wall_visible, trial):
    ''' Identify whether either of the relevant walls for this trial are visible at trial start
        Takes a boolean array of shape num_walls*timepoints which is True when a wall falls within the FoV
        range of the player
        Also takes the trial
        Returns 2 bools, reflecting wall visibility for wall 1 and wall 2 at trial start'''

    # local variables for logic
    wall1_visible = False
    wall2_visible = False
  
    # identify walls
    walls = get_indices.get_walls(trial=trial)
    # take the wall index instead of the wall number, to index wall_visible
    wall1_index = walls[0] - 1
    wall2_index = walls[1] - 1

    # identify which walls are initially visible
    if wall_visible[wall1_index,0]:
        wall1_visible = True
    if wall_visible[wall2_index,0]:
        wall2_visible = True


    return wall1_visible, wall2_visible


# In[ ]:


# Eventually might want to change this function to include whether the second wall becomes visible
def get_first_visible_wall(wall_visible, wall1_visible, wall2_visible, trial,
                                    debug=False):
    ''' Return the wall that becomes visible first
        Takes num_walls*timepoints boolean array of wall visibility, and bools for 
        whether wall 1 and wall 2 are visible at trial start
        Also takes the trial
        Returns 'wall1', 'wall2', 'both', or 'neither' '''
    
    if debug:
        start_time = time.time()

    # local variables for logic
    both_walls_initially_visible = False
    wall1_becomes_visible = False
    wall2_becomes_visible = False
    both_walls_become_visible = False
    neither_wall_becomes_visible = False
    wall1_visible_first = False
    wall2_visible_first = False
    
    # get trial walls
    walls = get_indices.get_walls(trial=trial)
    wall1_index = walls[0] - 1 # index, not wall number
    wall2_index = walls[1] - 1

    # check to see if both walls are already visible
    if wall1_visible and wall2_visible:
        both_walls_initially_visible = True
        if debug:
            print("Both walls already visible")
        return 'both'
    
    # for each wall, check which index of the trial the wall became visible on
    # Or, if the wall never became visible, keep wall_becomes_visible as False
    if wall1_visible: # wall immediately visible, so index is 0
        wall1_becomes_visible = True
        visible_index_wall1 = 0
        if debug:
            print("wall1_already visible")
    else:
        # convert the boolean wall visiblibilty array into an integer array, then use np.diff to compare
        # consecutive values for a difference.
        # If the array value ever changes from 0 to 1 there will be a diff of 1 at that timepoint
        # np.where then finds the index where this occurs
        wall_visibility_change_wall1 = np.where(np.diff(wall_visible[wall1_index,:].astype(int)) == 1)[0]
        if debug:
            print(f"wall_vis for wall 1: {wall_visible[wall1_index,:].astype(int)}")
            print(f"wall vis change wall1: {wall_visibility_change_wall1}")
        if wall_visibility_change_wall1.size > 0:
            wall1_becomes_visible = True
            if debug:
                print("wall1_becomes_visible")
            visible_index_wall1 = wall_visibility_change_wall1[0] + 1 # account for diff value being 1 index early
    
    if wall2_visible:
        wall2_becomes_visible = True
        visible_index_wall2 = 0
        if debug:
            print("wall2_already visible")
    else:
        wall_visibility_change_wall2 = np.where(np.diff(wall_visible[wall2_index,:].astype(int)) == 1)[0]
        if debug:
            print(f"wall_vis for wall 2: {wall_visible[wall2_index,:].astype(int)}")
            print(f"wall vis change wall2: {wall_visibility_change_wall2}")
        if wall_visibility_change_wall2.size > 0:
            wall2_becomes_visible = True
            if debug:
                print("wall2_becomes_visible")
            visible_index_wall2 = wall_visibility_change_wall2[0] + 1 # account for diff value being 1 index early
    
    
    # check if both walls eventually become visible in the trial
    if wall1_becomes_visible and wall2_becomes_visible:
        both_walls_become_visible = True
        if debug:
            print("both walls become visible")
    
    # If both walls become visible, identify which became visible first
    if both_walls_become_visible == True:
        if visible_index_wall1 < visible_index_wall2:
            wall1_visible_first = True
            if debug:
                print("wall1 visible first")
        elif visible_index_wall2 < visible_index_wall1:
            wall2_visible_first = True
            if debug:
                print("wall2 visible first")
        else:
            if debug:
                print("wall visible indices are equal, or a logical error")
            return 'both'
    # if no more than one wall ever becomes visible, identify it as the first visible wall
    elif wall1_becomes_visible == True and wall2_becomes_visible == False:
        wall1_visible_first = True
        if debug:
            print("wall1_visible_first")
    elif wall1_becomes_visible == False and wall2_becomes_visible == True:
        wall2_visible_first = True
        if debug:
            print("wall2_visible_first")
    # also account for neither wall becoming visible
    elif wall1_becomes_visible == False and wall2_becomes_visible == False:
        neither_wall_becomes_visible = True

    if debug:
        # output the time taken for this function
        if debug:
            end_time = time.time()
            print(f"Time taken for get_first_visible_wall (one trial, one player) is {end_time-start_time:.2f}")
    
    # now choose a return value: 'wall1' if wall1 becomes visible first, 'wall2' if wall2 becomes visible first, 'neither' if neither
    if wall1_visible_first:
        return 'wall1'
    elif wall2_visible_first:
        return 'wall2'
    elif neither_wall_becomes_visible:
        return 'neither'
    elif both_walls_initially_visible:
        if debug:
            print("returning 'both'")
        return 'both'
            

    else:
        raise ValueError("Function logic has failed.")
        


# In[ ]:


def get_wall_visibility_order(wall_visible, wall_initial_visibility, trial, 
                                    return_times=False, debug=False):
    ''' Return when walls becomes visible.
        Takes num_walls,timepoints boolean array of wall visibility,
        and num_walls boolean array of whether walls are visible at trial start.
        Also takes the trial.
        Returns int index at which this wall became visible relative to other walls. '''
    
    
    if debug:
        start_time = time.time()

    # get trial wall indices for the number of walls in the trial
    walls = get_indices.get_walls(trial=trial)
    num_walls = len(walls)
    wall_indices = np.empty(num_walls, dtype=int)
    for i in range(num_walls):
        wall_indices[i] = walls[i] - 1 # take index, not wall number

    
    # for each wall, find whether the wall becomes visible and on which time index of the trial
    # this occurs
    wall_becomes_visible_time = np.empty(num_walls)        # when does wall become visible
    wall_becomes_visible = np.empty(num_walls, dtype=bool)  # does wall become visible

    for wall_num in range(num_walls): # for each wall
        wall_index = wall_indices[wall_num] # find position in space that this wall appeared in for the trial
        
        if wall_initial_visibility[wall_num]: # wall immediately visible, so index is 0
            wall_becomes_visible_time[wall_num] = 0
            wall_becomes_visible[wall_num] = True
            if debug:
                print(f"wall {wall_num} already visible. Assigning an index of 0")
        
        else: # current wall not visible at slice onset
            # compare consecutive values of the wall visibility array for the time index at which this wall
            # (first) becomes visible 
            this_wall_visibility_change = np.where(
                                                    np.diff(
                                                             wall_visible[wall_index,:].astype(int)
                                                            ) == 1
                                                   )[0] 
            if debug:
                print(f"Wall visibility change wall {wall_num}: {this_wall_visibility_change}")

            # if wall visibility ever changes from negative
            if this_wall_visibility_change.size > 0:
                wall_becomes_visible[wall_num] = True 
                wall_becomes_visible_time[wall_num] = this_wall_visibility_change[0] + 1 # np.diff value is one index early
            else:
                wall_becomes_visible[wall_num] = False 
                wall_becomes_visible_time[wall_num] = np.nan # set index as nan if never visible


    # identify the order in which walls became visible this trial
    # nans will carry over to this array
    wall_becomes_visible_index = get_ordered_indices.get_ordered_indices(wall_becomes_visible_time)        
        
    if debug:
        # output the time taken for this function
        if debug:
            end_time = time.time()
            print(f"Time taken for get_wall_visibility (one trial, one player) is {end_time-start_time:.2f}")

    if not return_times:
        return wall_becomes_visible_index
    else:
        return wall_becomes_visible_index, wall_becomes_visible_time
    
            


# In[ ]:


def was_first_visible_wall_chosen_winner(wall, trial):
    ''' Identifies if the first visible wall for the winner was the wall chosen by the winner
        To be used in trials where one wall was visible to the player before the other
        Wall input currently accepts 'wall1' or 'wall2' 
        Returns bool '''

    # local variables
    if wall == 'wall1':
        df_wall = globals.WALL_1
    elif wall == 'wall2':
        df_wall = globals.WALL_2
    else:
        raise ValueError("wall parameter must take one of the values listed in the function docstring")

    # find the wall number for the first wall visible
    first_wall_visible = int(trial[df_wall].unique().item())

    # find the wall that was triggered on this trial
    # filter out nans that result from non-accepted triggers
    wall_triggered = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION][globals.WALL_TRIGGERED].unique()
    wall_triggered_filter_nans = wall_triggered[~np.isnan(wall_triggered)]
    wall_chosen = wall_triggered_filter_nans.item()

    if first_wall_visible == wall_chosen:
        print("first visible wall was the one chosen")
        first_visible_wall_chosen = True
    else:
        print("first visible wall was NOT the one chosen")
        first_visible_wall_chosen = False

    return first_visible_wall_chosen



# In[ ]:


def was_first_visible_wall_chosen_general(wall_num, trial):
    ''' Identifies if the first visible wall for the loser was the wall chosen by the loser
        To be used in trials where one wall was visible to the player before the other
        Wall input currently accepts 'wall1' or 'wall2' '''

        


# In[ ]:


# Note that this only takes the winners choices. Do I have a separate function for Loser's choice in losers_inferred_choice? check this.

# umbrella function for identifying if there was a first visible wall, whether it was the one chosen,
# and whether this was High or Low
def was_first_visible_wall_chosen_player(wall_visible, trial):
    ''' Umbrella function that identifies whether one wall became visible before the other, whether this wall
        was High, and then whether this first visible wall was chosen
        Takes boolean wall_num*timepoints wall_visible array that is True where a wall falls within the player's FoV
        Returns bools, first_visible_wall_chosen and first_visible_wall_high '''
    
    first_visible_wall = ''
    
    # identify which walls were visible at slice onset for this trial
    wall1_visible, wall2_visible = wall_visibility_player_slice_onset(wall_visible, trial)
        
    # if a single wall is visible, run through was_first_visible_wall_chosen_winner
    if wall1_visible != wall2_visible:
        print("only one wall visible at trial start")
        if wall1_visible:
            print("and this was wall1")
            first_visible_wall = 'wall1'
            first_visible_wall_chosen = was_first_visible_wall_chosen_winner('wall1', trial)
        elif wall2_visible:
            print("and this was wall2")
            first_visible_wall = 'wall2'
            first_visible_wall_chosen = was_first_visible_wall_chosen_winner('wall2', trial)    
    
    # if both walls are visible, not relevant for this analysis
    if wall1_visible == True and wall2_visible == True:
        print("both walls visible")
        first_visible_wall_chosen = 777
        pass
    
    # if no walls are visible, identify when and which wall was first visible
    if wall1_visible == False and wall2_visible == False:
        first_visible_wall = get_first_visible_wall(wall_visible, wall1_visible, wall2_visible, trial)
        # account for neither wall becoming visible, then not relevant for this analysis
        if first_visible_wall == 'neither':
            print("neither wall becomes visible")
            first_visible_wall_chosen = 777
        # if one wall does become visible, check whether this wall was the one chosen by the winner
        else:
            first_visible_wall_chosen = was_first_visible_wall_chosen_winner(first_visible_wall, trial)


    # identify whether the first visible wall was high
    if first_visible_wall == 'wall1':
        first_visible_wall_high = True
    elif first_visible_wall == 'wall2':
        first_visible_wall_high = False
    else:
        first_visible_wall_high = 777

    
    return first_visible_wall_chosen, first_visible_wall_high

    

