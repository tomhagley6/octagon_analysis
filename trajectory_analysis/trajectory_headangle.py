#!/usr/bin/env python
# coding: utf-8

# In[21]:


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


# In[2]:


## HEADANGLES THROUGHOUT TRAJECTORY ##


# In[3]:


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


# In[26]:


def get_smoothed_player_head_angle_vectors_for_trajectory(head_angle_vector_array, window_size=10):
    ''' Calculate smoothed player head angle vectors for a whole trajectory '''

    # head angle vectors with a mean average rolling window of window_size 
    window_size =10
    try:
        head_angle_vector_array_smoothed = np.zeros([2,head_angle_vector_array.shape[1]-window_size])
        for i in range(head_angle_vector_array.shape[1] - window_size):
            smoothed_head_angle_vector = np.mean(head_angle_vector_array[:,i:i+window_size], axis=1)
            head_angle_vector_array_smoothed[:,i] = smoothed_head_angle_vector
        
    except ValueError:
        print("head angle vector array too short to smooth, taking the raw array instead")
        head_angle_vector_array_smoothed = head_angle_vector_array
        

    return head_angle_vector_array_smoothed


# In[5]:


## HEAD ANGLE COMPARED TO WALL CENTRES ##


# In[6]:


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
    


# In[7]:


## WALL VISIBILITY ##


# In[8]:


def get_octagon_vertex_coordinates():
    ''' Return octagon vertex coordinates as a 2D array of shape 2*8
        The first point is the CCW vertex of wall 1 '''
    
    # get octagon vertex coordinates
    octagon_vertex_coords = plot_octagon.calculate_coordinates(vertex=True)
    
    # convert to array rows as x coords, y coords
    octagon_vertex_coords = np.vstack([octagon_vertex_coords[0], octagon_vertex_coords[1]]) 
    
    # remove repeated first coordinate
    octagon_vertex_coords = octagon_vertex_coords[:,:-1]
    
    # rearrange array so that north wall is at the beginning
    octagon_vertex_coords = np.hstack([octagon_vertex_coords[:,-1:], octagon_vertex_coords[:,:-1]])

    
    return octagon_vertex_coords


# In[9]:


def get_CW_CCW_vertex_coords(octagon_vertex_coords):
    ''' Take a 2*8 array of octagon vertex coordinates and return two arrays
        First is the 'clockwise' array, where the first column is the CCW vertex of wall 1
        Second is the 'counterlockwise' array, where the first column is the CW vertex of wall 1
        Both returned arrays are still shape 2*8 '''

    CW_octagon_vertex_coords = octagon_vertex_coords
    CCW_octagon_vertex_coords = np.hstack([octagon_vertex_coords[:,1::], octagon_vertex_coords[:,0:1:]])

    return CW_octagon_vertex_coords, CCW_octagon_vertex_coords


# In[10]:


def calculate_cross_product(smoothed_player_headangles_trial, player_to_alcove_vectors, num_walls=8):
    ''' Calculate the cross product between the head angle vector and the alcove vectors for each time
        point in a trajectory
        Cross product is positive if the second vector is CCW of the first, and negative if the second
        vector is CW of the first
        Return a num_walls*trajectory_length-1 shaped array '''

    cross_products_wall_headangle = np.zeros([num_walls,smoothed_player_headangles_trial.shape[1]])
    for timepoint in range(smoothed_player_headangles_trial.shape[1]):
        headangle_vector_x_coord = smoothed_player_headangles_trial[0, timepoint]
        headangle_vector_y_coord = smoothed_player_headangles_trial[1, timepoint]
        
        for wall_num in range(num_walls):
            wall_vector_x_coord = player_to_alcove_vectors[0, wall_num, timepoint]
            wall_vector_y_coord = player_to_alcove_vectors[1, wall_num, timepoint]
            cross_product_this_wall = headangle_vector_x_coord*wall_vector_y_coord - headangle_vector_y_coord*wall_vector_x_coord
            cross_products_wall_headangle[wall_num,timepoint] = cross_product_this_wall

    return cross_products_wall_headangle
    


# In[11]:


def is_wall_clockwise_of_player(cross_products_wall_headangle):
    ''' Return a boolean array of shape num_walls*player_headangles_trial.shape[1]
        which is True for when the wall is clockwise of the player's current headangle vector '''

    return cross_products_wall_headangle < 0


# In[12]:


def get_closest_wall_section_coords_trajectory(wall_is_clockwise, CW_octagon_vertex_coords, CCW_octagon_vertex_coords):
    ''' Taking the clockwise and counterclockwise octagon vertex coordinates (i.e., the coordinates of the
        vertices of each wall, 1-8, that would be seen first if rotating clockwise or counterclockwise)
        Create an array of shape wall_angular_direction.shape*2 that records the x/y coordinates of the wall
        for all timepoints, being either CW or CCW coordinate dictated by np.where(wall_is_clockwise)
        Where wall_is_clockwise is true when the wall is clockwise of the current headangle vector '''
    
    
    wall_coords_cross_product_dependent = np.zeros((*wall_is_clockwise.shape, 2)) # add a 3rd dimension of size
                                                                                     # 2 to store x/y coordinates
    
    # reshape and broadcast the x and y coordinates of octagon_vertex_coords to fit np.where
    CW_octagon_vertex_coords_x = CW_octagon_vertex_coords[0].reshape(8,1)
    CW_octagon_vertex_coords_x = CW_octagon_vertex_coords_x * np.ones((8,wall_is_clockwise.shape[1]))
    
    CCW_octagon_vertex_coords_x = CCW_octagon_vertex_coords[0].reshape(8,1)
    CCW_octagon_vertex_coords_x = CCW_octagon_vertex_coords_x * np.ones((8,wall_is_clockwise.shape[1]))
    
    CW_octagon_vertex_coords_y = CW_octagon_vertex_coords[1].reshape(8,1)
    CW_octagon_vertex_coords_y = CW_octagon_vertex_coords_y * np.ones((8,wall_is_clockwise.shape[1]))
    
    CCW_octagon_vertex_coords_y = CCW_octagon_vertex_coords[1].reshape(8,1)
    CCW_octagon_vertex_coords_y = CCW_octagon_vertex_coords_y * np.ones((8,wall_is_clockwise.shape[1]))
    
    
    # # Verify the shape of wall_angular_direction
    # print("wall_is_clockwise shape:", wall_is_clockwise.shape)
    
    # # Verify the shapes and contents of CW and CCW octagon vertex coordinates
    # print("CW_octagon_vertex_coords_x shape:", CW_octagon_vertex_coords_x.shape)
    # print("CCW_octagon_vertex_coords_x shape:", CCW_octagon_vertex_coords_x.shape)
    # print("CW_octagon_vertex_coords contents:", CW_octagon_vertex_coords)
    # print("CCW_octagon_vertex_coords contents:", CCW_octagon_vertex_coords)
    
    
    wall_coords_cross_product_dependent[:,:,0] = np.where(wall_is_clockwise,
                                                          CW_octagon_vertex_coords_x,
                                                          CCW_octagon_vertex_coords_x)
    wall_coords_cross_product_dependent[:,:,1] = np.where(wall_is_clockwise,
                                                          CW_octagon_vertex_coords_y,
                                                          CCW_octagon_vertex_coords_y)

    return wall_coords_cross_product_dependent


# In[23]:


def get_player_to_closest_wall_section_direction_vectors_for_trajectory(trajectory,
                                                                        wall_coords_cross_product_dependent,
                                                                        num_walls=8,
                                                                        debug=False):
    ''' Calculate the direction vector between player and the angularly closest wall coordinate (of each wall)
        Input requires the smoothed head angle vectors of the player for a full trajectory,
        and the wall coordinates to use, dependent on the current head angle
        The first array must be shape 2*timepoints, the second array must be
        shaped wall_num*timepoints*2
        Returns a 3-dimensional array of shape 2*num_walls*trajectory.shape[1] '''
    
    # calculate the vector between the closest wall section point and current player location
    vector_to_closest_wall_sections = np.zeros([2, num_walls, wall_coords_cross_product_dependent.shape[1]])
    for time_index in range(wall_coords_cross_product_dependent.shape[1]): # for each timepoint in trajectory
        player_x_loc = trajectory[0,time_index]
        player_y_loc = trajectory[1,time_index]
    
        for wall_num in range(num_walls): # for each wall
            vector_to_closest_wall_section = wall_coords_cross_product_dependent[wall_num, time_index, :] - trajectory[:, time_index]
            vector_to_closest_wall_sections[:,wall_num,time_index] = vector_to_closest_wall_section
            if (time_index == 10 and wall_num == 0):
                if debug:
                    print("at 10, wall 0")
                    print("vector_to_closest_wall_section: ", vector_to_closest_wall_section)
                    print("wall_coords_cross_product_dependent[0, 10, :] - trajectory[:, 10]: ",
                          wall_coords_cross_product_dependent[0, 10, :] - trajectory[:, 10])
                    print("vector_to_closest_wall_sections[:,0,10]: ", vector_to_closest_wall_sections[:,0,10])

    return vector_to_closest_wall_sections


# In[14]:


def wall_coords_cross_product_dependent(trial_list=None, trial_index=0, trial=None, player_id=0):
    ''' Umbrella function
        Taking the clockwise and counterclockwise octagon vertex coordinates (i.e., the coordinates of the
        vertices of each wall, 1-8, that would be seen first if rotating clockwise or counterclockwise)
        Create an array of shape wall_angular_direction.shape*2 that records the x/y coordinates of the wall
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
    trial_player_headangles = extract_trial_player_headangles(trial=trial, player_id=player_id)

    # get the smoothed headangles for this player, for this trial
    smoothed_trial_player_headangles = get_smoothed_player_head_angle_vectors_for_trajectory(trial_player_headangles,
                                                                                                          window_size=10)
    
    
    # get vectors from player to walls to identify whether a wall is CW or CCW of player headangle
    player_to_alcove_vectors = trajectory_vectors.get_player_to_alcove_direction_vectors_for_trajectory(trajectory)
    
    # find the cross product between the headangle vector and the vector to each wall to identify whether
    # each wall is CW or CCW at each timepoint (relative to player headangle vector)
    cross_products_wall_headangle = calculate_cross_product(smoothed_trial_player_headangles, player_to_alcove_vectors)

    
    # boolean array to record whether each wall is CW of the player's headangle vector (True) at each timepoint
    wall_is_clockwise = is_wall_clockwise_of_player(cross_products_wall_headangle)

    # cross-product dependent wall coords for all walls and timepoints. Take the CCW wall coord if the wall is 
    # CW of the player headangle vector, and vice versa
    wall_coords_cross_product_dependent = get_closest_wall_section_coords_trajectory(wall_is_clockwise,
                                                                                     CW_octagon_vertex_coords,
                                                                                     CCW_octagon_vertex_coords)

    return wall_coords_cross_product_dependent

        


# In[15]:


# Umbrella function for getting angle difference between FoV centre and angularly-closest section of wall for a player
# (similar to head_angle_to_walls_throughout_trajectory, see above)
def head_angle_to_closest_wall_section_throughout_trajectory(trajectory, head_angle_vector_array_trajectory,
                                                             wall_coords_cross_product_dependent,
                                                             window_size=10, num_walls=8):
    ''' From a trajectory, calculate the angles between the player head angle vector and 
        the player-to-closest-wall-coordinate vectors for an entire trial
        Returns an array of shape num_walls*timepoints '''

    # 1. find head angle unit vectors for a player at each timepoint, smoothed with a rolling window
    smoothed_player_head_angles = get_smoothed_player_head_angle_vectors_for_trajectory(head_angle_vector_array_trajectory,
                                                                                        window_size=10)
    # print("smoothed_player_head_angles.shape: ", smoothed_player_head_angles.shape)
    # print("smoothed_player_head_angles\n", smoothed_player_head_angles[:,110:120])

    # 2. find the player-to-closest-wall-coordinate vectors for each wall, for each timepoint
    player_to_closest_wall_section = get_player_to_closest_wall_section_direction_vectors_for_trajectory(trajectory,
                                                                                                     wall_coords_cross_product_dependent,    
                                                                                                     num_walls=num_walls)
    
    # print("player_to_closest_wall_section.shape: ", player_to_closest_wall_section.shape)
    # print("player_to_closest_wall_section\n", player_to_closest_wall_section[:,1,110:120])
    # print("player_to_closest_wall_section at 10\n", player_to_closest_wall_section[:,0,10])
    # 3. calculate the dot products between the two sets of vectors 
    dot_products_trajectory = trajectory_vectors.calculate_vector_dot_products_for_trajectory(player_to_closest_wall_section,
                                                                                   smoothed_player_head_angles,
                                                                                   num_walls=num_walls)

    # print("dot_products_trajectory.shape: ", dot_products_trajectory.shape)
    # print("dot_products_trajectory\n", dot_products_trajectory[:,110:120]) 


    
    # 4. calculate the norms for the two sets of vectors
    (head_angle_vector_norms_trajectory,
     player_to_closest_wall_section_vector_norms_trajectory) = trajectory_vectors.calculate_vector_norms_for_trajectory(player_to_closest_wall_section,
                                                                                                   smoothed_player_head_angles,
                                                                                                   num_walls=8)

    # print("head_angle_vector_norms_trajectory\n", head_angle_vector_norms_trajectory[110:120])
    # print("player_to_closest_wall_section_vector_norms_trajectory\n", player_to_closest_wall_section_vector_norms_trajectory[:,110:120])
    
    # print("head_angle_vector_norms_trajectory.shape: ", head_angle_vector_norms_trajectory.shape)
    # print("player_to_closest_wall_section_vector_norms_trajectory.shape: ", player_to_closest_wall_section_vector_norms_trajectory.shape)

    # 5. calculate cosine similarity for the head angle vector as compared to the vector from the player to each wall
    # this is done for all timepoints in a trajectory
    cosine_similairities_trajectory = trajectory_vectors.calculate_cosine_similarity_for_trajectory(dot_products_trajectory,
                                                                                             head_angle_vector_norms_trajectory,
                                                                                             player_to_closest_wall_section_vector_norms_trajectory,
                                                                                             num_walls=8)

    # print("cosine_similairities_trajectory.shape: ", cosine_similairities_trajectory.shape)

    # 6. calculate angles between player head direction and player-to-alcove vectors for each wall
    thetas = trajectory_vectors.calculate_thetas_for_trajectory(cosine_similairities_trajectory, num_walls=8)

    return thetas
    


# In[16]:


## WALL VISIBILITY ANALYSIS


# In[17]:


def wall_visibility_player_slice_onset(wall_visible, trial):
    ''' Identify whether either of the relevant walls for this trial are visible at trial start '''

    # local variables for logic
    wall1_visible = False
    wall2_visible = False
    both_walls_visible = False
  
    # identify walls
    walls = get_indices.get_walls(trial=trial)
    wall1_index = walls[0] - 1
    wall2_index = walls[1] - 1

    # identify which walls are initially visible
    if wall_visible[wall1_index,0]:
        wall1_visible = True
    if wall_visible[wall2_index,0]:
        wall2_visible = True


    return wall1_visible, wall2_visible


# In[24]:


def which_wall_becomes_visible_first(wall_visible, wall1_visible, wall2_visible, trial,
                                    debug=False):
    ''' Return the wall that becomes visible first
        Input requires information about whether a wall starts as visible
        which is retrieved from wall_visibility_player_slice_onset
        Returns 'wall1', 'wall2', 'both', or 'neither' '''
    

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
    wall1_index = walls[0] - 1
    wall2_index = walls[1] - 1

    # check to see if both walls are already visible
    if wall1_visible and wall2_visible:
        both_walls_initially_visible = True
    
    # for each wall, check which index of the trial the wall became visible on
    # Or, if the wall never became visible, keep wall_becomes_visible as False
    if wall1_visible:
        wall1_becomes_visible = True
        visible_index_wall1 = 0
        if debug:
            print("wall1_already visible")
    else:
        # convert the boolean 'wall visible' array into an integer array, then use np.diff to compare
        # consecutive values for a difference.
        # If the array value ever changes from 0 to 1 there will be a diff of 1 at that timepoint
        # np.where then finds the index where this occurs
        # NB: the index value for a wall is wall_number - 1
        wall_visibility_change_wall1 = np.where(np.diff(wall_visible[wall1_index,:].astype(int)) == 1)[0]
        if debug:
            print(f"wall_vis for wall 1: {wall_visible[wall1_index,:].astype(int)}")
            print(f"wall vis change wall1: {wall_visibility_change_wall1}")
        if wall_visibility_change_wall1.size > 0:
            wall1_becomes_visible = True
            if debug:
                print("wall1_becomes_visible")
            visible_index_wall1 = wall_visibility_change_wall1[0] + 1
    
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
            visible_index_wall2 = wall_visibility_change_wall2[0] + 1
    
    
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
    
    # now choose a return value: 'wall1' if wall1 becomes visible first, 'wall2' if wall2 becomes visible first, 'neither' if neither
    if wall1_visible_first:
        return 'wall1'
    elif wall2_visible_first:
        return 'wall2'
    elif neither_wall_becomes_visible:
        return 'neither'
    elif both_walls_initially_visible:
        return 'both'
    else:
        raise ValueError("Function logic has failed.")
        return None
        


# In[19]:


def was_first_visible_wall_chosen_winner(wall, trial):
    ''' Identifies if the first visible wall for the winner was the wall chosen by the winner
        To be used in trials where one wall was visible to the player before the other
        Wall input currently accepts 'wall1' or 'wall2' '''

    # local variables
    if wall == 'wall1':
        df_wall_index = 'data.wall1'
    elif wall == 'wall2':
        df_wall_index = 'data.wall2'
    else:
        raise ValueError("wall parameter must take one of the values listed in the function docstring")

    # find the first wall visible
    first_wall_visible_index = int(trial[df_wall_index].unique().item())

    wall_triggered = trial[globals.WALL_TRIGGERED].unique()
    wall_triggered_filter_nans = wall_triggered[~np.isnan(wall_triggered)]
    wall_chosen_index = wall_triggered_filter_nans.item()

    if first_wall_visible_index == wall_chosen_index:
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

        


# In[20]:


# umbrella function for identifying if there was a first visible wall, whether it was the one chosen,
# and whether this was High or Low
def was_first_visible_wall_chosen_player(wall_visible, trial):
    ''' Umbrella function that identifies whether one wall became visible before the other, whether this wall
        was High, and then whether this first visible wall was chosen
        Returns bools first_visible_wall_chosen and first_visible_wall_high '''
    
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
        first_visible_wall = which_wall_becomes_visible_first(wall1_visible, wall2_visible, trial)
        # account for neither wall becoming visible, then not relevant for this analysis
        if first_visible_wall == 'neither':
            print("neither wall becomes visible")
            first_visible_wall_chosen = 777
        # then run through was_first_visible_wall_chosen_winner
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

    

