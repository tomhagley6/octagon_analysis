#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import data_extraction.get_indices as get_indices
import parse_data.prepare_data as prepare_data
import globals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plotting import plot_octagon, plot_trajectory
import data_extraction.get_indices as get_indices
import plotting.plot_probability_chose_wall as plot_probability_chose_wall
import data_strings
import analysis.wall_visibility_and_choice as wall_visibility_and_choice
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import analysis.wall_choice as wall_choice
import data_extraction.extract_trial as extract_trial
import utils.pad_and_reshape_array as utils


# In[3]:


def get_visualisation_vector_coordinates(trial_list, trial_num, trajectory, trial_player_headangles_smoothed, player_to_alcove_vectors,
                                     player_to_closest_wall_section, start_index=0, vector_length=20,
                                     wall_index=None):
    ''' Return the coordinates of the vectors that originate at the player location and project towards
        the direction of headangle, the wall alcove, and the wall closest-wall-section.
        Input is the trial list, trajectory, smoothed player headangles, player-to-alcove vectors,
        and player-to-closest-wall-section vectors for the trial.
        Optional arguments specify the point in the trajectory to take as the origin, the vector length,
        and the wall to use as the wall_idex (default, real chosen wall for this trial).'''


    start_index=start_index
    if not wall_index:
        wall_index = int(get_indices.get_chosen_walls(trial_list)[trial_num] - 1)
    
    # get the start coordinates for the vector as the location of the player at the specified point
    # in the trajectory
    x_start = trajectory[0,start_index]
    y_start = trajectory[1,start_index]

    # calculate gradients for the vectors of: current headangle, player-to-alcove (given wall index)
    # and player-to-closest-wall-section (given wall index)
    x_gradient = trial_player_headangles_smoothed[0,start_index]
    y_gradient = trial_player_headangles_smoothed[1,start_index]
    x_gradient_alcove = player_to_alcove_vectors[0,wall_index,start_index]
    y_gradient_alcove = player_to_alcove_vectors[1,wall_index,start_index]
    x_gradient_closest_wall_section = player_to_closest_wall_section[0,wall_index,start_index]
    y_gradient_closest_wall_section = player_to_closest_wall_section[1,wall_index,start_index] 

    vector_length = vector_length

    # assign coordinate values to the start and end point for each vector
    start = [x_start, y_start]
    end_head_direction = [x_start + x_gradient*vector_length, y_start + y_gradient*vector_length]
    end_wall_alcove = [x_start + x_gradient_alcove, y_start + y_gradient_alcove]
    end_wall_section = [x_start + x_gradient_closest_wall_section, y_start + y_gradient_closest_wall_section]

    # zip start and end values and return as arrays
    head_direction_vector_coordinates = np.array(list(zip(start,end_head_direction)))
    alcove_direction_vector_coordinates = np.array(list(zip(start,end_wall_alcove)))
    closest_wall_section_vector_coordinates = np.array(list(zip(start,end_wall_section)))

    return (head_direction_vector_coordinates, alcove_direction_vector_coordinates,
             closest_wall_section_vector_coordinates, wall_index)


# In[4]:


def plot_octagon_visualisation_vectors(head_direction_vector_coordinates, alcove_direction_vector_coordinates,
                                        closest_wall_section_vector_coordinates, trajectory, wall_index, start_index=0,
                                        colours = ['r','g','orange'], axes=None):
    ''' Return plotted axes of octagon with the visualisation vectors for head angle, player-to-alcove, and 
        player-to-closest-wall-section for the given wall index and trajectory start index.
        Takes the start and end coordinates of these vectors, and the wall index (see 
        get_visualisation_vector_coordinates for details).'''
    
    # plot visualisation vectors over octagon base, head direction alcove direction and closest wall section
    # colours are decided by the colours array parameter
    colours= colours
    ax = plot_octagon.plot_octagon(ax=axes)
    ax.scatter(trajectory[0,:], trajectory[1,:], s=0.5)
    ax.plot(head_direction_vector_coordinates[0,:], head_direction_vector_coordinates[1,:], c=colours[0], linewidth=2)
    ax.plot(alcove_direction_vector_coordinates[0,:], alcove_direction_vector_coordinates[1,:], c=colours[1], linewidth=2)
    ax.plot(closest_wall_section_vector_coordinates[0,:], closest_wall_section_vector_coordinates[1,:], c=colours[2], linewidth=2)

    # change plot params
    for spine in ax.spines.values():
        spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)  # Turn off major ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return ax


# In[5]:


def get_angles_between_head_and_wall_locations(thetas_trajectory, thetas_closest_wall_section, wall_index, start_index=0):
    ''' Return the angles between the head direction and wall alcove centre, and head direction 
        and closest wall section'''

    # print the angle between head direction and wall_index, and closest_wall_section
    print(thetas_trajectory[wall_index,start_index], thetas_closest_wall_section[wall_index,start_index])

    return thetas_trajectory[wall_index,start_index], thetas_closest_wall_section[wall_index,start_index]


# In[6]:


def get_trajectory_related_information(trial=None, trial_list=None, trial_num=None, player_id=0):
    '''gather data for head angle and wall vector plots''' 

    # get trial
    trial = extract_trial.extract_trial(trial, trial_list, trial_num)

    # trajectory and headangle vectors for the trial 
    trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_id)
    headangles = trajectory_vectors.extract_trial_player_headangles(trial=trial, player_id=player_id)
    trial_player_headangles =  trajectory_headangle.get_player_headangle_vectors_for_trial(headangles)
    trial_player_headangles_smoothed = trajectory_headangle.get_smoothed_player_head_angle_vectors_for_trial(headangles, window_size=5)

    # vectors between player and alcove, for all walls and timepoints
    player_to_alcove_vectors = trajectory_vectors.get_player_to_alcove_direction_vectors_for_trajectory(trajectory,
                                                                                        num_walls=8)
    # get either the clockwise or counterclockwise octagon vertex coordinate, depending on whether the wall is CW or CCW of the current
    # player head direction
    wall_coords_cross_product_dependent = trajectory_headangle.get_wall_coords_cross_product_dependent(trial=trial, player_id=player_id)
    # using the above coordinates, get the vectors for each wall between head direction and CW or CCW wall coordinate
    player_to_closest_wall_section = trajectory_headangle.get_player_to_closest_wall_section_direction_vectors_for_trajectory(trajectory,
                                                                                                wall_coords_cross_product_dependent)

    # find angles between the head direction and the closest wall sections for each timepoint in the trajectory
    thetas_closest_wall_section = trajectory_headangle.head_angle_to_closest_wall_section_throughout_trajectory(trial=trial,
                                                                    player_id=player_id)
    thetas_closest_wall_section = np.rad2deg(thetas_closest_wall_section)

    # find the angles between the head direction and the alcove centre for each timepoint in the trajectory
    thetas_trajectory = trajectory_headangle.head_angle_to_walls_throughout_trajectory(trajectory,
                                                                                    headangles,
                                                                                    window_size=5, num_walls=8)
    thetas_trajectory = np.rad2deg(thetas_trajectory)

    return (trajectory, trial_player_headangles_smoothed, player_to_alcove_vectors, player_to_closest_wall_section,
            thetas_closest_wall_section, thetas_trajectory)


# In[7]:


def plot_multiple_trials_first_wall_visibility(trial_list, rows=12, cols=12, trial_num_offset=0, player_id=0,
                                               vector_length=20, wall_index=None, start_index=0):
    ''' Display a rows,cols figure of subplots showing the visualisation vectors for player head direction,
        player to alcove centre, and player to closest wall section, for trajectory index start_index, for
        wall index wall_index (default the true chosen wall of the trial), and vector length of vector_length.
        Takes trial list. '''

    fig, axes = plt.subplots(rows,cols, figsize=(20,20))
    index_out_of_range_flag = False
    exception_text = None

    theta_closest_wall_section_session = np.full((rows,cols), np.nan, dtype=float)
    theta_trajectory_session = np.full((rows,cols), np.nan, dtype=float)

    # loop through each trial index
    for i in range(rows):
        for j in range(cols):
            trial_num = i*rows + j + trial_num_offset
            wall_index = None # reset wall index each loop iteration

            try:
                # get all information needed for the plot for this trial
                (trajectory, trial_player_headangles_smoothed, 
                player_to_alcove_vectors, player_to_closest_wall_section,
                thetas_closest_wall_section, thetas_trajectory)= get_trajectory_related_information(trial_list=trial_list,
                                                                                                trial_num=trial_num,
                                                                                                    player_id=player_id)
                

                    
            except Exception as e:
                # print(f"Exception: {e}, no trials left?")
                index_out_of_range_flag = True
                exception_text = e
                axes[i, j].axis('off')
                continue
            
            # calculate coordinates for visualisation vectors for this trial
            (head_direction_vector_coordinates,
            alcove_direction_vector_coordinates,
            closest_wall_section_vector_coordinates,
            wall_index) = get_visualisation_vector_coordinates(trial_list, trial_num, trajectory,
                                                                trial_player_headangles_smoothed,
                                                                player_to_alcove_vectors,
                                                                player_to_closest_wall_section,
                                                                vector_length=vector_length,
                                                                wall_index=wall_index,
                                                                start_index=start_index
                                                                )
            
            
            # plot visualisation vectors for this trial
            axes[i,j] = plot_octagon_visualisation_vectors(head_direction_vector_coordinates, alcove_direction_vector_coordinates,
                                                    closest_wall_section_vector_coordinates, trajectory,
                                                    wall_index, axes=axes[i,j])

            # append the angles between vectors to arrays
            # take the angle for the relevant wall and start index, and then round it to 2 decimal places
            theta_closest_wall_section_session[i,j] = round(thetas_closest_wall_section[wall_index, start_index], 1)
            theta_trajectory_session[i,j] = round(thetas_trajectory[wall_index, start_index],1)           
            
    if index_out_of_range_flag:
        print(f"Exception: {exception_text}, no trials left?")
    
    # adjust layout to prevent overlap
    plt.tight_layout()

    # show the plot
    plt.show()

    return theta_closest_wall_section_session, theta_trajectory_session


# In[8]:


def plot_single_trial_first_wall_visibility(trial_list, trial_num, vector_length=20, start_index=0, player_id=0, wall_index=None):
    ''' Display a rows,cols figure of subplots showing the visualisation vectors for player head direction,
    player to alcove centre, and player to closest wall section, for trajectory index start_index, for
    wall index wall_index (default the true chosen wall of the trial), and vector length of vector_length.
    Takes trial list. '''

    fig, ax = plt.subplots()
    # loop through each trial index

    trial_num = trial_num
    try:
        # get all information needed for the plot for this trial
        (trajectory, trial_player_headangles_smoothed, 
        player_to_alcove_vectors, player_to_closest_wall_section,
        thetas_closest_wall_section, thetas_trajectory)= get_trajectory_related_information(trial_list=trial_list,
                                                                                        trial_num=trial_num,
                                                                                            player_id=player_id)
        
    except Exception as e:
        print(f"Exception: {e}, no trials left?")
    
    # calculate coordinates for visualisation vectors for this trial
    (head_direction_vector_coordinates,
    alcove_direction_vector_coordinates,
    closest_wall_section_vector_coordinates,
    wall_index) = get_visualisation_vector_coordinates(trial_list, trial_num, trajectory,
                                                        trial_player_headangles_smoothed,
                                                        player_to_alcove_vectors,
                                                        player_to_closest_wall_section,
                                                        vector_length=vector_length,
                                                        wall_index=wall_index,
                                                        start_index=start_index
                                                        )
    

    # plot visualisation vectors for this trial
    ax = plot_octagon_visualisation_vectors(head_direction_vector_coordinates, alcove_direction_vector_coordinates,
                                            closest_wall_section_vector_coordinates, trajectory,
                                            wall_index, axes=ax)
    
    

    # show the plot
    plt.show()

    # take the angle for the relevant wall and start index, and then round it to 2 decimal places
    theta_closest_wall_section = round(thetas_closest_wall_section[wall_index, start_index], 1)
    theta_trajectory = round(thetas_trajectory[wall_index, start_index], 1)

    return (theta_closest_wall_section, theta_trajectory)

