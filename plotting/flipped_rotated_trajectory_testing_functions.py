#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import plotting.wall_visibility_order_testing_functions as wall_visibility_order_testing_functions
import parse_data.flip_rotate_trajectories as flip_rotate_trajectories


# In[1]:


def get_trajectory_information_trial(chosen_walls_session, trial=None, trial_list=None, trial_index=None, player_id=0):
    '''Gather single trial data for trajectory rotation plots.
       Takes trial list, index, and chosen walls for the session
       Returns the trial df, rotation angle to be applied, rotated and flipped df,
       the walls for the trial, and the chosen wall for the trial.''' 

    # get trial
    trial = extract_trial.extract_trial(trial=trial, trial_list=trial_list, trial_index=trial_index)

   #  trajectory 
   #  trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_id)

    # trial rotation angle
    rotation_angle_trial = flip_rotate_trajectories.find_rotation_angle_trial(trial)

    rotated_flipped_trial = flip_rotate_trajectories.flip_rotate_trajectories(trial)

    # trial walls
    walls = get_indices.get_walls(trial_list=trial_list, trial_index=trial_index)
    chosen_wall = chosen_walls_session[trial_index]


    return (trial, rotation_angle_trial, rotated_flipped_trial, walls, chosen_wall)


# In[3]:


def plot_octagon_trajectories(trial, rotated_flipped_trial, label=False, axes=None):
    ''' Plot two subplots of single trial trajectories, with separate winner and loser
        colours.
        Left subplot is without flipping and rotating, right subplot is with.
        Takes the trial df, and the rotated and flipped trial df.'''
        
    axes[0] = plot_octagon.plot_octagon(ax=axes[0])
    axes[1] = plot_octagon.plot_octagon(ax=axes[1])

    axes[0] = plot_trajectory.plot_trial_trajectory(axes[0], trial=trial, label=label)
    axes[1] = plot_trajectory.plot_trial_trajectory(axes[1], trial=rotated_flipped_trial, label=label)

    # change plot params
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
            ax.tick_params(left=False, bottom=False)  # Turn off major ticks
            ax.set_xticklabels([])
            ax.set_yticklabels([])



# In[ ]:


def plot_single_trial_flip_rotate_trajectories(trial_list, chosen_walls_session, trial_index, player_id=0):
    ''' Umbrella function to plot trajectories for a single trial, pre- and post- flip/rotating respectively.
        Takes a list of trials, the chosen walls for a session, and the trial index.
        Returns High wall val, Low wall val, and the chosen wall val.'''

    fig, axes = plt.subplots(1,2)

    # get trajectory information
    (trial, rotation_angle_trial, rotated_flipped_trial, walls, chosen_wall) = get_trajectory_information_trial(trial_list=trial_list,
                                                                                                                 trial_index=trial_index,
                                                                                                                 chosen_walls_session=chosen_walls_session,
                                                                                                                   player_id=player_id)

    # plot trajectories for this trial
    axes = plot_octagon_trajectories(trial, rotated_flipped_trial, label=False, axes=axes)

    # show the plot
    plt.show()

    # return the High wall, Low wall, and chosen wall on this trial
    return (walls[0], walls[1], int(chosen_wall))


# In[ ]:


def plot_multiple_trials_flip_rotate_trajectories(trial_list, chosen_walls_session, rows=12, cols=12, trial_num_offset=0, player_id=0,
                                               vector_length=20, wall_index=None, start_index=0):
    ''' Display a rows,cols figure of subplots showing the flipped and rotated trajectories for both players in single trials.
        Plots are in pairs, with the first plotted being pre-flip and the second being post-flip.
        Takes trial list. '''

    fig, axes = plt.subplots(rows,cols, figsize=(20,20))
    index_out_of_range_flag = False
    exception_text = None


    # loop through each trial index
    for i in range(rows):
        # loop through -1 as we plot 2 subplots per loop
        for j in range(0,cols,2):
            trial_index = i*rows + j + trial_num_offset

            try:
                # get trajectory information
                (trial, rotation_angle_trial, rotated_flipped_trial, walls,
                    chosen_wall) = get_trajectory_information_trial(trial_list=trial_list, trial_index=trial_index,
                                                                                        chosen_walls_session=chosen_walls_session,
                                                                                            player_id=player_id)

                    
            except Exception as e:
                index_out_of_range_flag = True
                exception_text = e
                axes[i, j].axis('off')
                axes[i, j+1].axis('off')
                continue
            
            
            # plot visualisation vectors for this trial
            axes[i,j:j+2] = plot_octagon_trajectories(trial, rotated_flipped_trial, label=False, axes=axes[i,j:j+2])

            
    if index_out_of_range_flag:
        print(f"Exception: {exception_text}, no trials left?")
    
    # adjust layout to prevent overlap
    plt.tight_layout()

    # show the plot
    plt.show()


# In[ ]:


def plot_player_start_positions(rotated_flipped_trial, chosen_player, label=False, axes=None):
    ''' Plot two subplots of single trial trajectories, with separate winner and loser
        colours.
        Left subplot is without flipping and rotating, right subplot is with.
        Takes the trial df, and the rotated and flipped trial df.'''
        
    ax = plot_octagon.plot_octagon(ax=axes)

    ax = plot_trajectory.plot_trial_slice_onset_positions(ax, chosen_player, trial=rotated_flipped_trial, label=label)

    # change plot params
    for spine in ax.spines.values():
        spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)  # Turn off major ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])

