# %%
import globals
import parse_data.prepare_data as prepare_data
import data_extraction.trial_list_filters as trial_list_filters
from matplotlib.patches import Wedge
from matplotlib import pyplot as plt
import data_extraction.get_indices as get_indices
import parse_data.identify_filepaths as identify_filepaths
import globals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Polygon
from matplotlib import cm
import math
from parse_data import flip_rotate_trajectories
# import occupancy_and_strategy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import data_strings
import data_extraction.data_saving as data_saving
import plotting.plot_octagon as plot_octagon

# %%
# load_data = True
# data_folder = data_strings.DATA_FOLDER

# %%
# if load_data:
#     # load previously prepared data
#     trial_lists_solo = data_saving.load_data(r'trial_lists_solo_combined_standard_50')
#     trial_lists_social = data_saving.load_data(r'trial_lists_social_standard_50')

# else:

#     data_folder = data_strings.DATA_FOLDER

#     # combine consecutive solo sessions (pre- and post- for an individual player)
#     json_filenames_social, json_filenames_solo = identify_filepaths.get_filenames()
#     _, trial_lists = prepare_data.prepare_data(data_folder, json_filenames_solo, combine=False)
#     print(len(trial_lists))
#     trial_lists = [trial_lists[i] + trial_lists[i+1] for i in range(0, len(trial_lists), 2)]

# %%
def flip_rotate_trial_list(trial_list):
    ''' flip and rotate spatial positions for an entire trial list
      (does not affect wall numbers or other information) '''

    flip_rotated_trials = []
    for trial_index in range(len(trial_list)):
        trial = trial_list[trial_index]
        trial = flip_rotate_trajectories.flip_rotate_trajectories(trial=trial)
        flip_rotated_trials.append(trial)

    return flip_rotated_trials

# %%
# def trial_list_filter_valid_choice(flip_rotated_trial_list, num_players):
#     """ Filter a trial list to leave only valid choice trials for analysis
#         This means:
#         - trials filtered to only include HighLow trials
#         - trial filtered to only include those with a retrievable choice for each player 
        
#         Returns a nested list of valid trials indexed by player_id"""
    
#     trial_lists_choice_filtered = {}
#     trial_lists_choice_filtered_indices = {}

#     # across both player IDs
#     for player_id in range(num_players):

#         trial_lists_choice_filtered[player_id] = []
#         trial_lists_choice_filtered_indices[player_id] = []

#         # across trials
#         for trial_list_index in range(len(flip_rotated_trial_list)):
#             trial_list = flip_rotated_trial_list[trial_list_index]

#             # filter for high/low trials
#             trial_indices = get_indices.get_trials_trialtype(trial_list, trial_type='HighLow')
#             trial_list_filtered = [trial_list[i] for i in trial_indices]

#             # and then trials with a retrievable choice
#             (trial_list_filtered_player_choice_exists,
#               player_choice_exists_indices) = trial_list_filters.filter_trials_retrievable_choice(trial_list_filtered,
#                                                                                                 player_id=player_id,
#                                                                                                 inferred_choice=True)
            
#             trial_lists_choice_filtered[player_id].append(trial_list_filtered_player_choice_exists)
#             trial_lists_choice_filtered_indices[player_id].append(player_choice_exists_indices)
            
#     return trial_lists_choice_filtered, player_choice_exists_indices

# %%
def trial_list_filter_valid_choice(flip_rotated_trial_list, num_players):
    """ Filter a trial list to leave only valid choice trials for analysis
        This means:
        - trials filtered to only include HighLow trials
        - trial filtered to only include those with a retrievable choice for each player 
        
        Returns a nested list of valid trials indexed by player_id"""
    
    trial_list_choice_filtered = [[] for _ in range(num_players)]
    trial_list_choice_filtered_indices = [[] for _ in range(num_players)]

    # across both player IDs
    for player_id in range(num_players):

        trial_list = flip_rotated_trial_list

        # filter for high/low trials
        trial_indices = get_indices.get_trials_trialtype(trial_list, trial_type='HighLow')
        trial_list_filtered = [trial_list[i] for i in trial_indices]

        # and then trials with a retrievable choice
        (trial_list_filtered_player_choice_exists,
            player_choice_exists_indices) = trial_list_filters.filter_trials_retrievable_choice(trial_list_filtered,
                                                                                            player_id=player_id,
                                                                                            inferred_choice=True if num_players > 1 else False)
        
        trial_list_choice_filtered[player_id] = trial_list_filtered_player_choice_exists
        trial_list_choice_filtered_indices[player_id] = player_choice_exists_indices

    return trial_list_choice_filtered, trial_list_choice_filtered_indices

# %%
def trial_list_filter_wall_visibility(trial_list_choice_filtered, player_choice_exists_indices,
                                      num_players, first_visible_wall_index, current_fov = 110):
    """ Return a list of trials and of original indices, with num_players nested lists,
        which is filtered for a given first-visible wall """
    

    trial_list_visibility_filtered = [[] for _ in range(num_players)]
    trial_list_visibility_filtered_indices = [[] for _ in range(num_players)]
    # then, filter for trials where the wall [wall_first_vis_index] becomes visible first
    for player_id in range(num_players):
        (trial_list_filtered_wall_initially_visible,
        wall_initially_visible_indices) = trial_list_filters.filter_trials_one_wall_becomes_visible_first(trial_list_choice_filtered[player_id],
                                                                                                                player_id=player_id,
                                                                                                                current_fov=current_fov,
                                                                                                                wall_index=first_visible_wall_index,
                                                                                                                original_indices=player_choice_exists_indices[player_id])
        
        trial_list_visibility_filtered[player_id] = (trial_list_filtered_wall_initially_visible)
        trial_list_visibility_filtered_indices[player_id] = (wall_initially_visible_indices)

        
    return trial_list_visibility_filtered, trial_list_visibility_filtered_indices
        

# %%
def trial_list_filter_chose_wall(trial_list, num_players, chosen_wall_index):
    """ Filter a trial list to leave only trials where a given player chose 
        the chosen_wall_index wall
        Returns a nested list of valid trials indexed by player_id """

    trial_list_choice_filtered = [[] for _ in range(num_players)]
    trial_list_choice_filtered_indices = [[] for _ in range(num_players)]
    for player_id in range(num_players):
        # filter for trials where player chose high wall
        (trial_list_filtered_chose_wall,
        original_indices) = trial_list_filters.filter_trials_player_chose_given_wall(trial_list[player_id],
                                                                                      player_id=player_id,
                                                                    inferred_choice=True if num_players > 1 else False,
                                                                    given_wall_index=chosen_wall_index, # chose high
                                                                    original_indices=None,
                                                                    debug=False)
        
    
        trial_list_choice_filtered[player_id] = trial_list_filtered_chose_wall
        trial_list_choice_filtered_indices[player_id] = original_indices

    return trial_list_choice_filtered, trial_list_choice_filtered_indices

# %%
def filter_pipeline_p_high_first_seen_wall(trial_lists, num_players, first_visible_wall_index, chosen_wall_index):

    trial_list_vis_filtered = {}
    trial_list_vis_and_choice_filtered = {}


    for i, trial_list in enumerate(trial_lists):
        # Step 1: flip and rotate
        flip_rotated = flip_rotate_trial_list(trial_list)

        # Step 2: filter for valid choice trials
        valid_choice_filtered, valid_choice_filtered_indices = trial_list_filter_valid_choice(flip_rotated, num_players)

        # Step 3: filter for trials where a particular wall is first visible (first_visible_wall_index)
        vis_filtered, vis_filtered_indices = trial_list_filter_wall_visibility(
                                        valid_choice_filtered, valid_choice_filtered_indices,
                                          num_players, first_visible_wall_index
        )

        # Step 4: filter for trials where player chose the chosen_wall_index wall
        chose_wall_filtered, chose_wall_filtered_indices = trial_list_filter_chose_wall(vis_filtered, num_players, chosen_wall_index)

        # Store results indexed by player_id
        for player_id in range(num_players):
            trial_list_vis_filtered.setdefault(player_id, []).append(vis_filtered[player_id])
            trial_list_vis_and_choice_filtered.setdefault(player_id, []).append(chose_wall_filtered[player_id])

    return trial_list_vis_filtered, trial_list_vis_and_choice_filtered



