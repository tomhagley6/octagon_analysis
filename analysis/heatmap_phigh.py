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

# %% [markdown]
# ### Filter functions

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

# %% [markdown]
# ### Umbrella filtering function
# 

# %%
def filter_pipeline_p_high_first_seen_wall_no_filtering_vis(trial_lists, num_players, first_visible_wall_index, chosen_wall_index):

    trial_list_vis_filtered = {}
    trial_list_vis_and_choice_filtered = {}


    for i, trial_list in enumerate(trial_lists):
        # Step 1: flip and rotate
        flip_rotated = flip_rotate_trial_list(trial_list)

        # Step 2: filter for valid choice trials
        valid_choice_filtered, _ = trial_list_filter_valid_choice(flip_rotated, num_players)


        # Step 4: filter for trials where player chose the chosen_wall_index wall
        chose_wall_filtered, _ = trial_list_filter_chose_wall(valid_choice_filtered, num_players, chosen_wall_index)

        # Store results indexed by player_id
        for player_id in range(num_players):
            trial_list_vis_filtered.setdefault(player_id, []).append(valid_choice_filtered[player_id])
            trial_list_vis_and_choice_filtered.setdefault(player_id, []).append(chose_wall_filtered[player_id])

    return trial_list_vis_filtered, trial_list_vis_and_choice_filtered



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


# %%


# %% [markdown]
# ### post-processing functions

# %%
def heatmap_counts_division(bins_dict_wall_seen_wall_chosen, bins_dict_wall_seen):

    # element-wise division of numerator and denominator arrays, with handling for division by zero
    probabilities = np.divide(
        bins_dict_wall_seen_wall_chosen, bins_dict_wall_seen,
        out=np.zeros_like(bins_dict_wall_seen_wall_chosen, dtype=float),
        # boolean array to mask division for only non-zero denominator entries
        where=bins_dict_wall_seen > 0
    )

    return probabilities


def heatmap_lowbincounts_filter(probabilities, bin_counts, min_trials=10):
    ''' Filter out bins with fewer than min_trials trials by setting them to NaN
        bincounts: array or list of arrays of bin counts to use for filtering '''
    
    probabilities_filtered = np.copy(probabilities)

    if isinstance(bin_counts, list):
        for array in bin_counts:
            probabilities_filtered[array <= min_trials] = np.nan
    else:
            probabilities_filtered[bin_counts <= min_trials] = np.nan
    
    return probabilities_filtered

# %% [markdown]
# ### plotting function

# %%
def plot_heatmap_phigh(probabilities, n_rows=10, n_cols=10, difference=False,
                        cmap_name='inferno', diff_cmap_name='PiYG'):
    ''' Plot a heatmap of P(Choose High) across spatial bins
     
        probabilities: 2D array of probabilities across heatmap bin space
        Expects an array produced by heatmap_phigh functions, numerator divided by denominator,
        and filtered for a minimum number of trials per bin'''
    
    n_rows, n_cols = 10, 10
    x_min, x_max = -20, 20
    y_min, y_max = -20, 20
    grid_width = (x_max - x_min) / n_cols
    grid_height = (y_max - y_min) / n_rows

    octagon_vertex_coordinates = plot_octagon.return_octagon_path_points()

    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = cmap_name if not difference else diff_cmap_name
    cmap = cm.get_cmap(cmap).copy()
    cmap.set_bad(color='lightgrey')

    if not difference: # normalise probabilities to 0-1 if not a difference heatmap
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        im = ax.imshow(probabilities, extent=[x_min, x_max, y_min, y_max],
                    origin='lower', norm=norm, cmap=cmap)
    else: # keep probabilities within a fixed range for difference heatmap
        im = ax.imshow(probabilities, extent=[x_min, x_max, y_min, y_max],
                origin='lower', vmin=-1, vmax=1, cmap=cmap)

    patch = Polygon(octagon_vertex_coordinates, edgecolor='black', facecolor='none', lw=2)
    ax.add_patch(patch)
    im.set_clip_path(patch)

    ax.set_xlim([-22, 22])
    ax.set_ylim([-22, 22])
    ax.set_aspect(1.)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    #cbar.set_label("Average Occupancy", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.show()



