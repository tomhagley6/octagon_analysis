import parse_data.prepare_data as prepare_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import globals
import data_strings
import data_extraction.extract_trial as extract_trial
import utils.cosine_similarity as cosine_similarity
import analysis.wall_visibility_and_choice as wall_visibility_and_choice
import data_extraction.get_indices as get_indices
import plotting.plot_probability_chose_wall as plot_probability_chose_wall
import plotting.flipped_rotated_trajectory_testing_functions as fr_funcs

def filter_trials_other_visible(trial_list, other_visible_session, inverse=False, original_indices=None):
    ''' Return a filtered trial list and list of indices from the original trial list that
        conform to Other visible to player player_id at trial start.
        If inverse=True, return only trials where Other is not visible '''
    
    # if no original indices supplied, assume this is the original trial list
    if original_indices is None:
        original_indices = np.arange(len(trial_list))
    
    if not inverse:
        other_visible_mask = other_visible_session == True
    else:
        other_visible_mask = other_visible_session == False
    
    # filter original indices and current trial list based on the mask
    trial_list_filtered = [trial_list[i] for i in np.flatnonzero(other_visible_mask)]
    original_indices = original_indices[other_visible_mask]

    # return the list of filtered trials and the indices of these trials as relates to the original trial list
    return trial_list_filtered, original_indices
    
    

def filter_trials_retrievable_choice(trial_list, player_id, inferred_choice, original_indices=None,
                                     debug=False):
    ''' Return the filtered trial list and list of indices from the original trial list that
        conform with player player_id having a recorded choice.
        This is required for accurate probabilities, because we do cannot include trials (as negative)
        where we do not know what the player's choice would have been. '''
    
    # if no original indices supplied, assume this is the original trial list
    if original_indices is None:
        original_indices = np.arange(len(trial_list))
    
    # get player choice (wall number) for each trial
    # inferred choice can be used here
    player_choice = wall_visibility_and_choice.get_player_wall_choice(trial_list, player_id,
                                                                        inferred_choice, debug=False)
    
    if debug:
        print(f"player_choice, inferred status {inferred_choice} is:\n{player_choice}")
    
    # boolean mask of trial list to only include trials where this player had a recorded choice
    retrievable_choice_mask = ~np.isnan(player_choice)

    # filter original indices and current trial list based on the mask
    original_indices = original_indices[retrievable_choice_mask]
    trial_list_filtered = [trial_list[i] for i in np.flatnonzero(retrievable_choice_mask)]

    # return the list of filtered trials and the indices of these trials as relates to the original trial list
    return trial_list_filtered, original_indices

    

def filter_trials_one_wall_initially_visible(trial_list, player_id, wall_index, current_fov, original_indices=None):
    ''' Return a filtered trial list and list of indices from the original trial list that
        conform to a single trial wall being visible to player player_id at trial start,
        conferred by wall_index (e.g. 0 or 1 for wall1 or wall2) '''
    
    # if no original indices supplied, assume this is the original trial list
    if original_indices is None:
        original_indices = np.arange(len(trial_list))
    
    # find wall visibility (at the trial start timepoint) for the full session
    (wall1_visible_session,
    wall2_visible_session) = wall_visibility_and_choice.get_walls_initial_visibility_session(trial_list,
                                                                    player_id, current_fov, debug=False)
    
    # use np bitwise operators to find trials with only the relevant wall visible
    # also maintain a list of indices relative to the original trial list
    if wall_index == 0:
        given_wall_init_vis_mask = (wall1_visible_session == True) & (wall2_visible_session == False)

    elif wall_index == 1:
        given_wall_init_vis_mask = (wall2_visible_session == True) & (wall1_visible_session == False)
    
    # filter original indices and current trial list based on the mask
    trial_list_filtered = [trial_list[i] for i in np.flatnonzero(given_wall_init_vis_mask)]
    original_indices = original_indices[given_wall_init_vis_mask]

    # return the list of filtered trials and the indices of these trials as relates to the original trial list
    return trial_list_filtered, original_indices

def filter_trials_both_walls_initially_visible(trial_list, player_id, current_fov, original_indices=None):
    ''' Return a filtered trial list and list of indices from the original trial list that
        conform to both trial walls being visible to player player_id at trial start. '''
    
    # if no original indices supplied, assume this is the original trial list
    if original_indices is None:
        original_indices = np.arange(len(trial_list))
    
    # find wall visibility for the full session
    (wall1_visible_session,
    wall2_visible_session) = wall_visibility_and_choice.get_walls_initial_visibility_session(trial_list,
                                                                    player_id, current_fov,
                                                                    debug=False)
    

    both_walls_visible_init_vis_mask = (wall1_visible_session == True) & (wall2_visible_session == True)
    
    # filter original indices and current trial list based on the mask
    trial_list_filtered = [trial_list[i] for i in np.flatnonzero(both_walls_visible_init_vis_mask)]
    original_indices = original_indices[both_walls_visible_init_vis_mask]

    # return the list of filtered trials and the indices of these trials as relates to the original trial list
    return trial_list_filtered, original_indices

def filter_trials_one_wall_becomes_visible_first(trial_list, player_id,
                                                 current_fov, wall_index,
                                                 original_indices=None,
                                                 debug=False):
    ''' Return a filtered trial list and list of indices from the original trial list that
        conform to one of the trial walls becoming visible to player player_id during the trial
        before any other. '''
    

    # if no original indices supplied, assume this is the original trial list
    if original_indices is None:
        original_indices = np.arange(len(trial_list))

    # find whether given wall is visible first (and initially alone) for this player for a full session
    # TODO This can be checked again after having written
    given_wall_first_visible_session = wall_visibility_and_choice.get_given_wall_first_visible_session(trial_list,
                                                                                                       player_id,
                                                                                                       wall_index,
                                                                                                       current_fov,
                                                                                                       debug)
    
    print(f"filter_trials_one_wall_becomes_visible_first - given_wall_first_visible_session array:\n{given_wall_first_visible_session}")
    print(f"And the number of valid trials at this step is {np.sum(given_wall_first_visible_session == 1)}")

    given_wall_visibile_first_mask = given_wall_first_visible_session == True
    
    # filter original indices and current trial list based on the mask
    trial_list_filtered = [trial_list[i] for i in np.flatnonzero(given_wall_visibile_first_mask)]
    original_indices = original_indices[given_wall_visibile_first_mask]

    # return the list of filtered trials and the indices of these trials as relates to the original trial list
    return trial_list_filtered, original_indices

# But this function makes no distinction on which player_id was responsible for choosing the given wall? 
def filter_trials_player_chose_given_wall(trial_list, player_id, inferred_choice, given_wall_index, original_indices=None,
                                          debug=False):
    ''' Return a filtered trial list and list of indices from the original trial list 
        where player choice (winner + loser, or just winner) aligned with
        the given wall index (e.g., 0 for wall1) '''
    
    # if no original indices supplied, assume this is the original trial list
    if original_indices is None:
        original_indices = np.arange(len(trial_list))
    
    # get player choice (wall number) for each trial
    # inferred choice can be used here
    player_choice = wall_visibility_and_choice.get_player_wall_choice(trial_list, player_id,
                                                                        inferred_choice, debug=False)
    if debug:
        print(f"player choice array:\n{player_choice}")

    # get the truth array for whether the player choice wall aligns with the given wall parameter
    # to this function (NB. this is NOT the wall that was eventually chosen in the trial)
    given_wall_chosen_session = get_indices.was_given_wall_chosen(trial_list, player_choice,
                                                                  given_wall_index)
    if debug:
        print(f"given wall chosen array:\n{given_wall_chosen_session}")
    
    # find the indices of the trials in trial_list where the given wall was chosen by player player_id.
    # this will drop trials where the given wall was not chosen, and trials without retrievable choice information
    given_wall_chosen_mask = given_wall_chosen_session == True
    
    if debug:
        print(f"given wall chosen true indices:\n{ np.flatnonzero(given_wall_chosen_mask)}")
    
    # filter original indices and current trial list based on the mask
    trial_list_filtered = [trial_list[i] for i in np.flatnonzero(given_wall_chosen_mask)]
    original_indices = original_indices[given_wall_chosen_mask]

    # return the list of filtered trials and the indices of these trials as relates to the original trial list
    return trial_list_filtered, original_indices
    


def filter_trials_player_won(trial_list, player_id, original_indices=None):
    ''' Return a filtered trial list and list of indices from the original trial list 
        where player player_id won. '''
    
    # if no original indices supplied, assume this is the original trial list
    if original_indices is None:
        original_indices = np.arange(len(trial_list))
    
    # get trigger activators for this session
    trigger_activators = get_indices.get_trigger_activators(trial_list)
    
    # find the indices of the trials in trial_list where player player_id won
    this_player_wins_mask = trigger_activators == player_id
    
    # filter original indices and current trial list based on the mask
    trial_list_filtered = [trial_list[i] for i in np.flatnonzero(this_player_wins_mask)]
    original_indices = original_indices[this_player_wins_mask]

    # return the list of filtered trials and the indices of these trials as relates to the original trial list
    return trial_list_filtered, original_indices
    

