import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import globals
import data_extraction.get_indices as get_indices
import data_extraction.trial_list_filters as trial_list_filters
import analysis.opponent_visibility as opponent_visibility

def probability_trial_outcome_given_conditions(trial_list, player_id, 
                                               visible_wall_index, chosen_wall_index, other_visible,
                                               wall_initial_vis_only, inferred_choice, current_fov,
                                                 inverse_other_visible=False,
                                               return_trial_indices=False, data_size_cutoff=4,
                                                 debug=False):
    ''' Take a trial list and filter based on existence of player_id player choice,
        Other visibility at trial start (if other_visible is not None,
        with inverse_other_visible=True returning Other non-visibility),
        and whether a specific wall becomes visible alone first (if visible_wall_index is not None). 
        Also, filter this trial list by the player choice being for a specific trial wall 
        (if chosen_wall_index is not None) and take the proportion of trials remaining
        as the probability of making this choice conditioned on our chosen filters.
        Else, the trials remaining will be set to all winning trials for this player.
        Returns probability of choosing a trial wall. 
        inferred_choice decides whether to include inferred choice for identifying retrievable choice and chosen wall.
        visible_wall_index and chosen_wall_index control wall initial visibility and player choice filters respectively. '''
    
    if debug:
      print(f"Next session, player_id {player_id}")

    # keep a list of indices relative to the original trial list
    original_indices = np.arange(len(trial_list))
    
    if debug:
      print(f"initial original indices are: {original_indices}")

    # filter trials with a retrievable choice for this player (trials only valid for analysis if we 
    # have a recorded choice for the player) if a chosen_wall_index is specified
    if chosen_wall_index is not None:
      (trial_list_filtered,
      original_indices) = trial_list_filters.filter_trials_retrievable_choice(trial_list, player_id,
                                                            inferred_choice, original_indices=original_indices)

      if debug:
        print(f"Len 'player choice exists' indices: {len(original_indices)}")

    else:
       trial_list_filtered = trial_list

    
    if other_visible:
      # get Other visibility status for this session and player_id
      orientation_angle_to_other_session = opponent_visibility.get_angle_of_opponent_from_player_session(player_id, trial_list_filtered)
      other_visible_session = opponent_visibility.get_other_visible_session(orientation_angle_to_other_session, current_fov)
      # filter Other initially visible
      (trial_list_filtered,
      original_indices) = trial_list_filters.filter_trials_other_visible(trial_list_filtered, other_visible_session,
                                                      inverse=inverse_other_visible,
                                                      original_indices=original_indices)
      
      if debug:
        print(f"Len 'player other initially visible' indices with inverse status {inverse_other_visible}: {len(original_indices)}")
    
    # filter based on initial/first wall visibility if requested
    if visible_wall_index is not None:
      if wall_initial_vis_only:
        # filter with visible_wall_index wall initially visible (at slice onset)
        (trial_list_filtered,
        original_indices) = trial_list_filters.filter_trials_one_wall_initially_visible(trial_list_filtered, player_id,
                                                                    wall_index=visible_wall_index, current_fov=current_fov,
                                                                    original_indices=original_indices)
      
      elif not wall_initial_vis_only:
            # filter with visible_wall_index becomes visible first (during the trial)
            (trial_list_filtered,
            original_indices) = trial_list_filters.filter_trials_one_wall_becomes_visible_first(trial_list_filtered, player_id,
                                                                            current_fov, wall_index=visible_wall_index,
                                                                            original_indices=original_indices,
                                                                            debug=False)   
      if debug:
          print(f"Len '{visible_wall_index} with initial_vis_only {wall_initial_vis_only}': {len(original_indices)}")   
                                                                          

    # filter based on chosen wall if requested
    if chosen_wall_index is not None:
        # filter with chosen_wall_index chosen
        (trial_list_filtered_choice,
        original_indices_choice) = trial_list_filters.filter_trials_player_chose_given_wall(trial_list_filtered, player_id,
                                                                  inferred_choice,
                                                                  given_wall_index=chosen_wall_index,
                                                                  original_indices=original_indices)
    else: # otherwise, default to all wins for this player
        (trial_list_filtered_choice,
        original_indices_choice) = trial_list_filters.filter_trials_player_won(trial_list_filtered, player_id,
                                                            original_indices=original_indices)
       
        if debug:
          print(f"Len '{chosen_wall_index} index wall chosen by player {player_id}': {len(original_indices_choice)}")

    
    # return np.nan in place of probability if fraction denominator is below or equal to data_size_cutoff
    if len(original_indices) <= data_size_cutoff:
      print(f"fewer than {data_size_cutoff} trials in the denominator, returning np.nan instead of probability")
      if not return_trial_indices:
        return np.nan
      else:
        return np.nan, original_indices, original_indices_choice
    
    # find the probability of the player choosing the given wall index, considering only trials that are filtered
    # prior to choice filtering
    probability_chose_wall = calculate_probability_choose_wall(trial_list_filtered, trial_list_filtered_choice)

    if debug:
      print(f"Probability player chose wall given these conditions: {probability_chose_wall}")

    if debug:
       print(f"Final original indices are: {original_indices}")

    if not return_trial_indices:
      return probability_chose_wall
    else:
      return probability_chose_wall, original_indices, original_indices_choice

def probability_trial_outcome_given_conditions_all_sessions(trial_lists, inferred_choice, current_fov,
                                                            chosen_wall_index=None, visible_wall_index=None,
                                                            other_visible=None, wall_initial_vis_only=False,
                                                             solo=False, wall_sep=None, inverse_other_visible=False,
                                                            trial_type=globals.HIGH_LOW, debug=True):
    
    ''' Returns two dictionaries: probabilities and trial_data.
        probabilities contains the probability of trial outcome given conditions for each session and player.
        Arrays are of shape num_sessions, num_players.
        denominator and numerator contains the trial indices (relative to the original trial list for each session) of the 
        numerator and denominator trials that feed into the final probability calculation for that player and session.
        Lists contain 2 nested lists, one for each player.
        Takes a list of trial lists (one for each session), inferred choice, fov. 
        Optionally takes chosen_wall_index and visible_wall_index to specify choosing either High or Low,
        or see High or Low at the beginning of the session, respectively. Default is to not select on these.
        Also optionall takes other_visible, to specify whether to condition on initial opponent visibility,
        with inverse_other_visible instead condition on initial opponent non-visibility.
        Also adapts to solo or social data. However, be aware whether solo sessions from an individual player
        have been combined or not before passing. If not, consecutive pairs of entries will be pre- and post-
        solo sessions for one player.
        '''
    
    if not solo:
      # initialize arrays and lists to store probabilities and indices (respectively) for the condition
      probabilities =  np.full((len(trial_lists), 2), np.nan) # two columns for two players
      denominator = [[],[]]
      numerator = [[],[]]
    else:
      probabilities =  np.full((len(trial_lists)), np.nan) # one column, as one player per solo session
      denominator = []
      numerator = []

    
    # loop through each session in the trial list
    for trial_list_index, trial_list in enumerate(trial_lists):

        # filter trial list for specified trial type (e.g., HIGH_LOW)
        trial_indices = get_indices.get_trials_trialtype(trial_list, trial_type=trial_type)
        trial_list_filtered = [trial_list[i] for i in trial_indices]

        # also filter trial list by specific wall separation if specified
        if wall_sep:
            trial_indices = get_indices.get_trials_with_wall_sep(trial_list_filtered, wall_sep=wall_sep)
            trial_list_filtered = [trial_list_filtered[i] for i in trial_indices]


        # iterate over both player IDs (0 and 1)
        for player_id in [0,1]:
            
            # skip the player_id == 1 loop iteration if we are analysing solo sessions
            if solo:
                if player_id == 1:
                    continue

            # calculate probabilities and filtered indices for all specified conditions
            (probability,
            filtered_indices_visible,
            filtered_choice_indices_visible) = probability_trial_outcome_given_conditions(trial_list_filtered, player_id, 
                                                visible_wall_index, chosen_wall_index, other_visible, wall_initial_vis_only,
                                                inferred_choice, current_fov, inverse_other_visible,
                                                return_trial_indices=True, debug=debug)

            if not solo:
              # store probabilities for each player in the respective column (player 0 -> col 0, player 1 -> col 1)
              probabilities[trial_list_index, player_id] = probability
              # append filtered indices to the nested list for each player (player 0 -> index 0, player 1 -> index 1)
              denominator[player_id].append(filtered_indices_visible)
              numerator[player_id].append(filtered_choice_indices_visible)
            else:
              # store probabilities for solo player, one column
              probabilities[trial_list_index] = probability
              # append filtered indices for solo player, one column
              denominator.append(filtered_indices_visible)
              numerator.append(filtered_choice_indices_visible)


    return probabilities, numerator, denominator

def calculate_probability_choose_wall(trial_list, trial_list_choice_filtered):
    ''' Given a trial list (pre-filtered, but not for choice), calculate the probability that 
        a player will choose a given wall value as the proportion of trials from the trial
        list in which the player chose the wall value.
        More complex use of this function could involve e.g. filtering the trial list for 
        trials where Low was first seen and the Opponent is visible, and then further filtering
        for player choice being 'Low', to find probability of (choose Low | first visible) under the
        condition of Other visibility at trial start. '''
        
    
    # use the length of the trial list pre-choice filtering, and the length of the trial list post-choice
    # filtering (e.g. with filter_trials_player_chose_given_wall) to calculate the proportion of 
    # relevant trials that a player chose a specific wall
    try:
        probability_chose_wall = len(trial_list_choice_filtered)/len(trial_list)
    except ZeroDivisionError:
        probability_chose_wall = np.nan

    return probability_chose_wall
    
    
