# %%
import numpy as np
from analysis import opponent_visibility, wall_visibility_and_choice
from data_extraction import get_indices
from plotting import plot_octagon
from trajectory_analysis import trajectory_vectors
import os
import pickle
import pandas as pd
import globals

# %% [markdown]
# ### Regressor value extraction functions (for one session). To be used if not loading pre-generated analysis_results

# %%
def extract_wall_sep(trial_list, normalise=False):
    ''' Return wall separation for one session.
        1 for 45 degrees, 2 for 90 and 4 for 180. '''
    
    wall_sep = np.full(len(trial_list), np.nan)
    for i, trial in enumerate(trial_list):
        wall_sep_this_trial = get_indices.get_wall_difference(trial=trial)
        wall_sep[i] = wall_sep_this_trial

    if normalise:
        wall_sep = wall_sep/4

    return wall_sep


def extract_first_wall_seen(trial_list, player_id):
    ''' Return first visible walls for one player across one session.
        1 for WALL_1, 2 for WALL_2, np.nan for no visible wall (or both initially visible). '''
        
    high_wall_first_visible_session = wall_visibility_and_choice.get_given_wall_first_visible_session(trial_list,
                                                                                                        player_id,
                                                                                                        wall_index=0,
                                                                                                        current_fov=110)

    low_wall_first_visible_session = wall_visibility_and_choice.get_given_wall_first_visible_session(trial_list,
                                                                                                        player_id, 
                                                                                                        wall_index=1,
                                                                                                        current_fov=110)
    low_wall_first_visible_session = low_wall_first_visible_session*2
    
    first_visible_session = high_wall_first_visible_session + low_wall_first_visible_session

    first_visible_session[first_visible_session == 0] = np.nan

    return first_visible_session

def extract_first_wall_visibilities(trial_list, player_id, three_levels=False):
    ''' Return first visible walls for one player across one session.
        1 for WALL_1, 2 for WALL_2, and np.nan for no visible wall.
        If three_levels, 1 for WALL_1, 2 for WALL_2, 3 for both visible, and np.nan for neither. '''

    first_visible_session = np.full(len(trial_list), np.nan)
    for i, trial in enumerate(trial_list):
        wall_vis_order = wall_visibility_and_choice.get_wall_visibility_order_trial(player_id, trial, current_fov=110)

        # decide whether one wall is first visible, both were initially visible, or neither wall was visible
        # plus 1 to each index to match the wall number (1 and 2) rather than the index (0 and 1)
        if np.all(wall_vis_order == 0): # both walls visible at the start of the trial
            if three_levels:
                this_trial_first_visible = wall_vis_order.size + 1
            else:
                this_trial_first_visible = np.nan  # optionally set both walls visible to np.nan instead of 3
        elif np.all(np.isnan(wall_vis_order)): # neither wall visible at the start of the trial
            this_trial_first_visible = np.nan
        elif np.sum(wall_vis_order == 0) == 1: # one wall visible at the start of the trial
            this_trial_first_visible = np.where(wall_vis_order == 0)[0][0] + 1

        first_visible_session[i] = this_trial_first_visible

    return first_visible_session

# double check code
def extract_distances_to_walls(trial_list, player_id, normalise=False):
    ''' Return a trial_num, 2 sized array, where column 1
        is distance to WALL_1, and column 2 is distance to WALL_2.
        Data applies to one full session, and specified player_id.
        If normalise, returns distances as a proportion of the maximum
        possible in the arena '''
    
    # get octagon alcove coordinates
    alcove_coordinates = plot_octagon.return_alcove_centre_points()

    positions_session = np.full((len(trial_list), 2), np.nan)
    walls_session = np.full((len(trial_list), 2), np.nan)
    distances_session = np.full((len(trial_list), 2), np.nan)

    # get distances for each trial in the session
    for i, trial in enumerate(trial_list):
        # get WALL_1 and WALL_2 coordinates
        trial_walls = get_indices.get_walls(trial)
        high_wall_idx = trial_walls[0] - 1
        low_wall_idx = trial_walls[1] - 1
        trial_high_coordinates = alcove_coordinates[:,high_wall_idx]
        trial_low_coordinates = alcove_coordinates[:, low_wall_idx]

        # index trajectory at timepoint 0 to get player starting coordinates
        trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_id)
        trial_start_position = trajectory[:,0]

        # find distance between start position and WALL_1/WALL_2
        d2h = np.linalg.norm(trial_high_coordinates - trial_start_position) # WALL_1
        d2l = np.linalg.norm(trial_low_coordinates - trial_start_position) # WALL_2

        walls_session[i,:] = trial_walls 
        positions_session[i,:] = trial_start_position
        distances_session[i,:] = np.hstack((d2h, d2l))

    # normalise to maximum possible distance in octagon
    if normalise:
        distances_session = distances_session/plot_octagon.return_maximum_distance()

    return distances_session


def extract_opponent_visibility_slice_onset(trial_list, player_id, current_fov=110):
    ''' Return opponent visibility at slice onset for one player for one session.
        1 for opponent visible, 0 for opponent not visible '''
    
    # slice onset angle of Other from self centre FoV
    orientation_angle_to_other_session = opponent_visibility.get_angle_of_opponent_from_player_session(player_id, trial_list)

    # boolean array of Other visible
    other_visible_session = opponent_visibility.get_other_visible_session(orientation_angle_to_other_session, current_fov)
    other_visible_session = other_visible_session.astype(int) # converted to int for categorical regressor

    # does this return 1 and 0? 

    return other_visible_session


def extract_player_choice(trial_list, player_id, inferred_choice=True, debug=True):
    ''' Return (inferred by default) player choice for one player for one session.
        Where inferred and actual choice are both missing, values are np.nan '''

    # array of wall numbers where player choice is available, np.nan where it is not
    player_choice = wall_visibility_and_choice.get_player_wall_choice(trial_list, player_id,
                                                                        inferred_choice=inferred_choice, debug=debug)

    # 2 where player chose High, 0 where player chose Low, np.nan where lacking inferred choice
    high_wall_chosen_session = get_indices.was_given_wall_chosen(trial_list, player_choice,
                                                                    given_wall_index=0)
    high_wall_chosen_session = high_wall_chosen_session*2
    print(f"High wall chosen session:\n{high_wall_chosen_session}")

    # 1 where player chose Low, 0 where player chose High, np.nan where lacking inferred choice
    low_wall_chosen_session  = get_indices.was_given_wall_chosen(trial_list, player_choice,
                                                                    given_wall_index=1)
    
    print(f"Low wall chosen session:\n{low_wall_chosen_session}")

    # 1 or 2 where player chose Low or High respectively, np.nan where lacking inferred choice
    chosen_wall_session = high_wall_chosen_session + low_wall_chosen_session

    print(f"Overall chosen wall for this session:\n{chosen_wall_session}")

    # Does this switch to 0 or 1 respectively and np.nan? 
    chosen_wall_session = chosen_wall_session -1 

    return chosen_wall_session


def extract_trial_outcome(trial_list, player_id):
    ''' Return whether this player won the trial for one player for one session '''
    
    trigger_activators = get_indices.get_trigger_activators(trial_list)
    this_player_won_session = (trigger_activators-1)*-1 if player_id == 0 else trigger_activators

    return this_player_won_session

# %% [markdown]
# ### Analysis dictionary functions
# - Create a dictionary to hold session regressor values for each session and player
# - Populate this dictionary with regressor values
# - Save the dictionary 
# - Load the dictionary
# 

# %%
def generate_analysis_dict(num_experiments):
    ''' Generate a dictionary to hold all analysis results for each player and session type.
        This is used to store results from the GLM analysis. '''
    
    # Define player IDs and session types
    player_ids = [0,1]

    analysis_results = {
        experiment_id: {
            player_id: {
                session_type: {

                    'regressors': {
                        'wall_sep': None,
                        'first_seen': None,
                        'd2h': None,
                        'd2l': None,
                        'opponent_visible': None,
                        'd2h_opponent': None,
                        'd2l_opponent': None
                    },

                    'dependent': {
                        'choice': None
                    },

                    'misc': {
                        'valid_trial_indices': None,
                        'high_low_trial_indices': None
                    }
                    
                }
                for session_type in ['solo', 'social']
            }   
            for player_id in player_ids
        }
        for experiment_id in np.arange(num_experiments)
    }

    return analysis_results


def populate_analysis_dict(analysis_results, trial_lists_social, trial_lists_combined_solo, opponent_first_seen_wall):
    ''' Populate the analysis dictionary with data extracted from trial lists.
        This function processes each player's trial lists, extracts relevant data,
        and fills the analysis_results dictionary with regressors, dependent variables,
        and miscellaneous information for both social and solo sessions. '''

    # Loop through each experiment and player
    for experiment_id, players in analysis_results.items():
        for player_id, data in players.items():
            
            # get opponent_id
            opponent_id = 1 if player_id == 0 else 0
            
            # get the trial lists for this session and player
            trial_list_social = trial_lists_social[experiment_id]
            trial_list_solo = trial_lists_combined_solo[experiment_id*2 + player_id] # player_id used to select correct solo
            trial_lists = [trial_list_social, trial_list_solo]
            print(f"Trial list social length for experimentId {experiment_id} and playerId {player_id}: {len(trial_list_social)}")
            
            # filter the trial lists for HighLow trials
            original_indices_lists = []
            for i, trial_list in enumerate(trial_lists):
                original_indices = np.arange(len(trial_list))
                
                # identify indices of trial list with HighLow trials and filter
                high_low_trial_indices = get_indices.get_trials_trialtype(trial_list, trial_type=globals.HIGH_LOW)
                original_indices = original_indices[high_low_trial_indices]
                trial_list_filtered = [trial_list[i] for i in high_low_trial_indices]
                trial_lists[i] = trial_list_filtered
                original_indices_lists.append(original_indices)
                print(f"{high_low_trial_indices.size} high_low_trial_indices for player_id {player_id}, trail_list {i}\n out of {len(trial_list)} total trials")

            # re-assign trial lists
            trial_list_social = trial_lists[0]
            trial_list_solo = trial_lists[1]
            
            ## fill all social regressors
            ## social, use player_id == player_id and trial_list_social for functions
            # regressors social
            player_data = analysis_results[experiment_id][player_id]['social']
            distances = extract_distances_to_walls(trial_list_social, player_id, normalise=True)
            distances_opponent = extract_distances_to_walls(trial_list_social, player_id=opponent_id, normalise=True)
            player_data['regressors']['wall_sep'] = extract_wall_sep(trial_list_social, normalise=True)
            player_data['regressors']['first_seen'] = extract_first_wall_visibilities(trial_list_social, player_id, three_levels=False)
            player_data['regressors']['d2h'] = distances[:,0]
            player_data['regressors']['d2l'] = distances[:,1]
            player_data['regressors']['opponent_visible'] = extract_opponent_visibility_slice_onset(trial_list_social, player_id)
            player_data['regressors']['d2h_opponent'] = distances_opponent[:,0]
            player_data['regressors']['d2l_opponent'] = distances_opponent[:,1]
            if opponent_first_seen_wall:
                player_data['regressors']['first_seen_opponent'] = extract_first_wall_visibilities(trial_list_social, opponent_id)

            # dependent variable social
            player_data['dependent']['choice'] = extract_player_choice(trial_list_social, player_id, inferred_choice=True)

            # misc
            # player_data['misc']['valid_trial_indices'] = filtered_valid_trial_indices_social
            player_data['misc']['high_low_trial_indices'] = original_indices_lists[0] # social trial list indices


            ## fill all solo regressors
            ## solo, use player_id == 0 and trial_list_solo for functions
            # regressors solo
            player_data = analysis_results[experiment_id][player_id]['solo']
            distances = extract_distances_to_walls(trial_list_solo, player_id=0, normalise=True)
            player_data['regressors']['wall_sep'] = extract_wall_sep(trial_list_solo, normalise=True)
            player_data['regressors']['first_seen'] = extract_first_wall_visibilities(trial_list_solo, player_id=0, three_levels=False)
            player_data['regressors']['d2h'] = distances[:,0]
            player_data['regressors']['d2l'] = distances[:,1]

            # dependent variable solo
            player_data['dependent']['choice'] = extract_player_choice(trial_list_solo, player_id=0, inferred_choice=False) # no inferred for solo

            # misc
            player_data['misc']['high_low_trial_indices'] = original_indices_lists[1] # solo trial list indices


    return analysis_results


def save_analysis_dict(analysis_dict, analysis_file, analysis_dir='../data'):
    ''' Save the analysis dictionary to a file for later use. '''

    path = os.path.join(analysis_dir, analysis_file)
    with open(path, 'wb') as f:
        pickle.dump(analysis_dict, f)

        

def load_analysis_dict(analysis_file, analysis_dir='../data'):
    ''' Load the analysis dictionary from a file. '''
 
    filename = os.path.join(analysis_dir, analysis_file)
    with open(filename, 'rb') as f:
        analysis_results = pickle.load(f)

    return analysis_results

# %% [markdown]
# ### Dataframe generation from analysis dictionary (can I reduce repeated code here?)

# %%
# glm_df_social = pd.DataFrame()

# for session_id, players in analysis_results.items():
#     for player_id in players:
        
#         # take each filtered_regressor array and fill the relevant df field for this player
#         player_data = analysis_results[session_id][player_id]['social']['regressors']
#         choice = analysis_results[session_id][player_id]['social']['dependent']['choice']
#         opponent_player_id = 1 if player_id == 0 else 1
#         opponent_player_data = analysis_results[session_id][opponent_player_id]['social']['regressors']
#         df_player = pd.DataFrame(
#                     {
#                         "SessionID" : session_id,
#                         "PlayerID" : player_id,
#                         "GlmPlayerID" : session_id*2 + player_id,
#                         "ChooseHigh" : choice,
#                         "WallSep" : player_data['wall_sep'],
#                         "FirstSeenWall" : player_data['first_seen'],
#                         "D2H" : player_data['d2h'],
#                         "D2L" : player_data['d2l'],
#                         "OpponentVisible" : player_data['opponent_visible'],
#                         "OpponentFirstSeenWall" : player_data['first_seen_opponent'],
#                         "OpponentD2H" : player_data['d2h_opponent'],
#                         "OpponentD2L" : player_data['d2l_opponent']
                        
#                     }
#         )


#         # append this smaller dataframe to the the full dataframe
#         glm_df_social = pd.concat([glm_df_social, df_player], ignore_index=True)

# # convert to categorical variables, retaining np.nans
# glm_df_social["FirstSeenWall"] = glm_df_social["FirstSeenWall"].apply(lambda x: str(x) if pd.notna(x) else x)
# glm_df_social["OpponentFirstSeenWall"] = glm_df_social["OpponentFirstSeenWall"].apply(lambda x: str(x) if pd.notna(x) else x)
# glm_df_social["FirstSeenWall"] = glm_df_social["FirstSeenWall"].astype("category")
# glm_df_social["OpponentFirstSeenWall"] = glm_df_social["OpponentFirstSeenWall"].astype("category")

# # glm_df_social["WallSep"] = glm_df_social["WallSep"].astype("category") # now using continuous values for wall separation

# %%
# glm_df_solo_social = pd.DataFrame()

# for session_id, players in analysis_results.items():
#     for player_id in players:
        
#         # take each filtered_regressor array and fill the relevant df field for this player
#         player_data_solo = analysis_results[session_id][player_id]['solo']['regressors']
#         player_data_social = analysis_results[session_id][player_id]['social']['regressors']
#         choice_solo = analysis_results[session_id][player_id]['solo']['dependent']['choice']
#         choice_social = analysis_results[session_id][player_id]['social']['dependent']['choice']
#         df_player = pd.DataFrame(
#                     {
#                         "SessionID" : session_id,
#                         "PlayerID" : player_id,
#                         "GlmPlayerID" : session_id*2 + player_id,
#                         "ChooseHigh" : np.concatenate([choice_solo, choice_social]),
#                         "WallSep" :  np.concatenate([player_data_solo['wall_sep'], player_data_social['wall_sep']]),
#                         "FirstSeenWall" : np.concatenate([player_data_solo['first_seen'], player_data_social['first_seen']]),
#                         "D2H" : np.concatenate([player_data_solo['d2h'], player_data_social['d2h']]),
#                         "D2L" : np.concatenate([player_data_solo['d2l'], player_data_social['d2l']]),
#                         "SocialContext" : np.concatenate([np.ones(player_data_solo["wall_sep"].shape[0]) - 1, np.ones(player_data_social["wall_sep"].shape[0])]) # 0 for solo, 1 for social
#                     }
#         )

#         # append this smaller dataframe to the the full dataframe
#         glm_df_solo_social = pd.concat([glm_df_solo_social, df_player], ignore_index=True)

# # convert to categorical variables, retaining np.nans
# glm_df_solo_social["FirstSeenWall"] = glm_df_solo_social["FirstSeenWall"].apply(lambda x: str(x) if pd.notna(x) else x)
# glm_df_solo_social["FirstSeenWall"] = glm_df_solo_social["FirstSeenWall"].astype("category")
# # glm_df_solo_social["WallSep"] = glm_df_solo_social["WallSep"].astype("category")

# %%
# glm_df_solo = pd.DataFrame()

# for session_id, players in analysis_results.items():
#     for player_id in players:
        
#         # take each filtered_regressor array and fill the relevant df field for this player
#         player_data = analysis_results[session_id][player_id]['solo']['regressors']
#         choice = analysis_results[session_id][player_id]['solo']['dependent']['choice']
#         df_player = pd.DataFrame(
#                     {
#                         "SessionID" : session_id,
#                         "PlayerID" : player_id,
#                         "GlmPlayerID" : session_id*2 + player_id,
#                         "ChooseHigh" : choice,
#                         "WallSep" : player_data['wall_sep'],
#                         "FirstSeenWall" : player_data['first_seen'],
#                         "D2H" : player_data['d2h'],
#                         "D2L" : player_data['d2l']
#                     }
#         )

#         # append this smaller dataframe to the the full dataframe
#         glm_df_solo = pd.concat([glm_df_solo, df_player], ignore_index=True)

# # convert to categorical variables, retaining np.nans
# glm_df_solo["FirstSeenWall"] = glm_df_solo["FirstSeenWall"].apply(lambda x: str(x) if pd.notna(x) else x)
# glm_df_solo["FirstSeenWall"] = glm_df_solo["FirstSeenWall"].astype("category")

# # glm_df_solo["WallSep"] = glm_df_solo["WallSep"].astype(str).astype("category")


