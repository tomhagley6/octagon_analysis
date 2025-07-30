# %%
import numpy as np
from analysis import opponent_visibility, wall_visibility_and_choice
from data_extraction import get_indices
from plotting import plot_octagon
from trajectory_analysis import trajectory_vectors

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


