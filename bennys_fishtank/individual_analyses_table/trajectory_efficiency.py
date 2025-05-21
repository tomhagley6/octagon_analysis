import numpy as np
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import data_extraction.get_indices as get_indices

def player_direct_distance_trial(trial, player_index):
    ''' Return the direct distance in units between a slice onset location 
        and the final location for a single player.
        Takes a trial and player index. '''
    
    # get the player's trajectory
    trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_index, debug=False)

    # find distance between first and last trajectory points
    end_position = trajectory[:,-1]
    start_position = trajectory[:,0]
    distance = np.linalg.norm(end_position - start_position)

    return distance

def player_actual_distance_trial(trial, player_index, debug=False):
    ''' Return the actual distance in units throughout the player's entire
        trajectory.
        Takesa trial and player index. '''
    
    # get the player's trajectory
    trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_index, debug=False)

    # compute difference between consecutive columns (timepoints)
    position_diffs = np.diff(trajectory, axis=1) # shape 2, timepoints-1
    if debug:
        print(f"shape of np.diff output: {position_diffs.shape}")

    # square these differences, and sum across x and y for each column (timepoint)
    squared_distances = np.sum(position_diffs**2, axis=0) # shape timepoints-1,

    # take the square root to get the euclidean difference between each consecutive point
    distances = np.sqrt(squared_distances)

    if debug:
        print(f" shape of summed euclidean distance: {distances.shape}")
        print(f"summed euclidean distance[:5]: {distances[:5]}")

    return np.sum(distances)

def player_direct_distance_session(trial_list, player_index):
    ''' Return an array of len(trial_list) of the direct distance between a player's
        start and end location on each trial.
        Takes a list of trials. '''
    

    direct_distances = np.zeros(len(trial_list))
    for i, trial in enumerate(trial_list):
        trial = trial_list[i]
        distance = player_direct_distance_trial(trial, player_index)

        direct_distances[i] = distance

    return direct_distances


def player_actual_distance_session(trial_list, player_index):
    ''' Return an array of len(trial_list) of the direct distance between a player's
        start and end location on each trial.
        Takes a list of trials. '''
    

    actual_distances = np.zeros(len(trial_list))
    for i, trial in enumerate(trial_list):
        trial = trial_list[i]
        distance = player_actual_distance_trial(trial, player_index)

        actual_distances[i] = distance

    return actual_distances

def direct_distance_winner_loser_session(trial_list):
    ''' Return 4 arrays, each of len(trial_list), which are the direct distances in
        winning trials and losing trials respectively for player 0 and player 1 respectively.
        Indices where the trial outcome does match the array name are np.nan. '''
    
    # fill all arrays with the direct distances for the player
    player_0_win_direct_distances = player_direct_distance_session(trial_list, player_index=0)
    player_0_loss_direct_distances = player_0_win_direct_distances
    
    player_1_win_direct_distances = player_direct_distance_session(trial_list, player_index=1)
    player_1_loss_direct_distances = player_1_win_direct_distances

    # find the winners in this session
    winners_session = get_indices.get_trigger_activators(trial_list)
    losers_session = (winners_session-1)*-1

    # replace array elements with np.nan at indices where the trial outcome does not match the array name 
    player_0_win_direct_distances = np.where(winners_session == 0, player_0_win_direct_distances, np.nan)
    player_0_loss_direct_distances = np.where(losers_session == 0, player_0_loss_direct_distances, np.nan)

    player_1_win_direct_distances = np.where(winners_session == 1, player_1_win_direct_distances, np.nan)
    player_1_loss_direct_distances = np.where(losers_session == 1, player_1_loss_direct_distances, np.nan)

    return (player_0_win_direct_distances, player_0_loss_direct_distances, player_1_win_direct_distances,
            player_1_loss_direct_distances)



def actual_distance_winner_loser_session(trial_list):
    ''' Return 4 arrays, each of len(trial_list), which are the direct distances in
        winning trials and losing trials respectively for player 0 and player 1 respectively.
        Indices where the trial outcome does match the array name are np.nan. '''
    
    player_0_win_actual_distances = player_actual_distance_session(trial_list, player_index=0)
    player_0_loss_actual_distances = player_0_win_actual_distances
    
    player_1_win_actual_distances = player_actual_distance_session(trial_list, player_index=1)
    player_1_loss_actual_distances = player_1_win_actual_distances

        # find the winners in this session
    winners_session = get_indices.get_trigger_activators(trial_list)
    losers_session = (winners_session-1)*-1

    # replace array elements with np.nan at indices where the trial outcome does not match the array name 
    player_0_win_actual_distances = np.where(winners_session == 0, player_0_win_actual_distances, np.nan)
    player_0_loss_actual_distances = np.where(losers_session == 0, player_0_loss_actual_distances, np.nan)

    player_1_win_actual_distances = np.where(winners_session == 1, player_1_win_actual_distances, np.nan)
    player_1_loss_actual_distances = np.where(losers_session == 1, player_1_loss_actual_distances, np.nan)

    return (player_0_win_actual_distances, player_0_loss_actual_distances, player_1_win_actual_distances,
            player_1_loss_actual_distances)


def ratio_direct_to_absolute_distances_session(trial_list):
    ''' Return 4 values, the ratio of direct distances to absolute distances for 
        winner and then loser for first player 0 and then player 1 '''
    

    # find the actual distances for both players, win and loss
    (player_0_win_actual_distances, player_0_loss_actual_distances, player_1_win_actual_distances,
                player_1_loss_actual_distances) = actual_distance_winner_loser_session(trial_list)

    # find the direct distances for both players, win and loss
    (player_0_win_direct_distances, player_0_loss_direct_distances, player_1_win_direct_distances,
                player_1_loss_direct_distances) = direct_distance_winner_loser_session(trial_list)
    
    # calculate ratios
    ratio_player_0_win =  np.nanmean(player_0_win_direct_distances/player_0_win_actual_distances)
    ratio_player_0_loss =  np.nanmean(player_0_loss_direct_distances/player_0_loss_actual_distances)
    ratio_player_1_win = np.nanmean(player_1_win_direct_distances/player_1_win_actual_distances)
    ratio_player_1_loss =  np.nanmean(player_1_loss_direct_distances/player_1_loss_actual_distances)

    return (ratio_player_0_win, ratio_player_0_loss, ratio_player_1_win, ratio_player_1_loss)

def ratio_direct_to_absolute_distances_multiple_sessions(trial_lists):
    ''' Takes a list of trial lists (from multiple sessions) and returns
        a (4,num_sessions) array of trajectory effiency ratios.
        Columns are: player_0_win, player_0_loss, player_1_win, player_1_loss. '''
    
    trajectory_efficiency_ratios = np.zeros((len(trial_lists),4))

    for i, trial_list in enumerate(trial_lists):
        
        (ratio_player_0_win, ratio_player_0_loss,
        ratio_player_1_win, ratio_player_1_loss) = ratio_direct_to_absolute_distances_session(trial_list)

        trajectory_efficiency_ratios[i,:] = (ratio_player_0_win, ratio_player_0_loss,
                                            ratio_player_1_win, ratio_player_1_loss)
        
    return trajectory_efficiency_ratios

def ratio_direct_to_absolute_distances_solo_sessions(trial_lists):
    ''' Return 4 arrays, each of len(trial_list), which are the direct distances in
        all trials in the solo session fed to the function for the player.
    '''

    trajectory_efficiency_ratios = np.zeros((len(trial_lists),1))

    for i, trial_list in enumerate(trial_lists):

        player_direct_distances = player_direct_distance_session(trial_list, player_index=0)
        #print(player_direct_distances)
        player_actual_distances = player_actual_distance_session(trial_list, player_index=0)
        #print(player_actual_distances)

        ratio = np.nanmean(player_direct_distances/player_actual_distances)
        print(ratio)
        
        trajectory_efficiency_ratios[i,:] = ratio

    return trajectory_efficiency_ratios
