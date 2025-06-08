import numpy as np
import analysis.loser_inferred_choice as loser_inferred_choice
import analysis.wall_visibility_and_choice as wall_visibility_and_choice
import data_extraction.get_indices as get_indices
import data_extraction.trial_list_filters as trial_list_filters
import trajectory_analysis.trajectory_direction as trajectory_direction
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import globals
import math as math


def reaction_time_session(trial_list, player_id, current_fov, solo=False):
    '''
    Checks whether the player chose the high wall or the low wall.
    Gets the time (timepoint in trial) when the chosen wall becomes first visible.
    Gets the time (timepoint in trial) when the player's trajectory first becomes aligned
    with the chosen wall - alignment is based on a threshold of cosine similarity = 0.95.
    Finds the corresponding application times and computes reaction time as difference between them.
    Returns a list of reaction times for every trial in the given trial list and the mean reaction time for the whole trial list.
    Uses inferred choice.
    '''

    if solo:
        trial_list = trial_list
        player_wall_choice = get_indices.get_chosen_walls(trial_list)
    else:
        # filter trial list by retrievable choice
        trial_list, _ = trial_list_filters.filter_trials_retrievable_choice(trial_list, player_id, inferred_choice=True, original_indices=None, debug=False)

        # get choices for each trial within the given trial list
        # player_choice = wall_visibility_and_choice.get_player_wall_choice(trial_list, player_id, inferred_choice=True, debug=False)
        # I think the below function returns the same output
        # verify differences
        player_wall_choice = loser_inferred_choice.player_wall_choice_win_or_loss(trial_list, player_id, debug=False)

    # initialise empty reaction times list
    reaction_times = []
    mean_reaction_times = []

    # trial logic
    for trial_index in range(len(trial_list)):
        # define trial
        trial = trial_list[trial_index]

        # part 1. get time the chosen wall becomes first visible
        # get high and low walls for trial
        walls = get_indices.get_walls(trial=trial)
        # compare player wall choices with high and low wall indices
        # accessing player wall choices across all trials in trial list
        chosen_wall_number = int(player_wall_choice[trial_index])
        print(chosen_wall_number)
        # this is to check that the chosen wall is either the high wall or the low wall
        # if not reaction time is set to one and loop ends
        if chosen_wall_number in walls:
            print("chosen wall number is either high or low")
            for idx in range(len(walls)):
                if walls[idx] == chosen_wall_number:
                    choice_index = idx
        else: 
            print(f"chosen wall number neither high nor low. walls: {walls}")
            reaction_times.append(np.nan)
            mean_reaction_times.append(np.nan)
            continue
                
        print(idx)
        # for each timepoint was wall x with x in range(8) visible
        wall_visible_array_trial = trajectory_headangle.get_wall_visible(trial=trial, player_id=player_id, current_fov=current_fov)
        # returns bool for whether high and low walls were visible at slice onset
        (wall1_initially_visible, wall2_initially_visible) = wall_visibility_and_choice.get_walls_initial_visibility_trial(player_id=player_id, current_fov=current_fov,
                                                                        trial=trial, wall_visible_array_trial=wall_visible_array_trial,
                                                                        debug=False)
        wall_initial_visibility = np.vstack([wall1_initially_visible, wall2_initially_visible])
        # get the time that high and low walls were first seen
        wall_becomes_visible_index, wall_becomes_visible_time = trajectory_headangle.get_wall_visibility_order(wall_visible_array_trial, wall_initial_visibility, trial, return_times=True, debug=True)
        # get the time that the chosen wall was first visible
        print(f"choice index: {choice_index}, wall becomes visible time: {wall_becomes_visible_time}")
        chosen_wall_becomes_visible_time = wall_becomes_visible_time[choice_index]
        print(chosen_wall_becomes_visible_time)
        
        # get_wall_visibility_order does not always compute valid time values and sometimes returns nan
        # when this happen close the loop and set rt to nan
        if not math.isnan(chosen_wall_becomes_visible_time):
            chosen_wall_becomes_visible_time = int(chosen_wall_becomes_visible_time)
        else:
            print(f"chosen_wall_becomes_visible_time is {chosen_wall_becomes_visible_time}")
            reaction_times.append(np.nan)
            mean_reaction_times.append(np.nan)
            continue

        # note: head angles are smoothed in such a way that the last 5 timepoints don't have smoothed head angles 

        # part 2. find the time that the player first becomes aligned with the chosen wall
        # extract cosine similarities for the chosen wall (the wall number) for the whole trial trajectory
        trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_id)
        cosine_similarities_trajectory = trajectory_direction.cosine_similarity_throughout_trajectory(trajectory, window_size=10, num_walls=8, calculate_thetas=False, debug=False)
        cosine_similarities_array = cosine_similarities_trajectory[chosen_wall_number-1][~np.isnan(cosine_similarities_trajectory[chosen_wall_number-1])]
        # this accounts for the fact that some trials are too short to extract valid cosine similarities from
        if cosine_similarities_array.size == 0:
            print(f"trajectory too short: {trajectory}")
            reaction_times.append(np.nan)
            mean_reaction_times.append(np.nan)
            continue
        else:
            # get first index in cosine similarities array for which cosine similarity first crosses alignment threshold
            indices = []
            for idx in range(len(cosine_similarities_array)):
                if cosine_similarities_array[idx] > 0.95:
                    indices.append(idx)
            # handles case in which no value crosses the threshold and just takes the index for the highest cosine similarity
            if len(indices) == 0:
                max_index = np.argmax(cosine_similarities_array)
                indices.append(max_index)
                       
            first_alignment_index = np.min(indices)
            first_alignment_index = int(first_alignment_index)
            print(first_alignment_index)

        # part 3. compute reaction time
        # cut trials to only return timepoints between slice onset and server-selected trigger activation
        slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
        slice_onset_index = slice_onset.index[0] - trial.index[0]
        selected_trigger_activation = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
        selected_trigger_activation_index = selected_trigger_activation.index[0] - trial.index[0]

        # extract application time between slice onset and server-selected trigger activation
        time_app_from_onset_to_trigger = trial['timeApplication'].iloc[slice_onset_index:selected_trigger_activation_index]

        # subtract application time for when player becomes first aligned to chosen wall from when chosen wall becomes first visible
        delta = time_app_from_onset_to_trigger.iloc[chosen_wall_becomes_visible_time] - time_app_from_onset_to_trigger.iloc[first_alignment_index]
        reaction_time = delta.total_seconds()
        print(reaction_time)

        # append to list for list of reaction times for every trial in trial list
        if reaction_time >= 0:
            reaction_times.append(reaction_time)
        else:
            reaction_times.append(reaction_time*(-1))

        # compute mean reaction time across the session
    clean_rts = [i for i in reaction_times if not np.isnan(i)]
    print(clean_rts)
    #mean_reaction_time = np.mean(clean_rts)
    #mean_reaction_times.append(mean_reaction_time)

    return reaction_times, clean_rts
#, mean_reaction_times