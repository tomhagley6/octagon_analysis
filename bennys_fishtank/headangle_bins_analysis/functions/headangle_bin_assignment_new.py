import scipy
import math
import globals
import plotting.plot_octagon as plot_octagon
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import numpy as np
import parse_data.preprocess as preprocess


def define_45_degree_bins():
    
    #get the central points of the alcoves
    alcove_centers = plot_octagon.return_alcove_centre_points()

    #convert coordinates to angles
    alcove_angles = np.arctan2(alcove_centers[1], alcove_centers[0])

    #normalize angles to [0, 2π]
    alcove_angles = np.mod(alcove_angles, 2 * np.pi)

    #sort the angles to ensure bins are in order
    alcove_angles = np.sort(alcove_angles)

    #define the bin ranges (±22.5 degrees around each center)
    bin_ranges = [
        ((center - np.pi / 8) % (2 * np.pi), (center + np.pi / 8) % (2 * np.pi))
        for center in alcove_angles
    ]

    #find the bin that contains 0 radians
    for idx, (start, end) in enumerate(bin_ranges):
        if start < 0 < end or (start > end and (0 >= start or 0 <= end)):
            shift_index = idx
            break

    #circularly shift bins so the first bin contains 0 radians
    bin_ranges = bin_ranges[shift_index:] + bin_ranges[:shift_index]

    return bin_ranges






def sort_head_angle_into_bin(trial_list, trial_index, debug=False):
    '''not returning array for two players'''
    num_walls = globals.NUM_WALLS
    
    trial = trial_list[trial_index]
    num_players = preprocess.num_players(trial)
    
    bins = []


    #step 2: looping over players define trial_trajectory and head_angle_vector_array
    for player_id in range(num_players):
        
        trial_trajectory = trajectory_vectors.extract_trial_player_trajectory(
            trial_list=trial_list, trial_index=trial_index, trial=trial, 
            player_id=player_id
        )
        head_angle_vector_array = trajectory_vectors.extract_trial_player_headangles(
            trial_list=trial_list, trial_index=trial_index, trial=trial, 
            player_id=player_id
        )

        #step 2.1: get head angles to walls for the whole trial
        thetas = trajectory_headangle.head_angle_to_walls_throughout_trajectory(
            trajectory=trial_trajectory,
            head_angle_vector_array_trial=head_angle_vector_array,
            num_walls=num_walls,
            debug=debug
        )
        
        if thetas.size == 0 or thetas.shape[1] == 0:
            print("Thetas empty, skipping trial")
            return None


        #step 2.2: extract head angles at slice onset 
        head_angle_at_slice_onset = thetas[:, 0]
        
        #step 2.3: find the angle to the nearest wall
        min_angle_index = np.argmin(head_angle_at_slice_onset)  #index of the closest wall
        nearest_wall_angle = head_angle_at_slice_onset[min_angle_index]  #angle to the nearest wall

        #step 2.4: return the required data as an array
        bin_index = int(min_angle_index+1)
        bins.append(bin_index)
        
        if debug:
            print(f"Player {player_id}: Bin {bin_index}")
            
    return bins







def assign_bins_to_all_trials(trial_list, debug=False):

    num_walls = globals.NUM_WALLS

    bin_assignments_player0 = []
    bin_assignments_player1 = []


    for trial_index in range(len(trial_list)):
        if debug:
            print(f"Processing trial {trial_index + 1}/{len(trial_list)}")

        #assign bin for the trial
        
        bins = sort_head_angle_into_bin(trial_list, trial_index, debug=False)

        if bins is None or len(bins) == 0:
            print(f"Skipping bin assignment for trial {trial_index}, empty bin")
            continue
        
        bin_assignments_player0.append(bins[0])

        if len(bins) > 1:
            bin_assignments_player1.append(bins[1])

    if not bin_assignments_player1:
        return bin_assignments_player0


    return bin_assignments_player0, bin_assignments_player1

