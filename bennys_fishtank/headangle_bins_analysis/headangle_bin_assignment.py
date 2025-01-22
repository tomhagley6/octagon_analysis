import scipy
import math
import globals
import numpy as np
import plotting.plot_octagon as plot_octagon
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import trajectory_analysis.trajectory_vectors as trajectory_vectors

def define_45_degree_bins():
    
    # Get the central points of the alcoves
    alcove_centers = plot_octagon.return_alcove_centre_points()

    # Convert coordinates to angles
    alcove_angles = np.arctan2(alcove_centers[1], alcove_centers[0])

    # Normalize angles to [0, 2π]
    alcove_angles = np.mod(alcove_angles, 2 * np.pi)

    # Sort the angles to ensure bins are in order
    alcove_angles = np.sort(alcove_angles)

    # Define the bin ranges (±22.5 degrees around each center)
    bin_ranges = [
        ((center - np.pi / 8) % (2 * np.pi), (center + np.pi / 8) % (2 * np.pi))
        for center in alcove_angles
    ]

    # Find the bin that contains 0 radians
    for idx, (start, end) in enumerate(bin_ranges):
        if start < 0 < end or (start > end and (0 >= start or 0 <= end)):
            shift_index = idx
            break

    # Circularly shift bins so the first bin contains 0 radians
    bin_ranges = bin_ranges[shift_index:] + bin_ranges[:shift_index]

    return bin_ranges






def sort_head_angle_into_bin(trial_trajectory, head_angle_vector_array, num_walls=8, debug=False):
    """
    Extract the head angle at slice onset and the angle to the closest wall for a given trial.

    Args:
        trial_trajectory: Player's trajectory for the trial (2D coordinates).
        head_angle_vector_array: Head angle vector array for the trial.
        num_walls: Number of walls (default 8).
        debug: Whether to print debug information.

    Returns:
        result: Array containing the head angle at slice onset and the angle to the closest wall.
    """

    # Step 1: Get head angles to walls for the whole trial
    thetas = trajectory_headangle.head_angle_to_walls_throughout_trajectory(
        trajectory=trial_trajectory,
        head_angle_vector_array_trial=head_angle_vector_array,
        num_walls=num_walls,
        debug=debug
    )

    # Step 2: Extract head angles at slice onset
    head_angle_at_slice_onset = thetas[:, 0] 

    # Step 3: Find the angle to the nearest wall
    min_angle_index = np.argmin(head_angle_at_slice_onset)  # Index of the closest wall
    nearest_wall_angle = head_angle_at_slice_onset[min_angle_index]  # Angle to the nearest wall

    if debug:
        print(f"Head angle at slice onset: {head_angle_at_slice_onset}")
        print(f"Nearest wall angle: {np.degrees(nearest_wall_angle):.2f}°")
        print(f"Min angle index: {min_angle_index+1}")

    # Step 4: Return the required data as an array
    bin_index = [int(min_angle_index+1)]
    return bin_index






def assign_bins_to_all_trials(trial_list, player_id, num_walls=8, debug=False):
    """
    Assign bins to the head angle at slice onset for all trials in trial list

    Args:
       trial_list: list of trial dataframes
       num_walls: number of walls (default 8)
       debug: whether to print debug information

    Returns:
       list of bin indices corresponding to each trial

    Inputs: 
       requires head angles in degrees
    """

    bin_assignments = [] #list to store bin indices for all trials

    for trial_index, trial in enumerate(trial_list):
        if debug:
            print(f"Processing trial {trial_index + 1}/{len(trial_list)}")

        #extract trial trajectory and head angle vector array
        
        trajectory = trajectory_vectors.extract_trial_player_trajectory(
            trial_list=trial_list, trial_index=trial_index, trial=trial, player_id=player_id
        )
        print(f"Trajectory: {trajectory}")
        
        headangle_array = trajectory_vectors.extract_trial_player_headangles(
            trial_list=trial_list, trial_index=trial_index, trial=trial, player_id=player_id
        )
        print(f"Head Angles: {headangle_array}")


        #assign bin for the trial
        
        bin_index = sort_head_angle_into_bin(
            trial_trajectory=trajectory, head_angle_vector_array=headangle_array, num_walls=num_walls, 
            debug=debug
        )
        
        bin_assignments.append(bin_index[0])


    return bin_assignments


        
