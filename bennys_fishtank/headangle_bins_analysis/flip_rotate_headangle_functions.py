import json
import numpy as np
import pandas as pd
import parse_data.prepare_data as prepare_data
import parse_data.flip_rotate_trajectories as flip_rotate_trajectories
import data_extraction.get_indices as get_indices
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import globals
import scipy
import math




def flip_rotate_trial_headangles(trial_list, trial_index, player_id, theta, flip=True):
    '''Rotate yaw by theta
       Flip yaw if wall 1 CCW of wall 0
       Return altered yaw, single chosen player, whole trial'''

    num_walls = globals.NUM_WALLS

    altered_yaw_values = []
    trial = trial_list[trial_index]
    num_players = preprocess.num_players(trial)

    for player_id in range(num_players):
        
        y_rotation = trial[globals.PLAYER_ROT_DICT[player_id]['yrot']]
        head_angles = np.deg2rad(y_rotation)
        #print(f"head angle for trial {trial_index}: {head_angles}")
        
        player_altered_yaw = []
        for yaw in head_angles:
                new_yaw = (yaw - theta) % (2 * math.pi) #keeps new yaw within valid 0-2pi range
                #note: changed from yaw + theta to above, appears to work but run checks

                    #flip if needed
                if flip:
                    walls = get_indices.get_walls(trial=trial, trial_list=None, trial_index=None, num_walls=2)

                    if walls[1] < walls[0]:
                        counterclockwise_distance = walls[0] - walls[1]
                    else:
                        counterclockwise_distance = (num_walls - walls[1]) + walls[0] 
                    clockwise_distance = num_walls - counterclockwise_distance

                    if counterclockwise_distance < clockwise_distance:
                        new_yaw = flip_headangles(new_yaw)

                player_altered_yaw.append(new_yaw)
            
        altered_yaw_values.append(player_altered_yaw)
        
    return altered_yaw_values 






def flip_headangles(altered_yaw):
    ''' If wall 0 is CW of wall 1, flip the yaw around. This keeps wall 0
        CCW of wall 1 '''
    
    altered_yaw = -altered_yaw

    return altered_yaw







def replace_with_altered_yaws(trial_list, trial_index, altered_yaw_values, player_id):
    '''Replaces yaw values from slice onset to server selected trigger
    with altered yaw values from flip-rotate function
    Input: altered_yaw_values needs to be np.array of above altered_yaw_values[player_id]'''
    
    if trial_list is not None and trial_index is not None:
        trial = trial_list[trial_index]
    else:
        trial = trial_index 

    #trial = trial_list[trial_index]
    trial_copy = trial.copy()

    player_yaw_values = altered_yaw_values#[player_id]
    
    if len(player_yaw_values) != len(trial_copy):
        raise ValueError(f"Length of altered yaw values ({len(player_yaw_values)})does not match the number of rows in the DataFrame ({len(trial_copy)})")

    trial_copy[globals.PLAYER_ROT_DICT[player_id]['yrot']] = player_yaw_values

    return trial_copy






#umbrella function
def process_and_update_trials(trial_list, player_id):
    '''Changes yaw values and coordinates for each player in each trial in trial list 
    Returns new trial list with updated trials'''
    updated_trial_list = []
    
    for i in range(len(trial_list)):
    
        #step 1: calculate rotation angle
        theta = flip_rotate_trajectories.find_rotation_angle_trial(trial_list=trial_list, trial_index=i)
    
        #step 2: change yaw values 
        altered_yaw_values = flip_rotate_trial_headangles(trial_list=trial_list, trial_index=i, player_id=player_id, theta=theta)
        player_altered_yaw = np.array(altered_yaw_values[player_id])

        #step 3: create trial copy with new yaw values
        trial_copy = replace_with_altered_yaws(trial_list=trial_list, trial_index=i, altered_yaw_values=player_altered_yaw, player_id=player_id)
    
        #step 4: change coordinates
        altered_coords = flip_rotate_trajectories.flip_rotate_trial(trial_list=trial_list, trial_index=i, theta=theta, flip=True)
        altered_coords = np.array(altered_coords)
    
        #step 5: create trial copy with new coordinates
        trial_copy_coords = flip_rotate_trajectories.replace_with_altered_coordinates(trial_list=trial_list, trial_index=i, altered_coordinates=altered_coords)

        #step 6: combine all new values in a single trial copy
        trial_example = trial_copy
        for j in range(len(altered_coords)):
            trial_example[globals.PLAYER_LOC_DICT[j]['xloc']] = altered_coords[j][0] # x coordinates
            trial_example[globals.PLAYER_LOC_DICT[j]['yloc']] = altered_coords[j][1] # y coordinates

        #add the updated trial to the new trial list
        updated_trial_list.append(trial_example)

    return updated_trial_list



