import json
import numpy as np
import pandas as pd
import parse_data.prepare_data as prepare_data
import globals
import scipy
import math
import data_extraction.get_indices as get_indices
from assign_coordinates_to_walls import alcove_x, alcove_y
from parse_data import preprocess


# Prepare data - load json data and convert it into df and list of trials
data_folder = '/Users/benny/Desktop/MSc/Project/Git/repos/octagon_analysis/Json data'
json_filenames = ['2024-09-13_11-31-00_YansuJerrySocial.json']

df, trial_list = prepare_data.prepare_data(data_folder, json_filenames)

# DIRECT DISTANCE

def direct_distance(trial_index, chosen_player):
  
    this_trial = trial_list[trial_index]

    # isolate slice onset event, trigger event, and activating client
    slice_onset_event = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]
    trigger_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    
    # find index of slice onset and trigger event normalised to this trial
    slice_onset_idx = slice_onset_event.index[0]
    slice_onset_idx = int(slice_onset_idx - this_trial.index[0])
    trigger_idx = trigger_event.index[0]
    trigger_idx = int(trigger_idx - this_trial.index[0])

    # identify wall touched by winning player at trigger event 
    wall_triggered = trigger_event[globals.WALL_TRIGGERED].item()

    # define wall centers by finding midpoint between left and right edge of alcove 
    alcove_centers = {}
    num_walls = 8
    for i in range(num_walls):
        # Get the x and y coordinates for the current wall
        x1, x2 = alcove_x[i * 2], alcove_x[i * 2 + 1]
        y1, y2 = alcove_y[i * 2], alcove_y[i * 2 + 1]
        
        # Calculate the center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Store the center coordinates
        alcove_centers[i + 1] = (center_x, center_y)
   
    # Find wall center for the triggered wall
    alcove_center = alcove_centers[wall_triggered]
    
    num_players = preprocess.num_players(this_trial)
  
    # assign labels to values from this_trial from slice onset onwards
    coordinate_array_labels = []
    for i in range(num_players):
        coordinate_array_labels.extend((globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])) 
    coordinate_arrays = {label : this_trial[label].values[slice_onset_idx:] for label in coordinate_array_labels}

    # extract x and y values at the start looping over players
    d_distances = []
    for i in range(num_players):
        if chosen_player is None:
           x = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['xloc']][0]  # Player's x position
           y = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['yloc']][0]  # Player's y position
           starting_position = np.array([x, y])
    
           # calculate distances using alcove_center and starting position arrats
           d_distance = [np.linalg.norm(np.array(alcove_center) - starting_position)]
           d_distances.append(d_distance)
            
           print(f"Trial {trial_index}, Player {i}: x = {x}, y = {y}, d_distance = {d_distance}")
        
        else:
           if i != chosen_player:
              pass
           else:
              x = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['xloc']][0]  # Player's x position
              y = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['yloc']][0]  # Player's y position
              starting_position = np.array([x, y])
    
              # calculate distances using alcove_center and starting position arrats
              d_distance = [np.linalg.norm(np.array(alcove_center) - starting_position)]
              d_distances.append(d_distance)
               
              print(f"Trial {trial_index}, Player {i}: x = {x}, y = {y}, d_distance = {d_distance}")

    return d_distances
# above as stand-alone code not functional, needs below to execute but I'm not actually sure why, because it needs to loop over trials I guess?

# TEST----------
# loop over trials within trial_list and return distances for chosen player using direct_distance function, prints Error if not 
for trial_index in range(len(trial_list)): 
    try:
        d_distance = direct_distance(trial_index, chosen_player = 0)
        print("Direct distances from starting positions to the alcove center:", d_distance)
    except Exception as e:
        print(f"Error: {e}")


# PATH DISTANCE

# main function summing over euclidian distances between points on trajectory
def path_distance(trial_index, chosen_player = None):
  
    this_trial = trial_list[trial_index]

    # isolate slice onset event, trigger event, and activating client
    slice_onset_event = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]
    trigger_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    
    # find index of slice onset and trigger event normalised to this trial
    slice_onset_idx = slice_onset_event.index[0]
    slice_onset_idx = int(slice_onset_idx - this_trial.index[0])
    trigger_idx = trigger_event.index[0]
    trigger_idx = int(trigger_idx - this_trial.index[0])

    num_players = preprocess.num_players(this_trial)

    # define trajectory as the set of points on the path
    # create list containing labels for player coordinates from slice onset to trigger event
    coordinate_array_labels = []
    for i in range(num_players):
        coordinate_array_labels.extend((globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])) 
    coordinate_arrays = {label : this_trial[label].values[slice_onset_idx:trigger_idx] for label in coordinate_array_labels}
    # maybe update above coordinate_array_labels?


    a_distances = []
    for i in range(num_players):
        if chosen_player is None:
           x = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['xloc']]  # Player's x position
           y = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['yloc']]  # Player's y position
           trajectory = np.array([x,y])
        
           a_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=1), axis=0))
           a_distances.append(a_distance)
        
           print(f"Trial {trial_index}, Player {i}: x = {x}, y = {y}, a_distance = {a_distance}")
            
        else:
           if i != chosen_player:
              pass
           else:
              x = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['xloc']]  # Player's x position
              y = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['yloc']]  # Player's y position
              trajectory = np.array([x,y])
        
              a_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=1), axis=0))
              a_distances.append(a_distance)
        
              print(f"Trial {trial_index}, Player {i}: x = {x}, y = {y}, a_distance = {a_distance}")
               
    return a_distances
    
# TEST-------------

    
for trial_index in range(len(trial_list)): 
    try:
       a_distance = path_distance(trial_index, chosen_player = 0)
       print("Actual actual distance travelled from slice onset to trigger event:", a_distance)
    except Exception as e:
       print(f"Error: {e}")


# CALCULATE RATIO

def ratio_f(trial_index, chosen_player = 0):
   actual_distances = path_distance(trial_index, chosen_player)
   direct_distances = direct_distance(trial_index, chosen_player)

   a_distance = actual_distances[chosen_player]
   d_distance = direct_distances[chosen_player]
    
   if d_distance == 0:
     return a_distance
       
   return a_distance/d_distance

for trial_index in range(len(trial_list)): 
    try:
       ratio = ratio_f(trial_index)
       print("Ratio between actual distance and direct distance:", ratio)
    except Exception as e:
       print(f"Error: {e}")
        


