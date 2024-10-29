# Add imports
import json
import numpy as np
import pandas as pd
import parse_data.prepare_data as prepare_data
import globals
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import data_extraction.get_indices as get_indices
from assign_coordinates_to_walls import alcove_x, alcove_y
from parse_data import preprocess


# Prepare data - load json data and convert it into df and list of trials
data_folder = '/Users/benny/Desktop/MSc/Project/Git/repos/octagon_analysis/Json data'
json_filenames = ['2024-09-13_11-31-00_YansuJerrySocial.json']

df, trial_list = prepare_data.prepare_data(data_folder, json_filenames)


# Calculate distances between starting positions and alcove center
def direct_distance(trial_index):
  
    this_trial = trial_list[trial_index]

    # isolate slice onset event, trigger event, and activating client
    slice_onset_event = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]
    trigger_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    
    # find index of slice onset and trigger event normalised to this trial
    slice_onset_idx = slice_onset_event.index[0]
    slice_onset_idx = int(slice_onset_idx - this_trial.index[0])
    trigger_idx = trigger_event.index[0]
    trigger_idx = int(trigger_idx - this_trial.index[0])

    wall_triggered = trigger_event[globals.WALL_TRIGGERED].item()

    
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
   
    alcove_center = alcove_centers[wall_triggered]
    
    num_players = preprocess.num_players(this_trial)
  
    coordinate_array_labels = []
    for i in range(num_players):
        coordinate_array_labels.extend((globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])) 
    coordinate_arrays = {label : this_trial[label].values[slice_onset_idx:] for label in coordinate_array_labels}
    
    starting_positions = []
    for i in range(num_players):
        x = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['xloc']][0]  # Player's x position
        y = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['yloc']][0]  # Player's y position
        starting_position = np.array([x, y])
        starting_positions.append(starting_position)
    
    
    distances = [np.linalg.norm(np.array(alcove_center) - starting_position) for starting_position in starting_positions]

    return distances

for trial_index in range(len(trial_list)):  # Change this to the appropriate index of your trial
    try:
        distances = direct_distance(trial_index)
        print("Distances from starting positions to the alcove center:", distances)
    except Exception as e:
        print(f"Error: {e}")

# INCOMPLETE --------
def path_distance(trajectory):
  return np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))




