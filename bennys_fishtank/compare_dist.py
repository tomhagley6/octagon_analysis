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
import assign_coordinates_to_walls

data_folder = '/Users/benny/Desktop/MSc/Project/Git/repos/octagon_analysis/Json data'
json_filenames = ['2024-09-13_11-31-00_YansuJerrySocial.json']

df, trial_list = prepare_data.prepare_data(data_folder, json_filenames)

# Get trial from trial list
def extract_trial(trial, trial_list, trial_index):
    ''' isolate trial '''
    
    if not trial is None:
        this_trial = trial
    elif not trial_list is None:
        this_trial = trial_list[trial_index]
    else:
        raise ValueError("a list of trials and the chosen index must be given, or the trial itself must be given, but not neither.")

    return this_trial


# Key functions
def direct_distance(this_trial, alcove_center, trial_index):
  
    this_trial = extract_trial(trial, trial_list, trial_index)

    # isolate slice onset event, trigger event, and activating client
    slice_onset_event = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]
    trigger_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    trigger_activating_client = trigger_event[globals.TRIGGER_CLIENT].values[0]
    
    # find index of slice onset and trigger event normalised to this trial
    slice_onset_idx = slice_onset_event.index[0]
    slice_onset_idx = int(slice_onset_idx - this_trial.index[0])
    trigger_idx = trigger_event.index[0]
    trigger_idx = int(trigger_idx - this_trial.index[0])

    num_players = preprocess.num_players(this_trial)
  
    coordinate_array_labels = []
    for i in range(num_players):
        coordinate_array_labels.extend((globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])) 
    coordinate_arrays = {label : this_trial[label].values[slice_onset_idx:] for label in coordinate_array_labels}
    
    starting_positions = []
    for i in range(num_players):
        x = coordinates[globals.PLAYER_LOC_DICT[i]['xloc']][0]  # Player's x position
        y = coordinates[globals.PLAYER_LOC_DICT[i]['yloc']][0]  # Player's y position
        starting_position = np.array([x, y])
        starting_positions.append(starting_position)

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
        alcove_center[i + 1] = (center_x, center_y)

  
    distances = [np.linalg.norm(np.array(alcove_center)-starting_position) for starting_position in starting_positions]
    return distances

def path_distance(trajectory):
  return np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))




