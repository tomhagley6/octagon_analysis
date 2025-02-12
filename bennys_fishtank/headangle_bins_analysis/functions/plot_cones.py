# Plot cones function

import globals
from plotting import plot_octagon
from parse_data import preprocess
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

def extract_trial(trial, trial_list, trial_index):
    ''' isolate trial '''
    
    if not trial is None:
        this_trial = trial
    elif not trial_list is None:
        this_trial = trial_list[trial_index]
    else:
        raise ValueError("a list of trials and the chosen index must be given, or the trial itself must be given, but not neither.")

    return this_trial

def plot_cones(ax, trial_list=None, trial_index=0, trial=None):
  this_trial = extract_trial(trial, trial_list, trial_index)
  
  slice_onset_event = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]
  trigger_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]

  slice_onset_idx = slice_onset_event.index[0]
  slice_onset_idx = int(slice_onset_idx - this_trial.index[0])

  num_players = preprocess.num_players(this_trial)

  diameter = 36.21
  radius = diameter/2

  coordinate_array_labels = []
  for i in range(num_players):
    coordinate_array_labels.extend((globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'], globals.PLAYER_ROT_DICT[i]['yrot'])) 
  coordinate_arrays = {label: this_trial[label].values[slice_onset_idx:] for label in coordinate_array_labels}

  for i in range(num_players):
        x = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['xloc']][0]  # Player's x position
        y = coordinate_arrays[globals.PLAYER_LOC_DICT[i]['yloc']][0]  # Player's y position
        rot = coordinate_arrays[globals.PLAYER_ROT_DICT[i]['yrot']][0]  # Player's yaw angle

        # Convert yaw to radians
        rot_rad = np.deg2rad(rot)

        # Calculate cone endpoints
        cone_angle_left = rot_rad - np.pi / 4  # 45 degrees to the left
        cone_angle_right = rot_rad + np.pi / 4  # 45 degrees to the right

        # Endpoints of the cone
        cone_base_left = (x + diameter * np.cos(cone_angle_left), y + diameter * np.sin(cone_angle_left))
        cone_base_right = (x + diameter * np.cos(cone_angle_right), y + diameter * np.sin(cone_angle_right))

        # Plot the cone as a triangle
        ax.fill([x, cone_base_left[0], cone_base_right[0]], 
                [y, cone_base_left[1], cone_base_right[1]], 
                alpha=0.3)  # Adjust alpha for transparency

  return ax

# Plot cones

import parse_data.prepare_data as prepare_data
import globals
from plotting import plot_octagon_updated
from assign_coordinates_to_walls import alcove_coordinates
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import data_extraction.get_indices as get_indices


data_folder = '/Users/benny/Desktop/MSc/Project/Git/repos/octagon_analysis/Json data'
json_filenames = ['2024-09-13_11-31-00_YansuJerrySocial.json']

df, trial_list = prepare_data.prepare_data(data_folder, json_filenames)

for i in range(len(trial_list)):
  fig, ax = plt.subplots()
  ax, alcove_x, alcove_y = plot_octagon_updated.plot_octagon(ax)
  this_trial = extract_trial(trial=None, trial_list=trial_list, trial_index=i)
  slice_onset_event = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]
  if not slice_onset_event.empty:
     slice_onset_idx = slice_onset_event.index[0]

  active_walls = df.loc[slice_onset_idx, ['data.wall1', 'data.wall2']].values
  print("Active Walls (as integers):", active_walls)

  for wall in active_walls:
        wall = int(wall)
        if wall in alcove_coordinates:
            x_coords, y_coords = alcove_coordinates[wall]
            print(f"Filling wall {wall} with coordinates: {x_coords}, {y_coords}")
            ax.fill(x_coords, y_coords, alpha=0.5, color='red', linewidth=5)
        else:
            print(f"Wall {wall} not found in alcove_coordinates.")
    
  ax = plot_cones(ax, trial_list=trial_list, trial_index = i)
  ax.set_title("Player point of view and active walls at slice onset")

plt.show()
