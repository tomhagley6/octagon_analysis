import plotting.plot_trajectory as plot_trajectory
import matplotlib as mpl
import numpy as np
import globals
import plotting.plot_octagon as plot_octagon
from matplotlib.patches import Circle
#import pandas as pd

def percent_trajectory_in_circle(trial_list=None, trial=None, trial_index=None, radius=0, num_players=2):
    
    coordinate_arrays, _ = plot_trajectory.plot_interim_trajectory_colour_map(ax=None, trial_list=trial_list, trial_index=trial_index, cmap_winner=mpl.cm.spring, cmap_loser=mpl.cm.summer,s=0.5, trial=trial)
    percent_inside = [0] * num_players
    if coordinate_arrays is not None:
        for i in range(num_players):
            x_vals, y_vals = coordinate_arrays[
                (globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])
            ]

            distances = np.sqrt(x_vals**2 + y_vals**2)
            inside_circle = distances <= radius
            percent_inside_player = np.sum(inside_circle) / len(distances) * 100
            percent_inside[i] = percent_inside_player

    return percent_inside


def distance_travelled_in_iti(trial_list=None, trial=None, trial_index=None, num_players=2):
    '''Computes the total distance travelled in the time between previous trial trigger activation and slice onset of current trial'''

    coordinate_arrays, _ = plot_trajectory.plot_interim_trajectory_colour_map(ax=None, trial_list=trial_list, trial_index=trial_index, cmap_winner=mpl.cm.spring, cmap_loser=mpl.cm.summer,s=0.5, trial=trial)
    distances_travelled = [0] * num_players
    if coordinate_arrays is not None:
        for i in range(num_players):
            x_vals, y_vals = coordinate_arrays[
                (globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])
            ]
            # Compute differences between consecutive points
            dx = np.diff(x_vals)
            dy = np.diff(y_vals)

            # Compute Eucl distances and sum
            distance_travelled = np.sum(np.sqrt(dx**2 + dy**2))
            distances_travelled[i] = distance_travelled
    
    return distances_travelled

def plot_trajectory_with_circle(trial_list=None, trial=None, trial_index=None, radius=0, num_players=2):
    '''Plots the trajectory between previous trial trigger activation and slice onset of current trial within circle bounds'''
    
    if trial_index:
        trial = trial_list[trial_index]
    this_trial = plot_trajectory.extract_trial(trial, trial_list, trial_index)
    
    ax = plot_octagon.plot_octagon()
    coordinate_arrays, _ = plot_trajectory.plot_interim_trajectory_colour_map(ax=ax, trial_list=trial_list, trial_index=trial_index, cmap_winner=mpl.cm.spring, cmap_loser=mpl.cm.summer, s=0.5, trial=this_trial, plot=False)

    # Add circle
    circle = Circle((0, 0), radius=radius, edgecolor='black', facecolor='none', linewidth=0.5)
    ax.add_patch(circle)

    # Set equal aspect ratio to keep circle round
    ax.set_aspect('equal')

    # Find current trial winner
    trigger_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    trigger_activating_client = trigger_event[globals.TRIGGER_CLIENT].values[0]
    
    # Plot trajectory
    for i in range(num_players):
        x_vals, y_vals = coordinate_arrays[
            (globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])
        ]
        for j in range(len(x_vals)):
            distance = np.sqrt(x_vals[j]**2 + y_vals[j]**2)
            if i == trigger_activating_client:
                if distance <= radius:
                    ax.scatter(x_vals[j], y_vals[j], c='green', label='Inside', s=0.5)
                else: ax.scatter(x_vals[j], y_vals[j], c='red', label='Outside', s=0.5)
            else: 
                if distance <= radius:
                    ax.scatter(x_vals[j], y_vals[j], c='blue', label='Inside', s=0.5)
                else: ax.scatter(x_vals[j], y_vals[j], c='orange', label='Outside', s=0.5)
            
    #ax.legend()
    return ax