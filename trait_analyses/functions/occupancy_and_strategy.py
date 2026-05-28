import plotting.plot_trajectory as plot_trajectory
import matplotlib as mpl
import numpy as np
import globals
import plotting.plot_octagon as plot_octagon
from matplotlib.patches import Circle
import data_extraction.extract_trial as extract_trial
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

def return_maximum_distance():
    ''' Return the maximum distance between 2 points within the Octagon arena '''
    
    # get octagon arena coordinates
    # global Unity coordinates (so, local Unity coordinates, multiplied by arena scale factor)
    octagon_coordinates = plot_octagon.return_octagon_path_points()

    # separate x and y coordinates
    x_coords = [coordinates[0] for coordinates in octagon_coordinates]
    y_coords = [coordinates[1] for coordinates in octagon_coordinates]

    # find the L2 norm between the minimum and maximum x location, at y=0
    # (octagon is symmetric, so any opposites points can be used)
    max_dist = np.linalg.norm(np.array([min(x_coords), 0]) - np.array([max(x_coords), 0]))

    return max_dist

def central_tendency_trial(trial_list=None, trial=None, trial_index=0, num_players=2):
    '''Length of vector between player position at slice onset and arena centre 
    divided by maximum vector, between arena centre and alcove inner wall'''

    trial = extract_trial.extract_trial(trial, trial_list, trial_index)
    
    # define slice onset and trigger activation events
    slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    if not isinstance(slice_onset, int):
        slice_onset_index = slice_onset.index[0] - trial.index[0]
    else:
        slice_onset_index = slice_onset.index - trial.index[0]

    # define arena centre
    centre = (0,0)

    # define maximum distance from centre
    max_dist = return_maximum_distance()
    max_dist_from_centre = max_dist/2

    # initialise central tendency array
    central_tendencies = [0] * num_players

    for player_id in range(num_players):
    # define player location at slice onset
        x_coordinate = trial[globals.PLAYER_LOC_DICT[player_id]['xloc']].iloc[slice_onset_index]
        y_coordinate = trial[globals.PLAYER_LOC_DICT[player_id]['yloc']].iloc[slice_onset_index]
        position_at_slice_onset = (x_coordinate, y_coordinate)

        # define player to centre vector
        vector = np.array(centre) - np.array(position_at_slice_onset)
        player_dist = np.linalg.norm(vector)

        # define centre occupancy metric
        central_tendency = 1 - (player_dist/max_dist_from_centre)

        #print(f"max distance: {max_dist_from_centre}, player distance: {player_dist}")

        # populate array
        central_tendencies[player_id] = central_tendency

    return central_tendencies

def central_tendency_trial_trajectory(trial_list=None, trial=None, trial_index=0, num_players=2):
    '''Computes normalised distance from the centre from slice onset to trigger activation for each player
    mean_central_tendency[i] → overall tendency for player i for that trial
    central_tendencies_trajectory[i] → array of tendency values per timepoint (one per frame)
    '''

    # define arena centre
    centre = (0,0)

    # define maximum distance from centre
    max_dist = return_maximum_distance()
    max_dist_from_centre = max_dist/2

    # get trajectory coordinates
    coordinate_arrays, _ = plot_trajectory.plot_trial_trajectory_colour_map(ax=None, trial_list=trial_list, trial_index=trial_index, cmap_winner=mpl.cm.spring, cmap_loser=mpl.cm.summer,s=0.5, trial=trial)
    mean_central_tendency = []
    central_tendencies_trajectory = []
    
    if coordinate_arrays is not None:
        for i in range(num_players):
            x_label = globals.PLAYER_LOC_DICT[i]['xloc']
            y_label = globals.PLAYER_LOC_DICT[i]['yloc']

            try:
                x_vals = coordinate_arrays[x_label]
                y_vals = coordinate_arrays[y_label]

                distances = np.sqrt(x_vals**2 + y_vals**2)
                normalised_tendency = 1 - (distances / max_dist_from_centre)

                central_tendencies_trajectory.append(normalised_tendency)
                mean_central_tendency.append(np.nanmean(normalised_tendency))
            
            except KeyError:
                central_tendencies_trajectory.append(np.nan)
                mean_central_tendency.append(np.nan)
        
    return mean_central_tendency, central_tendencies_trajectory

def central_tendency_multiple_sessions(trial_lists=None, num_players=2):
    ''' Compute the mean central tendency at slice onset for a whole session for both players '''

    mean_central_tendencies_multiple_sessions = [[] for _ in range(len(trial_lists))]
    central_tendencies_multiple_sessions = [[] for _ in range(len(trial_lists))]

    for session_idx, trial_list in enumerate(trial_lists):

        session_player_values = [[] for _ in range(num_players)]

        for trial_index in range(len(trial_list)):
            if trial_index == len(trial_list)-1:
                continue
            central_tendencies = central_tendency_trial(trial_list=trial_list, trial=None, trial_index=trial_index, num_players=num_players)
            if num_players==2:
                if np.isnan(central_tendencies[0]) or np.isnan(central_tendencies[1]):
                    continue
            for player_id in range(num_players):
                session_player_values[player_id].append(central_tendencies[player_id])
            
        mean_per_player = [np.nanmean(session_player_values[player_id]) for player_id in range(num_players)]
        
        mean_central_tendencies_multiple_sessions[session_idx] = mean_per_player
        central_tendencies_multiple_sessions[session_idx] = session_player_values

    return mean_central_tendencies_multiple_sessions, central_tendencies_multiple_sessions

def central_tendency_in_trial_multiple_sessions(trial_lists=None, num_players=2):
    '''Compute mean central tendency from slice onset to trigger activation for each player across multiple sessions.'''

    mean_central_tendencies_multiple_sessions = [[] for _ in range(len(trial_lists))]
    central_tendencies_multiple_sessions = [[] for _ in range(len(trial_lists))]

    for session_idx, trial_list in enumerate(trial_lists):
        session_player_values = [[] for _ in range(num_players)]  
        
        for trial_index in range(len(trial_list)):
            _, central_tendencies = central_tendency_trial_trajectory(
                trial_list=trial_list, trial=None, trial_index=trial_index, num_players=num_players
            )

            if not isinstance(central_tendencies, list) or len(central_tendencies) < num_players:
                continue  

            for player_id in range(num_players):
                if isinstance(central_tendencies[player_id], np.ndarray):
                    session_player_values[player_id].append(central_tendencies[player_id])

        mean_per_player = [
            np.nanmean(np.concatenate(session_player_values[player_id])) if session_player_values[player_id] else np.nan
            for player_id in range(num_players)
        ]

        mean_central_tendencies_multiple_sessions[session_idx] = mean_per_player
        central_tendencies_multiple_sessions[session_idx] = session_player_values

    return mean_central_tendencies_multiple_sessions, central_tendencies_multiple_sessions




def central_tendency_interim_trajectory(trial_list=None, trial=None, trial_index=0, num_players=2):
    '''Computes normalised distance from the centre for each timepoint'''

    # define arena centre
    centre = (0,0)

    # define maximum distance from centre
    max_dist = return_maximum_distance()
    max_dist_from_centre = max_dist/2

    # get trajectory coordinates
    coordinate_arrays, _ = plot_trajectory.plot_interim_trajectory_colour_map(ax=None, trial_list=trial_list, trial_index=trial_index, cmap_winner=mpl.cm.spring, cmap_loser=mpl.cm.summer,s=0.5, trial=trial)
    
    mean_central_tendency = [np.nan] * num_players
    central_tendencies_trajectory = [np.nan] * len(coordinate_arrays)
    #print(coordinate_arrays)
    if coordinate_arrays is not None:
        for i in range(num_players):
            x_vals, y_vals = coordinate_arrays[
                (globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])
            ]

            distances = np.sqrt(x_vals**2 + y_vals**2)
            #print(f"distances: {distances}")
            normalised_tendency = 1 - (distances / max_dist_from_centre)
            #print(normalised_tendency)
            central_tendencies_trajectory[i] = normalised_tendency
            mean_central_tendency[i] = np.mean(normalised_tendency)
        
    return mean_central_tendency, central_tendencies_trajectory

def central_tendency_multiple_sessions_trajectory(trial_lists=None, num_players=2):
    ''' Compute the mean central tendency throughout ITI trajectory for a whole session for both players '''

    mean_central_tendencies_multiple_sessions = [[] for _ in range(len(trial_lists))]
    central_tendencies_multiple_sessions = [[] for _ in range(len(trial_lists))]

    for session_idx, trial_list in enumerate(trial_lists):

        session_player_values = [[] for _ in range(num_players)]

        for trial_index in range(1,len(trial_list)):
            trajectory_mean_central_tendency, central_tendencies_trajectory = central_tendency_interim_trajectory(trial_list=trial_list, trial=None, trial_index=trial_index, num_players=num_players)
            
            for player_id in range(num_players):
                session_player_values[player_id].append(central_tendencies_trajectory[player_id])
            
        mean_per_player = [
            np.nanmean(np.concatenate(session_player_values[player_id])) if session_player_values[player_id] else np.nan
            for player_id in range(num_players)
            ]
        
        mean_central_tendencies_multiple_sessions[session_idx] = mean_per_player
        central_tendencies_multiple_sessions[session_idx] = session_player_values

    return mean_central_tendencies_multiple_sessions, central_tendencies_multiple_sessions

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
