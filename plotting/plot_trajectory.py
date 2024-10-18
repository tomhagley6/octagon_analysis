#!/usr/bin/env python
# coding: utf-8

# In[1]:


import globals
from plotting import plot_octagon
from parse_data import preprocess
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def extract_trial(trial, trial_list, trial_index):
    ''' isolate trial '''
    
    if not trial is None:
        this_trial = trial
    elif not trial_list is None:
        this_trial = trial_list[trial_index]
    else:
        raise ValueError("a list of trials and the chosen index must be given, or the trial itself must be given, but not neither.")

    return this_trial


# In[3]:


def plot_trial_trajectory_colour_map(ax, trial_list=None, trial_index=0, cmap_winner=mpl.cm.spring, cmap_loser=mpl.cm.summer,
                   s=0.5, trial=None):
    ''' Plot the trajectories of all players for a single trial '''
    
    # isolate trial
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

    # find number of players to plot for
    num_players = preprocess.num_players(this_trial)

    # create an array of (column) labels to index the dataframe for each player's trajectory
    coordinate_array_labels = []
    for i in range(num_players):
        coordinate_array_labels.extend((globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])) 
    coordinate_arrays = {label : this_trial[label].values[slice_onset_idx:trigger_idx] for label in coordinate_array_labels}
    
    # create colormap data
    cmap_winner = mpl.cm.spring 
    cmap_loser = mpl.cm.summer
    timestamps = np.arange(len(coordinate_arrays[globals.PLAYER_0_XLOC]))
    min_val, max_val = min(timestamps), max(timestamps)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)

    # scatter each players trajectory, with a unique colour map for the winning player
    cmaps = [cmap_winner, cmap_loser]
    for i in range(num_players):
        cmap_index = 0 if i == trigger_activating_client else 1
        ax.scatter(coordinate_arrays[coordinate_array_labels[2*i]], coordinate_arrays[coordinate_array_labels[2*i+1]], s=0.5, c=timestamps, cmap=cmaps[cmap_index], norm=norm)


    return ax
    


# In[4]:


def plot_trial_trajectory(ax, trial_list=None, trial_index=0, colour_winner='c', colour_loser='m',
                   s=0.5, trial=None):
    ''' Plot the trajectories of each player for a single trial, with separate colours for winner and loser '''
    
    # isolate trial
    this_trial = extract_trial(trial, trial_list, trial_index)
    
    # isolate trigger event and activating client
    slice_onset_event = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]
    trigger_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    trigger_activating_client = trigger_event['data.triggerClient'].values[0]
    
    # find index of trigger event normalised to this trial
    slice_onset_idx = slice_onset_event.index[0]
    slice_onset_idx = int(slice_onset_idx - this_trial.index[0])
    trigger_idx = trigger_event.index[0]
    trigger_idx = int(trigger_idx - this_trial.index[0])

    # find number of players to plot for
    num_players = preprocess.num_players(this_trial)

    # create an array of (df column) labels to index the dataframe for each player's trajectory
    coordinate_array_labels = []
    for i in range(num_players):
        coordinate_array_labels.extend((globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])) 
    coordinate_arrays = {label : this_trial[label].values[slice_onset_idx:trigger_idx] for label in coordinate_array_labels}

    # scatter each players trajectory, with a unique colour map for the winning player
    colours = [colour_winner, colour_loser]
    for i in range(num_players):
        colour_index = 0 if i == trigger_activating_client else 1
        ax.plot(coordinate_arrays[coordinate_array_labels[2*i]], coordinate_arrays[coordinate_array_labels[2*i+1]], markersize=1, color=colours[colour_index])


    return ax


# In[10]:


def plot_trial_winning_trajectory(ax, trial_list=None, trial_index=0, colour_wall1='blue', colour_wall2='blueviolet',
                                          trial=None, loser=False):
    ''' Plot only the winning trajectory for a single trial, with optional flag to plot only the losing trajectories
        Separate colours for wall 1 and wall 2 '''
    
    # isolate trial
    this_trial = extract_trial(trial, trial_list, trial_index)

    # isolate trigger event, activating client, which wall was triggered
    slice_onset_event = this_trial[this_trial['eventDescription'] == globals.SLICE_ONSET]
    trigger_event = this_trial[this_trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    trigger_activating_client = trigger_event['data.triggerClient'].values[0]
    wall_triggered = trigger_event[globals.WALL_TRIGGERED].item()
    wall1_triggered = True if wall_triggered == this_trial.iloc[0]['data.wall1'] else False
    
    # find index of trigger event normalised to this trial
    slice_onset_idx = slice_onset_event.index[0]
    slice_onset_idx = int(slice_onset_idx - this_trial.index[0])

    trigger_idx = trigger_event.index[0]
    trigger_idx = int(trigger_idx - this_trial.index[0])

    # find number of players to plot for
    num_players = preprocess.num_players(this_trial)

    # extract all of the relevant players' coordinates from the dataframe (winner or loser) by first
    # creating an array of correct labels for extraction
    if not loser:
        # create an array of (df column) labels to index the dataframe for each player's trajectory
        coordinate_array_labels = [globals.PLAYER_LOC_DICT[trigger_activating_client]['xloc'],
                                   globals.PLAYER_LOC_DICT[trigger_activating_client]['yloc']]
    
    else:
        # create a list of of labels for the players that did not win the trial
        coordinate_array_labels = []
        for i in range(num_players):
            if i == trigger_activating_client:
                pass
            else:
                coordinate_array_labels.extend((globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])) 

    # use the labels array to extract the relevant player trajectory coordinates
    coordinate_arrays = {label : this_trial[label].values[slice_onset_idx:trigger_idx] for label in coordinate_array_labels}

    # plot relevant players trajectory, with colours separating wall1 and wall2 trajectories
    colours = [colour_wall1, colour_wall2]
    x_coordinates = coordinate_arrays[coordinate_array_labels[0]]
    y_coordinates = coordinate_arrays[coordinate_array_labels[1]]
    ax.plot(x_coordinates, y_coordinates, markersize=1, color=colours[0 if wall1_triggered else 1])

    # testing = x_coordinates[0], y_coordinates[0]
    # print(testing)
    
    return ax 


# In[6]:


def plot_session_trajectory(ax, df, colour_player_1='yellow', colour_player_2='turquoise', alpha=0.7, chosen_player=None):
    ''' Plot the continuous trajectory for an entire session for each player '''

    # find number of players to plot for
    num_players = preprocess.num_players(df)

    # extract all of the relevant players' coordinates from the dataframe by first
    # creating an array of correct labels for extraction
    coordinate_array_labels = []
    for i in range(num_players):
        coordinate_array_labels.extend((globals.PLAYER_LOC_DICT[i]['xloc'], globals.PLAYER_LOC_DICT[i]['yloc'])) 
    
    # use the labels array to extract the relevant player trajectory coordinates 
    coordinate_arrays = {label : df[label].values for label in coordinate_array_labels}
    
    # plot each players trajectory
    # if choosing a specific player, only plot this player's trajectory
    colours = [colour_player_1, colour_player_2, 'g', 'r', 'b', 'y']
    for i in range(num_players):
        if chosen_player is None:
            ax.plot(coordinate_arrays[coordinate_array_labels[2*i]], coordinate_arrays[coordinate_array_labels[2*i+1]], markersize=1, color=colours[i], alpha=alpha, label=f'player {i}')
        else:
            if i != chosen_player:
                pass
            else:
                ax.plot(coordinate_arrays[coordinate_array_labels[2*i]], coordinate_arrays[coordinate_array_labels[2*i+1]], markersize=1, color=colours[i], alpha=alpha, label=f'player {i}')

    # add title
    main_title = "Whole session trajectory" 
    title_supp = f" for player {chosen_player}" if chosen_player is not None else ""
    title_string = main_title + title_supp
    ax.set_title(title_string)
        

    return ax

