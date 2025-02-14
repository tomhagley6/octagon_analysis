import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import data_extraction.extract_trial as extract_trial
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import plotting.plot_octagon as plot_octagon
import data_extraction.get_indices as get_indices




def plot_trial_headangle_vectors_colour_map(ax, trial_list=None, trial_index=0, trial=None, player_id=0, 
                                            vector_length=3, step=3, cmap=mpl.cm.plasma):
    '''
    Plots head angle vectors for a given player in a trial, using a color map based on timestamps.

    Parameters:
    - ax: Matplotlib axis to plot on
    - trial_list: List of trials
    - trial_index: Index of trial in the list
    - trial: A single trial
    - player_id: ID of the player whose head angles we want to plot
    - vector_length: Scale factor for head angle vectors
    - step: How frequently to plot head angle vectors (e.g., every 3rd point)
    - cmap: Colormap for timestamps.

    Returns:
    - ax: Updated axis with head angle vectors plotted
    '''

    # extract trial data
    this_trial = extract_trial.extract_trial(trial, trial_list, trial_index)

    trajectory = trajectory_vectors.extract_trial_player_trajectory(trial=trial, player_id=player_id)

    # extract head angles for the player
    headangles = trajectory_vectors.extract_trial_player_headangles(trial=this_trial, player_id=player_id)

    # get smoothed head angle vectors
    trial_player_headangles_smoothed = trajectory_headangle.get_smoothed_player_head_angle_vectors_for_trial(headangles, window_size=5)

    # generate timestamps
    timestamps = np.arange(trial_player_headangles_smoothed.shape[1])

    # normalize timestamps for colormap
    norm = mpl.colors.Normalize(vmin=min(timestamps), vmax=max(timestamps))
    color_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # scatter trajectory
    ax.scatter(trajectory[0,:], trajectory[1,:], s=0.5)


    # plot head angle vectors at specified intervals with colormap
    for time_index in range(0, trial_player_headangles_smoothed.shape[1], step):
        x_start = trajectory[0,time_index]
        y_start = trajectory[1,time_index]
        x_gradient = trial_player_headangles_smoothed[0, time_index]
        y_gradient = trial_player_headangles_smoothed[1, time_index]

        start = [x_start, y_start]
        end = [x_start + x_gradient * vector_length, y_start + y_gradient * vector_length]

        this_head_angle_vector_coordinates = np.array([start, end])

        ax.plot(this_head_angle_vector_coordinates[:, 0], 
                this_head_angle_vector_coordinates[:, 1], 
                c=color_map.to_rgba(timestamps[time_index]), linewidth=1.3)

    # plot active walls
    alcove_coordinates = plot_octagon.return_alcove_centre_points()
    walls = get_indices.get_walls(trial=this_trial)
    wall1_index = walls[0] - 1
    wall2_index = walls[1] - 1

    # scatter wall points
    plt.scatter(alcove_coordinates[0,wall1_index], alcove_coordinates[1,wall1_index], c='r', s=15)
    plt.scatter(alcove_coordinates[0,wall2_index], alcove_coordinates[1,wall2_index], c='b', s=15)


    # hide spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # ddd colorbar
    cbar = plt.colorbar(color_map, ax=ax, orientation="vertical")
    cbar.set_label("Time Index")

    return ax