import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import globals
import data_extraction.extract_trial as extract_trial
import utils.cosine_similarity as cosine_similarity
import data_extraction.get_indices as get_indices
import data_extraction.trial_list_filters as trial_list_filters
import analysis.conditioned_player_choice as conditioned_player_choice
from analysis import opponent_visibility



def get_player_positions(player_id, trial=None, trial_list=None, trial_index=None):
    '''Return player positions of the player from slice onset to trigger activation.
    Input player id and single trial'''

    trial = extract_trial.extract_trial(trial=trial, trial_list=trial_list, trial_index=trial_index)
    assert isinstance(trial, pd.DataFrame)

    # get slice onset and trigger activation indices
    slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    slice_onset_index = slice_onset.index - trial.index[0]

    trigger_activation = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    trigger_activation_index = trigger_activation.index - trial.index[0]

    # crop the trial to the time window of interest
    trial_cropped = trial.iloc[slice_onset_index[0]:trigger_activation_index[0]]

    # get player positions
    x_coords = trial_cropped[globals.PLAYER_LOC_DICT[player_id]['xloc']].iloc[slice_onset_index[0]:trigger_activation_index[0]].to_numpy()
    y_coords = trial_cropped[globals.PLAYER_LOC_DICT[player_id]['yloc']].iloc[slice_onset_index[0]:trigger_activation_index[0]].to_numpy()

    player_position_coords = np.stack((x_coords, y_coords), axis=-1)
    
    return player_position_coords



def get_player_headangle_vectors(player_id, trial=None, trial_list=None, trial_index=None):
    '''Return player head angle of the player from slice onset to trigger activation.
       Input player id and single trial'''

    trial = extract_trial.extract_trial(trial=trial, trial_list=trial_list, trial_index=trial_index)
    assert isinstance(trial, pd.DataFrame)

    # get slice onset and trigger activation indices
    slice_onset = trial[trial['eventDescription'] == globals.SLICE_ONSET]
    slice_onset_index = slice_onset.index - trial.index[0]

    trigger_activation = trial[trial['eventDescription'] == globals.SELECTED_TRIGGER_ACTIVATION]
    trigger_activation_index = trigger_activation.index - trial.index[0]

    # crop the trial to the time window of interest
    trial_cropped = trial.iloc[slice_onset_index[0]:trigger_activation_index[0]]

    # get player head angle vectors
    y_rotation = trial_cropped[globals.PLAYER_ROT_DICT[player_id]['yrot']].iloc[slice_onset_index[0]:trigger_activation_index[0]].to_numpy()
    head_angle = np.deg2rad(y_rotation) # convert to radians

    x_component = np.sin(head_angle)
    z_component = np.cos(head_angle)
    head_angle_vectors = np.stack((x_component, z_component), axis=-1)
    
    return head_angle_vectors



def calculate_angle_to_opponent_timepoints(self_positions, other_positions, self_head_angle_vectors):
    '''Use cosine similarity (angle between vectors, length invariant) to return the angle
       between the cevtor of self head angle and vector from self to other in the time 
       between slice onset and selected trigger activation.'''
    
    # euclidian vector from self to other
    # self and other positions are size len(active_trial) arrays of size 2 (x,y)
    self_other_vectors = other_positions - self_positions

    thetas = []

    # dot product between head angle vector and self-other vector for each timepoint
    for i in range(len(self_other_vectors)):

        self_other_vector = self_other_vectors[i]
        self_head_angle_vector = self_head_angle_vectors[i]

        dot_product_vectors = np.dot(self_other_vectors[i], self_head_angle_vectors[i])
        
        # vector norms for both self_other_vector and self_head_angle_vector
        (self_other_vector_norm,
        self_head_angle_vector_norm) = opponent_visibility.calculate_vector_norms_for_timepoint(self_other_vector, 
                                                                          self_head_angle_vector)
        
        # cosine similarity between the two vectors
        vector_cosine_similarity = cosine_similarity.calculate_cosine_similarity_two_vectors(dot_product_vectors,
                                                                                         self_other_vector_norm,
                                                                                         self_head_angle_vector_norm)
        # angle for cosine similarity
        theta = cosine_similarity.calculate_angle_from_cosine_similarity(vector_cosine_similarity)
        thetas.append(theta)
    
    return thetas



def get_two_player_positions_slice_onset(player_id, trial=None, trial_list=None, trial_index=None):
    ''' Return the Self position and Other position for all trial timepoints'''

    opponent_id = 1 if player_id == 0 else 0

    self_position = get_player_positions(player_id, trial, trial_list, trial_index)

    other_position = get_player_positions(opponent_id, trial, trial_list, trial_index)

    return self_position, other_position   



def get_angle_of_opponent_from_player_trial_timepoints(player_id, trial=None, trial_list=None, trial_index=None):
    ''' For a single trial, return the angles from player head direction to opponent player.
        Takes the player_id of Self, and the trial. '''

    # find self and other positions 
    self_positions, other_positions = get_two_player_positions_slice_onset(player_id, trial=trial,
                                                                           trial_list=trial_list,
                                                                           trial_index=trial_index)

    # find general self head angle vector
    self_head_angle_vectors = get_player_headangle_vectors(player_id, trial=trial,
                                                           trial_list=trial_list,
                                                           trial_index=trial_index)

    # calculate angle from cosine similarity between self_head_angle_vector and self_to_other vector
    # (self_to_other_vector is calculated in this function as the difference between Self and Other position)
    thetas = calculate_angle_to_opponent_timepoints(self_positions, other_positions, self_head_angle_vectors)
  
    return thetas



def get_angle_of_opponent_from_player_session(player_id, trial_list):
    ''' For all trials in a session, return the angles from player head direction to opponent player.
        Takes the player_id of Self (persistent throughout session) and the trial list. '''
    
    # get the angle for each trial in session, for a persistent Self player_id
    orientation_angle_to_other_session = {}

    for trial_index, trial in enumerate(trial_list):
    
        theta = get_angle_of_opponent_from_player_trial_timepoints(player_id, trial=trial)

        orientation_angle_to_other_session[trial_index] = theta

    return orientation_angle_to_other_session



def get_other_visible_session_timepoints(orientation_angle_to_other_session, current_fov):
    ''' Return a boolean array for whether Other is visible to Self at each trial timepoint.
        Takes the angle of orientation from Self to Other as an array for the session (in radians)
        and the visible fov for this dataset (in degrees) '''
    
    other_visible = {}

    # convert orientation_angle_to_other_session from radians to degrees to match current_fov
    for i in range(len(orientation_angle_to_other_session)):
        orientation_angle_to_other_session_deg = np.rad2deg(orientation_angle_to_other_session[i])

        # if Other is visible, the angle to orient Other into Self central view must be less than half the current
        # field-of-view.
        # At the threshold, Other enters visual periphery
        other_visible_session = orientation_angle_to_other_session_deg < current_fov/2
        other_visible[i] = other_visible_session

    return other_visible




def proportion_of_time_other_visible_session(other_visible):
    '''Return proportion of time in trial (slice onset to selected trigger activation)
       that opponent was in view.'''
    
    proportions = []

    for i in range(len(other_visible)):

        trial_visible = other_visible[i]
        total_length = len(trial_visible)
        is_visible_length = len(trial_visible[trial_visible == True])

        if total_length == 0:
            proportion_trial_visible = np.nan
        else:
            proportion_trial_visible = is_visible_length / total_length

        proportions.append(proportion_trial_visible)
    
    return proportions





def get_other_visible_session_trial_lists(trial_lists, current_fov):
    ''' Return a boolean array for whether Other is visible to Self at each trial timepoint.
        Takes the angle of orientation from Self to Other as an array for the session (in radians)
        and the visible fov for this dataset (in degrees) '''
    
    other_visible_dict = {}
    proportions_dict = {}

    for trial_list_idx, trial_list in enumerate(trial_lists):

        other_visible_dict[trial_list_idx] = {}
        proportions_dict[trial_list_idx] = {}

        for player_id in range(2):

            orientation_angle_to_other_session = get_angle_of_opponent_from_player_session(player_id, trial_list)

            other_visible = get_other_visible_session_timepoints(orientation_angle_to_other_session, current_fov)

            proportions = []

            for i in range(len(other_visible)):

                trial_visible = other_visible[i]
                total_length = len(trial_visible)
                is_visible_length = len(trial_visible[trial_visible == True])

                if total_length == 0:
                    proportion_trial_visible = np.nan
                else:
                    proportion_trial_visible = is_visible_length / total_length

                proportions.append(proportion_trial_visible)
                    
            other_visible_dict[trial_list_idx][player_id] = other_visible
            proportions_dict[trial_list_idx][player_id] = proportions

    return other_visible_dict, proportions_dict