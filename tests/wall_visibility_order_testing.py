#!/usr/bin/env python
# coding: utf-8

# TODO:
# - Testing to verify that head angle at trial start is correct (Then all head angle to wall code is verified correct)

# In[52]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import data_extraction.get_indices as get_indices
import parse_data.prepare_data as prepare_data
import globals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plotting import plot_octagon, plot_trajectory
import data_extraction.get_indices as get_indices
import plotting.plot_probability_chose_wall as plot_probability_chose_wall
import data_strings
import analysis.wall_visibility_and_choice as wall_visibility_and_choice
import trajectory_analysis.trajectory_vectors as trajectory_vectors
import trajectory_analysis.trajectory_headangle as trajectory_headangle
import analysis.wall_choice as wall_choice
import data_extraction.extract_trial as extract_trial
import utils.pad_and_reshape_array as utils
import plotting.wall_visibility_order_testing_functions as wall_visibility_order_testing_functions


# In[53]:


# notebook global variables

wall_sep = None
trial_type = globals.HIGH_LOW
player_id = 0
n_cols = 12


# In[54]:


# prepare the data

data_folder = data_strings.DATA_FOLDER
json_filenames_all_social = data_strings.JSON_FILENAMES_SOCIAL
json_filenames_all_solo = data_strings.JSON_FILENAMES_SOLO

# specify session number
json_filename = json_filenames_all_social[3]

_, trial_list = prepare_data.prepare_data(data_folder, json_filename, combine=True)


# In[55]:


# filter the trial list for trialtype and wallsep

# filter trial list for given_wallLow trialtype
trial_indices = get_indices.get_trials_trialtype(trial_list, trial_type=trial_type)
trial_list_filtered = [trial_list[i] for i in trial_indices]

# filter trial list for wall separations if specified
if wall_sep:
    trial_indices = get_indices.get_trials_with_wall_sep(trial_list_filtered, wall_sep=wall_sep)
    trial_list_filtered = [trial_list_filtered[i] for i in trial_indices]


# In[56]:


# gather data for the first visible wall for the session

# condition here is 'first wall visible is High, High is chosen', no inferred choice
(condition_fulfilled_session,
  player_chose_given_wall_session) = wall_visibility_and_choice.given_wall_chosen_conditioned_on_visibility(trial_list, player_id=player_id,
                                                                                                            given_wall_index=0, given_wall_first_vis=True,
                                                                                                              current_fov=110, wall_sep=None, trial_type=globals.HIGH_LOW,
                                                                                                                inferred_choice=False, debug=False)


player_wall_choice = wall_choice.player_wall_choice_wins_only(trial_list_filtered, player_id=player_id)


# In[57]:


# reshape relevant arrays to fit with n_rows,n_cols grid (pad with np.nan) 

condition_fulfilled_session_reshaped = utils.pad_and_reshape_array(condition_fulfilled_session, n_cols)
player_chose_given_wall_session_reshaped = utils.pad_and_reshape_array(player_chose_given_wall_session, n_cols)
player_wall_choice_reshaped = utils.pad_and_reshape_array(player_wall_choice, n_cols)


# In[ ]:


(thetas_closest_wall_section,
 thetas_trajectory)  = wall_visibility_order_testing_functions.plot_single_trial_first_wall_visibility(trial_list_filtered, trial_index=2,
                                                                vector_length=20, start_index=0, wall_index=None, player_id=0)

print(f"Angle to closest wall section and alcove: {thetas_closest_wall_section}, {thetas_trajectory}")


# In[59]:


(thetas_closest_wall_section_session,
 thetas_trajectory_session) = wall_visibility_order_testing_functions.plot_multiple_trials_first_wall_visibility(trial_list_filtered, player_id=0)


# In[60]:


print(f"condition fulfilled:\n {condition_fulfilled_session_reshaped}\n"), 
print(f"player chose given wall:\n {player_chose_given_wall_session_reshaped}\n") 
print(f"player wall choice:\n {player_wall_choice_reshaped}")


# In[61]:


np.set_printoptions(suppress=True)
print(f"angle to closest wall section:\n {thetas_closest_wall_section_session}")


# In[62]:


print(f"angle to alcove:\n {thetas_trajectory_session}")

