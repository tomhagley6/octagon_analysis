#!/usr/bin/env python
# coding: utf-8

# In[9]:


import parse_data.preprocess as preprocess
import parse_data.loading as loading
import parse_data.combine_sessions as combine_sessions
import parse_data.split_session_by_trial as split_session_by_trial
from analysis.response_times import calculate_response_times, plot_response_times
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import globals


# In[10]:


def prepare_single_session_data(data_folder, json_filename):
    ''' Load and preprocess a single dataframe
        Return the full dataframe and a list of trials '''
    
    # Load JSON file into pandas df with collapsed data dictionary and adjustments based on date of recording 
    df = loading.loading_pipeline(data_folder, json_filename)

    # Pre-process data 
    df = preprocess.standard_preprocessing(df)

    trial_list = split_session_by_trial.split_session_by_trial(df, drop_trial_zero=True)

    return df, trial_list


# In[11]:


def prepare_combined_session_data(data_folder, json_filenames):
    ''' Load and preprocess multiple dataframes, and concatenate
        Return the full dataframe and a list of trials '''

    df = combine_sessions.combine_sessions(data_folder, json_filenames)
    
    trial_list = split_session_by_trial.split_session_by_trial(df, drop_trial_zero=False)

    return df, trial_list


# In[12]:


# umbrella function
def prepare_data(data_folder, json_filenames):
    ''' Prepare a full dataframe and list of trial dataframe from either a single
        or set of sessions, given as filepaths '''
    
    if len(json_filenames) == 1:
        # index the list for its only item
        df, trial_list = prepare_single_session_data(data_folder, json_filenames[0])
    elif len(json_filenames) > 1:
        df, trial_list = prepare_combined_session_data(data_folder, json_filenames)
    else:
        print("json_filenames must be a list of strings of len >= 1")
        return None

    return df, trial_list
        

