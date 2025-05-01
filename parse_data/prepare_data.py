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
from ipywidgets import IntProgress
from IPython.display import display
import time


# In[10]:


def prepare_single_session_data(data_folder, json_filename):
    ''' Load and preprocess data from a single session
        Returns: full dataframe, list of trials '''
    
    # Load JSON file into pandas df with collapsed data dictionary and adjustments based on date of recording 
    # (parse_data/loading.py)
    df = loading.loading_pipeline(data_folder, json_filename)

    # Pre-process data 
    # (parse_data/preprocess.py)
    df = preprocess.standard_preprocessing(df)

    # (parse_data/split_session_by_trial.py)
    trial_list = split_session_by_trial.split_session_by_trial(df, drop_trial_zero=True)

    return df, trial_list


# In[ ]:


def prepare_combined_session_data(data_folder, json_filenames, drop_trial_zero=True):
    ''' Load and preprocess multiple dataframes, and concatenate
        Returns: full dataframe, list of trials '''

    # (parse_data/combine_sessions.py)
    df = combine_sessions.combine_sessions(data_folder, json_filenames)

    # (parse_data/split_session_by_trial.py)
    trial_list = split_session_by_trial.split_session_by_trial(df, drop_trial_zero=drop_trial_zero)

    return df, trial_list


# In[ ]:


# umbrella function
def prepare_data(data_folder, json_filenames, combine=False):
    ''' Input: data folder and json_filename string or list of json_filename strings.
        Returns: full dataframe, list of trials.
        Adapts to: a single session, multiple sessions combined, multiple sessions kept separate in a list '''
    
    if isinstance(json_filenames, str):  # handle a single session

        df, trial_list = prepare_single_session_data(data_folder, json_filenames)
   
    elif isinstance(json_filenames, list): # handle multiple sessions
        
        if combine: # keep sessions in one dataframe and one list
            df, trial_list = prepare_combined_session_data(data_folder, json_filenames)
        
        else: # separate sessions in separate dfs and separate trial lists
            
            # use a progress bar to show loading progress
            max_count = len(json_filenames)
            f = IntProgress(min=0, max=max_count) # instantiate the bar
            display(f) # display the bar
            
            df = []
            trial_list = []
            for filename in json_filenames:
                this_df, this_trial_list = prepare_single_session_data(data_folder, filename)
                df.append(this_df)
                trial_list.append(this_trial_list)
                
                f.value += 1 # signal to increment the progress bar
    else:
        print("json_filenames must be a list of strings of len >= 1, or a string")
        return None

    return df, trial_list
        

