#!/usr/bin/env python
# coding: utf-8

# In[5]:


import parse_data.preprocess as preprocess
import parse_data.loading as loading
import parse_data.split_session_by_trial as split_session_by_trial
from analysis.response_times import calculate_response_times, plot_response_times
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import globals


# In[7]:


def load_multiple_sessions(data_folder, json_filenames):
    ''' load more than one session into a pandas df 
        with collapsed data dictionary and adjustments 
        based on date of recording '''
    
    loaded_sessions = []
    for json_filename in json_filenames:
        df = loading.loading_pipeline(data_folder, json_filename)
        loaded_sessions.append(df)

    return loaded_sessions


# In[8]:


def preprocess_multiple_sessions(loaded_sessions):
    ''' pre-process more than 1 session '''
    
    preprocessed_sessions = []
    for session in loaded_sessions:
        df = preprocess.standard_preprocessing(session)
        preprocessed_sessions.append(df)

    return preprocessed_sessions


# In[9]:


def split_and_reconcatenate_sessions(preprocessed_sessions):
    ''' split trials and remove the first (and last, if incomplete)
        of each session before re-concatenating '''

    split_trial_sessions = []
    for session in preprocessed_sessions:
        trial_list = split_session_by_trial.split_session_by_trial(session) # This will remove the first trial and any unfinished final trial
        split_trial_sessions.append(trial_list)
    
    reconcatenated_sessions = []
    for trial_list in split_trial_sessions:
        reconcatenated_session = pd.concat(trial_list)
        reconcatenated_sessions.append(reconcatenated_session)

    return reconcatenated_sessions


# In[10]:


def create_continuity_between_sessions(reconcatenated_sessions):
    ''' Create continuity between sessions for time
        time fields and trial numbers '''
    
    sessions_with_continuity = []
    for i in range(len(reconcatenated_sessions)):
    
        # copy dataframe to edit
        df = reconcatenated_sessions[i].copy()
    
        # increment current dataframe's data values by previous dataframe's final values
        if i > 0:
            df['timeApplication'] = df['timeApplication'] + final_application_time
            df['timeReferenced'] = df['timeReferenced'] + final_relative_time
            df[globals.TRIAL_NUM] = df[globals.TRIAL_NUM] + final_trial_num
    
        # record final data values for the current dataframe
        final_application_time = df['timeApplication'].iloc[-1]
        final_relative_time = df['timeReferenced'].iloc[-1]
        final_trial_num = df[globals.TRIAL_NUM].iloc[-1]
    
        sessions_with_continuity.append(df)

    return sessions_with_continuity


# In[11]:


# umbrella function
def combine_sessions(data_folder, json_filenames):
    ''' Provide a list of filenames, and a data folder
        Sessions will be loaded, preprocessed (including removal of first
        and last trials), and concatenated with continuity '''
    
    loaded_sessions = load_multiple_sessions(data_folder, json_filenames)

    preprocessed_sessions = preprocess_multiple_sessions(loaded_sessions)
    
    reconcatenated_sessions = split_and_reconcatenate_sessions(preprocessed_sessions)

    sessions_with_continuity = create_continuity_between_sessions(reconcatenated_sessions)

    combined_sessions = pd.concat(sessions_with_continuity)

    return combined_sessions

