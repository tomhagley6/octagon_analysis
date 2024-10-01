#!/usr/bin/env python
# coding: utf-8

# In[21]:


import json
import os
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from handle_specific_datasets import playerinfo_playerposition_conversion

pd.set_option('display.precision', 9)
pd.set_option('display.width', 1000)  # Adjust to ensure there's enough room for all data
pd.set_option('display.max_columns', None)  # Show all columns

json_normalise = True


# In[22]:b


### Load JSON data and prepare it for analysis ###


# In[23]:


# paths
# data_folder = '/home/tom/Documents/SWC/data' # desktop Duan Lab
# json_filename = '240913_Yansu_Jerry/2024-09-13_11-23-37_YansuFirstSolo.json' 
data_folder = r'D:\Users\Tom\OneDrive\PhD\SWC\data' # desktop home
json_filename = r'first_experiments_2409\240913\2024-09-13_11-23-37_YansuFirstSolo.json'
filepath = data_folder + os.sep + json_filename


# In[24]:


## Load JSON file into pandas df with collapsed data dictionary ##


# In[25]:


# Note json_normalize requires the json file, whereas read_json requires the filepath
if json_normalise == True:
    with open(filepath) as f:
        file = json.load(f)
        df = pd.json_normalize(file)
else:
    with open(filepath) as f:
        df = pd.read_json(f)
        print(type(df))


# In[26]:


# Convert time columns into datetime format
df['timeLocal'] = pd.to_datetime(df['timeLocal'], format='%H:%M:%S:%f')

# Use to_timedelta instead as a vectorised function (lambdas are python loops)
# df['timeApplication'] = df['timeApplication'].apply(lambda x: timedelta(seconds=int(x) + (x - int(x))))
df['timeApplication'] = pd.to_numeric(df['timeApplication']) 
df['timeApplication'] = pd.to_timedelta(df['timeApplication'], unit='s')


# In[27]:


## handle data based on date ##


# In[29]:


# find date string in filename
pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
match = re.search(pattern, json_filename)

# convert date string to datetime
timestamp_dt = datetime.strptime(match.group(), "%Y-%m-%d_%H-%M-%S")

# list of all dates with data that needs specific handling
date_first_experiment = datetime.strptime("2024-09-13", "%Y-%m-%d")

# conditional statements based on date of data
if timestamp_dt < date_first_experiment + timedelta(days=1):
    print(f"Data is from period before {timestamp_dt}")
    df = playerinfo_playerposition_conversion(df)
    print(f"Running dataframe through playerinfo_playerposition_conversion.")


# In[30]:


### Pre-process data


# In[31]:


# Take time in reference to start time
def reference_application_time(df):
    df2 = df.copy()
    start_time = df['timeApplication'].iloc[0]
    df2['timeReferenced'] = df['timeApplication'] - start_time

    return df2


# In[32]:


df = reference_application_time(df)


# In[33]:


# Fill nans in trialNum with the correct trial number (starting at 0 for pre-trial data)
# This is needed because trialNum is only recorded at the single timepoint that trialNum changes
def fill_trial_zero(df):
    df2 = df.copy()
    df2.loc[0, 'data.trialNum'] = 0 # Manually change first entry to 0 and fill forward
                                     # This means nans after trial 1 will not be set to 0
    df2['data.trialNum'] = df2['data.trialNum'].ffill()

    return df2


# In[34]:


# Fill player scores 
def fill_player_scores_solo(df):
    df2 = df.copy()
    df2.loc[0, 'data.playerScores.0'] = 0 
    df2['data.playerScores.0'] = df2['data.playerScores.0'].ffill()

    return df2


# In[35]:


# Fill current trial type and account for data pre trial 1
def fill_trial_type(df):
    df2 = df.copy()
    df2.loc[0, 'data.trialType'] = 'pre-trials'
    df2['data.trialType'] = df2['data.trialType'].ffill()

    return df2


# In[36]:


# Fill data past the final trial end with post-trials label
def fill_post_final_trial_type(df):
    df2 = df.copy()

    # Find the indices for the final trial end and final trial start log events
    final_trial_end_idx = df2[df2['eventDescription'] == 'trial end'].index[-1] if not df[df['eventDescription'] == 'trial end'].empty else None
    final_trial_start_idx = df2[df2['eventDescription'] == 'trial start'].index[-1] if not df[df['eventDescription'] == 'trial start'].empty else None

    # Fill indices after final trial end with a post-trial label if there is a subsequent trial start
    if final_trial_end_idx < final_trial_start_idx:
        df2.loc[final_trial_end_idx + 1:, 'data.trialType'] = 'post-trials'
    
    return df2


# In[37]:


# Fill current trial type for all rows
def fill_trial_type_full(df):
    df = fill_trial_type(df)
    df = fill_post_final_trial_type(df)

    return df


# In[38]:


# Fill currently active walls with values throughout the trial 
# But only between start of trial and end of trial indices
def fill_trial_walls(df): 
    df2 = df.copy()
    
    trial_start_indices = df2[df2['eventDescription'] == 'trial start'].index
    slice_onset_indices = df2[df2['eventDescription'] == 'slice onset'].index
    trial_end_indices = df2[df2['eventDescription'] == 'trial end'].index    

    for idx in range(len(trial_start_indices) -1):
        # Forward fill the wall numbers from slice onset to end trial
        df2.loc[slice_onset_indices[idx]:trial_end_indices[idx], 'data.wall1'] = df2.loc[slice_onset_indices[idx]:trial_end_indices[idx], 'data.wall1'].ffill()
        df2.loc[slice_onset_indices[idx]:trial_end_indices[idx], 'data.wall2'] = df2.loc[slice_onset_indices[idx]:trial_end_indices[idx], 'data.wall2'].ffill()

        # Backwards fill the wall numbers from slice onset to start trial
        df2.loc[trial_start_indices[idx]:slice_onset_indices[idx], 'data.wall1'] = df2.loc[trial_start_indices[idx]:slice_onset_indices[idx], 'data.wall1'].bfill()
        df2.loc[trial_start_indices[idx]:slice_onset_indices[idx], 'data.wall2'] = df2.loc[trial_start_indices[idx]:slice_onset_indices[idx], 'data.wall2'].bfill()
    
    return df2
    


# In[39]:


## Data cleaning functions


# In[40]:


df = fill_trial_zero(df)


# In[41]:


df = fill_trial_type_full(df)


# In[42]:


df = fill_player_scores_solo(df)


# In[43]:


df = fill_trial_walls(df)


# In[44]:


df.tail()

