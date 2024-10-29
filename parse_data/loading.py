#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import re
import os
import pandas as pd
from datetime import datetime, timedelta
from parse_data.handle_specific_datasets import playerinfo_playerposition_conversion, remove_zero_wall_numbers


# In[1]:


def match_filename_to_filesystem(json_filename):
    json_filename_parts = json_filename.split('\\')
    json_filename_rejoined = os.path.join(*json_filename_parts)

    return json_filename_rejoined
    


# In[4]:


# given a filepath, load dataframe from .json, with nesting flattened or not
def load_df_from_file(data_folder, json_filename, json_normalise=True):
    filepath = data_folder + os.sep + json_filename
    if json_normalise == True:
        with open(filepath) as f:
            file = json.load(f)
            df = pd.json_normalize(file)
    else:
        with open(filepath) as f:
            df = pd.read_json(f)
    
    return df


# In[5]:


# Convert time columns into datetime format
def convert_time_strings(df):
    df2 = df.copy()
    df2['timeLocal'] = pd.to_datetime(df2['timeLocal'], format='%H:%M:%S:%f')

    # Use to_timedelta instead as a vectorised function (lambdas are python loops)
    # df['timeApplication'] = df['timeApplication'].apply(lambda x: timedelta(seconds=int(x) + (x - int(x))))
    df2['timeApplication'] = pd.to_numeric(df2['timeApplication']) 
    df2['timeApplication'] = pd.to_timedelta(df2['timeApplication'], unit='s')

    return df2


# In[2]:


# check the date of the file against any date conditionals, and then run the relevant functions
def handle_date_sensitive_processing(df, json_filename):
    # find date string in filename
    pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
    match = re.search(pattern, json_filename)

    # convert date string to datetime
    timestamp_dt = datetime.strptime(match.group(), "%Y-%m-%d_%H-%M-%S")

    # list of all dates with data that needs specific handling
    date_first_experiment = datetime.strptime("2024-09-13", "%Y-%m-%d") # merging playerinfo
    date_fourth_experiment = datetime.strptime("2024-10-18", "%Y-%m-%d") # removing zeros from wallnums

    # conditional statements based on date of data
    df2 = df.copy()

    # merging playerinfo dictionary into playerposition
    if timestamp_dt < date_first_experiment + timedelta(days=1):
        print(f"Data is from period before {date_first_experiment}")
        df2 = playerinfo_playerposition_conversion(df2)
        print(f"Running dataframe through playerinfo_playerposition_conversion.")

    # # currently treating this a standard preprocessing step
    # # removing any zeros from recorded wall numbers and replacing them with nans
    # if timestamp_dt < date_fourth_experiment + timedelta(days=1):
    #     print(f"Data is from period before {date_fourth_experiment}")
    #     df2 = remove_zero_wall_numbers(df2)
    #     print("Running dataframe through remove_zero_wall_numbers")
    

    print("Loading complete.")

    return df2


# In[2]:


def loading_pipeline(data_folder, json_filename, json_normalise=True):
    json_filename = match_filename_to_filesystem(json_filename)
    df = load_df_from_file(data_folder, json_filename, json_normalise=True)
    df = convert_time_strings(df)
    df = handle_date_sensitive_processing(df, json_filename)

    return df

