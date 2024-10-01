#!/usr/bin/env python
# coding: utf-8

# In[47]:


import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('display.precision', 9)
pd.set_option('display.width', 1000)  # Adjust to ensure there's enough room for all data
pd.set_option('display.max_columns', None)  # Show all columns

json_normalise = True


# In[48]:


## Load JSON file into pandas df with collapsed data dictionary ##


# In[49]:


## paths
# data_folder = '/home/tom/Documents/SWC/data' # desktop Duan Lab
# json_filename = '240913_Yansu_Jerry/2024-09-13_11-23-37_YansuFirstSolo.json' 
# data_folder = r'D:\Users\Tom\OneDrive\PhD\SWC\data' # desktop home
# json_filename = r'first_experiments_2409\240913\2024-09-13_11-23-37_YansuFirstSolo.json'
# filepath = data_folder + os.sep + json_filename


# # In[50]:


# # Note json_normalize requires the json file, whereas read_json requires the filepath
# if json_normalise == True:
#     with open(filepath) as f:
#         file = json.load(f)
#         df = pd.json_normalize(file)
# else:
#     with open(filepath) as f:
#         df = pd.read_json(f)
#         print(type(df))


# In[51]:


# ##  Convert time columns into datetime format
# df['timeLocal'] = pd.to_datetime(df['timeLocal'], format='%H:%M:%S:%f')

# # Use to_timedelta instead as a vectorised function (lambdas are python loops)
# # df['timeApplication'] = df['timeApplication'].apply(lambda x: timedelta(seconds=int(x) + (x - int(x))))
# df['timeApplication'] = pd.to_numeric(df['timeApplication']) 
# df['timeApplication'] = pd.to_timedelta(df['timeApplication'], unit='s')


# In[52]:


## Handle trial start events using a different key for player location information - pre-240927 data ##


# In[1]:


def playerinfo_playerposition_conversion(df, solo=True):
    
    # List of data affected by issue (relative paths)
    if solo:
        columns_to_merge = ['0.location.x', '0.location.y', '0.location.z', '0.rotation.x', '0.rotation.y', '0.rotation.z']
    else:
        columns_to_merge = ['0.location.x', '0.location.y', '0.location.z', '0.rotation.x', '0.rotation.y', '0.rotation.z', +
                            '1.location.x', '1.location.y', '1.location.z', '1.rotation.x', '1.rotation.y', '1.rotation.z']
    # Replace the current playerPosition column with one in which the trial start events are filled (instead of NaN)
    # Do this by filtering the relevant 2 columns, ffilling across columns (so the playerPosition column has its NaNs
    # replaced by the values in playerInfo, and taking only this column with iloc
    df2 = df.copy()
    for name in columns_to_merge:
        df2[f'data.playerPosition.{name}'] = df.filter(like=name).ffill(axis=1).iloc[:,-1]

    # Remove the redundant columns from the dataframe
    cols_to_drop = [col for col in df.columns.to_list() if 'playerInfo' in col]
    df2 = df2.drop(cols_to_drop, axis=1)


    return df2


# In[58]:


# df2 = playerinfo_playerposition_conversion(df, solo=True)


# In[59]:


# df2.tail()

