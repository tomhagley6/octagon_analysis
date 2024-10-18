#!/usr/bin/env python
# coding: utf-8

# In[4]:


import parse_data.preprocess as preprocess
import parse_data.loading as loading
from analysis.response_times import calculate_response_times, plot_response_times
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import globals


# In[5]:


# paths
# data_folder = '/home/tom/Documents/SWC/data' # desktop Duan Lab
# json_filename = '240913_Yansu_Jerry/2024-09-13_11-53-34_YansuSecondSolo.json' 
# json_filename = '240913_Yansu_Jerry/2024-09-13_11-31-00_YansuJerrySocial.json'
data_folder = r'D:\Users\Tom\OneDrive\PhD\SWC\data' # desktop home
json_filenames = [r'first_experiments_2409\240913\2024-09-13_11-23-37_YansuFirstSolo.json',
                  r'second_experiments_2409\240927\2024-09-27_14-25-20_SaraEmilySocial.json']


# In[6]:


# Load JSON file into pandas df with collapsed data dictionary and adjustments based on date of recording 
sessions = []
for json_filename in json_filenames:
    df = loading.loading_pipeline(data_folder, json_filename)
    sessions.append(df)


# In[ ]:


# Pre-process data 
preprocessed_sessions = []
for session in sessions:
    df = preprocess.standard_preprocessing(session)
    preprocessed_sessions.append(df)

