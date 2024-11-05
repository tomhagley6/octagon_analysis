#!/usr/bin/env python
# coding: utf-8

# In[27]:


import parse_data.preprocess as preprocess
import parse_data.loading as loading
from analysis.response_times import calculate_response_times, plot_response_times
import globals
from plotting import plot_octagon
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl


# In[31]:


## Extract single trials ##


# In[3]:


def split_session_by_trial(df, drop_trial_zero=True):
    
    # groupby produces an iterable of tuples with the group key and the dataframe 
    trials_list = [data for _, data in df.groupby('data.trialNum')]

    if drop_trial_zero:
        # exclude trial 0 (could also exclude trial 1)
        trials_list = trials_list[1:]

    # if final  trial does not contain a server selected trigger activation, discard it
    if not globals.TRIAL_END in trials_list[-1]['eventDescription'].unique():
        trials_list = trials_list[:-1]

    return trials_list

