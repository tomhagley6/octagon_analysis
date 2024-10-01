#!/usr/bin/env python
# coding: utf-8

# In[1]:


import parse_data.preprocess as preprocess
import parse_data.loading as loading
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt


# In[10]:


class ResponseTimes:

    def __init__(self, response_times, mean, median, iqr,
                 *social_response_times, **social_stats):
        self.response_times = response_times
        self.mean = mean
        self.median = median
        self.iqr = iqr

        # store extra social information
        self.social_response_times = social_response_times
        self.social_summary_stats = social_stats

        # validate structure
        self._validate()

    def _validate(self):
        # ensure only one extra response_times list for each player
        if len(self.social_response_times) not in [0,2]:
            raise ValueError("Require exactly 2 extra positional response_times Series for social," +
                             " p1 followed by p2")
        if len(self.social_summary_stats) not in [0,6]:
            raise ValueError("Require mean_p1, mean_p2, median_p1, median_p2, and iqr_p1, iqr_p2 keyword arguments"
                             + " for each player for social")

    def is_social(self):
        return bool(self.social_response_times or self.social_summary_stats)

    def print_summary_stats(self):
        if self.is_social():

            summary_labels = ['mean_p1', 'mean_p2', 'median_p1', 'median_p2', 'iqr_p1', 'iqr_p2']
            summary_stats = {label: self.social_summary_stats.get(label, 'Not provided') for label in summary_labels}


            print(f"Mean response time: p1 - {summary_stats['mean_p1']}, p2  - {summary_stats['mean_p2']}.")
            print(f"Median response time: p1 - {summary_stats['median_p1']}, p2  - {summary_stats['median_p2']}")
            print(f"Response time range: p1 - {summary_stats['iqr_p1']}, p2  - {summary_stats['iqr_p2']}")

            print(f"Mean response time (combined): {self.mean}.")
            print(f"Median response time (combined): {self.median}.")
            print(f"Response time range (combined): {self.iqr}.\n")
    
        else:

            print(f"Mean response time: {self.mean}.")
            print(f"Median response time: {self.median}.")
            print(f"Response time range: {self.iqr}.")


# In[11]:


def calculate_response_times(df):
    
    # response times are calculated as difference between slice onset and selected trigger activation
    # discard the first trial 
    df_slice_onset = df[df['eventDescription'] == 'slice onset'].iloc[1:]
    df_selected_trigger_activation = df[df['eventDescription'] == 'server-selected trigger activation'].iloc[1:]

    # only index as many values as there were triggers (completed trials)
    num_triggers = len(df_selected_trigger_activation['timeApplication'].values)

    # calcuate rt
    response_times = df_selected_trigger_activation['timeApplication'].values - df_slice_onset['timeApplication'].values[:num_triggers]
    response_times = pd.Series(response_times)

    # handle social
    trigger_activating_client = df_selected_trigger_activation['data.triggerClient'].values
    
    if 1 in trigger_activating_client:
        social = True
    else:
        social = False

    # create separate response times Series for each player in social
    if social:
        trial_idxs_p1 = np.where(trigger_activating_client == 0)[0]
        trial_idxs_p2 = np.where(trigger_activating_client == 1)[0]

        response_times_p1 = response_times[trial_idxs_p1]
        response_times_p2 = response_times[trial_idxs_p2]

    # summary statistics (general)
    mean = np.mean(response_times).to_timedelta64().astype('timedelta64[ms]')
    median = np.median(response_times).astype('timedelta64[ms]')
    iqr = scipy.stats.iqr(response_times).astype('timedelta64[ms]')

    # separate summary statistics for each player (social)
    if social:
        mean_p1 = np.mean(response_times_p1).to_timedelta64().astype('timedelta64[ms]')
        mean_p2 = np.mean(response_times_p2).to_timedelta64().astype('timedelta64[ms]')

        median_p1 = np.median(response_times_p1).astype('timedelta64[ms]')
        median_p2 = np.median(response_times_p2).astype('timedelta64[ms]')

        iqr_p1 = scipy.stats.iqr(response_times_p1).astype('timedelta64[ms]')
        iqr_p2 = scipy.stats.iqr(response_times_p2).astype('timedelta64[ms]')


    # Return an instance of ResponseTimes class which handles social vs solo
    if social:
        return ResponseTimes(response_times, mean, median, iqr, response_times_p1, response_times_p2, 
                             mean_p1=mean_p1, mean_p2=mean_p2, median_p1=median_p1, median_p2=median_p2, 
                             iqr_p1=iqr_p1, iqr_p2=iqr_p2)
    else:
        return ResponseTimes(response_times, mean, median, iqr)


# In[4]:


def plot_response_times(response_times):

    if response_times.is_social():
        plt.plot(response_times.response_times_p1.astype('timedelta64[ms]')) # plot in milliseconds (default ns)
        plt.plot(response_times.response_times_p2.astype('timedelta64[ms]'))
    else:
    
        plt.plot(response_times.response_times.astype('timedelta64[ms]')) # plot in milliseconds (default ns)
    
    
    plt.ylabel('Time (ms)')
    plt.xlabel('Trial num')

    return None


# In[ ]:




