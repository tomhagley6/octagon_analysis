#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import globals


# In[47]:


## Pre-process data ##


# In[48]:


# Take time in reference to start time
def reference_application_time(df):
    df2 = df.copy()
    start_time = df['timeApplication'].iloc[0]
    df2['timeReferenced'] = df['timeApplication'] - start_time

    return df2


# In[49]:


# Fill nans in trialNum with the correct trial number (starting at 0 for pre-trial data)
# This is needed because trialNum is only recorded at the single timepoint that trialNum changes
def fill_trial_zero(df):
    df2 = df.copy()
    df2.loc[0, 'data.trialNum'] = 0 # Manually change first entry to 0 and fill forward
                                     # This means nans after trial 1 will not be set to 0
    df2['data.trialNum'] = df2['data.trialNum'].ffill()

    return df2


# In[50]:


def is_social(df):
    return globals.PLAYER_1_XLOC in df.columns
        


# In[51]:


def num_players(df):
    return len(df.filter(like=globals.XLOC).columns)


# In[52]:


# Fill player scores 
def fill_player_scores(df, num_players):
    df2 = df.copy()
    
    df2.loc[0, 'data.playerScores.0'] = 0 
    df2['data.playerScores.0'] = df2['data.playerScores.0'].ffill()

    if num_players == 2:
        df2.loc[0, 'data.playerScores.1'] = 0 
        df2['data.playerScores.1'] = df2['data.playerScores.1'].ffill()


    return df2


# In[53]:


# Fill current trial type and account for data pre trial 1
def fill_trial_type(df):
    df2 = df.copy()
    df2.loc[0, 'data.trialType'] = 'pre-trials'
    df2['data.trialType'] = df2['data.trialType'].ffill()

    return df2


# In[54]:


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


# In[55]:


# Fill current trial type for all rows
def fill_trial_type_full(df):
    df = fill_trial_type(df)
    df = fill_post_final_trial_type(df)

    return df


# In[56]:


def fill_trial_walls(df): 
    '''  Fill currently active walls with values throughout the trial 
         But only between start of trial and end of trial indices '''
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
    


# In[10]:


def fill_trial_walls_fully(df):
    '''  Fill currently active walls with values throughout the trial 
         Past the end trial index and throughout the ITI '''
    
    df2 = df.copy()
    
    trial_start_indices = df2[df2['eventDescription'] == 'trial start'].index
    slice_onset_indices = df2[df2['eventDescription'] == 'slice onset'].index

    for idx in range(len(trial_start_indices) -1):
        this_slice_onset = slice_onset_indices[idx]
        first_index_in_trial = trial_start_indices[idx]
        last_index_in_trial = trial_start_indices[idx+1] - 1
        
        # Forward fill the wall numbers from slice onset to next start trial
        df2.loc[this_slice_onset:last_index_in_trial, 'data.wall1'] = df2.loc[this_slice_onset:last_index_in_trial, 'data.wall1'].ffill()
        df2.loc[this_slice_onset:last_index_in_trial, 'data.wall2'] = df2.loc[this_slice_onset:last_index_in_trial, 'data.wall2'].ffill()

        # Backwards fill the wall numbers from slice onset to start trial
        df2.loc[first_index_in_trial:this_slice_onset, 'data.wall1'] = df2.loc[first_index_in_trial:this_slice_onset, 'data.wall1'].bfill()
        df2.loc[first_index_in_trial:this_slice_onset, 'data.wall2'] = df2.loc[first_index_in_trial:this_slice_onset, 'data.wall2'].bfill()

    # account for there being a fully complete trial at the end without a new trial start (i.e., recording ends
    # on ITI phase
    trial_end_indices = df2[df2['eventDescription'] == globals.TRIAL_END].index
    
    if len(trial_end_indices) == len(trial_start_indices):
        this_slice_onset = slice_onset_indices[len(trial_start_indices) -1]
        first_index_in_trial = trial_start_indices[len(trial_start_indices) -1]
        last_index_in_trial = df.index[-1]
        
        # Forward fill the wall numbers from slice onset to next start trial
        df2.loc[this_slice_onset:last_index_in_trial, 'data.wall1'] = df2.loc[this_slice_onset:last_index_in_trial, 'data.wall1'].ffill()
        df2.loc[this_slice_onset:last_index_in_trial, 'data.wall2'] = df2.loc[this_slice_onset:last_index_in_trial, 'data.wall2'].ffill()
        
        # Backwards fill the wall numbers from slice onset to start trial
        df2.loc[first_index_in_trial:this_slice_onset, 'data.wall1'] = df2.loc[first_index_in_trial:this_slice_onset, 'data.wall1'].bfill()
        df2.loc[first_index_in_trial:this_slice_onset, 'data.wall2'] = df2.loc[first_index_in_trial:this_slice_onset, 'data.wall2'].bfill()   
        
    
    return df2


# In[5]:


def remove_zero_wall_numbers(df):
    ''' When a trigger activation occurred that was not selected by the server, it will trial walls as 0,0
        Remove these values and replace with nans to allow forward and backward filling of wall numbers '''

    df2 = df.copy()
    df2.loc[df2['data.wall1'] == 0, 'data.wall1'] = np.nan
    df2.loc[df2['data.wall2'] == 0, 'data.wall2'] = np.nan

    return df2


# In[59]:


# Create a column to reflect the current trial epoch for each row
def create_trial_epoch_column(df, col_name='trial_epoch'):
    df2 = df.copy()
    
    # create column
    df2[col_name] = np.nan
    # cast from float64 to object dtype to allow including strings without complaining
    df2[col_name] = df2[col_name].astype('object')

    # define the eventDescription triggers that lead to epoch transitions, the indices where these occur,
    # and the labels of the epoch periods
    epoch_transition_triggers = ['trial start', 'slice onset', globals.SELECTED_TRIGGER_ACTIVATION, 'trial end']
    epoch_transition_idxs = [df2.index[df2['eventDescription'] == trigger] for trigger in epoch_transition_triggers]
    epoch_transition_labels = [globals.TRIAL_STARTED, globals.SLICES_ACTIVE, globals.POST_CHOICE, globals.ITI]

    # insert the epoch period label at all indexes where this transition occurs
    # and do this for all epoch periods
    for i in range(len(epoch_transition_idxs)):
        df2.loc[epoch_transition_idxs[i], col_name] = epoch_transition_labels[i]
    # add a pre-trials label to the very beginning of recording
    df2.loc[0, col_name] = globals.PRE_TRIALS

    # forward fill trial epoch to replace all nans and make the labels continuous
    df2[col_name] = df2[col_name].ffill()

    return df2


# In[2]:


# umbrella function for the above preprocessing
def standard_preprocessing(df):
    df = reference_application_time(df)
    df = fill_trial_zero(df)
    df = fill_trial_type_full(df)
    social = is_social(df)
    n_players = num_players(df)
    df = fill_player_scores(df, num_players)
    df = remove_zero_wall_numbers(df)
    df = fill_trial_walls_fully(df)
    df = create_trial_epoch_column(df)

    print("Preprocessing complete.")
    
    return df

