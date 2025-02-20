#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import globals


# In[47]:


## Pre-process data ##


# In[48]:


def reference_application_time(df):
    ''' Return a dataframe with additional column of the time referenced to
        the time at row 0 '''
    
    df2 = df.copy()
    start_time = df['timeApplication'].iloc[0]
    df2['timeReferenced'] = df['timeApplication'] - start_time

    return df2


# In[49]:


# This is needed because trialNum is only recorded at the single timepoint that trialNum changes
def fill_trial_zero(df):
    ''' Replace np.nan with the relevant trial number, for all nans in trialNum column
        with 0 in place of any nan values pre-trial 1
        Returns dataframe with these replacements '''
    
    df2 = df.copy()
    df2.loc[0, 'data.trialNum'] = 0 # Manually change first entry to 0 and fill forward
                                     # This means nans after trial 1 will not be set to 0
    df2['data.trialNum'] = df2['data.trialNum'].ffill()

    return df2


# In[50]:


def is_social(df):
    ''' Return boolean value for whether dataframe contains social session data '''
    
    return globals.PLAYER_1_XLOC in df.columns
        


# In[51]:


def num_players(df):
    ''' Return int number of players '''
    
    return len(df.filter(like=globals.XLOC).columns)


# In[52]:


def fill_player_scores(df, num_players):
    ''' Return a dataframe with the player scores value filled at all indices
        Functional for up to 2 players '''
    
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
    ''' Return a dataframe with the trial type column filled for all indices
        including 'pre-trials' for indices before the first trial began '''
    
    df2 = df.copy()
    df2.loc[0, 'data.trialType'] = 'pre-trials'
    df2['data.trialType'] = df2['data.trialType'].ffill()

    return df2


# In[54]:


def fill_post_final_trial_type(df):
    ''' Return a dataframe with the trial type column filled with 'post-trials'
        for all indices after a final trial start that has no subsequent 
        trial end ''' 
    
    df2 = df.copy()

    # Find the indices for the final trial end and final trial start log events
    # Take the final index or return None if there are no 'trial end' or 'trial start' events
    final_trial_end_idx = df2[df2['eventDescription'] == 'trial end'].index[-1] if not df[df['eventDescription'] == 'trial end'].empty else None
    final_trial_start_idx = df2[df2['eventDescription'] == 'trial start'].index[-1] if not df[df['eventDescription'] == 'trial start'].empty else None

    # Fill indices after final trial end with a post-trial label if there is a subsequent trial start
    if final_trial_end_idx < final_trial_start_idx:
        df2.loc[final_trial_end_idx + 1:, 'data.trialType'] = 'post-trials'
    
    return df2


# In[55]:


def fill_trial_type_full(df):
    ''' Return dataframe with trial type column filled for all indices '''

    # fill trial type from before first trial and until last trial
    df = fill_trial_type(df)
    # fill trial type past the final trial end event
    df = fill_post_final_trial_type(df)

    return df


# In[56]:


# Currently unused as following function will check to include the final trial start index
def fill_trial_walls(df): 
    '''  Return dataframe where wall number columns are filled with values throughout the trial 
         But only between start of trial and end of trial indices '''
    
    df2 = df.copy()

    # Note, start of trial and not slice onset (walls are decided at the start of trial but not 
    # displayed to subjects until slice onset)
    trial_start_indices = df2[df2['eventDescription'] == 'trial start'].index
    slice_onset_indices = df2[df2['eventDescription'] == 'slice onset'].index
    trial_end_indices = df2[df2['eventDescription'] == 'trial end'].index    

    # for all trial indices excepting the final trial start (check if this excluding agrees with other functions)
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
    '''  Return dataframe with filled wall value columns with values throughout the trial 
         Past the end trial index and throughout the ITI '''
    
    df2 = df.copy()
    
    trial_start_indices = df2[df2['eventDescription'] == 'trial start'].index
    slice_onset_indices = df2[df2['eventDescription'] == 'slice onset'].index

    # for all trial indices excepting the final trial start (check if this excluding agrees with other functions)
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

    # ordinarily the final trial start is ignored, but here check for there being a matching trial end
    # and if so, include the final trial as well as all the previous
    if len(trial_end_indices) == len(trial_start_indices):
        this_slice_onset = slice_onset_indices[len(trial_start_indices) -1]
        first_index_in_trial = trial_start_indices[len(trial_start_indices) -1]
        last_index_in_trial = df.index[-1]
        
        # Forward fill the wall numbers from slice onset to the final index in the dataframe
        df2.loc[this_slice_onset:last_index_in_trial, 'data.wall1'] = df2.loc[this_slice_onset:last_index_in_trial, 'data.wall1'].ffill()
        df2.loc[this_slice_onset:last_index_in_trial, 'data.wall2'] = df2.loc[this_slice_onset:last_index_in_trial, 'data.wall2'].ffill()
        
        # Backwards fill the wall numbers from slice onset to start trial
        df2.loc[first_index_in_trial:this_slice_onset, 'data.wall1'] = df2.loc[first_index_in_trial:this_slice_onset, 'data.wall1'].bfill()
        df2.loc[first_index_in_trial:this_slice_onset, 'data.wall2'] = df2.loc[first_index_in_trial:this_slice_onset, 'data.wall2'].bfill()   
        
    
    return df2


# In[5]:


def remove_zero_wall_numbers(df):
    ''' When a trigger activation occurred that was not selected by the server, it will set trial walls as 0,0
        Remove these values and replace with nans to allow forward and backward filling of wall numbers
        Note that for any analysis of server-rejected trigger activations I will need to avoid this function '''

    df2 = df.copy()
    df2.loc[df2['data.wall1'] == 0, 'data.wall1'] = np.nan
    df2.loc[df2['data.wall2'] == 0, 'data.wall2'] = np.nan

    return df2


# In[59]:


def create_trial_epoch_column(df, col_name='trial_epoch'):
    ''' Return dataframe with a new str column that reflects the trial epoch at 
        each index '''
    
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

    # insert the epoch period label at all specific indices where this transition occurs
    # and do this for all epoch periods
    for i in range(len(epoch_transition_idxs)):
        df2.loc[epoch_transition_idxs[i], col_name] = epoch_transition_labels[i]
    # add a pre-trials label to the very beginning of recording
    df2.loc[0, col_name] = globals.PRE_TRIALS

    # forward fill trial epoch to replace all nans and make the labels continuous
    df2[col_name] = df2[col_name].ffill()

    return df2


# In[ ]:


# umbrella function for the above preprocessing
def standard_preprocessing(df):
    df = reference_application_time(df)
    df = fill_trial_zero(df)
    df = fill_trial_type_full(df)
    social = is_social(df)
    n_players = num_players(df)
    # df = fill_player_scores(df, n_players)
    df = remove_zero_wall_numbers(df)
    df = fill_trial_walls_fully(df)
    df = create_trial_epoch_column(df)

    print("Preprocessing complete.")
    
    return df

