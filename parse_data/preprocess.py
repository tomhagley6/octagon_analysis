#!/usr/bin/env python
# coding: utf-8

# In[30]:


## Pre-process data ##


# In[31]:


# Take time in reference to start time
def reference_application_time(df):
    df2 = df.copy()
    start_time = df['timeApplication'].iloc[0]
    df2['timeReferenced'] = df['timeApplication'] - start_time

    return df2


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
    


# In[1]:


# umbrella function for the above preprocessing
def standard_preprocessing(df):
    df = reference_application_time(df)
    df = fill_trial_zero(df)
    df = fill_trial_type_full(df)
    df = fill_player_scores_solo(df)
    df = fill_trial_walls(df)

    print("Preprocessing complete.")
    
    return df

