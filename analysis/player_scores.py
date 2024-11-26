#!/usr/bin/env python
# coding: utf-8

# In[1]:


import parse_data.prepare_data as prepare_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import globals
import os
import data_extraction.get_indices as get_indices



# In[ ]:


def proportion_score_sessions(data_folder, json_filenames):
    ''' Return num_sessions*num_players array for proportion of score
        each player earned in a session.
        Takes the data folder and a list of session filenames '''

    proportion_scores_all_sessions = np.zeros((len(json_filenames), 2))
    for json_filenames_index in range(len(json_filenames)):
        json_filename = json_filenames[json_filenames_index]
        print(data_folder + os.sep + json_filename)
        _, trials_list = prepare_data.prepare_data(data_folder, json_filename)
        
        # access final trial event log event for the final player scores
        final_trial = trials_list[-1]
        final_trial_trial_end = final_trial[final_trial['eventDescription'] == 'trial end']
        
        # flexibly index player scores
        player0_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[0]['score']].item()
        player1_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[1]['score']].item()
        total_score = player0_score + player1_score
        
        proportion_score_player0 = player0_score/total_score
        proportion_score_player1 = player1_score/total_score

        proportion_scores_all_sessions[json_filenames_index, 0] = proportion_score_player0
        proportion_scores_all_sessions[json_filenames_index, 1] = proportion_score_player1

    return proportion_scores_all_sessions


# In[ ]:


def proportion_score_sessions_df(trial_lists):
    ''' Return num_sessions*num_players array for proportion of score
        each player earned in a session.
        Takes a list of pre-processed trial lists '''
    
    proportion_scores_all_sessions = np.zeros((len(trial_lists), 2))
    for trial_list_index in range(len(trial_lists)):
        trial_list = trial_lists[trial_list_index]

        # access final trial event log event for the final player scores
        final_trial = trial_list[-1]
        final_trial_trial_end = final_trial[final_trial['eventDescription'] == 'trial end']
        
        # flexibly index player scores
        player0_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[0]['score']].item()
        player1_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[1]['score']].item()
        total_score = player0_score + player1_score
        
        proportion_score_player0 = player0_score/total_score
        proportion_score_player1 = player1_score/total_score

        proportion_scores_all_sessions[trial_list_index, 0] = proportion_score_player0
        proportion_scores_all_sessions[trial_list_index, 1] = proportion_score_player1

    return proportion_scores_all_sessions


# In[ ]:


def player_scores_sessions_df(trial_lists):
    ''' Return num_sessions*num_players array of player score vals
        each player earned in a session.
        Takes a list of pre-processed trial lists '''
    
    player_scores_all_sessions = np.zeros((len(trial_lists), 2))
    for trial_list_index in range(len(trial_lists)):
        trial_list = trial_lists[trial_list_index]
        
        # access final trial event log event for the final player scores
        final_trial = trial_list[-1]
        final_trial_trial_end = final_trial[final_trial['eventDescription'] == 'trial end']
        
        # flexibly index player scores
        player0_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[0]['score']].item()
        player1_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[1]['score']].item()

        player_scores_all_sessions[trial_list_index, 0] = player0_score
        player_scores_all_sessions[trial_list_index, 1] = player1_score

    return player_scores_all_sessions


# In[ ]:


def proportion_wins_sessions(trial_lists):
    ''' Return num_sessions*num_players array of proportion wins 
        each player earned in a session.
        Takes a list of pre-processed trial lists '''

    proportion_wins_array = np.zeros((len(trial_lists), 2))
    for i in range(len(trial_lists)):
        trial_list = trial_lists[i]
        winners = get_indices.get_trigger_activators(trial_list)
        proportion_wins_player_0 = np.sum(winners == 0)/winners.size
        proportion_wins_player_1 = 1 - proportion_wins_player_0
        proportion_wins_array[i, :] = proportion_wins_player_0, proportion_wins_player_1

    return proportion_wins_array

