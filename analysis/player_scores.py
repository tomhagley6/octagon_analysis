# %%
import parse_data.prepare_data as prepare_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import globals
import os
import data_extraction.get_indices as get_indices



# %%
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

# %%
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

# %%
# %%
# make sure this is getting the total score of the run of trials, not including the scores before!

# have made edits to address the above. Now run checks to see if this workss

def proportion_score_subset_sessions_df(trial_lists, num_trials, debug=False):
    ''' Return num_sessions*num_players array for proportion of score
        each player earned in a random string of num_trials trials in a session.
        Takes a list of pre-processed trial lists '''
    
    proportion_scores_all_sessions = np.zeros((len(trial_lists), 2))
    
    for trial_list_index in range(len(trial_lists)):

        # grab the trial list for this session
        trial_list = trial_lists[trial_list_index]
        num_trials_total = len(trial_list)
        if num_trials > num_trials_total:
            raise ValueError(f'num_trials {num_trials} exceeds total number of trials {num_trials_total} in session {trial_list_index}')
        
        # pick a random trial index
        initial_trial_index = np.random.randint(0, num_trials_total - num_trials + 1)

        # # initialise player scores in this subset of trials
        # player0_score, player1_score = 0,0

        # # loop through the trials in this subset and accumulate player scores
        # for trial_index in range(initial_trial_index, initial_trial_index + num_trials):
        #     trial = trial_list[trial_index]
        #     trial_trial_end_event = trial[trial['eventDescription'] == 'trial end']
        #     player0_score += trial_trial_end_event[globals.PLAYER_SCORE_DICT[0]['score']].item()
        #     player1_score += trial_trial_end_event[globals.PLAYER_SCORE_DICT[1]['score']].item()

        starting_trial = trial_list[initial_trial_index]
        ending_trial = trial_list[initial_trial_index + num_trials - 1]
        starting_trial_trial_end_event = starting_trial[starting_trial['eventDescription'] == 'trial end']
        ending_trial_trial_end_event = ending_trial[ending_trial['eventDescription'] == 'trial end']
        final_score_player1 = ending_trial_trial_end_event[globals.PLAYER_SCORE_DICT[1]['score']].item()
        final_score_player0 = ending_trial_trial_end_event[globals.PLAYER_SCORE_DICT[0]['score']].item()
        initial_score_player1 = starting_trial_trial_end_event[globals.PLAYER_SCORE_DICT[1]['score']].item()
        initial_score_player0 = starting_trial_trial_end_event[globals.PLAYER_SCORE_DICT[0]['score']].item()
        accumulated_score_player0 = final_score_player0 - initial_score_player0
        accumulated_score_player1 = final_score_player1 - initial_score_player1

        # Debug prints to inspect values for this subset
        print(f"Session {trial_list_index}: total_trials={num_trials_total}, subset_length={num_trials}, start_index={initial_trial_index}")
        print(f"Starting trial end scores -> player0: {initial_score_player0}, player1: {initial_score_player1}")
        print(f"Ending   trial end scores -> player0: {final_score_player0}, player1: {final_score_player1}")
        print(f"Accumulated scores -> player0: {accumulated_score_player0}, player1: {accumulated_score_player1}")

        # sanity checks
        if accumulated_score_player0 < 0 or accumulated_score_player1 < 0:
            print("Warning: negative accumulated score detected")

        total_check = accumulated_score_player0 + accumulated_score_player1
        print(f"Accumulated total: {total_check}")

        if total_check == 0:
            print("Warning: accumulated total is 0. Setting accumulated scores to NaN to avoid division by zero.")
            accumulated_score_player0 = np.nan
            accumulated_score_player1 = np.nan
        
        # calculate proportion scores
        total_score = accumulated_score_player0 + accumulated_score_player1
        proportion_score_player0 = accumulated_score_player0/total_score
        proportion_score_player1 = 1 - proportion_score_player0 

        print(f"Proportion accumulated total earned by player0: {proportion_score_player0}") 
        

        # store in an array across all sessions
        proportion_scores_all_sessions[trial_list_index, 0] = proportion_score_player0
        proportion_scores_all_sessions[trial_list_index, 1] = proportion_score_player1

    return proportion_scores_all_sessions

# %%
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

# %%
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

# %%
# I think this should be correct
# Check results for bugs

def proportion_score_solo_sessions_df(trial_lists_player0, trial_lists_player1, debug=False):
    ''' Return num_sessions*num_players array for proportion of score
        each player earned in a the final solo session.
        Takes a list of pre-processed trial lists '''
    
    proportion_scores_all_sessions = np.zeros((len(trial_lists_player0), 2))
    
    for trial_list_index in range(len(trial_lists_player0)):
        player0_trial_list = trial_lists_player0[trial_list_index]
        player1_trial_list = trial_lists_player1[trial_list_index]
        score_player0 = 0
        score_player1 = 0


        for player_index, trial_list in enumerate([player0_trial_list, player1_trial_list]):

            # access final trial event log event for the final player scores
            final_trial_this_player = trial_list[-1]
            final_trial_trial_end = final_trial_this_player[final_trial_this_player['eventDescription'] == 'trial end']

            # flexibly index player scores
            score_this_player = final_trial_trial_end[globals.PLAYER_SCORE_DICT[0]['score']].item()
            
            # assign score to player
            if player_index == 0:
                player0_score = score_this_player
            else:
                player1_score = score_this_player

        total_score = player0_score + player1_score
        proportion_score_player0 = player0_score/total_score
        proportion_score_player1 = 1 - proportion_score_player0

        proportion_scores_all_sessions[trial_list_index, 0] = proportion_score_player0
        proportion_scores_all_sessions[trial_list_index, 1] = proportion_score_player1

        if debug:
            # brief verification prints
            print(f"Session {trial_list_index}: player0_score={player0_score}, player1_score={player1_score}, total_score={total_score}")
            print(f"Session {trial_list_index}: proportion_player0={proportion_score_player0:.4f}, proportion_player1={proportion_score_player1:.4f}")

            # simple sanity checks
            if total_score == 0:
                print(f"Warning: Session {trial_list_index} total_score is 0 — proportions may be invalid")
            elif not (0.0 <= proportion_score_player0 <= 1.0):
                print(f"Warning: Session {trial_list_index} proportion_player0 out of bounds: {proportion_score_player0}")

    return proportion_scores_all_sessions

# %%
def is_winner(trial_lists, debug=False):
    ''' Return num_sessions*num_players boolean array
        for the winner of each trial'''
    
    is_winner_all_sessions = np.zeros((len(trial_lists), 2))
    
    for trial_list_index in range(len(trial_lists)):

        # grab the trial list for this session
        trial_list = trial_lists[trial_list_index]
        final_trial = trial_list[-2] # use penultimate trial to avoid end of session issues
        trial_end_event = final_trial[final_trial['eventDescription'] == 'trial end']

        player0_final_score = trial_end_event[globals.PLAYER_SCORE_DICT[0]['score']].item()
        player1_final_score = trial_end_event[globals.PLAYER_SCORE_DICT[1]['score']].item()

        if player0_final_score > player1_final_score:
            is_winner_player0 = 1
            is_winner_player1 = 0
        elif player1_final_score > player0_final_score:
            is_winner_player0 = 0
            is_winner_player1 = 1
        elif player0_final_score == player1_final_score:
            is_winner_player0 = 0.5
            is_winner_player1 = 0.5
            print("Tiebreaker social game")
        else:
            print(f"player0_final_score={player0_final_score}, player1_final_score={player1_final_score}")
            raise ValueError("Unexpected condition in determining winner")
        


        # store in an array across all sessions
        is_winner_all_sessions[trial_list_index, 0] = is_winner_player0
        is_winner_all_sessions[trial_list_index, 1] = is_winner_player1

        if debug:
            # brief print checks for winner and loser scores
            if is_winner_player0 == 1:
                print(f"Session {trial_list_index}: Player 0 wins — winner_score={player0_final_score}, loser_score={player1_final_score}")
            elif is_winner_player1 == 1:
                print(f"Session {trial_list_index}: Player 1 wins — winner_score={player1_final_score}, loser_score={player0_final_score}")
            else:
                # tie case (0.5 / 0.5)
                print(f"Session {trial_list_index}: Tie — both players score={player0_final_score}")

    return is_winner_all_sessions
    


