#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import parse_data.prepare_data as prepare_data
import analysis.wall_visibility_and_choice as wall_visibility_and_choice
import globals
import data_extraction.get_indices as get_indices
from scipy.stats import pearsonr


# ### Paired boxplots of probability of choosing a wall across any number of conditions

# In[ ]:


def boxplot_probability_choose_wall(wall_choice_probabilities, wall_choice_labels, ylabel):
    ''' Plotting function to plot wall choice probability paired data across any number
        of conditions.
        Assumes each datapoint in the pair is from a single subject's session data.
        Takes a list of probabilities (for wall choice) and a list of labels for plotting.
        List arrays must be of shape num_sessions*num_players. '''


    # LVs
    num_datasets = len(wall_choice_probabilities)
    dataset_size = wall_choice_probabilities[0].size

    # Ensure input consistency
    assert len(wall_choice_probabilities) == len(wall_choice_labels), \
        "Number of probabilities and labels must match."

    # Reshape data and create labels
    data = np.concatenate([arr.ravel() for arr in wall_choice_probabilities])
    labels = [np.full(arr.size, label) for arr, label in zip(wall_choice_probabilities, wall_choice_labels)]
    labels = np.concatenate(labels)

    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        "Probability": data,
        "Condition": labels
    })

    # # Generate distinct colors for each individual
    # colors = plt.cm.viridis(np.linspace(0, 1, (first_wall_seen).size))

    # Plot
    plt.figure(figsize=(4*num_datasets, 5))
    sns.boxplot(x="Condition", y="Probability", data=df, palette="Paired", width=.8)
    
    # Draw lines connecting paired data points
    for i in range(dataset_size):
        # print(f"{len(wall_choice_labels)}, {len([dataset.ravel()[i] for dataset in wall_choice_probabilities])}")
        plt.plot(
            wall_choice_labels, # x-coordinates
            [dataset.ravel()[i] for dataset in wall_choice_probabilities], # y-coordinates
            color='k',  # Get color from colormap
            linestyle='-',  # Solid line
            marker='x',  # Marker for the endpoints
            linewidth=1,
            alpha=0.4
        )

    # plt.title("Probability of Choosing First Wall Seen vs. First Wall Seen (Low)")
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.ylim(0.0, 1)  # Set y-axis limits for probabilities
    plt.gca().set_aspect(3)    
    plt.tight_layout()

    # Remove top and bottom spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


# ### Plot ratio of player performance against the ratio of probability of players choosing Low when first visible 

# In[26]:


def plot_performance_against_probability_low_when_first_visible(data_folder, json_filenames_all, correlation_line=True):
    '''Plot the graph of session performance against session probability for players choosing low when it is first visible.
       One data point for each session to avoid replicating data from within a session.
       Data is taken as the ratio player0:player1 for proportion score and for probability of choice '''

    # get probability of choosing the low wall when it is first visible, and the proportion of score within the session
    # these are both recorded per player and session, shape num_sessions*num_players
    probability_low_when_first_visible, _, _ = wall_visibility_and_choice.probability_first_wall_chosen_and_low_multiple_sessions(data_folder, json_filenames_all)
    proportion_scores_all_sessions = get_proportion_scores(data_folder, json_filenames_all)

    print(f"Probability low when first visible: \n {probability_low_when_first_visible}")
    print(f"Proportion of scores for all sessions \n {proportion_scores_all_sessions}")

    # from the above arrays, find the probability of choosing the low wall when it is first visible in the ratio player0:player1
    # also find the proportion of total session score in the ratio player0:player1
    ratio_probability_low_when_first_visible = probability_low_when_first_visible[:,0]/probability_low_when_first_visible[:,1]
    proportion_scores_player_0 = proportion_scores_all_sessions[:,0]/proportion_scores_all_sessions[:,1] # use ratio or just player 1 proportion here?

    x = ratio_probability_low_when_first_visible.ravel()
    y = proportion_scores_player_0.ravel()

    plt.scatter(x, y)

    if correlation_line:
        # Fit a line to the data
        slope, intercept = np.polyfit(x, y, 1)  # 1st-degree polynomial (linear fit)
        line = slope * x + intercept

        # Plot the correlation line
        plt.plot(x, line, color='red', label=f'Fit line: y = {slope:.2f}x + {intercept:.2f}')

    plt.title("Performance in session against the probability of choosing\n the Low wall when the Low wall is the first visible")
    plt.xlabel("Probability")
    plt.ylabel("Performance")

    # Remove top and bottom spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


# In[ ]:


def plot_performance_against_probability_low_when_first_visible(data_folder, json_filenames_all, correlation_line=True):
    '''Plot the graph of session performance against session probability for players choosing low when it is first visible.
       One data point for each session to avoid replicating data from within a session.
       Data is taken as the ratio player0:player1 for proportion score and for probability of choice '''

    # get probability of choosing the low wall when it is first visible, and the proportion of score within the session
    # these are both recorded per player and session, shape num_sessions*num_players
    probability_low_when_first_visible, _, _ = wall_visibility_and_choice.probability_first_wall_chosen_and_low_multiple_sessions_social(data_folder, json_filenames_all)
    proportion_scores_all_sessions = get_proportion_scores(data_folder, json_filenames_all)

    print(f"Probability low when first visible: \n {probability_low_when_first_visible}")
    print(f"Proportion of scores for all sessions \n {proportion_scores_all_sessions}")

    # from the above arrays, find the probability of choosing the low wall when it is first visible in the ratio player0:player1
    # also find the proportion of total session score in the ratio player0:player1
    ratio_probability_low_when_first_visible = probability_low_when_first_visible[:,0]/probability_low_when_first_visible[:,1]
    proportion_scores_player_0 = proportion_scores_all_sessions[:,0]/proportion_scores_all_sessions[:,1] # use ratio or just player 1 proportion here?

    # x = ratio_probability_low_when_first_visible.ravel()
    x = ratio_probability_low_when_first_visible.ravel()
    y = proportion_scores_all_sessions[:,0].ravel()

    plt.scatter(x, y)

    if correlation_line:
        # Fit a line to the data
        slope, intercept = np.polyfit(x, y, 1)  # 1st-degree polynomial (linear fit)
        line = slope * x + intercept

        # Plot the correlation line
        plt.plot(x, line, color='red', label=f'Fit line: y = {slope:.2f}x + {intercept:.2f}')

    plt.title("Performance in session against the probability of choosing\n the Low wall when the Low wall is the first visible")
    plt.xlabel("Probability")
    plt.ylabel("Performance")

    # Remove top and bottom spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


# In[ ]:


def plot_performance_against_probability_low_when_first_visible_df(trial_lists, inferred_choice=False, correlation_line=True, print_correlation=True):
    '''Plot the graph of session performance against session probability for players choosing low when it is first visible.
       One data point for each session to avoid replicating data from within a session.
       Data is taken as the ratio player0:player1 for probability of choice, and player0 value proportion score '''

    # get probability of choosing the low wall when it is first visible, and the proportion of score within the session
    # these are both recorded per player and session, shape num_sessions*num_players
    probability_low_when_first_visible, _, _ = wall_visibility_and_choice.probability_first_wall_chosen_and_low_multiple_sessions_social(trial_lists, inferred_choice=inferred_choice)
    proportion_scores_all_sessions = get_proportion_scores_df(trial_lists)

    print(f"Probability low when first visible: \n {probability_low_when_first_visible}")
    print(f"Proportion of scores for all sessions \n {proportion_scores_all_sessions}")

    # from the above arrays, find the probability of choosing the low wall when it is first visible in the ratio player0:player1
    # also find the proportion of total session score in the ratio player0:player1
    ratio_probability_low_when_first_visible = probability_low_when_first_visible[:,0]/probability_low_when_first_visible[:,1]
    proportion_scores_player_0 = proportion_scores_all_sessions[:,0]/proportion_scores_all_sessions[:,1] # use ratio or just player 1 proportion here?

    # x = ratio_probability_low_when_first_visible.ravel()
    x = ratio_probability_low_when_first_visible.ravel()
    y = proportion_scores_all_sessions[:,0].ravel()

    plt.scatter(x, y)

    if correlation_line:
        # Fit a line to the data
        slope, intercept = np.polyfit(x, y, 1)  # 1st-degree polynomial (linear fit)
        line = slope * x + intercept

        # Plot the correlation line
        plt.plot(x, line, color='red', label=f'Fit line: y = {slope:.2f}x + {intercept:.2f}')

    if print_correlation:
        corr_coeff_pearsonr, pval_pearsonr = pearsonr(x,y)
        print(f"Pearson correlation coefficient is: {corr_coeff_pearsonr}")
        print(f"P-value is: {pval_pearsonr}")

    plt.title("Performance in session against the probability of choosing\n the Low wall when the Low wall is the first visible")
    plt.xlabel("Probability")
    plt.ylabel("Performance")

    # Remove top and bottom spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


# In[29]:


# helper function for plot_performance_against_probability_low_when_first_visible
def get_proportion_scores(data_folder, json_filenames_all):
    ''' Returns a float array of shape num_session*num_players with the proportion of
        total session score attributed to each player
        Takes the data folder path string and list of filenames for JSON datasets '''

    # go through every session and find the proportion of score in the session that players achieved
    proportion_scores_all_sessions = np.zeros((len(json_filenames_all), 2))
    
    for json_filenames_index in range(len(json_filenames_all)):
        # get data for session this loop index
        json_filenames = json_filenames_all[json_filenames_index]
        print(data_folder + os.sep + json_filenames)
        _, trials_list = prepare_data.prepare_data(data_folder, [json_filenames])

        # identify the overall session score from the final trial end log event
        final_trial = trials_list[-1]
        final_trial_trial_end = final_trial[final_trial['eventDescription'] == 'trial end']
        
        player0_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[0]['score']].item()
        player1_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[1]['score']].item()
        total_score = player0_score + player1_score
        
        # find the proportion of the total session score attributed to each player
        proportion_score_player0 = player0_score/total_score
        proportion_score_player1 = player1_score/total_score

        proportion_scores_all_sessions[json_filenames_index, 0] = proportion_score_player0
        proportion_scores_all_sessions[json_filenames_index, 1] = proportion_score_player1

    return proportion_scores_all_sessions


# In[30]:


# helper function for plot_performance_against_probability_low_when_first_visible
def get_proportion_scores_df(trial_lists):
    ''' Returns a float array of shape num_session*num_players with the proportion of
        total session score attributed to each player
        Takes the data folder path string and list of filenames for JSON datasets '''

    # go through every session and find the proportion of score in the session that players achieved
    proportion_scores_all_sessions = np.zeros((len(trial_lists), 2))
    
    for trial_list_idx in range(len(trial_lists)):
        # get data for session this loop index
        trial_list = trial_lists[trial_list_idx]

        # identify the overall session score from the final trial end log event
        final_trial = trial_list[-1]
        final_trial_trial_end = final_trial[final_trial['eventDescription'] == 'trial end']
        
        player0_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[0]['score']].item()
        player1_score = final_trial_trial_end[globals.PLAYER_SCORE_DICT[1]['score']].item()
        total_score = player0_score + player1_score
        
        # find the proportion of the total session score attributed to each player
        proportion_score_player0 = player0_score/total_score
        proportion_score_player1 = player1_score/total_score

        proportion_scores_all_sessions[trial_list_idx, 0] = proportion_score_player0
        proportion_scores_all_sessions[trial_list_idx, 1] = proportion_score_player1

    return proportion_scores_all_sessions


# ### Plot the probability of choosing High compared between solo and social conditions (combined and separated solo)

# In[ ]:


def plot_probability_choose_high_solo_social(social_p_choose_high, *solo_p_choose_high, black_lines=False):
    ''' Plot paired data line graph of the probability of choosing High across
        solo and social conditions. 
        Takes a num_sessions*num_players social array and a 1D solo array of the same size.
        Depending on how many solo arrays are passed, will plot combined or separated solo graphs.
        Drops points if they are nan (subject to low n in probability calculation). '''
    
    # convert social array to a single dimension for plotting
    social_p_choose_high = social_p_choose_high.ravel()

    # Number of individuals
    individuals = np.arange(len(social_p_choose_high))

    # Generate distinct colors for each individual
    if black_lines:
        colors = ['k']*len(individuals)
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(individuals)))

    # Plotting
    if len(solo_p_choose_high) == 1: # plot for combined solo data
        
        solo_p_choose_high = solo_p_choose_high[0]

        plt.figure(figsize=(4, 5))

        # Plot lines for each individual
        for i in individuals:

            # check for any nan values in probabilities. Do not plot it.
            probabilities = np.array([solo_p_choose_high[i], social_p_choose_high[i]])
            conditions = np.array([0,1])
            nan_mask = np.isnan(probabilities)
            if np.any(nan_mask): # if nan value present
                print(f"NaN value in probabilities: {probabilities}. Dropping this point from the combined plot.")
            
            plt.plot(conditions[~nan_mask], probabilities[~nan_mask], 
                    marker='o', linestyle='-', color=colors[i], alpha=0.7)

        plt.plot([0,1], [np.nanmean(solo_p_choose_high), np.nanmean(social_p_choose_high)],
                            marker='x', color='red', label='Average', linewidth=2, linestyle='--')

        plt.ylabel('P(Choose High)')
        plt.xticks([0, 1], [' Combined Solo', 'Social'])
        plt.ylim(0, 1.1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()

    elif len(solo_p_choose_high) == 2: # plot for separated pre- and post solo data

        solo_first_session_p_choose_high = solo_p_choose_high[0]
        solo_second_session_p_choose_high = solo_p_choose_high[1]
        
        # Plotting
        plt.figure(figsize=(6, 5))

        # Plot lines for each individual
        for i in individuals:

            # check for any nan values in probabilities. Do not plot it.
            probabilities = np.array([solo_first_session_p_choose_high[i], social_p_choose_high[i], solo_second_session_p_choose_high[i]])
            conditions = np.array([0,1,2])
            nan_mask = np.isnan(probabilities)

            if np.any(nan_mask): # if nan value present
                print(f"NaN value in probabilities: {probabilities}. Dropping this point from the separated plot.")

            plt.plot(conditions[~nan_mask], probabilities[~nan_mask], 
                    marker='o', linestyle='-', color=colors[i], alpha=0.7)

        plt.plot([0,1,2], [np.nanmean(solo_first_session_p_choose_high), np.nanmean(social_p_choose_high), np.nanmean(solo_second_session_p_choose_high)],
                            marker='x', color='red', label='Average', linewidth=2, linestyle='--')

        plt.ylabel('P(Choose High)')
        plt.xticks([0, 1, 2], ['First Solo', 'Social', 'Second Solo'])
        plt.ylim(0, 1.1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()



# In[33]:


# helper function for plot_probability_choose_high_solo_social
def get_probability_chose_high_social_df(trial_list, trial_type=globals.HIGH_LOW, wall_sep=None):
    ''' Find the probability that each player chose High in a social context.
        Optionally specify the trial type and wall separation type to use.
        This does not include inferred choices.
        Assumes one session only. '''


    # filter trial list to include HighLow trials only
    if trial_type is not None:
        trial_list_indices = get_indices.get_trials_trialtype(trial_list, trial_type=trial_type)
        trial_list = [trial_list[i] for i in trial_list_indices]
    # print(f"len trial list = {len(trial_list)}")

    # filter trial list to include specific wall separation
    if wall_sep is not None:
        trial_list_indices =  get_indices.get_trials_with_wall_sep(trial_list, wall_sep=wall_sep)
        trial_list = [trial_list[i] for i in trial_list_indices]

    # find the high wall trials and the indices where each player won
    high_wall_chosen = get_indices.was_high_wall_chosen(trial_list)
    # print(f"high_wall_chosen = {high_wall_chosen}")
    player0_win_indices = get_indices.get_player_win_indices(trial_list, player_id=0)
    # print(f"player0_win_indices = {player0_win_indices}")
    player1_win_indices = get_indices.get_player_win_indices(trial_list, player_id=1)
    # print(f"player1_win_indices = {player1_win_indices}")

    # create an array of size player_win_indices that is True where this win was a High wall choice 
    player0_wins_high = np.zeros(player0_win_indices.size)
    for i in range(player0_win_indices.size):
        trial_idx = player0_win_indices[i]
        player0_wins_high[i] = True if high_wall_chosen[trial_idx] else False

    player1_wins_high = np.zeros(player1_win_indices.size)
    for i in range(player1_win_indices.size):
        trial_idx = player1_win_indices[i]
        player1_wins_high[i] = True if high_wall_chosen[trial_idx] else False

    probability_player0_choose_high = player0_wins_high[player0_wins_high == True].size/player0_wins_high.size
    probability_player1_choose_high = player1_wins_high[player1_wins_high == True].size/player1_wins_high.size

    return probability_player0_choose_high, probability_player1_choose_high


# In[ ]:


# helper function for plot_probability_choose_high_solo_social
def get_probability_chose_high_solo_df(trial_list, trial_type=globals.HIGH_LOW, wall_sep=None, cut_trials=10, data_size_cutoff=4):
    ''' Find the probability that the player chose High in a solo context
        Takes a data folder string and JSON filename.
        Optionally specify the trial and wall separation type to use.
        Cut the first cut_trials trials to reduce effect of learning controls/associations. 
        Return np.nan if filtering and cut_trials leaves the trial list at size < dat_size_cutoff'''
    
    # cut first cut_trials trials (learning controls/associations)
    trial_list = trial_list[cut_trials:]

    # filter trial list to include HighLow trials only
    if trial_type is not None:
        trial_list_indices = get_indices.get_trials_trialtype(trial_list, trial_type=trial_type)
        trial_list = [trial_list[i] for i in trial_list_indices]


    # filter trial list to include specific wall separation
    if wall_sep is not None:
        trial_list_indices =  get_indices.get_trials_with_wall_sep(trial_list, wall_sep=wall_sep)
        trial_list = [trial_list[i] for i in trial_list_indices]
    
    high_wall_chosen = get_indices.was_high_wall_chosen(trial_list)

    # if calling this function leaves too few relevant trials, return np.nan
    if trial_list_indices.size <= data_size_cutoff:
        return np.nan
    else:
        probability_choose_high = high_wall_chosen[high_wall_chosen == True].size/trial_list_indices.size
        return probability_choose_high


# In[ ]:


# helper function for plot_probability_choose_high_solo_social
def get_probability_chose_high_solo_social_all_sessions_df(trial_lists_solo, trial_lists_social, wall_sep=None, trial_type=globals.HIGH_LOW, cut_solo_trials=10):
    ''' Get probabilities of choosing the High wall for each participant for each session, and split by social and solo.
        Takes a list of trial lists for solo sessions, and for social sessions.
        Assumes the solo trial list is complete, and that second sessions follow directly from first sessions.
        Returns 4 floats: P(choose High) in social, combined solo, first solo session, second solo session.
        These floats may be np.nan if low n in the probability calculation.'''


    # 1. social
    # loop through all social sessions
    probability_choose_high_social_array = np.zeros((len(trial_lists_social), 2))
    for trial_list_idx in range(len(trial_lists_social)):

        # get the dataframe for this session
        trial_list = trial_lists_social[trial_list_idx]

        # find the probability of choosing high for each player
        probability_player0_choose_high, probability_player1_choose_high = get_probability_chose_high_social_df(trial_list,
                                                                                                        trial_type=trial_type,
                                                                                                        wall_sep=wall_sep)

        # add this to the sessions array
        probability_choose_high_social_array[trial_list_idx,:] = [probability_player0_choose_high, probability_player1_choose_high]
    
    # 2. solo combined
    # loop through all solo sessions
    # get solo choice data for combined pre- and post-
    probability_choose_high_solo_array = np.zeros((int(len(trial_lists_solo)/2)))
    for trial_list_idx in range(0, len(trial_lists_solo), 2):

        # concatenate the trial lists for the 2 solos of this session
        trial_list_combined = trial_lists_solo[trial_list_idx] + trial_lists_solo[trial_list_idx + 1]

        # find the probability of choosing high for each player
        probability_choose_high = get_probability_chose_high_solo_df(trial_list_combined, trial_type=trial_type, wall_sep=wall_sep, cut_trials=cut_solo_trials)

        # add this to the sessions array
        probability_choose_high_solo_array[int(trial_list_idx/2)] = probability_choose_high

    # 3. solo separated
    # loop through all solo sessions
    # get solo choice data for separated pre- and post
    probability_choose_high_solo_array_separated_sessions = np.zeros(int(len(trial_lists_solo)))
    for trial_list_idx in range(0, len(trial_lists_solo)):

        # get the dataframe for this session
        trial_list = trial_lists_solo[trial_list_idx]

        # find the probability of choosing high for each player
        probability_choose_high = get_probability_chose_high_solo_df(trial_list, trial_type=trial_type, wall_sep=wall_sep, cut_trials=cut_solo_trials)

        # add this to the sessions array
        probability_choose_high_solo_array_separated_sessions[int(trial_list_idx)] = probability_choose_high

    probability_choose_high_solo_array_first_session = probability_choose_high_solo_array_separated_sessions[0::2]
    probability_choose_high_solo_array_second_session = probability_choose_high_solo_array_separated_sessions[1::2]

    
    return (probability_choose_high_social_array, probability_choose_high_solo_array,
            probability_choose_high_solo_array_first_session, probability_choose_high_solo_array_second_session)


# ### Sandbox

# In[38]:


# Data arrays
first_wall_seen = np.array([
    [0.76923077, 0.75490196],
    [0.78378378, 0.67088608],
    [0.609375, 0.85714286],
    [0.69911504, 0.78014184]
])

first_wall_seen_low = np.array([
    [0.75, 0.73913043],
    [0.86363636, 0.53846154],
    [0.46428571, 0.82051282],
    [0.70689655, 0.671875]
])

test_array_please_ignore = np.array([
    [0.55, 0.34],
    [0.34, 0.453],
    [0.25, 0.67],
    [0.76, 0.46]
])


# In[39]:


wall_choice_probabilities = [first_wall_seen, first_wall_seen_low, test_array_please_ignore]
wall_choice_labels = ['First Wall Seen', 'First Wall Seen (Low)', 'test label please ignore']

# boxplot_probability_choose_wall(wall_choice_probabilities, wall_choice_labels)


# In[40]:


data_folder = r'D:\Users\Tom\OneDrive\PhD\SWC\data' # desktop home
json_filenames_all = [r'first_experiments_2409\240913\2024-09-13_11-31-00_YansuJerrySocial.json',
               r'second_experiments_2409\240927\2024-09-27_14-25-20_SaraEmilySocial.json',
               r'third_experiments_2410\241017\2024-10-17_14-28-40_ShamirAbigailSocial.json',
               r'fourth_experiments_2410\241017\2024-10-17_16-41-38_ZimoElsaSocial.json']


# In[41]:


# plot_performance_against_probability_low_when_first_visible(data_folder, json_filenames_all, correlation_line=True)

