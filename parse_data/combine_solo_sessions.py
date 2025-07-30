# %% [markdown]
# ### Combine pre- and post-solo sessions, dropping the first 5 trails (to reduce learning-associated effects)

# %%
def combine_solo_sessions(trial_lists_solo, drop_first_n_trials=5):
    ''' Combine pre- and post- social solo sessions, removing 5 trials from each pre 
    Args:
        trial_lists_solo (list): List of lists containing the  trials for each solo session.
        Post-social solo sessions are expected to be at even indices, and pre-social solo sessions at odd indices,
        with contiguous indices representing the same individual.
        Returns:
        trial_lists_combined_solo (list): List of lists containing the combined trials for each individual,
        with the first 5 trials of the pre-social solo sessions removed.
    '''
    # create a list of combined pre- and post- social solo sessions, removing 5 trials from each pre
    trial_lists_combined_solo = []
    cut_trials = drop_first_n_trials
    for i in range(0,len(trial_lists_solo), 2): # iterate over each individual
        # get the trial lists for both solo sessions
        trial_list_first_solo = trial_lists_solo[i]
        trial_list_second_solo = trial_lists_solo[i + 1]

        # cut first cut_trials trials (learning controls/associations) from the first solo
        trial_list_first_solo = trial_list_first_solo[cut_trials:]

        # combine trial lists from the first and second solo sessions (the current and consecutive index)
        trial_list = trial_list_first_solo + trial_list_second_solo

        trial_lists_combined_solo.append(trial_list)

    return trial_lists_combined_solo


