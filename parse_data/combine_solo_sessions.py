# %% [markdown]
# ### Combine pre- and post-solo sessions, dropping the first 5 trails (to reduce learning-associated effects)

# %%
def combine_solo_sessions(trial_lists_solo, drop_first_n_trials=5):
    ''' Combine each player's two solo sessions, cutting the opening trials of the first.
        Takes the per-session solo trial lists and the number of trials to cut.

        The first (pre-social) solo is expected at even indices and the second (post-social) solo at
        odd indices, with consecutive indices belonging to the same player. That is the order
        identify_filepaths.get_filenames returns: session, then player, then FirstSolo before
        SecondSolo.

        Only the first solo is cut, because it is the player's first exposure to the task and those
        choices are confounded by learning the controls. The second solo is never cut.

        Returns one combined trial list per player, so 404 solo sessions become 202 entries.
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


