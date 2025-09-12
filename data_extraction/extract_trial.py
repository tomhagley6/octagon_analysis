#!/usr/bin/env python
# coding: utf-8

# In[1]:


def extract_trial(trial=None, trial_list=None, trial_index=None):
    ''' isolate trial '''
    
    if not trial is None:
        this_trial = trial
    elif not trial_list is None:
        this_trial = trial_list[trial_index]
    else:
        raise ValueError("a list of trials and the chosen index must be given, or the trial itself must be given, but not neither.")

    return this_trial

