#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# condition = np.array([4,4,4,4])
# chosen = np.array([1,2,3,4])
# data = {
#     'Condition' : condition,
#     'Chosen' : chosen
# }

# plot_counts_condition_vs_chosen(data)


# In[13]:


def plot_counts_condition_vs_chosen(data, title='Conditions fulfilled and then chosen across individuals', define_ylim=None):
    ''' Takes a data dictionary containing 'Condition' and 'Chosen' fields
        Plots a stacked bar plot of these 2 counts for each individual '''
    



    # handle data
    data['Individuals'] = np.arange(data['Condition'].size)
    individuals = data['Individuals']
    condition = data['Condition']
    chosen = data['Chosen']
    not_chosen = condition - chosen  # find the 'not chosen' counts

    # Plot setup
    x = individuals # X-axis positions
    width = 0.6  # Bar width

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot 'not chosen' bar
    ax.bar(x, not_chosen, width, label='Not chosen', color='lightgray')

    # plot 'chosen' bar
    ax.bar(x, chosen, width, bottom=not_chosen, label='chosen', color='dodgerblue', hatch='//')

    # customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(individuals)
    ax.set_xlabel('Players')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # set ylim based on a 2-value tuple if provided by the input
    if define_ylim:
        ax.set_ylim(define_ylim)

    # display plot
    # plt.tight_layout()

    plt.show()

