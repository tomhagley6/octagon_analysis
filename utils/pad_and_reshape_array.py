#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


def pad_and_reshape_array(array, n_cols, pad_val=np.nan):
    ''' Take an array of any size, and a specified number of columns for the output
        array.
        Return the same array, padded with np.nan (default) so that the array is full
        to the specified number of columns '''
    
    if array.size % n_cols != 0:
        padding = n_cols - (array.size % n_cols)
        array_padded = np.pad(array, (0,padding), mode='constant', constant_values=np.nan)

    array_reshaped = array_padded.reshape((-1,n_cols))

    return array_reshaped

