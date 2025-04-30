#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[ ]:


def get_ordered_indices(values_array):
    ''' Takes an array with numeric or np.nan values.
        Returns an array with a value in each index relating to
        the size order of the numeric values in the input array.
        Where two indices have the same numeric value, the returned array
        will have the same order value.
        Lowest value will be 0, next lowest will be 1, etc.
        Np.nans will carry over. '''
    
    # get mask of nan values
    non_nan_mask = ~np.isnan(values_array)

    # get all non-nan values
    non_nan_values = values_array[non_nan_mask]

    # return sorted unique values, and an array of indices of the unique array that can reconstruct the input array
    # these indices correspond to the order (because np.unique sorts) of the original values, and will be the same for 
    # two identical values (because we are taking unique values)
    unique_values, inverse_indices = np.unique(non_nan_values, return_inverse=True)

    # create return array filled with np.nan
    ordered_indices_array = np.full(values_array.shape, np.nan)

    # place the inverse indices in the corresponding indices
    ordered_indices_array[non_nan_mask] = inverse_indices

    return ordered_indices_array

    

