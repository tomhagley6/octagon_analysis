#!/usr/bin/env python
# coding: utf-8

# In[41]:


import os
import re
import data_strings


# In[ ]:


def get_relative_paths(match_string, data_folder=data_strings.DATA_FOLDER):
    ''' Find all relative paths for files that contain match_string in
        subfolders of data_folder. Store these filenames in a list '''

    datafile_paths = []

    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        
        # check that the item is a directory
        if os.path.isdir(subfolder_path):
            
            # for each subfolder, check for .json files that contain the matched string
            for filename in os.listdir(subfolder_path):
                
                if filename.endswith('.json') and match_string in filename:
                    # add each relative filepath to the list
                    relative_path = os.path.join(subfolder, filename)
                    datafile_paths.append(relative_path)
                
    return datafile_paths
    


# In[ ]:


def get_relative_paths_regex(match_string, data_folder=data_strings.DATA_FOLDER):
    ''' Find all relative paths for files that contain match_string in
        subfolders of data_folder. Store these filenames in a dictionary
        with the filename pseudonym as a key. (Useful for combining different
        files from the same player, e.g. solos) '''
    
    datafile_paths = {}

    # regex for identifier pseudonym 
    pattern = re.compile(r'([A-Za-z]{2}\d{2})')

    # check that the item is a directory
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        
        # check that the item is a directory
        if os.path.isdir(subfolder_path):
            
            # for each subfolder, check for .json files that contain the matched string
            for filename in os.listdir(subfolder_path):
                
                if filename.endswith('.json') and match_string in filename:
                    match = pattern.search(filename)
                    if match:
                        pseudonym = match.group(1) 
                        full_path = os.path.join(subfolder, filename)

                        if pseudonym not in datafile_paths:
                            datafile_paths[pseudonym] = []
                            datafile_paths[pseudonym].append(full_path)

    return datafile_paths


# In[ ]:


def match_orders_solo(first_solos, second_solos, data_folder=data_strings.DATA_FOLDER):
    '''Ensures that the first solo sessions are correctly ordered w.r.t second solos.
       Takes 2 dictionaries, with key:value as pseudonym:filename, one for each type of solo
       session.'''

    # sort identifiers based on second_solos
    ordered_pseudonyms = sorted(second_solos.keys(), key=lambda pseudo: second_solos[pseudo])

    matched_first_solos = []
    matched_second_solos = []

    # append all values of first_solos to a list in the order of second_solos
    for pseudonym in ordered_pseudonyms:
        if pseudonym in first_solos and len(first_solos[pseudonym]) == 1:
            matched_first_solos.extend(first_solos[pseudonym])

    # also convert the second solo session filenames into a list of strings
    for pseudonym in second_solos.keys():
        matched_second_solos.extend(second_solos[pseudonym])


    return matched_first_solos, matched_second_solos


# In[ ]:


def get_all_relative_paths(data_folder=data_strings.DATA_FOLDER):
    social = get_relative_paths('Social', data_folder=data_folder)
    first_solo = get_relative_paths_regex('FirstSolo', data_folder=data_folder)
    second_solo = get_relative_paths_regex('SecondSolo', data_folder=data_folder)
    ordered_first_solo, ordered_second_solo = match_orders_solo(first_solo, second_solo)

    return social, ordered_first_solo, ordered_second_solo

