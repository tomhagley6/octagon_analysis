# %%
import os
import re
import data_strings 
from collections import defaultdict

# %% [markdown]
# #### Functions to extract filepaths for social and solo sessions from a root directory. Socials are returned taken in session order. Solos are returned in session order, prioritise Host, and then prioritising FirstSolo

# %%
def extract_sort_key(path):
    ''' Extract a sorting key from the folder name in the path.
        The folder name is expected to be in the format 'YYMMDD_SessionNumber'.
        For example, '230101_1' would yield (23, 1, 1, 1).
    '''
    folder_name = os.path.dirname(path).split(os.sep)[-1]  # Get the folder name
    match = re.match(r'(\d{2})(\d{2})(\d{2})_(\d+)', folder_name)
    if match:
        year, month, day, session_num = match.groups()
        return (int(year), int(month), int(day), int(session_num))
    else:
        # If the folder name doesn't match the expected format, return a default key
        return (0, 0, 0, 0)

# %%
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
    
    # Sort the paths based on the folder name (date and session number)
    datafile_paths.sort(key=lambda path: extract_sort_key(path))

    # Check for whitespace in any of the paths and print a warning if found
    for path in datafile_paths:
        if re.search(r'\s', path):
            print(f"Warning: Whitespace found in path: {path}")
            
    # Check for double underscores in any of the paths and print a warning if found
    for path in datafile_paths:
        if re.search(r'__', path):
            print(f"Warning: Double underscores found in path: {path}")
                
    return datafile_paths
    

# %%


# %%
def get_filenames(data_folder=data_strings.DATA_FOLDER):
    ''' Take a root folder, and extracts all social and solo filenames from
        all subfolders. Uses the social filename pseudonym order to order solo
        files, such that the structure priotises session, then host, then first solo.
        Returns a list of all Social session files in the directory, and a list of all
        ordered Solo session files in the directory '''
    
    
    # First list of individual files
    solo_files = get_relative_paths('Solo', data_folder)
    # Second list of social files (with desired pseudonym order)
    social_files = get_relative_paths('Social', data_folder)

    # 1. Create a dict of sessions with nested pseudonyms
    session_order = {}
    for sf in social_files:
        # match the session number and the pseudonym string
        match = re.search(r'(\d+_\d)[\\/].*?_.*?_(.*?)_Social\.json', sf)
        if match:
            session, pseudonyms = match.groups()
            pseudonym_list = pseudonyms.split('_')
            session_order[session] = pseudonym_list

    # 2. Group solo filenames by session and pseudonym 
    # create dictionary structure to initiate any new entry to the dictionary as
    # a default dict, which will contain an empty list
    # Note that the argument to a defaultdict is what that defaultdict initialises for 
    # any new entries
    session_pseudo_files = defaultdict(lambda: defaultdict(list))
    for f in solo_files:
        match = re.search(r'(\d+_\d)[\\/].*?_(\w+?)_(?:First|Second)Solo\.json', f)
        if match:
            session, pseudonym = match.groups()
            session_pseudo_files[session][pseudonym].append(f) # append the entire filename

    # 3. Sort each pseudonym's files by timestamp (ensure FirstSolo always comes first)
    for session in session_pseudo_files:
        for pseudonym in session_pseudo_files[session]:
            session_pseudo_files[session][pseudonym].sort()

    # 4. Reconstruct final ordered list
    ordered_solos_list = []
    for session in session_order:
        for pseudonym in session_order[session]:
            if pseudonym not in session_pseudo_files[session]:
                print(f"Warning: Pseudonym {pseudonym} missing in session {session}")
            # extend the list with each session's pseudonym's files, or extend by an empty list
            # if these do not exist
            ordered_solos_list.extend(session_pseudo_files[session].get(pseudonym, []))



    return social_files, ordered_solos_list


