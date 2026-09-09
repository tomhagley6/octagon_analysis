# %%
import json
import re
import os
import pandas as pd
from datetime import datetime, timedelta
from parse_data.handle_specific_datasets import playerinfo_playerposition_conversion, remove_zero_wall_numbers
import globals

# %%
def match_filename_to_filesystem(json_filename):
    ''' Input a filename string using default Windows filesep ('\')
        Return the filename string with the native os file separator instead '''
    
    json_filename_parts = json_filename.split('\\')
    json_filename_rejoined = os.path.join(*json_filename_parts)

    return json_filename_rejoined
    

# %%
def load_df_from_file(data_folder, json_filename, json_normalise=True):
    ''' Takes a full filepath for a JSON dataset
        Returns a dataframe with any dictionaries flattened into individual columns '''
    
    filepath = data_folder + os.sep + json_filename
    if json_normalise == True:
        with open(filepath) as f:
            print(f"filepath: {filepath}")
            file = json.load(f)
            df = pd.json_normalize(file)
    else:
        with open(filepath) as f:
            df = pd.read_json(f)
    
    return df

# %%
def convert_time_strings(df):
    ''' Covert df time columns into a Datetime format '''
    
    df2 = df.copy()
    df2['timeLocal'] = pd.to_datetime(df2['timeLocal'], format='%H:%M:%S:%f')

    # Use to_timedelta instead as a vectorised function (lambdas are python loops)
    # df['timeApplication'] = df['timeApplication'].apply(lambda x: timedelta(seconds=int(x) + (x - int(x))))
    df2['timeApplication'] = pd.to_numeric(df2['timeApplication']) 
    df2['timeApplication'] = pd.to_timedelta(df2['timeApplication'], unit='s')

    return df2

# %%
def handle_date_sensitive_processing(df, json_filename):
    ''' Compare the date of the file against conditionals for specific dates
        Run any relevant functions to handle data from these date ranges '''
    
    # find date string in filename
    pattern = r'\d{4}-\d{1,2}-\d{1,2}_\d{1,2}-\d{1,2}-\d{1,2}'
    match = re.search(pattern, json_filename)

    # convert date string to datetime
    timestamp_dt = datetime.strptime(match.group(), "%Y-%m-%d_%H-%M-%S")

    # list of all dates with data that needs specific handling
    date_first_experiment = datetime.strptime("2024-09-13", "%Y-%m-%d") # merging playerinfo
    # date_fourth_experiment = datetime.strptime("2024-10-18", "%Y-%m-%d") # removing zeros from wallnums

    # conditional statements based on date of data
    df2 = df.copy()

    # merging playerinfo dictionary into playerposition
    if timestamp_dt < date_first_experiment + timedelta(days=1):
        print(f"Data is from period before {date_first_experiment}"
              "\nRunning dataframe through playerinfo_playerposition_conversion.")
        solo = globals.PLAYER_1_XLOC not in df2.columns
        df2 = playerinfo_playerposition_conversion(df2, solo=solo)
        

    # # currently treating this a standard preprocessing step
    # # removing any zeros from recorded wall numbers and replacing them with nans
    # if timestamp_dt < date_fourth_experiment + timedelta(days=1):
    #     print(f"Data is from period before {date_fourth_experiment}")
    #     df2 = remove_zero_wall_numbers(df2)
    #     print("Running dataframe through remove_zero_wall_numbers")
    

    print("Loading complete.")

    return df2

# %%
import pandas as pd
dataframe = pd.DataFrame({'col1': ['test1', 'test2'], 'col2': [1, 2]})

# %%
# Umbrella function
def loading_pipeline(data_folder, json_filename, json_normalise=True):
    ''' Convert the filepath from Windows to the native OS, load JSON data
        into a pandas dataframe, convert time data to DateTime format, and
        run any functions associated with specific date ranges for the data 
        Return a dataframe '''
    
    json_filename = match_filename_to_filesystem(json_filename)
    df = load_df_from_file(data_folder, json_filename, json_normalise=True)
    df = convert_time_strings(df)
    df = handle_date_sensitive_processing(df, json_filename)

    return df


