# %%
import os
import pickle
import data_strings

# %% [markdown]
# ### data saving and loading

# %%
analysis_dir = data_strings.DATAFRAME_ROOT + os.sep + data_strings.DATAFRAME_DIR

# %%
def save_data(data_to_save, analysis_filename, analysis_dir=analysis_dir): 

    path = os.path.join(analysis_dir, analysis_filename + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data_to_save, f)

def load_data(analysis_filename, analysis_dir=analysis_dir):
    path = os.path.join(analysis_dir, analysis_filename + '.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)



