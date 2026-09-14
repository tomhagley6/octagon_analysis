# %%
import os
import pickle
import data_strings

# %% [markdown]
# ### data saving and loading

# %%
analysis_dir = data_strings.DATAFRAME_ROOT + os.sep + data_strings.DATAFRAME_DIR

# %%
def save_data(data_to_save, analysis_filename, analysis_dir=analysis_dir, overwrite=False):

    path = os.path.join(analysis_dir, analysis_filename + '.pkl')
    # 'xb' refuses to replace an existing file, which protects the session pickles from
    # being overwritten by accident. Cached analysis results are rewritten, so they pass overwrite
    with open(path, 'wb' if overwrite else 'xb') as f:
        pickle.dump(data_to_save, f)

def load_data(analysis_filename, analysis_dir=analysis_dir):
    path = os.path.join(analysis_dir, analysis_filename + '.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


# %% [markdown]
# ### caching slow analysis results
#
# The results themselves are small, a few arrays each, but they take minutes to compute. Cache them
# outside the repo so figures can be tweaked without rerunning the analysis.

# %%
cache_dir = analysis_dir + os.sep + data_strings.ANALYSIS_CACHE_DIR


# %%
def cached(analysis_filename, compute, recompute=False, stamp=None, analysis_dir=cache_dir):
    ''' Return a saved analysis result, computing and saving it if it is not already there.
        Takes a name, a function of no arguments that does the work, and a recompute flag.

        stamp is anything cheap that describes the inputs, such as the number of sessions. It is
        saved with the result and checked on load, so a cache built from different data is
        recomputed instead of silently returned. '''

    os.makedirs(analysis_dir, exist_ok=True)

    # use the saved result unless it is being forced, missing, or built from different inputs
    if not recompute:
        try:
            entry = load_data(analysis_filename, analysis_dir)
        except FileNotFoundError:
            print(f"{analysis_filename}: not cached, computing")
        else:
            if entry['stamp'] == stamp:
                print(f"{analysis_filename}: loaded from cache")
                return entry['result']
            print(f"{analysis_filename}: cached under {entry['stamp']}, now {stamp}. Recomputing")

    result = compute()
    save_data({'stamp': stamp, 'result': result}, analysis_filename, analysis_dir, overwrite=True)
    print(f"{analysis_filename}: computed and cached")

    return result


# %%
def clear_cache(analysis_dir=cache_dir):
    ''' Delete every cached analysis result. Takes the cache directory. '''

    if not os.path.isdir(analysis_dir):
        print("no cache to clear")
        return

    removed = [f for f in sorted(os.listdir(analysis_dir)) if f.endswith('.pkl')]
    for filename in removed:
        os.remove(os.path.join(analysis_dir, filename))

    print(f"cleared {len(removed)} cached results from {analysis_dir}")



