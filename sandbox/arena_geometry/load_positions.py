import sys, os
sys.path.insert(0, '/home/tom/repos/octagon_analysis')
import numpy as np, pandas as pd
import data_strings, globals
from parse_data import prepare_sim_data

OUT = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(data_strings.SIM_DATA_FOLDER, '2607', '260723')
MODELS = [1, 23, 33, 51]

rows = []
for m in MODELS:
    df, trial_list = prepare_sim_data.prepare_single_model_data(m, sim_data_folder=FOLDER)
    # drop the first logged trial (trial-drop convention)
    d = pd.concat(trial_list[1:])
    for pid in (0, 1):
        x = d[globals.PLAYER_LOC_DICT[pid]['xloc']].to_numpy(float)
        z = d[globals.PLAYER_LOC_DICT[pid]['yloc']].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(z)
        rows.append(pd.DataFrame(dict(model=m, player=pid, x=x[ok], z=z[ok])))
    print(m, len(trial_list) - 1, 'trials', len(d), 'rows')

pos = pd.concat(rows, ignore_index=True)
pos.to_pickle(os.path.join(OUT, 'positions_260723_peripheral.pkl'))
print(pos.groupby(['model', 'player']).size())
print(pos[['x', 'z']].describe())
