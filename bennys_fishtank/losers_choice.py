import numpy as np
import pandas as pd
from parse_data import prepare_data
from trajectory_direction import cosine_similarity_throughout_trajectory
import globals

data_folder = '/Users/benny/Desktop/MSc/Project/Git/repos/octagon_analysis/Json data'
json_filenames = ['2024-09-13_11-31-00_YansuJerrySocial.json']

df, trial_list = prepare_data.prepare_data(data_folder, json_filenames)

all_alignment_values = []
for trial_index in range(len(trial_list)):
    this_trial = trial_list[trial_index]
    alignment_values = cosine_similarity_throughout_trajectory(this_trial, window_size=10, num_walls=8, calculate_thetas=False)
    all_alignment_values.append(alignment_values)

# Slice resulting array to include only second half of the trial's time points
def split_alignment_values(all_alignment_values):
    half = len(all_alignment_values) // 2
    return all_alignment_values[half:]

split_values = [split_alignment_values(values) for values in all_alignment_values]

# Filter resulting array to include only active walls
active_walls = df.loc[slice_onset_idx, ['data.wall1', 'data.wall2']].values
print("Active Walls (as integers):", active_walls)
      
alignment_values_for_active_walls = []
for walls in active_walls:
    for wall in walls:
        wall_index = int(wall)
        new_alignment_values = split_values[wall_index]
        alignment_values_for_active_walls.append(new_alignment_values)

# Get alignment value for each parcel to wall 1 and 2
alignment_values_for_wall_1 = []
alignment_values_for_wall_2 = []

# Unsure whether I can split evenly, resolve by iterating up to last even index, however this ignores last value
for i in range(0, len(alignment_values_for_active_walls) - 1, 2):
    alignment_values_for_wall_1.append(alignment_values_for_active_walls[i])     
    alignment_values_for_wall_2.append(alignment_values_for_active_walls[i + 1])

if len(alignment_values_for_active_walls) % 2 != 0:
    print("Warning: Odd number of values.")

# Make lists for subsequent
all_aligned_to_w1 = []
all_aligned_to_w2 = []

# Ensure that alignment values difference exceeds interval [-0.3, 0.3] and determine alignment
for trial_index in range(len(trial_list)):
    this_trial = trial_list[trial_index]
    
    difference = np.subtract(alignment_values_for_wall_1, alignment_values_for_wall_2) 
   
    # Boolean mask for differences that exceed the specified intervals
    sufficiently_different_mask = (difference < -0.3) | (difference > 0.3)
    
    # Filter alignment values based on mask
    alignment_w1 = alignment_values_for_wall_1[sufficiently_different_mask]
    alignment_w2 = alignment_values_for_wall_2[sufficiently_different_mask]

    # Check values
    print(f"Trial {trial_index}: Wall 1 Alignment Values:", alignment_w1)
    print(f"Trial {trial_index}: Wall 2 Alignment Values:", alignment_w2)
    
    difference = np.subtract(alignment_w1, alignment_w2) 
    
    # Boolean mask for differences above and below zero
    assess_alignment_to_w1 = difference > 0
    assess_alignment_to_w2 = difference < 0
    
    # Filter alignment values based on mask
    aligned_to_w1 = alignment_w1[assess_alignment_to_w1]
    aligned_to_w2 = alignment_w2[assess_alignment_to_w2]

    # Extend previously created lists
    all_aligned_to_w1.extend(aligned_to_w1)
    all_aligned_to_w2.extend(aligned_to_w2)

    # Check values
    print(f"Trial {trial_index}: Aligned to Wall 1 Values:", aligned_to_w1)
    print(f"Trial {trial_index}: Aligned to Wall 2 Values:", aligned_to_w2)
    
# Get number of 'aligned to w1' and 'aligned to w2' values and calculate n/N_total
# If n/N_total >= 0.8 for ratio_w1/w2 determine loser's choice as w1/2
n_aligned_to_w1 = len(all_aligned_to_w1)
n_aligned_to_w2 = len(all_aligned_to_w2)
N_total = n_aligned_to_w1 + n_aligned_to_w2

# Obtain loser's choice
if N_total > 0:
    ratio_w1 = n_aligned_to_w1/N_total
    ratio_w2 = n_aligned_to_w2/N_total
    
    if ratio_w1 >= 0.8:
      print("Loser's choice: Wall 1")
    if ratio_w2 >= 0.8:
      print("Loser's choice: Wall 2")
    

