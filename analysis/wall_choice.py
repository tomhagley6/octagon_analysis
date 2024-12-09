import parse_data.prepare_data as prepare_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import globals
import data_strings
import time
import data_extraction.get_indices as get_indices


def player_wall_choice_wins_only(trials_list, player_id, debug=False):
    ''' Logic for identifying the player's chosen wall if they won the trial (no inferred choice).
        Returns int array of size len(trials_list) of chosen wall numbers, or of np.nan for
        trials where player_id was not the winner. '''
    
    if debug:
        start_time = time.time()

    winning_player = get_indices.get_trigger_activators(trials_list)
    chosen_walls = get_indices.get_chosen_walls(trials_list)
    current_player_wall_choice = np.zeros(len(trials_list))
    
    # set wall_chosen for each trial to the trail's chosen wall only if this player won the trial. If not, np.nan
    for trial_index in range(len(trials_list)):
        if player_id == winning_player[trial_index]:
            wall_chosen = chosen_walls[trial_index]
        else:
            wall_chosen = np.nan

        current_player_wall_choice[trial_index] = wall_chosen

    # output the time taken for this function
    if debug:
        end_time = time.time()
        print(f"Time taken for player_wall_choice_wins_only (one session for one player) is {end_time-start_time:.2f}")

    return current_player_wall_choice
