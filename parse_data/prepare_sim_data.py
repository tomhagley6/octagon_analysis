#!/usr/bin/env python
# coding: utf-8

# Umbrella for turning RL agent simulation results into dataframes + trial lists.
#
# This mirrors parse_data.prepare_data for the human task, but:
#   - resolves model subfolders via parse_data.identify_sim_filepaths, and
#   - uses parse_data.preprocess_sim (which also fills the extra trial-end reward fields).
# Loading and trial-splitting reuse the existing human-pipeline functions unchanged.

import data_strings
import parse_data.loading as loading
import parse_data.preprocess_sim as preprocess_sim
import parse_data.split_session_by_trial as split_session_by_trial
import parse_data.identify_sim_filepaths as identify_sim_filepaths


def prepare_single_model_data(model_number, sim_data_folder=data_strings.SIM_DATA_FOLDER,
                              drop_trial_zero=True):
    ''' Load and preprocess a single model's inference run.
        Returns: full dataframe, list of trials. '''

    data_folder, json_filename = identify_sim_filepaths.get_model_filepaths(sim_data_folder)[model_number]

    # reuse the human loading pipeline (json_normalize captures the extra trial-end fields)
    df = loading.loading_pipeline(data_folder, json_filename)

    # standard preprocessing + fill the simulation-only trial-end reward fields
    df = preprocess_sim.standard_preprocessing_sim(df)

    trial_list = split_session_by_trial.split_session_by_trial(df, drop_trial_zero=drop_trial_zero)

    return df, trial_list


def prepare_models_data(model_numbers=None, sim_data_folder=data_strings.SIM_DATA_FOLDER,
                        drop_trial_zero=True):
    ''' Load and preprocess multiple models, keeping each model separate.

        If model_numbers is None, every model found in sim_data_folder is loaded.
        Returns a dict mapping model_number -> (dataframe, trial_list). '''

    if model_numbers is None:
        model_numbers = [num for num, _ in identify_sim_filepaths.get_model_folders(sim_data_folder)]

    models_data = {}
    for model_number in model_numbers:
        print(f"Preparing model {model_number}")
        models_data[model_number] = prepare_single_model_data(
            model_number, sim_data_folder=sim_data_folder, drop_trial_zero=drop_trial_zero)

    return models_data
