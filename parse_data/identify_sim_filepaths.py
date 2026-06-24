#!/usr/bin/env python
# coding: utf-8

# Enumerate RL agent simulation results.
#
# The simulation results directory mirrors the human raw-data layout closely enough to
# reuse the human loading/preprocessing pipeline, but the folder structure is different:
# each model has its own subfolder (named e.g. '260515_0001') containing a single dated
# .json logging file (plus a DONE.txt marker and an mlagents_stdout.log we ignore).
#
# These helpers turn that directory into (data_folder, json_filename) pairs compatible
# with parse_data.loading.loading_pipeline, and provide a configurable grouping of models
# for between-model comparisons.

import os
import re

import data_strings


# model subfolders look like '<YYMMDD>_<4-digit model number>', e.g. '260515_0001'
MODEL_FOLDER_PATTERN = re.compile(r'^(\d{6})_(\d{4})$')


def _model_number(folder_name):
    ''' Return the integer model number encoded in a model subfolder name, or None
        if the folder name does not match the expected '<date>_<NNNN>' pattern. '''

    match = MODEL_FOLDER_PATTERN.match(folder_name)
    if match is None:
        return None
    return int(match.group(2))


def get_model_folders(sim_data_folder=data_strings.SIM_DATA_FOLDER):
    ''' Return a list of (model_number, folder_name) tuples for every model subfolder
        in sim_data_folder, sorted by model number. '''

    model_folders = []
    for entry in os.listdir(sim_data_folder):
        if not os.path.isdir(os.path.join(sim_data_folder, entry)):
            continue
        model_num = _model_number(entry)
        if model_num is None:
            continue
        model_folders.append((model_num, entry))

    model_folders.sort(key=lambda pair: pair[0])
    return model_folders


def get_model_json_filename(folder_path):
    ''' Return the single results .json filename inside a model subfolder.
        Raises if zero or more than one .json is found, so silent mis-loads are avoided. '''

    json_files = [f for f in os.listdir(folder_path)
                  if f.endswith('.json') and not f.startswith('._')]

    if len(json_files) != 1:
        raise ValueError(
            f"Expected exactly one .json in {folder_path}, found {len(json_files)}: {json_files}")

    return json_files[0]


def get_model_filepaths(sim_data_folder=data_strings.SIM_DATA_FOLDER):
    ''' Return a dict mapping model_number -> (data_folder, json_filename) for every model
        subfolder, where data_folder is the absolute path to that model's subfolder.

        The (data_folder, json_filename) pair is exactly what loading.loading_pipeline and
        prepare_data expect, so a single model is loaded with:
            data_folder, json_filename = get_model_filepaths()[model_number]
    '''

    filepaths = {}
    for model_num, folder_name in get_model_folders(sim_data_folder):
        folder_path = os.path.join(sim_data_folder, folder_name)
        json_filename = get_model_json_filename(folder_path)
        filepaths[model_num] = (folder_path, json_filename)

    return filepaths


# default model groups for the between-model comparison.
# ranges are inclusive of both endpoints. groups overlap/leave gaps exactly as described in
# the analysis brief (1-10 vs 20-30 vs 30-40); adjust here as more models are added.
DEFAULT_MODEL_GROUPS = {
    'reference':  (0, 10),
    'curriculum':    (20, 30),
    'higherRND':   (30, 40),
}


def group_models(model_numbers=None, groups=DEFAULT_MODEL_GROUPS,
                 sim_data_folder=data_strings.SIM_DATA_FOLDER):
    ''' Assign model numbers to named groups based on inclusive (low, high) ranges.

        If model_numbers is None, every model found in sim_data_folder is used.
        Returns a dict mapping group name -> sorted list of model numbers that exist and
        fall within that group's range. Models present on disk but outside every range are
        omitted; ranges that reference not-yet-generated models simply yield fewer entries. '''

    if model_numbers is None:
        model_numbers = [num for num, _ in get_model_folders(sim_data_folder)]

    model_numbers = sorted(model_numbers)

    grouped = {}
    for name, (low, high) in groups.items():
        grouped[name] = [num for num in model_numbers if low <= num <= high]

    return grouped
