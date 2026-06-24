#!/usr/bin/env python
# coding: utf-8

# Simulation-specific preprocessing.
#
# RL agent inference runs are logged in the same format as the human task, so the standard
# human preprocessing (parse_data.preprocess.standard_preprocessing) applies unchanged. The
# only structural difference is that the 'trial end' event carries extra reward fields that
# do not exist in the human logs:
#     data.trialScores.<id>   per-trial terminal reward (reward for the wall reached)
#     data.trialRewards.<id>  overall trial reward (terminal + step/shaping rewards)
# (data.playerScores.<id>, the cumulative terminal reward, is also present in the human logs.)
#
# pd.json_normalize already turns these into columns automatically, but the values land only
# on the single 'trial end' row of each trial. This module fills them across every row of
# their trial, mirroring how wall numbers and player scores are filled, so per-trial values
# are available throughout the trial (and on whichever row downstream extraction reads).

import pandas as pd

import globals
import parse_data.preprocess as preprocess


# columns that, when present, should be filled across each trial
SIM_TRIAL_END_FIELDS = [
    globals.PLAYER_0_SCORE,
    globals.PLAYER_1_SCORE,
    globals.PLAYER_0_TRIAL_SCORE,
    globals.PLAYER_1_TRIAL_SCORE,
    globals.PLAYER_0_TRIAL_REWARD,
    globals.PLAYER_1_TRIAL_REWARD,
]


def fill_trial_end_fields(df, fields=SIM_TRIAL_END_FIELDS):
    ''' Return a dataframe in which each simulation trial-end reward field is filled across
        every row of its trial.

        Each trial contributes exactly one non-null value per field (logged on its 'trial end'
        row), so filling forward then backward within each data.trialNum group propagates that
        single value to the whole trial without leaking across trial boundaries. Fields absent
        from the dataframe (e.g. player 1 columns in a solo/single-agent run) are skipped. '''

    df2 = df.copy()

    present_fields = [field for field in fields if field in df2.columns]
    if not present_fields:
        return df2

    # ffill then bfill within each trial fills the whole trial from its single trial-end value
    df2[present_fields] = (
        df2.groupby(globals.TRIAL_NUM)[present_fields]
           .transform(lambda group: group.ffill().bfill())
    )

    return df2


def standard_preprocessing_sim(df):
    ''' Simulation preprocessing: run the standard human preprocessing, then fill the extra
        trial-end reward fields across each trial. Returns the preprocessed dataframe. '''

    df = preprocess.standard_preprocessing(df)
    df = fill_trial_end_fields(df)

    print("Simulation preprocessing complete.")

    return df
