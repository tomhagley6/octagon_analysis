# globals.py
import math

NUM_WALLS = 8

## Unity project logging event strings
SELECTED_TRIGGER_ACTIVATION = 'server-selected trigger activation'
SLICE_ONSET = 'slice onset'
TRIAL_START = 'trial start'
TRIAL_END = 'trial end'

## trial epoch name strings
PRE_TRIALS = 'pre trials'
TRIAL_STARTED = 'trial started' 
SLICES_ACTIVE = 'slices active'
POST_CHOICE = 'post choice'
ITI = 'ITI'

# df column names 
PLAYER_0_XLOC = 'data.playerPosition.0.location.x'
PLAYER_1_XLOC = 'data.playerPosition.1.location.x'
PLAYER_0_YLOC = 'data.playerPosition.0.location.z'
PLAYER_1_YLOC = 'data.playerPosition.1.location.z'

# Only yrot is a real per-player value. yrot is the body transform, which is networked (SyncRotAngleY on
# FirstPersonPlayer.prefab). xrot and zrot are the camera, which is not networked and not held per player:
# ArenaLogger.BuildPlayerPosition calls GetComponentInChildren<Camera>(), which finds the host's own camera
# for both players, so in every social session player 1's xrot and zrot are an exact copy of player 0's.
# Checked over all 100 social sessions in tests/test_log_data_properties.py. Player 1 head pitch does not
# exist in the data and cannot be recovered. Solo logs have no player 1 columns at all.
PLAYER_0_XROT = 'data.playerPosition.0.rotation.x'
PLAYER_1_XROT = 'data.playerPosition.1.rotation.x'     # copy of player 0, see above
PLAYER_0_YROT = 'data.playerPosition.0.rotation.y'
PLAYER_1_YROT = 'data.playerPosition.1.rotation.y'
PLAYER_0_ZROT = 'data.playerPosition.0.rotation.z'
PLAYER_1_ZROT = 'data.playerPosition.1.rotation.z'     # copy of player 0, see above

PLAYER_0_SCORE = 'data.playerScores.0'
PLAYER_1_SCORE = 'data.playerScores.1'

## Simulation-only fields, logged in the 'trial end' event of RL agent inference runs.
## trialScores  = per-trial terminal reward (reward for the wall reached this trial)
## trialRewards = overall trial reward (terminal reward summed with step/shaping rewards)
## playerScores = cumulative terminal reward across trials (also present in human data)
PLAYER_0_TRIAL_SCORE = 'data.trialScores.0'
PLAYER_1_TRIAL_SCORE = 'data.trialScores.1'
PLAYER_0_TRIAL_REWARD = 'data.trialRewards.0'
PLAYER_1_TRIAL_REWARD = 'data.trialRewards.1'



WALL_1 = 'data.wall1'
WALL_2 = 'data.wall2'
WALL_3 = 'data.wall3'
WALL_4 = 'data.wall4'

WALL_TRIGGERED = 'data.wallTriggered'
TRIGGER_CLIENT = 'data.triggerClient'
TRIAL_NUM = 'data.trialNum'
TRIAL_TYPE = 'data.trialType'

# trial types
HIGH_LOW = 'HighLow'

PLAYER_LOC_DICT = {
                     0: {'xloc': PLAYER_0_XLOC, 'yloc': PLAYER_0_YLOC},
                     1: {'xloc': PLAYER_1_XLOC, 'yloc': PLAYER_1_YLOC}
                  } 

PLAYER_ROT_DICT = {
                     0: {'xrot': PLAYER_0_XROT, 'yrot': PLAYER_0_YROT, 'zrot': PLAYER_0_ZROT},
                     1: {'xrot': PLAYER_1_XROT, 'yrot': PLAYER_1_YROT, 'zrot': PLAYER_1_ZROT}
                  } 

PLAYER_SCORE_DICT = {
                     0: {'score': PLAYER_0_SCORE},
                     1: {'score': PLAYER_1_SCORE}
                    }

# simulation-only per-trial reward fields, keyed by player/agent id
PLAYER_TRIAL_REWARD_DICT = {
                     0: {'trial_score': PLAYER_0_TRIAL_SCORE, 'trial_reward': PLAYER_0_TRIAL_REWARD},
                     1: {'trial_score': PLAYER_1_TRIAL_SCORE, 'trial_reward': PLAYER_1_TRIAL_REWARD}
                    }

XLOC = 'location.x'

RECORDING_FREQUENCY = 50

## arena geometry as distances the player centre can reach, not technically wall positions
## measured from 260723 agent logs (models 1, 23, 33, 51)

OCTAGON_WALLFACE_RADIUS = 16.747 
OCTAGON_VERTEX_RADIUS = 16.747 / math.cos(math.radians(22.5)) # 18.127
OCTAGON_ALCOVE_RADIUS = 19.747
ALCOVE_HALF_WIDTH = 1.90
ALCOVE_DEPTH = OCTAGON_ALCOVE_RADIUS - OCTAGON_WALLFACE_RADIUS

## player radius (fitted from alcove-mouth corners; CharacterController radius 0.6)
## add to the reachable distances above to get physical wall positions, e.g. for line of sight
PLAYER_RADIUS = 0.61


