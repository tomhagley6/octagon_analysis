# globals.py

NUM_WALLS = 8

## Unity project logging event strings
SELECTED_TRIGGER_ACTIVATION = 'server-selected trigger activation'
SLICE_ONSET = 'slice onset'

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
PLAYER_0_XROT = 'data.playerPosition.0.rotation.x'
PLAYER_1_XROT = 'data.playerPosition.1.rotation.x'
PLAYER_0_YROT = 'data.playerPosition.0.rotation.y'
PLAYER_1_YROT = 'data.playerPosition.1.rotation.y'

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

XLOC = 'location.x'
YLOC = 'rotation.y'

PLAYER_ROT_DICT = {
                     0: {'yrot': PLAYER_0_YROT},
                     1: {'yrot': PLAYER_1_YROT}
                  } 
