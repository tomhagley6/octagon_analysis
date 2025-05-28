# %%
import pandas as pd
import numpy as np


# %% [markdown]
# #### Populate a dataframe, with a row for each trial, and fields for regressors (only including trials with fully-populated regressors)

# %%
def populate_dataframe(analysis_results, analysis_type):
    
    ''' Populate a dataframe with trial data for a given analysis type.

    Parameters:
    - analysis_results (dict): A nested dictionary containing session, player, and analysis data.
    - analysis_type (str): The type of analysis to process. Options are "solo", "social", "solosocial".

    Returns:
    - pd.DataFrame: A dataframe containing trial data with fields for regressors and other relevant information. '''
    
    glm_df = pd.DataFrame()

    for session_id, players in analysis_results.items():
        for player_id in players:

            if analysis_type == 'solosocial':
                # take each filtered_regressor array and fill the relevant df field for this player
                player_data_solo = analysis_results[session_id][player_id]['solo']['regressors']
                player_data_social = analysis_results[session_id][player_id]['social']['regressors']
                choice_solo = analysis_results[session_id][player_id]['solo']['dependent']['choice']
                choice_social = analysis_results[session_id][player_id]['social']['dependent']['choice']
                df_player = pd.DataFrame(
                        {
                            "SessionID" : session_id,
                            "PlayerID" : player_id,
                            "GlmPlayerID" : session_id*2 + player_id,
                            "ChooseHigh" : np.concatenate([choice_solo, choice_social]),
                            "WallSep" :  np.concatenate([player_data_solo['wall_sep'], player_data_social['wall_sep']]),
                            "FirstSeenWall" : np.concatenate([player_data_solo['first_seen'], player_data_social['first_seen']]),
                            "D2H" : np.concatenate([player_data_solo['d2h'], player_data_social['d2h']]),
                            "D2L" : np.concatenate([player_data_solo['d2l'], player_data_social['d2l']]),
                            "SocialContext" : np.concatenate([np.ones(player_data_solo["wall_sep"].shape[0]) - 1, np.ones(player_data_social["wall_sep"].shape[0])]) # 0 for solo, 1 for social
                        }
                    )
            
            # not solo-social analysis (solo or social only)
            else:
            
                # take each filtered_regressor array and fill the relevant df field for this player
                player_data = analysis_results[session_id][player_id][analysis_type]['regressors']
                choice = analysis_results[session_id][player_id][analysis_type]['dependent']['choice']
                
                # standard fields for all analysis types
                df_player = pd.DataFrame(
                            {
                                "SessionID" : session_id,
                                "PlayerID" : player_id,
                                "GlmPlayerID" : session_id*2 + player_id,
                                "ChooseHigh" : choice,
                                "WallSep" : player_data['wall_sep'],
                                "FirstSeenWall" : player_data['first_seen'],
                                "D2H" : player_data['d2h'],
                                "D2L" : player_data['d2l']
                                
                            }
                )
            
                # social-specific fields
                if analysis_type == 'social':
                    
                    df_player["OpponentVisible"] = player_data['opponent_visible']
                    df_player["OpponentFirstSeenWall"] = player_data['first_seen_opponent']
                    df_player["OpponentD2H"] = player_data['d2h_opponent']
                    df_player["OpponentD2L"] = player_data['d2l_opponent']
                    
                    # convert columns to categorical types for input to lmer
                    df_player["OpponentFirstSeenWall"] = df_player["OpponentFirstSeenWall"].apply(lambda x: str(x) if pd.notna(x) else x)
                    df_player["OpponentFirstSeenWall"] = df_player["OpponentFirstSeenWall"].astype("category")
            
            # append this smaller dataframe to the the full dataframe
            glm_df = pd.concat([glm_df, df_player], ignore_index=True)


    # convert columns to categorical types for input to lmer
    glm_df["FirstSeenWall"] = glm_df["FirstSeenWall"].apply(lambda x: str(x) if pd.notna(x) else x)
    glm_df["FirstSeenWall"] = glm_df["FirstSeenWall"].astype("category")
    # glm_df["WallSep"] = glm_df["WallSep"].astype(str).astype("category")

    return glm_df


