
import pandas as pd

def first_non_null(series):
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) > 0 else pd.NA

#removed time_column='time_x' from args
def process_questionnaire_scores(AQ_path, GAD_path, BIS_path):

    questionnaire_dict = {
        'AQ': AQ_path, 
        'GAD': GAD_path,
        'BIS': BIS_path
    }
    traits_dict = {
        'AQ': 10,  # AQ has 10 questions
        'GAD': 7,  # GAD has 7 questions
        'BIS': 8   # BIS has 8 questions
    }

    AQ_answers_dict = {
        'Definitely agree':1,
        'Slightly agree':1,
        'Definitely disagree':0,
        'Slightly disagree':0
    }

    AQ_answers_dict_reverse = {
        'Definitely agree':0,
        'Slightly agree':0,
        'Definitely disagree':1,
        'Slightly disagree':1
    }

    GAD_answers_dict = {
        'Not at all':0,
        'Several days':1,
        'More than half the days':2,
        'Nearly every day':3
    }

    BIS_answers_dict = {
        'Rarely/Never':1,
        'Occasionally':2,
        'Often':3,
        'Almost Always/Always':4
    }

    BIS_answers_dict_reverse = {
        'Rarely/Never':4,
        'Occasionally':3,
        'Often':2,
        'Almost Always/Always':1
    }

    answers_dict = {
        'GAD':GAD_answers_dict,
        'BIS': BIS_answers_dict,
        'AQ': AQ_answers_dict

    }

    # AQ specific scoring
    AQ_positive_scoring = {1,7,8,10}
    AQ_negative_scoring = {2,3,4,5,6,9}

    # BIS specific scoring
    BIS_positive_scoring = {3,4,6,8}
    BIS_negative_scoring = {1,2,5,7}

    dfs={}
    
    for questionnaire, path_to_questionnaire in questionnaire_dict.items():
        print(path_to_questionnaire)
        
        dfs[f'df_{questionnaire}']= pd.read_csv(path_to_questionnaire)
        #df_merged = dfs[f'df_{list(questionnaire_dict.keys())[0]}']
    
    df_merged = dfs['df_AQ']

    for questionnaire in ['GAD', 'BIS']:
        df_merged = df_merged.merge(
            dfs[f'df_{questionnaire}'], 
            on=["first_name", "last_name"], 
            how="outer",
            suffixes=('', f'_{questionnaire}')
        )

    #for questionnaire in list(questionnaire_dict.keys())[1:]:
        #df_merged = df_merged.iloc[:,:-1]
        #df_merged = df_merged.merge(dfs[f'df_{questionnaire}'], on=["first_name", "last_name"], how="outer")
        #df_merged = df_merged.iloc[:,:-1]
    
    for col in ['email', 'player_id', 'Session identifier']:
        matching_cols = [c for c in df_merged.columns if c == col or c.startswith(f'{col}_')]

        if matching_cols:
            df_merged[col] = df_merged[matching_cols].bfill(axis=1).iloc[:, 0]
            df_merged = df_merged.drop(columns=[c for c in matching_cols if c != col])
    
    df = df_merged.copy()

    # remove participants with no behavioural/session data
    df = df[
        df['Session identifier'].notna()
        & df['Session identifier'].astype(str).str.strip().ne("")
        & df['Session identifier'].astype(str).str.lower().ne("nan")
    ].copy()

    for trait, num_questions in traits_dict.items():
        if trait in answers_dict and trait == 'GAD':
            for i in range(1, num_questions+1):
                column_name = f'{trait}{i}'
                if column_name in df.columns:
                    df[column_name] = df[column_name].replace(answers_dict[trait]).astype("Int64")

    # for AQ-specific scoring
    for i in range(1, traits_dict['AQ'] + 1):
        column_name = f'AQ{i}'
        if column_name in df.columns:
            if i in AQ_positive_scoring:
                df[column_name] = df[column_name].replace(AQ_answers_dict).astype("Int64")
            elif i in AQ_negative_scoring:
                df[column_name] = df[column_name].replace(AQ_answers_dict_reverse).astype("Int64")

    # for BIS specific scoring
    for i in range(1, traits_dict['BIS'] + 1):
        column_name = f'BIS{i}'
        if column_name in df.columns:
            if i in BIS_positive_scoring:
                df[column_name] = df[column_name].replace(BIS_answers_dict).astype("Int64")
            elif i in BIS_negative_scoring:
                df[column_name] = df[column_name].replace(BIS_answers_dict_reverse).astype("Int64")

    for trait, num_questions in traits_dict.items():
        df[f'{trait}_total'] = df[[f'{trait}{i}' for i in range(1, num_questions+1)]].sum(axis=1, min_count=1)

    selected_columns = ['first_name','last_name','email','player_id','Session identifier','GAD_total','AQ_total','BIS_total']

    df = df[selected_columns]

    df = (
        df.groupby(
            ['first_name', 'last_name', 'Session identifier'],
            dropna=False,
            as_index=False
        )
        .agg({
            'email': first_non_null,
            'player_id': first_non_null,
            'GAD_total': first_non_null,
            'AQ_total': first_non_null,
            'BIS_total': first_non_null
        })
    )

    return df

# initialise empty set to track pseudonyms
existing_pseudonyms = set()
# generate pseudonyms
name_date_to_pseudonym = {}
               
#umbrella function
def preprocess_data(AQ_path, GAD_path, BIS_path):

    existing_pseudonyms = set()
    name_date_to_pseudonym = {}

    def generate_pseudonym(row):

        first_name = str(row['first_name']).strip().upper()
        last_name = str(row['last_name']).strip().upper()

        key = (first_name, last_name, row['date'])

        if key in name_date_to_pseudonym:
            return name_date_to_pseudonym[key]


        first_letter_name = first_name[0]
        day_of_month = str(row['date'].day).zfill(2)

        print(f"Generating pseudonym for {row['first_name']} {row['last_name']} with date {row['date']}")


        for char in last_name:
            pseudonym = first_letter_name + char + day_of_month
            print(f"first letter: {first_letter_name}, second letter: {char}, day: {day_of_month}")

            if pseudonym not in existing_pseudonyms:
                existing_pseudonyms.add(pseudonym)
                name_date_to_pseudonym[key] = pseudonym
                return pseudonym

        raise ValueError(f"Could not generate unique pseudonym for {key}")

    df = process_questionnaire_scores(AQ_path, GAD_path, BIS_path)

    # drop rows without session identifier
    df = df[
        df['Session identifier'].notna()
        & df['Session identifier'].astype(str).str.strip().ne("")
    ].copy()

    # ignore rows where Session identifier starts with R_ or is "risky"
    df = df[
        ~df['Session identifier'].astype(str).str.startswith('R_', na=False)
        & ~df['Session identifier'].astype(str).str.lower().eq('risky')
    ].copy()

    df['date'] = pd.to_datetime(
        df['Session identifier']
          .astype(str)
          .str[:10]
          .str.replace('_', '-', regex=False),
        errors='coerce'
    )

    df = df.dropna(subset=['date'])

    df['pseudonym'] = df.apply(generate_pseudonym, axis=1)

    df = (
        df.groupby(
            ['date', 'pseudonym', 'Session identifier'],
            dropna=False,
            as_index=False
        )
        .agg({
            'GAD_total': first_non_null,
            'AQ_total': first_non_null,
            'BIS_total': first_non_null
        })
    )

    selected_columns = [
        'date',
        'pseudonym',
        'Session identifier',
        'GAD_total',
        'AQ_total',
        'BIS_total'
    ]

    df = df[selected_columns]

    return df[selected_columns]

def process_items(AQ_path, GAD_path, BIS_path):

    questionnaire_dict = {
        'AQ': AQ_path, 
        'GAD': GAD_path,
        'BIS': BIS_path
    }
    traits_dict = {
        'AQ': 10,  # AQ has 10 questions
        'GAD': 7,  # GAD has 7 questions
        'BIS': 8   # BIS has 8 questions
    }

    AQ_answers_dict = {
        'Definitely agree':1,
        'Slightly agree':2,
        'Definitely disagree':3,
        'Slightly disagree':4
    }

    AQ_answers_dict_reverse = {
        'Definitely agree':4,
        'Slightly agree':3,
        'Definitely disagree':2,
        'Slightly disagree':1
    }

    GAD_answers_dict = {
        'Not at all':1,
        'Several days':2,
        'More than half the days':3,
        'Nearly every day':4
    }

    BIS_answers_dict = {
        'Rarely/Never':1,
        'Occasionally':2,
        'Often':3,
        'Almost Always/Always':4
    }

    BIS_answers_dict_reverse = {
        'Rarely/Never':4,
        'Occasionally':3,
        'Often':2,
        'Almost Always/Always':1
    }

    answers_dict = {
        'GAD':GAD_answers_dict,
        'BIS': BIS_answers_dict,
        'AQ': AQ_answers_dict

    }

    # AQ specific scoring
    AQ_positive_scoring = {1,7,8,10}
    AQ_negative_scoring = {2,3,4,5,6,9}

    # BIS specific scoring
    BIS_positive_scoring = {3,4,6,8}
    BIS_negative_scoring = {1,2,5,7}


    answers_dict = {
        'GAD':GAD_answers_dict,
        'BIS': BIS_answers_dict,
        #'AQ': AQ_answers_dict

    }

    # AQ specific scoring
    AQ_positive_scoring = {1,7,8,10}
    AQ_negative_scoring = {2,3,4,5,6,9}


    dfs={}

    for questionnaire, path_to_questionnaire in questionnaire_dict.items():
        print(path_to_questionnaire)
        dfs[f'df_{questionnaire}']= pd.read_csv(path_to_questionnaire)
   
    df_merged = dfs['df_AQ']
        
    for questionnaire in ['GAD', 'BIS']:
        df_merged = df_merged.merge(
            dfs[f'df_{questionnaire}'],
            on=['first_name', 'last_name'],
            how='outer',
            suffixes=('', f'_{questionnaire}')
        )
        
    # combine duplicated metadata columns
    for col in ['email', 'player_id', 'Session identifier']:
        matching_cols = [
            c for c in df_merged.columns
            if c == col or c.startswith(f'{col}_')
        ]

        if matching_cols:
            df_merged[col] = df_merged[matching_cols].bfill(axis=1).iloc[:, 0]
            df_merged = df_merged.drop(
                columns=[c for c in matching_cols if c != col]
            )

    df = df_merged.copy()

    for trait, num_questions in traits_dict.items():
        if trait in answers_dict and trait == 'GAD':
            for i in range(1, num_questions+1):
                column_name = f'{trait}{i}'
                if column_name in df.columns:
                    df[column_name] = df[column_name].replace(answers_dict[trait]).astype("Int64")

    # for AQ-specific scoring
    for i in range(1, traits_dict['AQ'] + 1):
        column_name = f'AQ{i}'
        if column_name in df.columns:
            if i in AQ_positive_scoring:
                df[column_name] = df[column_name].replace(AQ_answers_dict).astype("Int64")
            elif i in AQ_negative_scoring:
                df[column_name] = df[column_name].replace(AQ_answers_dict_reverse).astype("Int64")

    
    # for BIS specific scoring
    for i in range(1, traits_dict['BIS'] + 1):
        column_name = f'BIS{i}'
        if column_name in df.columns:
            if i in BIS_positive_scoring:
                df[column_name] = df[column_name].replace(BIS_answers_dict).astype("Int64")
            elif i in BIS_negative_scoring:
                df[column_name] = df[column_name].replace(BIS_answers_dict_reverse).astype("Int64")

    group_cols = ['first_name', 'last_name', 'Session identifier']

    item_cols = [
        col for col in df.columns
        if col.startswith('GAD') or col.startswith('AQ') or col.startswith('BIS')
    ]

    metadata_cols = ['pseudonym', 'email', 'player_id']

    agg_dict = {col: first_non_null for col in metadata_cols + item_cols if col in df.columns}

    df = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(agg_dict)
    )

    return df

def process_items_pseudo(AQ_path, GAD_path, BIS_path):

    existing_pseudonyms = set()
    name_date_to_pseudonym = {}

    def generate_pseudonym(row):

        first_name = str(row['first_name']).strip().upper()
        last_name = str(row['last_name']).strip().upper()

        key = (first_name, last_name, row['date'])

        if key in name_date_to_pseudonym:
            return name_date_to_pseudonym[key]


        first_letter_name = first_name[0]
        day_of_month = str(row['date'].day).zfill(2)

        print(f"Generating pseudonym for {row['first_name']} {row['last_name']} with date {row['date']}")


        for char in last_name:
            pseudonym = first_letter_name + char + day_of_month
            print(f"first letter: {first_letter_name}, second letter: {char}, day: {day_of_month}")

            if pseudonym not in existing_pseudonyms:
                existing_pseudonyms.add(pseudonym)
                name_date_to_pseudonym[key] = pseudonym
                return pseudonym

        raise ValueError(f"Could not generate unique pseudonym for {key}")

    df = process_items(AQ_path, GAD_path, BIS_path)
    df_totals = preprocess_data(AQ_path, GAD_path, BIS_path)

    # drop rows without session identifier
    df = df[
        df['Session identifier'].notna()
        & df['Session identifier'].astype(str).str.strip().ne("")
    ].copy()

    # ignore rows where Session identifier starts with R_ or is "risky"
    df = df[
        ~df['Session identifier'].astype(str).str.startswith('R_', na=False)
        & ~df['Session identifier'].astype(str).str.lower().eq('risky')
    ].copy()

    # allow dates written as 2025_05_12 or 2025-05-12
    df['date'] = pd.to_datetime(
        df['Session identifier']
          .astype(str)
          .str[:10]
          .str.replace('_', '-', regex=False),
        errors='coerce'
    )

    # remove rows where date could not be parsed
    df = df.dropna(subset=['date'])

    # apply function to DataFrame
    df['pseudonym'] = df.apply(generate_pseudonym, axis=1)

    # construct column names dynamically
    gad_items = [f'GAD{i}' for i in range(1, 8)]
    aq_items = [f'AQ{i}' for i in range(1, 11)]
    bis_items = [f'BIS{i}' for i in range(1, 9)]

    def first_non_null(series):
        non_null = series.dropna()
        return non_null.iloc[0] if len(non_null) > 0 else pd.NA

    # collapse duplicate participant rows
    df = (
        df.groupby(['pseudonym', 'date', 'Session identifier'],
                dropna=False,
                as_index=False)
        .agg({
            **{
                col: first_non_null
                for col in gad_items + aq_items + bis_items
                if col in df.columns
            }
        })
    )

    # bring in df_totals on 'pseudonym'
    df = df.merge(
        df_totals[['Session identifier', 'GAD_total', 'AQ_total', 'BIS_total']],
        on='Session identifier',
        how='left'
    )

    # add fixed columns
    selected_columns = ['pseudonym', 'date'] + gad_items + aq_items + bis_items + ['GAD_total', 'AQ_total', 'BIS_total']

    # subset dataframe
    df = df[selected_columns]

    return df


def process_personal_data(AQ_path, GAD_path, BIS_path):

    questionnaire_dict = {
        'AQ': AQ_path,
        'GAD': GAD_path,
        'BIS': BIS_path
    }

    dfs={}
    for questionnaire, path_to_questionnaire in questionnaire_dict.items():
        print(path_to_questionnaire)
        
        dfs[f'df_{questionnaire}']= pd.read_csv(path_to_questionnaire)
        df_merged = dfs[f'df_{list(questionnaire_dict.keys())[0]}']
        
    for questionnaire in list(questionnaire_dict.keys())[1:]:
        #df_merged = df_merged.iloc[:,:-1]
        df_merged = df_merged.merge(dfs[f'df_{questionnaire}'], on=["first_name", "last_name"], how="outer")
        #df_merged = df_merged.iloc[:,:-1]
        
    df = df_merged.dropna()

    df['date'] = pd.to_datetime(df['Session identifier'].str[:10])

    selected_columns = ['date','first_name','last_name','email']
    df = df[selected_columns]

    prescreen_df = pd.read_excel('prescreen_results_250402.xlsx')
    prescreen_columns = ['first_name','last_name','gender','age']
    prescreen_df = prescreen_df[prescreen_columns]

    df = df.merge(prescreen_df, on=['first_name','last_name'], how='left')

    df['pseudonym'] = df.apply(generate_pseudonym, axis=1)

    df.to_csv('personal_data.csv', index=False)

    return df