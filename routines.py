"""
This script contains routines used to convert raw data into pairwise interactions and to compute the prediction accuracy
of fitted AHPI parameters on test data.
"""

import os
import ast
import json
import logging
import pandas       as pd
import numpy        as np

from collections    import Counter


def get_dir():
    '''
    Returns the path to the folder where all data is stored (in particular cases_df.csv.gz). User needs to hard code the
    relevant path in the config.json file. 
    '''

    try:                                                    # Load the config file
        with open('config.json', 'r') as f:
            config  = json.load(f)
    except FileNotFoundError:                               # Raise error if file not found
        raise       FileNotFoundError("The 'config.json' file was not found. Please create the file.")
    
    path            = config.get('data_dir', '')            # Get the path from the config file
    
    if not path:                                            # Raise error if path is not set
        raise       ValueError("Data directory path is not set in config.json. Please add the path.")

    fn              =  f'{path}cases_df.csv.gz'             # Check if the cases_df.csv.gz file exists
    assert os.path.isfile(fn), f'File {fn} does not exist. Please check the path.'

    return path

def convert_to_interactions():
    '''
    Convert the law firms and roles in cases_df to interactions between law firms. Returns a list of interactions.
    '''
    # read and convert cases_df
    cases_df                    = pd.read_csv(f'{get_dir()}cases_df.csv.gz', compression='gzip')
    cases_df['extracted_roles'] = cases_df['extracted_roles'].apply(ast.literal_eval)
    cases_df['extracted_firms'] = cases_df['extracted_firms'].apply(ast.literal_eval)

    # create interactions
    interactions    = []
    for i in range(len(cases_df)):

        # extract all plaintiff and defendant firms
        def_firms, pla_firms    = [], []
        for j in range(len(cases_df['extracted_roles'][i])):
            if   cases_df['extracted_roles'][i][j]   == 'plaintiff':  pla_firms += cases_df['extracted_firms'][i][j]
            elif cases_df['extracted_roles'][i][j]   == 'defendant':  def_firms += cases_df['extracted_firms'][i][j]
        
        # create pairwise interactions
        if def_firms and pla_firms:
            for def_firm in def_firms:
                for pla_firm in pla_firms:
                    interactions.append((def_firm, pla_firm, round(cases_df['predict_proba'][i]), \
                                         cases_df['label'][i],cases_df['label'][i]))
    return interactions


def Q_fact_games(list_games, Q=1, verbose=False):
    '''
    Given a list of games and a Q factor, achieves this Q factor by iteratively removing the firms participating in the
    smallest number of interactions. Returns 0 if ends up with empty list of games due to removals.

    :param list_games:      list of games formatted as [(def, pla, proba),...] cf. case_fitting.py
    :param Q:               Q factor which should be achieved
    :param verbose:         whether or not to print progress to the output
    :return:                list of games achieving the Q factor
    '''
    if not list_games:                                              # Case where list_games is empty
        raise ValueError("The Q-factor of {Q} is not reached as the list of games is empty.")

    df                  = pd.DataFrame(list_games)                  # Convert list_games to a DataFrame
    def_col, pla_col    = df.columns[:2]                            # Assume the first two columns are 'def' and 'pla'

    all_firms           = df[[def_col, pla_col]].values.flatten()   # Find all firms
    firm_frequency      = Counter(all_firms)                        # Count frequency of each firm

    curr_Q              = len(df) / len(set(all_firms))             # Current Q factor

    if curr_Q           >= Q: return df.values.tolist()             # Return list of games if curr_Q >= Q

    min_freq            = min(firm_frequency.values())              # Min frequency
    firms_low_frequency = {firm for firm, freq in firm_frequency.items() if freq == min_freq}   # Min frequency firms

    df_filtered         = df[~df[[def_col, pla_col]].isin(firms_low_frequency).any(axis=1)]# Remove min frequency firms
    if verbose:         print('Q_fact_games iterated with min=', min_freq)                 # Print current min frequency

    return Q_fact_games(df_filtered.values.tolist(), Q=Q)                                  # Recursively call function


def balance_dataframe(df_balancing, column_to_balance, random_state=42):
    '''
    Balance the dataframe so that every value in the column_to_balance has the same number of rows

    param df_balancing:         dataframe to be balanced
    param column_to_balance:    column to be balanced
    param random_state:         random state for reproducibility
    :return:                    balanced dataframe
    '''
    value_counts  = df_balancing[column_to_balance].value_counts()              # Count occurrences of each unique value
    min_count     = value_counts.min()                                          # Determine minimum count among groups
    balanced_list = [df_balancing [df_balancing[column_to_balance] == value]                 # Sample min_count rows
            .sample(n=min_count, random_state=random_state) for value in value_counts.index] # for each unique value
    balanced_df   = pd.concat(balanced_list)                                    # Concatenate all sampled DataFrames
    balanced_df   = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True) # Shuffle the DataFrame
    return balanced_df                                                          # Return the balanced DataFrame


def prediction_accuracy(testing_file,              run_id                 = 'run_1',         balancing     = False,
                           included_firms = None,     included_intervals     = [(0, 1.0)],      external_data = False,
                           pred_cases     = False,    return_bin_win_rate    = False):
    """
    Predict outcomes for a test set of cases.
    Unless external_data is provided, the function loads the fitted scores, privileges and valence probabilities from
    csv files where run_id is the column identifier.
    
    1. Only consider firms that are in 'included_firms' (if provided).
    2. For each row/interaction in testing_file, if both 'def' and 'pla' exist in the scores,
       fetch their scores. Via the row's 'case_type', get the corresponding privilege shift and
       valence probability. Adjust the defendant's score via privilege (defendant is privileged) and compute
       the winning probability using a logistic function and a weighted combination.
    3. Iterate over all rows of testing_file, and if both firms have scores, compute the predicted winning probability.
    4. Compare the real winner and the predicted probability within the included_intervals, to compute accuracy.

    :param run_id:              Column identifier of the fitted values to use for predictions
    :param testing_file:        DataFrame of test cases with columns 'def', 'pla', 'case_type', 'winner'
    :param balancing:           Boolean, if True, balance the test data
    :param included_firms:      List of firms to include in the predictions, if None, all firms are included
    :param included_intervals:  List of intervals in which predict_proba must be to be included
    :param external_data:       Tuple of (scores, privileges, valence probabilities, test_df) to use instead of loading
                                from testing_file and fitted_scores, fitted_privileges, fitted_valence_probabs
    :param return_bin_win_rate: Boolean, if True, return defendant win rate in the included intervals/bin
    :return:                    Accuracy, defendant win rate, number of test cases, excess accuracy(, binned win rate)
    """
    ####################################################################################################################
    # load data

    if external_data:   # if external_data is provided, use it
        scores, priv_dict, val_dict, test_df = external_data
    else:               # else load data from csv files
        ################################################################################################################
        # load and format data
        df_scores  = pd.read_csv(f'{get_dir()}fitted_scores.csv',           index_col=0)
        df_priv    = pd.read_csv(f'{get_dir()}fitted_privileges.csv',       index_col=0)
        df_valence = pd.read_csv(f'{get_dir()}fitted_valence_probabs.csv',  index_col=0)
        
        if included_firms is not None:                                      # If included_firms provided,
            df_scores   = df_scores[df_scores.index.isin(included_firms)]   # keep firms contained in included_firms
        scores          = df_scores[run_id]                                 # get fitted scores for run_id
        priv_dict       = df_priv[run_id].to_dict()                         # get fitted privileges for run_id
        val_dict        = df_valence[run_id].to_dict()                      # get fitted valence probas for run_id

        ################################################################################################################
        test_df             = testing_file.copy()                           # load test data

        test_df['winner']   = test_df['predict_proba'].round().astype(int)  # rounds predict_proba to 0 or 1
        test_df             = test_df.drop(columns=['predict_proba','proba_case_type','date'])# drop columns,'id_number'

        test_df.loc[test_df['case_type'].isin(['real_property', 'prisoner_petitions']), 'case_type'] = 'other'

    ####################################################################################################################
    # compute predict proba and, as a benchmark, defendant_win_rate

    def compute_winning_proba(row):
        '''
        Use fitted values to compute the predicted winning probability for a test interaction.
        
        :param row:     row of the test dataframe with columns 'def', 'pla', 'case_type', 'winner'
        :return:        predicted winning probability if both 'def' and 'pla' firms have fitted scores, else np.nan
        '''
        firm_def, firm_pla, case_type   = row['def'], row['pla'], row['case_type']# extract def and pla firm, case type

        if firm_def in scores.index and firm_pla in scores.index:# If both firms have fitted scores
            score_def           = scores.loc[firm_def]                  # get fitted score for defendant
            score_pla           = scores.loc[firm_pla]                  # get fitted score for plaintiff
            privilege           = priv_dict.get(case_type,   0)         # get fited privilege, default 0
            q                   = val_dict.get( case_type,   0.5)       # get fitted valence proba, default 0.5

            diff                = (score_def + privilege) - score_pla   # difference shifted by privilege

            prob_favoured       = 1 / (1 + np.exp(-diff))               # sigmoid of diff as probability
            full_winning_prob   = prob_favoured * q + (1 - prob_favoured) * (1 - q)

            return full_winning_prob                             # 1 means defendant, encoded as 0, likely winner
        else:
            return np.nan                                        # if either firm does not have fitted score, return nan

    test_df['predict_proba']    = test_df.apply(compute_winning_proba, axis=1)  # compute_winning_proba for each row
    logging.info(f"Predicted winning probabilities computed for {len(test_df)} test interactions")
    
    ####################################################################################################################
    # Compute defendant win rate

    if pred_cases:              # if pred_cases, average win rate across cases (identified by id_number)
        defendant_win_rate      = (test_df.groupby('id_number', as_index=False)[['winner']].mean()['winner']==0).mean()
    elif balancing:        
        defendant_win_rate      = 0.5                               # if balanced, the defendant win rate = 0.5
    else:     
        defendant_win_rate      = (test_df['winner'] == 0).mean()   # defendant win rate across all interactions
    
    ####################################################################################################################
    # Construct the test dataframe

    test_df                     = test_df.dropna(subset=['predict_proba'])   # keep only rows with predict_proba

    if pred_cases:          # if pred_cases, average predict_proba and winner for every case (identified by id_number)
        test_df = test_df.groupby('id_number', as_index=False)[['predict_proba', 'winner']].mean()

    if balancing:   test_df = balance_dataframe(test_df, 'winner')          # if balancing, balance the test data

    ####################################################################################################################
    # Include only predictions in specified intervals 
    def is_in_intervals(x, intervals):
        '''check if x is in any of the given intervals'''
        return any(lower <= x <= upper for lower, upper in intervals)
    
    filtered_df = test_df[test_df['predict_proba'].apply(lambda x: is_in_intervals(x, included_intervals))].copy()

    ####################################################################################################################
    filtered_df['predicted_winner'] = filtered_df['predict_proba'].round().astype(int) # Round predict proba to 0 or 1

    if filtered_df.empty:       # if no interactions in the specified intervals, return nan                              
        logging.info(f"No interactions in the specified intervals {included_intervals}.")
        if return_bin_win_rate:     return np.nan, np.nan, np.nan, np.nan, np.nan
        else:                       return np.nan, np.nan, np.nan, np.nan

    excess_accuracy = (                                                                        # compute excess accuracy
            (filtered_df['predicted_winner'] == (1 - filtered_df['winner'])).astype(float)
                - defendant_win_rate * (filtered_df['predicted_winner'] == 1).astype(float)
            + (defendant_win_rate - 1) * (filtered_df['predicted_winner'] == 0).astype(float)).mean()
    
    accuracy            = (filtered_df['predicted_winner'] == 1-filtered_df['winner']).mean()  # compute accuracy

    card_tests          = len(filtered_df)                                                     # number of test cases

    if return_bin_win_rate:              # if return_bin_win_rate, compute win rate for specific interval/bin
        binned_win_rate = 1-filtered_df['winner'].mean()
        return accuracy, defendant_win_rate, card_tests, excess_accuracy, binned_win_rate
    else:
        return accuracy, defendant_win_rate, card_tests, excess_accuracy