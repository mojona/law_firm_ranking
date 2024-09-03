"""
This script contains routines used to convert raw data into pairwise interactions.
"""

import os
import ast
import json
import pandas as pd

from collections import Counter


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
