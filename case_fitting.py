"""
This script uses the AHPI algorithm to estiamte scores, valence probabilities and privileges for legal data. 
"""

import ast
import logging

import pandas as pd

from AHPI     import AHPI
from routines import get_dir, Q_fact_games

def convert_to_interactions():
    '''
    Convert the law firms and roles in cases_df to interactions between law firms. Returns a list of interactions.
    '''
    # read and convert cases_df
    ####################################################################################################################
    cases_df                    = pd.read_csv(f'{get_dir()}cases_df.csv.gz', compression='gzip')
    cases_df['extracted_roles'] = cases_df['extracted_roles'].apply(ast.literal_eval)
    cases_df['extracted_firms'] = cases_df['extracted_firms'].apply(ast.literal_eval)

    # create interactions
    ####################################################################################################################
    interactions    = []
    for i in range(len(cases_df)):

        # extract all plaintiff and defendant firms
        ################################################################################################################
        def_firms, pla_firms    = [], []
        for j in range(len(cases_df['extracted_roles'][i])):
            if   cases_df['extracted_roles'][i][j]   == 'plaintiff':  pla_firms += cases_df['extracted_firms'][i][j]
            elif cases_df['extracted_roles'][i][j]   == 'defendant':  def_firms += cases_df['extracted_firms'][i][j]
        
        # create pairwise interactions
        ################################################################################################################
        if def_firms and pla_firms:
            for def_firm in def_firms:
                for pla_firm in pla_firms:
                    interactions.append((def_firm, pla_firm, round(cases_df['predict_proba'][i]), \
                                         cases_df['label'][i],cases_df['label'][i]))
    return interactions

if __name__ =='__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Convert the law firms and roles in cases_df to pairwise interactions between law firms and construct a subsample
    # with a Q-factor (#interactions/#firms) of 60. Save the subsample.
    # Estimate the exponential scores (with which we work for convenience) for the firms via AHPI and save them.
    ####################################################################################################################
    logging.info('Starting conversion of firms and roles to interactions and extract subsample with Q-factor of 60.')

    list_games        =  convert_to_interactions()                 # Convert firms and roles in cases_df to interactions
    list_games        =  Q_fact_games(list_games, Q=60)            # Construct a subsample with a Q-factor of 60
    df                =  pd.DataFrame(list_games, columns=['priv', 'unpriv', 'win_index', 'val_type', 'priv_type'])

    exp_scores, _, _  =  AHPI(df)                                   # Estimate exponential scores for the firms
    exp_scores        =  pd.DataFrame.from_dict(exp_scores, orient='index', columns=["Exp Score"])
    _                 =  exp_scores.to_csv(f'{get_dir()}exp_scores.csv.gz', index=False, compression='gzip' )