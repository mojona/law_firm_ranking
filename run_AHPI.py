"""
In this script we import the AHPI algorithm which forms the core of our methods. We also implement a function that 
generates synthetic data with known ground truth to test the implementation. 
"""

import logging
import pandas       as pd
import numpy        as np
import scipy.stats  as stats

from scipy.special  import expit

from routines       import get_dir, prediction_accuracy
from AHPI           import AHPI

def generate_synthetic_data(R, scores=(0,1,10), val_probs=(0.95,0.05,1), privileges=(1.5,1,1)):
    '''
    This function generates synthetic data for R asymmetric heterogenous (Q types) pairwise interactions. The latent
    scores, valence probabilities and privileges are generated as normally distributed. The winner of an interaction is 
    probabilistically determined by using a generliazed Bradley-Terry model in line with AHPI.

    :param R:           Number of interactions
    :param Q:           Number of interaction types
    :param Q_equals_E:  0 = Q valence probas, Q privileges
                        1 = Q valence probas, 1 privilege
                        2 = 1 valence proba , Q privileges
    :param sigma:       Standard deviation of scores
    :param mean_val:    Mean of valence probabilities
    :param sigma_val:   Standard deviation of valence probabilities
    :param mean_eps:    Mean of privileges
    :param sigma_eps:   Standard deviation of privileges
    :param scores:      Scores passed for the data generation (generate randomly if None).
    :param val_probs:   Valence probabilities passed for the data generation (generate randomly if None).
    :param privileges:  Privileges passed for the data generation (generate randomly if None).
    :return:            priv  unpriv  win_index  val_type  priv_type
    '''

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # synthetic scores, valence probabilities, privileges
    ####################################################################################################################
    if isinstance(scores,tuple):
        mean,       sigma,      N   =   scores                          # Unpack tuple
        scores      =   np.random.normal(mean, sigma, N)                # Normally distributed scores
    else: N         =   len(scores)
    if isinstance(val_probs,tuple):
        mean_val,   sigma_val,  Q   =   val_probs                       # Unpack tuple
        assert 0 <= mean_val <= 1, "mean_val must be between 0 and 1"   # Assert mean of valence probabilities in [0,1]
        val_probs   =   np.random.normal(mean_val, sigma_val, Q)        # Normally distributed valence probabilities
        val_probs   = np.clip(val_probs, 0, 1)                          # Clip values not between 0, 1
    else: Q         =   len(val_probs)
    if isinstance(privileges,tuple):
        mean_eps,   sigma_eps,  P   =   privileges                      # Unpack tuple
        privileges  =   np.random.normal(mean_eps, sigma_eps, P)        # Normally distributed privileges
    else: P         =   len(privileges)

    assert  P==Q or P==1 or Q==1,       "only P==Q or P==1 or Q==1 is supported"
    if      P==Q:       Q_equals_E=3    # Q_equals_E determines if there is only 1 type of privilege, 1 type of valence
    elif    P==1:       Q_equals_E=1    # probability, or if both occur P=Q times. In this case the two types are set
    else:               Q_equals_E=2    # to be equal for every interaction
    
    # create DataFrame for interactions
    ####################################################################################################################
    interactions = []
    for _ in range(R):
        priv, unpriv     = np.random.choice(N, 2, replace=False)            # Randomly choose two different individuals

        random_Q         = np.random.choice(max(Q,P))                       # Choose a random valence/privilege type
        if    Q_equals_E == 1: val_type,  priv_type = random_Q, 0           # Only 1 privilege, Q valence probabilities
        elif  Q_equals_E == 2: val_type,  priv_type = 0,        random_Q    # Only 1 valence probability, Q privileges
        else:                  val_type = priv_type = random_Q              # Q privileges, Q valence probabilities

        interactions.append((priv, unpriv, val_type, priv_type))

    df = pd.DataFrame(interactions, columns=['priv', 'unpriv', 'val_type', 'priv_type'])
    
    # calculate winning probabilities and determine winners
    ####################################################################################################################
    win_index = []
    for i in range(R):
        # Extract individuals, privilege type, valence type from every interaction
        priv,                                        unpriv,                   priv_type,                   val_type = \
           df.loc[i, 'priv'], df.loc[i, 'unpriv'], df.loc[i, 'priv_type'], df.loc[i, 'val_type']

        p_favoured = expit(scores[priv] + privileges[priv_type] - scores[unpriv])   # sigmoid of scores with privilege:
        if np.random.rand() < p_favoured: favoured, unfavoured = priv,      unpriv  # used this as proba to assign
        else:                             favoured, unfavoured = unpriv,    priv    # favoured and unfavoured individual

        fav_win_prob = val_probs[val_type]  # valence probability = proba of favoured winning

        # choose winnter by randomly choosing the favoured over the unfavoured with proba = fav_win_prob
        win_idx = np.random.choice([favoured, unfavoured], p=[fav_win_prob, 1 - fav_win_prob])
        win_index.append(0 if win_idx == priv else 1)

    df['win_index'] = win_index       # integrate winner into df
    df=df[['priv', 'unpriv', 'win_index', 'val_type', 'priv_type']]     # reorder columns

    # create dictionaries for exp_scores, fitted privileges, fitted valence probabilities
    ####################################################################################################################
    exp_scores          = {idx: np.exp(score) for idx, score in enumerate(scores)}  # Dictionary for exp_scores
    fitted_privileges   = {idx: priv for idx, priv in enumerate(privileges)}# Dictionary for fitted privileges
    fitted_val_probs    = {idx: val for idx, val in enumerate(val_probs)}   # Dictionary of fitted valence probabilities

    return df, exp_scores, fitted_privileges, fitted_val_probs


if __name__=='__main__':

    # Create synthatic data with known ground truth. We work with the exponential of the scores for convenience.
    ####################################################################################################################
    scores                          =  (0,1,20) # mean, standard deviation, number of scores for synthetic data
    R                               =  500      # number of interactions. Consequently, Q = 500/20 = 25
    # generate synthetic data ( mean, sigma and cardinality for val_probs=(0.95,0.05,1), for privileges=(1.5,1,1) ))
    df, exp_scores, pri, val  =  generate_synthetic_data(R = R, scores = scores)
    _                               =  df.to_csv(f'{get_dir()}synthetic_data.csv.gz', \
                                                   index=False, compression='gzip' )
    exp_scores_df                   =  pd.DataFrame.from_dict(exp_scores, orient='index', columns=["Exp Score"])
    _                               =  exp_scores_df.to_csv(f'{get_dir()}synthetic_scores.csv.gz', \
                                                    index = False, compression='gzip')
    # split into test and train data
    train_inter, test_inter         = df.iloc[:int(0.8*R)], df.iloc[int(0.8*R):]
    logging.info(f'Synthetic data generated with privilege {pri[0]} and valence probability {val[0]}.')
    
    # Estimating exponential scores via AHPI and calculating Kendall's tau between fitted and synthetic scores.
    ####################################################################################################################
    scores_fit, val_prob_fit, _     =  AHPI(train_inter)

    # Knowing that the ground truth has a valence probability > 0.5, we check if the fitted valence probability 
    # is < 0.5. In this case, all estimated values have to be transformed in line with AHPI's underlying symmetry.
    # For the (exponential) scores, this means that they have to be inverted.
    ####################################################################################################################
    if val_prob_fit[0] < 0.5:   # Case where the ranking is inverted. The ground truth has a valence probability > 0.5.
        scores_fit = {key: value * -1 for key, value in scores_fit.items()}         # invert the scores
    keys                            = scores_fit.keys() & exp_scores.keys()         # find common keys
    fitted_values                   = [scores_fit[key] for key in keys]             # ordered fitted values
    exp_values                      = [exp_scores[key] for key in keys]             # ordered synthetic values
    tau, p_value                    = stats.kendalltau(fitted_values, exp_values)   # calculate Kendall's tau

    logging.info(f"Kendall's tau: {tau}, p-value: {p_value}")

    # Test the prediction accuracy of the fitted scores on the test data.
    ####################################################################################################################
    test_inter                      =\
        test_inter.rename(columns={'priv': 'def', 'unpriv': 'pla', 'val_type': 'case_type', 'win_index': 'winner'})
    series_exp_scores               = pd.Series(exp_scores)     # convert to series
    series_scores                   = np.log(series_exp_scores) # transform from exp to scores
    accuracy_0_1, benchmark, _,_,_  =\
        prediction_accuracy(test_inter, series_scores, pri, val, included_intervals = [(0, 1.0)])   # accuracy overall
    accuracy_08_1, _, _, _, _       =\
        prediction_accuracy(test_inter, series_scores, pri, val, included_intervals = [(0.8, 1.0)]) # accuracy [0.8,1.0]
    
    logging.info(f"For a benchmark of {benchmark} the overall prediction accuracy on test data is {accuracy_0_1:.3f}.")
    logging.info(f"The accuracy for predicted winning propensities in [0.8, 1.0] is {accuracy_08_1:.3f}.")