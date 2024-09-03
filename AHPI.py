"""
In this script we implement the AHPI algorithm which forms the core of our methods. We also implement a function that 
generates synthetic data with known ground truth to test the implementation. 
"""

import logging
import pandas       as pd
import numpy        as np
import scipy.stats  as stats

from scipy.special  import expit
from scipy.optimize import fsolve

from routines import get_dir


def AHPI(df_inter, MII=50, MIO=50, minimum_iterations=10, convergence_threshold=0.01, fit_valence_prob=True,
         fit_privilege=True):
    '''
    AHPI for asymmetric heterogenous pariwise interactions uses a generalized Bradley-Terry model to fit scores, valence
    probabilities and privileges based on pairwise interactions of the form 'defendant, plaintiff, winner,
    interaction_type'. The model is fitted using an Expectation-Maximization algorithm.

    :param df_inter:                Dataframe of pairwise interactions with columns 'priv', 'unpriv', 'win_index' (int),
                                    'val_type', 'priv_type' representing the privileged individual, unprivileged
                                    individual, index of winning individual (0=winner privileged or 1 else), valence
                                    type, privilege type
    :param minimum_iterations:      Minimum iterations in either loop
    :param MII:                     Maximum iterations inner loop
    :param MIO:                     Maximum iterations outer loop
    :param convergence_threshold:   Convergence threshold for scores, valence probabilities, privileges
    :param fit_valence_prob:        Boolean, determinig if the valence probability should be fitted or not
    :param fit_privilege:           Boolean, determinig if the privilege should be fitted or not
    :return:                        Dictionaries of exponentials of the scores, valence probabilities, privileges
    '''

    df_inter = df_inter.copy()      # Create a copy of the DataFrame to avoid modifying the original one

    # create mappings: the assigned index will be the index also used when calling fitted scores, valence probabilities,
    # privileges
    ####################################################################################################################
    indiv_map     = {value: idx for idx, value in enumerate(pd.concat([df_inter['priv'], df_inter['unpriv']]).unique())}
    val_type_map  = {value: idx for idx, value in enumerate(df_inter['val_type'].unique())}
    priv_type_map = {value: idx for idx, value in enumerate(df_inter['priv_type'].unique())}

    # create repositories for fitted ln scores, valence probabilities, privileges
    ####################################################################################################################
    exp_scores  = np.full(len(indiv_map),0.9)                                        # dictionary for ln scores
    val_probs   = np.full(len(val_type_map),0.5) if fit_valence_prob \
             else np.full(len(val_type_map),1.0)                                     # dictionary for valence probs
    privileges  = np.full(len(priv_type_map),0.0)                                    # dictionary for privileges

    # Map the individuals, valence types and privilege types
    ####################################################################################################################
    df_inter['priv']        = df_inter['priv'].map(indiv_map)
    df_inter['unpriv']      = df_inter['unpriv'].map(indiv_map)
    df_inter['val_type']    = df_inter['val_type'].map(val_type_map)
    df_inter['priv_type']   = df_inter['priv_type'].map(priv_type_map)

    # assign u (winning individual), v (losing individual), c for latter computations
    ####################################################################################################################
    df_inter['u'] = np.where(df_inter['win_index'] == 0, df_inter['priv'],df_inter['unpriv'])
    df_inter['v'] = np.where(df_inter['win_index'] == 1, df_inter['priv'],df_inter['unpriv'])
    df_inter['c'] = np.where(df_inter['win_index'] == 0, -1, 1)
    df_inter.drop(columns=['priv', 'unpriv'], inplace=True)

    # initialise fitted valence probabilities and privileges with initial guesses
    ####################################################################################################################
    df_inter['q']   = val_probs[df_inter['val_type']]
    df_inter['eps'] = privileges[df_inter['priv_type']]

    class ConvergenceChecker:
        '''
        class checking the convergence for inputs: initialised with maximum_iterations allowed and convergence_threshold
        being the maximum absolute difference in subsequent iterations
        updated via update(): current_lambda, current_epsilon, current_q
        '''
        def __init__(self, maximum_iterations, minimum_iterations = minimum_iterations,
                     convergence_threshold = convergence_threshold):
            self.maximum_iterations     = maximum_iterations                    # Store the maximum number of iterations
            self.minimum_iterations     = minimum_iterations                    # Store the minimum number of iterations
            self.convergence_threshold  = convergence_threshold                 # Store the convergence threshold
            self.old_lambdas            = []                                    # Initialize list for past lambda's
            self.old_epsilons           = []                                    # Initialize list for past epsilon's
            self.old_q_s                = []                                    # Initialize list for past q's
            self.loop_number            = 0                                     # Initialize loop counter

        def update(self, current_lambda, current_epsilon, current_q):  # Update method to check for convergence
            '''
            Takes current_lambda, current_epsilon, current_q as inputs.
            -updates loop number, old_lambdas, old_epsilons, old_q_s
            -test if:   1. loop_number > maximum_iterations
                        2. Kendall correlation not changing and larger than 0.999
                        3. max abs difference in lambdas, epsilons, privileges smaller than convergence_threshold
            :return:    -if 1. or 2.+3.: 0, loop_number
                        -else:           1, loop_number
            '''
            self.loop_number += 1                                       # Increment the loop counter

            if self.loop_number > self.maximum_iterations: return 0, self.loop_number # case: maximum iterations

            self.old_lambdas.append(np.copy(current_lambda))        # Append lambdas
            self.old_epsilons.append(np.copy(current_epsilon))      # Append epsilons
            self.old_q_s.append(np.copy(current_q))                 # Append q_s

            if len(self.old_lambdas) > 3:  # Keep only the last three lambda, epsilon and q values
                self.old_lambdas.pop(0), self.old_epsilons.pop(0), self.old_q_s.pop(0)

            if self.loop_number < self.minimum_iterations: return 1, self.loop_number  # case: minimum iterations

            if len(self.old_lambdas) >= 3:  # Check if there are at least 3 iterations
                kendall_corr_1 = stats.kendalltau(self.old_lambdas[-1], self.old_lambdas[-2])[0]
                kendall_corr_2 = stats.kendalltau(self.old_lambdas[-2], self.old_lambdas[-3])[0]

                if kendall_corr_1 > 0.999:  # Kendall_corr>0.999 and not changing
                    max_abs_diff_lambda     = max(abs(np.subtract(self.old_lambdas[-1], self.old_lambdas[-2])))
                    max_abs_diff_epsilon    = max(abs(np.subtract(self.old_epsilons[-1], self.old_epsilons[-2])))
                    max_abs_diff_q          = max(abs(np.subtract(self.old_q_s[-1], self.old_q_s[-2])))

                    if (max_abs_diff_lambda      < self.convergence_threshold and
                            max_abs_diff_epsilon < self.convergence_threshold and
                            max_abs_diff_q       < self.convergence_threshold):  # Check if all differences < threshold
                        return 0, self.loop_number  # Return if convergence criteria met

            return 1, self.loop_number  # Return the updated state

    card_q_t        = df_inter['val_type'].value_counts().sort_index().values
    outer_checker   = ConvergenceChecker(maximum_iterations=MIO)

    logging.info(f'Starting with the optimisation.')
    while True:   

        # assign current lambda of scores to u (winning individual), v (losing individual),
        ################################################################################################################
        df_inter['lambda_u']    = exp_scores[df_inter['u']]
        df_inter['lambda_v']    = exp_scores[df_inter['v']]
        
        # calculate pi
        ################################################################################################################
        df_inter['pi'] = np.exp(df_inter['c'] * df_inter['eps'])    * df_inter['lambda_u'] * df_inter['q'] \
                         / (df_inter['lambda_u'] * np.exp(df_inter['c'] * df_inter['eps']) * df_inter['q'] +
                            df_inter['lambda_v'] * (1 - df_inter['q']))
        
        # fit valence probability
        ################################################################################################################
        if fit_valence_prob:
            for idx in range(len(val_probs)):
                val_probs[idx] = df_inter.loc[df_inter['val_type'] == idx, 'pi'].sum() / card_q_t[idx]
            df_inter['q'] = val_probs[df_inter['val_type']]
        
        # fit privileges
        ################################################################################################################
        if fit_privilege:
            for idx in range(len(privileges)):
                mask = df_inter['priv_type'] == idx
                df_idx = df_inter[mask]
                pi_idx, c_idx,lambda_u_idx,lambda_v_idx = \
                    df_idx['pi'], df_idx['c'],  df_idx['lambda_u'], df_idx['lambda_v']
                def func_epsilon(x):
                    y = (1 - np.exp(x)) / (1 + np.exp(x))
                    y += np.sum(pi_idx * c_idx - lambda_u_idx * np.exp(c_idx * x) * c_idx
                                 / (lambda_u_idx * np.exp(c_idx * x) + lambda_v_idx))
                    return y

                privileges[idx] = fsolve(func_epsilon, 0.0)[0]

            df_inter['eps'] = privileges[df_inter['priv_type']]
        
        # fit scores
        ################################################################################################################
        inner_checker = ConvergenceChecker(maximum_iterations=MII)
        while True:
            for idx in range(len(exp_scores)):
                ###
                df_u_r          = df_inter.query('u == @idx')
                df_v_r          = df_inter.query('v == @idx')
                # convert to arrays
                win_index_u_r   = df_u_r['win_index'].values
                eps_u_r         = df_u_r['eps'].values
                pi_u_r          = df_u_r['pi'].values
                lambda_v_u_r    = df_u_r['lambda_v'].values

                win_index_v_r   = df_v_r['win_index'].values
                eps_v_r         = df_v_r['eps'].values
                pi_v_r          = df_v_r['pi'].values
                lambda_u_v_r    = df_v_r['lambda_u'].values

                gamma_r_u_r     = np.where(win_index_u_r == 1, np.exp(eps_u_r), np.exp(-eps_u_r))
                gamma_r_v_r     = np.where(win_index_v_r == 1, np.exp(eps_v_r), np.exp(-eps_v_r))

                numerator       = 1 + np.sum(pi_u_r) + np.sum(1 - pi_v_r)
                denominator     = 2 / (1 + exp_scores[idx]) + \
                              np.sum(gamma_r_u_r / (gamma_r_u_r * exp_scores[idx] + lambda_v_u_r)) + \
                              np.sum(1 / (gamma_r_v_r * lambda_u_v_r + exp_scores[idx]))

                exp_scores[idx] = numerator / denominator
            converged_i, _   = inner_checker.update(exp_scores,privileges,val_probs)
            if converged_i   == 0: break
        ################################################################################################################

        converged_o,_  = outer_checker.update(exp_scores,privileges,val_probs)

        current_val_probs   =   {k: val_probs[v]      for k, v in val_type_map.items()}
        current_privileges  =   {k:-privileges[v]     for k, v in priv_type_map.items()}
        logging.info(f'Reached iteration {_}. '
                     f'Current valence probabilities: {current_val_probs}. '
                     f'Current privileges: {current_privileges}. ')


        if converged_o == 0: break

    # convert scores, valence probabilities, privileges back to the original values
    ####################################################################################################################
    exp_scores  = {k: exp_scores[v]     for k, v in indiv_map.items()}
    val_probs   = {k: val_probs[v]      for k, v in val_type_map.items()}
    privileges  = {k:-privileges[v]     for k, v in priv_type_map.items()}

    return exp_scores, val_probs, privileges


def generate_synthetic_data(R, scores=(0,1,10), val_probs=(0.95,0.05,1), privileges=(0,1,1)):
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

    df_inter = pd.DataFrame(interactions, columns=['priv', 'unpriv', 'val_type', 'priv_type'])
    
    # calculate winning probabilities and determine winners
    ####################################################################################################################
    win_index = []
    for i in range(R):
        # Extract individuals, privilege type, valence type from every interaction
        priv,                                        unpriv,                   priv_type,                   val_type = \
           df_inter.loc[i, 'priv'], df_inter.loc[i, 'unpriv'], df_inter.loc[i, 'priv_type'], df_inter.loc[i, 'val_type']

        p_favoured = expit(scores[priv] + privileges[priv_type] - scores[unpriv])   # sigmoid of scores with privilege:
        if np.random.rand() < p_favoured: favoured, unfavoured = priv,      unpriv  # used this as proba to assign
        else:                             favoured, unfavoured = unpriv,    priv    # favoured and unfavoured individual

        fav_win_prob = val_probs[val_type]  # valence probability = proba of favoured winning

        # choose winnter by randomly choosing the favoured over the unfavoured with proba = fav_win_prob
        win_idx = np.random.choice([favoured, unfavoured], p=[fav_win_prob, 1 - fav_win_prob])
        win_index.append(0 if win_idx == priv else 1)

    df_inter['win_index'] = win_index       # integrate winner into df_inter
    df_inter=df_inter[['priv', 'unpriv', 'win_index', 'val_type', 'priv_type']]     # reorder columns

    # create dictionaries for exp_scores, fitted privileges, fitted valence probabilities
    ####################################################################################################################
    exp_scores          = {idx: np.exp(score) for idx, score in enumerate(scores)}  # Dictionary for exp_scores
    fitted_privileges   = {idx: priv for idx, priv in enumerate(privileges)}# Dictionary for fitted privileges
    fitted_val_probs    = {idx: val for idx, val in enumerate(val_probs)}   # Dictionary of fitted valence probabilities

    return df_inter, exp_scores, fitted_privileges, fitted_val_probs


if __name__=='__main__':

    # Create synthatic data with known ground truth. We work with the exponential of the scores for convenience.
    ####################################################################################################################
    scores                      =  (0,1,20) # mean, standard deviation, number of scores for synthetic data
    R                           =  500      # number of interactions. Consequently, Q = 500/20 = 25
    # generate synthetic data ( mean, sigma and cardinality for val_probs=(0.95,0.05,1), for privileges=(0,1,1) ))
    df_inter, exp_scores, p, v  =  generate_synthetic_data(R = R, scores = scores)
    _                           =  df_inter.to_csv(f'{get_dir()}synthetic_data.csv.gz', \
                                                   index=False, compression='gzip' )
    exp_scores_df               =  pd.DataFrame.from_dict(exp_scores, orient='index', columns=["Exp Score"])
    _                           =  exp_scores_df.to_csv(f'{get_dir()}synthetic_scores.csv.gz', \
                                                    index = False, compression='gzip')
    logging.info(f'Synthetic data generated with privilege {p[0]} and valence probability {v[0]}.')
    
    # Estimating exponential scores via AHPI and calculating Kendall's tau between fitted and synthetic scores.
    ####################################################################################################################
    scores_fit, val_prob_fit, _ =  AHPI(df_inter)

    # Knowing that the ground truth has a valence probability > 0.5, we check if the fitted valence probability 
    # is < 0.5. In this case, all estimated values have to be transformed in line with AHPI's underlying symmetry.
    # For the (exponential) scores, this means that they have to be inverted.
    ####################################################################################################################
    if val_prob_fit[0] < 0.5:   # Case where the ranking is inverted. The ground truth has a valence probability > 0.5.
        scores_fit = {key: value * -1 for key, value in scores_fit.items()}     # invert the scores
    keys                        = scores_fit.keys() & exp_scores.keys()         # find common keys
    fitted_values               = [scores_fit[key] for key in keys]             # ordered fitted values
    exp_values                  = [exp_scores[key] for key in keys]             # ordered synthetic values
    tau, p_value                = stats.kendalltau(fitted_values, exp_values)   # calculate Kendall's tau

    logging.info(f"Kendall's tau: {tau}, p-value: {p_value}")