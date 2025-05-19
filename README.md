# Data-Driven Law Firm Rankings to Reduce Information Asymmetry in Legal Disputes

Code to reproduce the results from the paper ([preprint available](http://arxiv.org/abs/2408.16863)).

## File overview and instructions: 

* Set up the environment with the environment.yml and activate with "conda activate law_firm_ranking".
* "config.json" stores the path to the data folder where cases_df.csv.gz is stored. Please add this path to "config.json.example" and rename it "config.json".
* "routines.py" lists routines used to convert raw data into pairwise interactions as well as a function used to test AHPI's prediction accuracy on test data. The cases_df.csv.gz added to this repo contains only a subset of the data for testing purposes. Our full source data will be made public upon publication.
* "AHPI.py" implements the AHPI algorithm which generalizes the Bradley-Terry model. This represents the core implementation of our paper. As an illustration, we fit AHPI on synthetic data with known ground truth and compare fitted to true scores using correlations.
* "extraction_clustering.py" extracts and clusters the roles and law firms from the attorney strings in the cases_df. The file appends additional columns to cases_df.csv.gz which contain information on the names and roles of the law firms involved.
* "case_fitting.py" uses the AHPI to fit scores, valence probabilities and privileges for the cases in cases_df.csv.gz. With these fitted values, the generalized Bradley-Terry model underlying the AHPI algorithm can be used to predict outcomes of cases with known law firms and case types.
* "cases_df.csv.gz" is a subsample of cases_df.csv.gz and serves to test the implementation. The full dataset will be made available upon publication.
* "exp_scores.csv.gz" is generated via "case_fitting.py" and contains fitted exponential scores of all legal cases.
* "synthetic_data.csv.gz" and "synthetic_scores.csv.gz" can be generated via "AHPI.py" and contain synthetic pairwise interactions and synthetic scores respectively.

Reach out to slera@mit.edu in case of any issues.





# Data-Driven Law Firm Rankings to Reduce Information Asymmetry in Legal Disputes

This repository contains code to reproduce the results from our paper  
**[Data-Driven Law Firm Rankings to Reduce Information Asymmetry in Legal Disputes](http://arxiv.org/abs/2408.16863)**.

We present a ranking algorithm **AHPI** which assigns scores to entities (e.g. law firms) competing against each other in pairwise interactions (e.g. trials). In 2022, the case where these pairwise interactions are of different “types” (e.g. civil rights trials as opposed to torts trials)<sup>1</sup>  has been addressed with an algorithm based on the Bradley–Terry model. **AHPI** generalizes this 2022 algorithm to account for asymmetry in interactions (e.g. a defendant has a priori higher winning odds than the plaintiff).  
We assign strength scores to law firms based on historical outcomes.

---

## Repository Overview

- `environment.yml`: Required dependencies. Create the environment with  
  ```bash
  conda env create -f environment.yml
  conda activate law_firm_ranking
  ```

- `config.json`: Add the path to your data folder (where `cases_df.csv.gz` is stored) into `config.json.example`, then rename it to `config.json`.

- `cases_df.csv.gz`: A small illustrative subset of the full case dataset for testing purposes.

- `exp_scores.csv.gz`: Fitted exponential scores computed from `case_fitting.py`.

- `synthetic_data.csv.gz` and `synthetic_scores.csv.gz`: Pairwise synthetic case data and corresponding scores. Generated using the synthetic example in `AHPI.py`.

- `routines.py`: Contains helper routines for data preprocessing and for testing the prediction performance of AHPI.

- `AHPI.py`: Core implementation of the **AHPI algorithm**. Includes functionality to generate synthetic datasets with known ground truth, fit AHPI, and evaluate performance.

- `extraction_clustering.py`: Extracts law firm and role information from the attorney strings in the case data and appends this structured metadata to `cases_df`.

- `case_fitting.py`: Fits AHPI scores, valence probabilities, and privilege parameters to the real dataset (`cases_df.csv.gz`), enabling out-of-sample predictions.

---

## Example: Using AHPI on Synthetic Data

An example illustrating the use of AHPI on synthetic data is given in `AHPI.py` and below.

```python

# Create synthatic data with known ground truth. We work with the exponential of the scores for convenience.
####################################################################################################################
scores                          =  (0,1,20) # mean, standard deviation, number of scores for synthetic data
R                               =  500      # number of interactions. Consequently, Q = 500/20 = 25
# generate synthetic data ( mean, sigma and cardinality for val_probs=(0.95,0.05,1), for privileges=(1.5,1,1) ))
df_inter, exp_scores, pri, val  =  generate_synthetic_data(R = R, scores = scores)
_                               =  df_inter.to_csv(f'{get_dir()}synthetic_data.csv.gz', \
                                                index=False, compression='gzip' )
exp_scores_df                   =  pd.DataFrame.from_dict(exp_scores, orient='index', columns=["Exp Score"])
_                               =  exp_scores_df.to_csv(f'{get_dir()}synthetic_scores.csv.gz', \
                                                index = False, compression='gzip')
# split into test and train data
train_inter, test_inter         = df_inter.iloc[:int(0.8*R)], df_inter.iloc[int(0.8*R):]
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
````````
---

## References

[1]: M. E. J. Newman, Ranking with multiple types of pairwise comparisons,
Proceedings of the Royal Society A 478 (2266) (2022) 20220517.

## Questions?

For questions, bug reports, or collaboration opportunities, please contact:
**slera@mit.edu**
