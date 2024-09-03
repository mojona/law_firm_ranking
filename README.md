Code to reproduce the results from the paper **Addressing Information Asymmetry in Legal Disputes through Data-Driven Law Firm Rankings** ([preprint available](http://arxiv.org/abs/2408.16863)).

## File overview and instructions: 

* Set up the environment with the enviornment.yml and activate with "conda activate law_firm_ranking".
* "config.json" stores the path to the data folder where cases_df.csv.gz is stored. Please add this path to "config.json.example" and rename it "config.json".
* "routines.py" lists routines used to convert raw data into pairwise interactions. The cases_df.csv.gz added to this repo only contains only a subset of the data for testing purposes. Our full source data will be made public upon publication.
* "AHPI.py" implements the AHPI algorithm which generalizes the Bradley-Terry model. This represents the core implementation of our paper. As an illustration, we fit AHPI on synthetic data with known ground truth and compare fitted to true scores using correlations.
* "extraction_clustering.py" extracts and clusters the roles and law firms from the attorney strings in the cases_df. The file appends additional columns to cases_df.csv.gz which contain information on the names and roles of the law firms involved.
* "case_fitting.py" uses the AHPI to fit scores, valence probabilities and privileges for the cases in cases_df.csv.gz.
* "cases_df.csv.gz" is a subsample of cases_df.csv.gz and serves to test the implementation. The full dataset will be made available upon publicaton.
* "exp_scores.csv.gz" is generated via "case_fitting.py" and contains fitted exponential scores of all legal cases.
* "synthetic_data.csv.gz" and "synthetic_scores.csv.gz" can be generated via "AHPI.py" and contain synthetic pairwise interactions and synthetic scores respectively.

Reach out to slera@mit.edu in case issues.
