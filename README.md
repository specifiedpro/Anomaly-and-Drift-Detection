
# Data Drift Detection

This repository contains the python files and a jupyter notebook for performing Data Drift Detection using two methods: Autoencoder and the statistical method provided by the evidentlyAI package.

Here is a brief explanation of the contents of each file:

## DATALOADER.py

This file contains functions for loading and preprocessing the data.

- `read_files(train_folder_path, pred_folder_path, output_path)`: This function reads the data files from the provided paths. The function returns `reference` and `current` dataframes.

- `clean_data(reference, current, output_path, val_size)`: This function cleans the data by removing unwanted columns and normalizing the data. It also splits the data into training and validation sets. The function returns a dictionary, `data_dict`, which contains the cleaned and split data.

## DRIFT_MODEL.py

This file contains functions for training the Autoencoder model and initializing the detector.

- `init_detector(data_dict, default_weight, model_path, hyper_params)`: This function initializes the drift detector using the provided data and hyperparameters. It returns the trained detector and the scaled data.

## LOSS_ANALYSIS.py

This file contains functions for calculating loss and percent change.

- `cal_loss(data, pred, feature_names, cal_type, output_path)`: This function calculates the loss between the data and the prediction. It returns a dictionary, `loss_result`, which contains the calculated loss and other related information.

- `pp_change(loss_base, loss_pred, feature_names)`: This function calculates the percentage change in loss. It returns a dataframe, `ae_df`, which contains the calculated percentage change for each feature.

## STATS_TEST.py

This file contains functions for performing statistical tests to detect drift.

- `stats_test(ref, cur, output_path)`: This function performs statistical tests on the reference and current data to detect any drift. It returns a dataframe, `drift_score`, which contains the calculated drift scores for each feature.

## Main.ipynb

This jupyter notebook contains the main workflow of the drift detection process. It uses the functions from the above-mentioned files to perform the following steps:

1. Read and preprocess the data.
2. Initialize the drift detector.
3. Inference the model.
4. Analyze the loss.
5. Perform statistical tests.
6. Combine the results from the Autoencoder model and the statistical tests, and plot the results.

