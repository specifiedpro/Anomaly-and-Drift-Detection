

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Non-interactive backend
matplotlib.use('agg')

# Local imports
from src.data_loader import read_files, clean_data
from src.drift_model import init_detector
from src.loss_analysis import cal_loss, pp_change
from src.stats_test import stats_test
from sklearn.preprocessing import MinMaxScaler


def main(args):
    # 0. Environment & Paths (from argparse arguments)
    train_folder_path = args.train_folder
    pred_folder_path = args.pred_folder
    output_path = args.output_path
    model_path = args.model_path
    
    # Make sure output_path/figures exists
    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)

    # 1. Read Files
    reference, current = read_files(
        train_folder_path=train_folder_path,
        pred_folder_path=pred_folder_path,
        sep='|',
        encoding='big5',
        dropna=False
    )
    print(f"[INFO] Loaded training data shape: {reference.shape}")
    print(f"[INFO] Loaded prediction data shape: {current.shape}")

    # 2. Clean & Preprocess Data
    # Adjust drop_columns/impute_zero_cols as needed
    data_dict = clean_data(
        reference=reference,
        current=current,
        val_size=0.2,
        drop_columns=['MOST_USE_BRAND', 'SUBSCR_ID', 'STATIS_MN', 'L6MN_MP_CNT', 'L3MN_MP_CNT'],
        impute_zero_cols=['L1MN_L3MN_GAME_SOCIAL_CNT_RT', 'L1MN_L3MN_GAME_SOCIAL_AMT_RT'],
        rename_prefix="col",
        random_state=42
    )

    feature_names = data_dict['val'].columns.tolist()
    print(f"[INFO] Cleaned data reference shape: {data_dict['ref'].shape}")
    print(f"[INFO] Cleaned data validation shape: {data_dict['val'].shape}")
    print(f"[INFO] Cleaned data current shape: {data_dict['cur'].shape}")

    # 3. Scale Data (for Autoencoder)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(data_dict['ref'])
    X_val = (
        scaler.transform(data_dict['val'])
        if not data_dict['val'].empty else np.empty((0, 0))
    )
    X_test = scaler.transform(data_dict['cur'])

    scaled_data = {'train': X_train, 'val': X_val, 'test': X_test}

    # 4. Initialize Detector (Autoencoder)
    detector, scaled_dict = init_detector(
        data_dict=scaled_data,
        model_path=model_path,
        default_weight=True,  # Switch to False if you want to train from scratch
        hyper_params={
            'n_epochs': 20,
            'batch_size': 128,
            'layer_dims': (128, 64, 32),
            'activation': 'leaky_relu',
            'final_activation': 'sigmoid'
        }
    )
    print("[INFO] Autoencoder model ready.")

    # 5. Inference (Reconstruction)
    pred_base = detector.predict(scaled_dict['train'])
    pred_test = detector.predict(scaled_dict['test'])
    print("[INFO] Reconstruction done.")

    # 6. Loss Analysis
    base_loss_info = cal_loss(
        original_X=scaled_dict['train'],
        reconstructed_X=pred_base,
        output_path=output_path,
        feature_names=feature_names,
        threshold=None,    # auto-calc (mean + 6*std)
        cal_type='Base',
        n_std_thresh=6.0
    )
    test_loss_info = cal_loss(
        original_X=scaled_dict['test'],
        reconstructed_X=pred_test,
        output_path=output_path,
        feature_names=feature_names,
        threshold=base_loss_info['threshold'],  # use baseline threshold
        cal_type='Predict',
        n_std_thresh=6.0
    )

    # 7. AE-based Percentage Change
    ae_df = pp_change(
        loss_percentage_base=base_loss_info['loss_percentage'],
        loss_percentage_pred=test_loss_info['loss_percentage'],
        feature_names=feature_names
    )
    print("[INFO] AE-based drift analysis completed.")

    # 8. Statistical Tests (Evidently)
    stats_df = stats_test(
        reference=data_dict['ref'],
        current=data_dict['cur'],
        output_path=output_path,
        test_params={
            'method_num': 'wasserstein',
            'method_cat': 'jensenshannon',
            'num_threshold': 0.1,
            'cat_threshold': 0.1
        }
    )
    print("[INFO] Evidently-based drift analysis completed.")

    # 9. Combine Results & Plot
    result_df = pd.merge(ae_df, stats_df, on='feature', how='left').fillna(0)
    plot_df = result_df.melt(id_vars='feature', var_name='Score Type', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='feature', y='Score', hue='Score Type')
    plt.xticks(rotation=90)
    plt.title("Combined Drift Scores (AE & Stats)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "figures", "Combined_Drift_Score.png"))
    plt.close()

    print("[INFO] Drift detection workflow completed. See output folder for results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Drift Detection Pipeline")

    parser.add_argument(
        "--train_folder",
        type=str,
        required=True,
        help="Path to the folder containing training CSV files."
    )
    parser.add_argument(
        "--pred_folder",
        type=str,
        required=True,
        help="Path to the folder containing prediction CSV files."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to directory where outputs (plots, CSVs, etc.) will be stored."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Autoencoder model .h5 file."
    )

    args = parser.parse_args()
    main(args)
