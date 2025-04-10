import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def cal_loss(
    original_X: np.ndarray,
    reconstructed_X: np.ndarray,
    output_path: str,
    feature_names: list,
    threshold: Optional[float] = None,
    cal_type: str = 'Base',
    n_std_thresh: float = 6.0
) -> Dict[str, Any]:
    """
    Calculate and plot reconstruction loss statistics.

    :param original_X: Numpy array of original data.
    :param reconstructed_X: Numpy array of reconstructed data.
    :param output_path: Path to save figures.
    :param feature_names: List of feature names (must match original_X shape).
    :param threshold: If None, will be computed as mean + n_std_thresh * std.
    :param cal_type: 'Base' or 'Predict' used for plot labeling and file naming.
    :param n_std_thresh: Number of std deviations to set threshold above mean if threshold is not provided.
    :return: Dictionary containing loss info (loss array, threshold, outliers, etc.).
    """
    if original_X.shape != reconstructed_X.shape:
        raise ValueError("original_X and reconstructed_X must have the same shape.")

    # Calculate per-sample reconstruction loss
    losses = np.mean(np.sqrt((original_X - reconstructed_X) ** 2), axis=1)
    avg_loss = np.mean(losses)
    std_dev = np.std(losses)

    # Determine threshold
    if threshold is None:
        threshold = avg_loss + n_std_thresh * std_dev

    # Classify outliers
    outlier_mask = losses > threshold
    outliers = original_X[outlier_mask]
    outliers_pred = reconstructed_X[outlier_mask]
    normalities = original_X[~outlier_mask]
    normalities_pred = reconstructed_X[~outlier_mask]

    # Compute loss by feature
    if cal_type == 'Base':
        diff_matrix = np.sqrt((original_X - reconstructed_X) ** 2)
    else:  # 'Predict'
        diff_matrix = np.sqrt((outliers - outliers_pred) ** 2) if len(outliers) else np.array([])

    if diff_matrix.size > 0:
        loss_by_feature = diff_matrix.mean(axis=0)
    else:
        loss_by_feature = np.zeros(shape=(original_X.shape[1],))

    loss_sum = np.sum(loss_by_feature) if loss_by_feature.size > 0 else 1e-9  # avoid zero division
    loss_percentage = loss_by_feature / loss_sum

    # Prepare info dict
    loss_info_dict = {
        'loss_ary': losses,
        'avg_loss': avg_loss,
        'std_dev': std_dev,
        'threshold': threshold,
        'outliers': outliers,
        'normalities': normalities,
        'loss_percentage': loss_percentage
    }

    # Plots
    # (1) Bar plot of feature-wise loss contribution
    plt.figure(figsize=(10, 3))
    plt.bar(feature_names, loss_percentage)
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.ylabel('Percentage')
    plt.title(f'{cal_type}: Loss by Percentage')
    plt.tight_layout()
    plt.savefig(f'{output_path}/figures/Loss_Contribution_{cal_type}.png')
    plt.close()

    # (2) Scatter plot (Control Limit)
    plt.figure(figsize=(10, 3))
    plt.scatter(range(len(losses)), losses, s=2, label='Normality')
    plt.scatter(
        [i for i, x in enumerate(losses) if x > threshold],
        [x for x in losses if x > threshold],
        s=2, c='red', label='Anomaly'
    )
    plt.axhline(avg_loss, color='black', linestyle='--', label='Mean')
    plt.axhline(threshold, color='green', linestyle='--', label='Threshold')
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.title(f'{cal_type} Loss Control Limits')
    plt.legend()
    plt.yscale('log')  # log-scale if needed
    plt.tight_layout()
    plt.savefig(f'{output_path}/figures/Control_Limit_{cal_type}.png')
    plt.close()

    return loss_info_dict


def pp_change(
    loss_percentage_base: np.ndarray,
    loss_percentage_pred: np.ndarray,
    feature_names: list
) -> pd.DataFrame:
    """
    Calculate absolute difference between two loss-percentage arrays.

    :param loss_percentage_base: Array of feature-wise loss percentages for baseline.
    :param loss_percentage_pred: Array of feature-wise loss percentages for new data.
    :param feature_names: List of feature names, must match size of arrays.
    :return: DataFrame with columns: ['feature', 'ae_score'].
    """
    if len(loss_percentage_base) != len(loss_percentage_pred):
        raise ValueError("Both loss_percentage arrays must have the same length.")

    diff = np.abs(loss_percentage_base - loss_percentage_pred)
    ae_df = pd.DataFrame({
        'feature': feature_names,
        'ae_score': diff
    })
    return ae_df
