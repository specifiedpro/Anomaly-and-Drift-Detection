import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def cal_loss(original_X, reconstructed_X, output_path, feature_names, threshold=0, cal_type='Base'): #

    # Calculate Statistics.
    losses = np.mean(np.sqrt(np.square(original_X - reconstructed_X)), axis = 1)
    avg_loss = np.mean(losses)
    std_dev = np.std(losses)
    # Control Limit.
    if threshold == 0:
        threshold = avg_loss + 6*std_dev
    # Classifiy datapoints.
    outliers = original_X[losses > threshold]
    outliers_pred = reconstructed_X[losses > threshold]
    
    normalities = original_X[losses <= threshold]
    normalities_pred = reconstructed_X[losses <= threshold]
    if cal_type == 'Base':
        loss_matrix_base = np.sqrt(np.square(original_X - reconstructed_X))
    elif cal_type == 'Predict':
        loss_matrix_base = np.sqrt(np.square(outliers - outliers_pred))
    loss_by_feature_base = np.mean(loss_matrix_base, axis=0)
    loss_sum = np.sum(loss_by_feature_base)
    loss_percentage_base = loss_by_feature_base / loss_sum
    loss_info_dict = {
        'loss_ary':losses,
        'avg_loss':avg_loss,
        'std_dev':std_dev,
        'threshold':threshold,
        'outliers':outliers,
        'normalities':normalities,
        'loss_percentage':loss_percentage_base
    }
    # Plot: Barplot
    plt.figure(figsize=(10,3))
    plt.bar(feature_names, loss_percentage_base)
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.ylabel('Percentage')
    plt.title(f'{cal_type}: Loss by Percentage')
    plt.savefig(f'{output_path}/figures/Loss_Contribution_{cal_type}.png')
    # Plot: Control Limit
    plt.figure(figsize=(10, 3))
    plt.scatter(range(len(losses)), losses, s=0.1, label='Normality')
    plt.scatter(range(len(losses)), [loss if loss > threshold else None for loss in losses] 
                , s=0.1, c='red', label='Anomaly(Red)')
    plt.axhline(avg_loss, color='black', linestyle='--', label='Mean')
#     plt.axhline(avg_loss, color='red', linestyle='--', label='Shifted Mean')
    # plt.axhline(avg_loss_base - std_dev_base, color='green', linestyle='--', label='Lower Bound')
    plt.axhline(threshold, color='green', linestyle='--', label='Upper Bound')
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.title(f'{cal_type} Loss Control Limits')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{output_path}/figures/Control_Limit_{cal_type}.png')
    return loss_info_dict

# 1.1 Get baseline loss
def pp_change(loss_percentage_base, loss_percentage_pred, feature_names):
    pp_change = np.abs(loss_percentage_base - loss_percentage_pred)
    ae_df = pd.DataFrame({'feature': feature_names, 'ae_score':pp_change})
    return ae_df