import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from typing import Dict, Any


def extract_drift_scores(evidently_result: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract drift scores from the JSON-like structure produced by Evidently.

    :param evidently_result: Dictionary from report.as_dict().
    :return: dict of {column_name: drift_score}
    """
    drift_scores = {}

    for metric in evidently_result.get('metrics', []):
        result = metric.get('result', {})
        if 'drift_by_columns' in result:
            columns_data = result['drift_by_columns']
            for col_name, col_info in columns_data.items():
                drift_scores[col_name] = col_info.get('drift_score', 0.0)

    return drift_scores


def stats_test(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str,
    test_params: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Perform statistical tests (Data Drift) using Evidently.

    :param reference: Reference (historical) DataFrame.
    :param current: Current (new) DataFrame.
    :param output_path: Path to save the HTML report.
    :param test_params: Dictionary of methods/thresholds for numeric/cat drift detection.
                       Example: {
                           'method_num': 'wasserstein',
                           'method_cat': 'jensenshannon',
                           'num_threshold': 0.1,
                           'cat_threshold': 0.1
                       }
    :return: DataFrame with columns ['feature', 'stats_drift_score'].
    """
    if test_params is None:
        test_params = {
            'method_num': 'wasserstein',
            'method_cat': 'jensenshannon',
            'num_threshold': 0.1,
            'cat_threshold': 0.1
        }

    # Create an Evidently report using the DataDriftPreset
    report = Report(metrics=[
        DataDriftPreset(
            num_stattest=test_params['method_num'],
            cat_stattest=test_params['method_cat'],
            num_stattest_threshold=test_params['num_threshold'],
            cat_stattest_threshold=test_params['cat_threshold']
        )
    ])

    # Run the report
    report.run(reference_data=reference, current_data=current)
    result_dict = report.as_dict()

    # Extract drift scores
    drift_score_dict = extract_drift_scores(result_dict)

    # Save HTML
    report.save_html(f'{output_path}/result.html')

    # Construct DF
    drift_df = pd.DataFrame(
        list(drift_score_dict.items()),
        columns=['feature', 'stats_drift_score']
    )
    drift_df.sort_values(by='stats_drift_score', ascending=False, inplace=True)

    # Optionally save as CSV
    drift_df.to_csv(f'{output_path}/stats_drift_scores.csv', index=False)

    return drift_df
