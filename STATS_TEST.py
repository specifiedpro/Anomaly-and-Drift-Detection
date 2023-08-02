from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, RegressionTestPreset
from evidently.tests import *

import pandas as pd

def extract_drift_scores(data):
    drift_scores = {}
    for metric in data['metrics']:
        if 'drift_by_columns' in metric['result']:
            drifted_columns = metric['result']['drift_by_columns']
            for column_name, column_data in drifted_columns.items():
                drift_score = column_data['drift_score']
                drift_scores[column_name] = [drift_score]
    return drift_scores

def stats_test(reference, current, output_path, test_params={'method_num':'wasserstein',
                                                                'method_cat':'jensenshannon',
                                                                'num_threshold':0.1,
                                                                'cat_threshold':0.1}):

    report = Report(metrics=[
        DataDriftPreset(num_stattest = test_params['method_num'], # 選擇連續變數檢定 預設
                        cat_stattest = test_params['method_cat'], # 選擇類別變數檢定
                        num_stattest_threshold = test_params['num_threshold'],
                        cat_stattest_threshold = test_params['cat_threshold']),
    ])
    """
    reference_data: 歷史資料
    current_data: 最新資料
    """
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()
    drift_score = extract_drift_scores(result)
    
    report.save_html(f'{output_path}/result.html')
    evi_df = pd.DataFrame(drift_score).T.reset_index().rename({0:f'stats_drift_score', 'index':'feature'}, axis='columns')
    evi_df.sort_values(by='stats_drift_score', ascending=False)
    evi_df.to_csv(f'{output_path}/stats_drift_scores.csv', index=False)
    return evi_df
    #     report.show(mode='inline')