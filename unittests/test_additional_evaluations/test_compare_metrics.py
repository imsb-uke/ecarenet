import pandas as pd
import unittest

from additional_evaluations.metrics_comparison_barplot import compare_models_barplot
from unittests.unittest_helpers import get_model_directory


class TestCompareMetrics(unittest.TestCase):
    def test_compare_models_errorbar(self):
        df_info = [['1', 'model1', 'tab:blue'],
                   ['2', 'model1', 'tab:blue'],
                   ['3', 'model2', 'tab:red']
                   ]
        df_info = pd.DataFrame(df_info, columns=['index', 'info', 'color'])
        df_info.set_index('index', inplace=True)
        model_directory = get_model_directory()
        res = compare_models_barplot(['1', '2', '3'], ['cdauc', 'brier', 'cindex', 'dcalib'], df_info,
                                     model_directory, save_fig=get_model_directory(), limit=None)
        print(res)
