import unittest

from additional_evaluations.risk_stratification import find_best_risk_intervals, evaluate_risk_groups
from unittests.unittest_helpers import get_model_directory


class TestRiskStratification(unittest.TestCase):
    def test_find_best_risk_intervals(self):
        find_best_risk_intervals([4], get_model_directory(),  'valid', save_plot=True,
                                 number_intervals=2, step_intervals=0.02,
                                 logrank_limit=0.05, possible_intervals=False, log_weight=True)

    def test_evaluate_risk_curves(self):
        # not a well trained model here, so risk scores are all very close, but for testing of method it's sufficient
        evaluate_risk_groups(4, get_model_directory(), 'valid', save_plot=True,
                             interval_limits=[0.880, 0.881], log_weight=True)
