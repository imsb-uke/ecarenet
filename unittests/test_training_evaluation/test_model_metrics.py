import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc
from training_evaluation.model_metrics import tf_f1_score_wrap, cohens_kappa_wrap, \
    brier_score_censored_core, concordance_td_core, d_calibration, cdauc_uno, cd_auc_uno_plot


class TestModelMetricsClassification(unittest.TestCase):
    def test_f1_score(self):
        # this is the example from the web page
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        y_true = [[0, 0],
                  [1, 0],
                  [1, 1],
                  [0, 0],
                  [1, 0],
                  [1, 1]]
        y_pred = [[0, 0],
                  [1, 1],
                  [1, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0]]
        f1 = tf_f1_score_wrap(None)
        f1_metric = f1(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
        self.assertAlmostEqual(0.2666666666666666, float(np.array(f1_metric)))

    def test_cohens_kappa(self):
        # this is the example from the web page
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]
        y_true = [[0, 0],
                  [1, 0],
                  [1, 1],
                  [0, 0],
                  [1, 0],
                  [1, 1]]
        y_pred = [[0, 0],
                  [1, 1],
                  [1, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0]]
        kappa = cohens_kappa_wrap(None)
        kappa = kappa(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))
        # returns a scalar value
        self.assertEqual(kappa.shape, [])


class TestModelMetricsSurvival(unittest.TestCase):
    def test_brier_perfect(self):
        y_true = np.array([[9, 0],
                           [13, 0]])
        y_pred = pd.DataFrame([[1, 1, 1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 0]],
                              columns=[2, 4, 6, 8, 10, 12, 14]).transpose()
        brier = brier_score_censored_core(y_true, y_pred)
        self.assertEqual(0, brier)

        y_true = pd.DataFrame([[9, 0], [13, 0]], columns=['event_month', 'is_censored'])
        brier = brier_score_censored_core(y_true, y_pred)
        self.assertEqual(0, brier)

    def test_brier_imperfect(self):
        y_true = np.array([[3.1], [4.9]])
        censored = np.array([[0], [0]])
        y_true = np.concatenate((y_true, censored), 1)
        y_pred = np.array([[0.9, 0.9, 0.9, 0.1, 0.1],
                           [0.9, 0.9, 0.1, 0.1, 0.1]]).transpose()
        y_pred = pd.DataFrame(y_pred)
        y_pred.index = [1, 2, 3, 4, 5]

        brier = brier_score_censored_core(y_true, y_pred, False)
        mse_0101 = (0.1**2 + 0.1**2)/2
        mse_0109 = (0.9**2 + 0.1**2)/2
        expected_mse = [mse_0101, mse_0101, mse_0109, mse_0109, mse_0101]
        expected_brier = auc([1, 2, 3, 4, 5], expected_mse) / (5-1)
        self.assertAlmostEqual(expected_brier, brier)

    def test_cindex_perfect(self):
        y_true = np.array([[9, 0],
                           [13, 0]])
        y_pred = pd.DataFrame([[0.8, 0.8, 0.7, 0.5, 0.1, 0.1, 0.1],
                               [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2]],
                              columns=[2, 4, 6, 8, 10, 12, 14]).transpose()
        cindex = concordance_td_core(y_true, y_pred)
        self.assertEqual(1, cindex)

        y_true = pd.DataFrame([[9.8, 0], [13.4, 0]],
                              columns=['event_month', 'is_censored'])
        cindex = concordance_td_core(y_true, y_pred)
        self.assertEqual(1, cindex)
        y_pred = pd.DataFrame([[1, 1, 1, 1, 0.1, 0.1, 0.1],
                               [1, 1, 1, 1, 0.9, 0.9, 0.2]],
                              columns=[2, 4, 6, 8, 10, 12, 14]).transpose()
        cindex = concordance_td_core(y_true, y_pred)
        self.assertEqual(1, cindex)

    def test_d_calibration_perfect(self):

        # intervals where events happen:
        y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # probability to survive: 1, 0.8, 0.6, 0.4, 0.2
        censored = np.zeros(len(y_true))
        y_true = np.concatenate(([y_true], [censored]), 0).transpose()
        # survival curves:
        y_pred = pd.DataFrame(
            np.repeat([[0.99, 0.8, 0.72, 0.6, 0.51, 0.4, 0.3, 0.2, 0.1, 0.01]], 20, 0), columns=[1,2,3,4,5,6,7,8,9,10])
        mse, b, b_norm = d_calibration(y_true, y_pred, 10, True)
        self.assertEqual(0, mse)

    def test_d_calibration_imperfect(self):
        target = pd.DataFrame({"event_days": [15, 35], "has_event": [1, 0]})
        target['is_censored'] = 1-target['has_event']
        predictions = pd.DataFrame({0:  [1,   1],
                                    10: [0.8, 0.9],
                                    20: [0.6, 0.3],
                                    30: [0.4, 0.2],
                                    40: [0,   0.1]}).transpose()
        dcal, b, b_norm = d_calibration(target, predictions, 10, return_b=True,
                                        event_time_col='event_days', censored_col='is_censored')

        # event 15: 0.6  # no event 35: 0.1
        expected_b = np.array([0.1/0.1, (0.1-0.1)/0.1, 0, 0, 0, 0, 1, 0, 0, 0])
        expected_b_norm = expected_b/2
        expected_dcal = np.sum((expected_b_norm - 0.1)**2)

        np.testing.assert_array_equal(expected_b_norm, b_norm)
        np.testing.assert_array_equal(expected_b, b)
        self.assertEqual(expected_dcal, dcal)

    def test_cdauc_perfect(self):
        y_true = np.array([[9, 0],
                           [13, 0]])
        y_pred = pd.DataFrame([[0.8, 0.8, 0.7, 0.5, 0.1, 0.1, 0.1],
                               [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2]],
                              columns=[2, 4, 6, 8, 10, 12, 14]).transpose()
        cdauc, _ = cdauc_uno(y_true, y_pred)
        self.assertEqual(1, cdauc)

        y_true = pd.DataFrame([[9.8, 0], [13.4, 0]],
                              columns=['event_month', 'is_censored'])
        cdauc, _ = cdauc_uno(y_true, y_pred)
        self.assertEqual(1, cdauc)
        y_pred = pd.DataFrame([[1, 1, 1, 1, 0.1, 0.1, 0.1],
                               [1, 1, 1, 1, 0.9, 0.9, 0.2]],
                              columns=[2, 4, 6, 8, 10, 12, 14]).transpose()
        cdauc, _ = cdauc_uno(y_true, y_pred)
        self.assertEqual(1, cdauc)

    def test_cdauc_equal(self):
        y_true = np.array([[9, 0],
                           [13, 0]])
        y_pred = pd.DataFrame([[0.8, 0.8, 0.7, 0.5, 0.1, 0.1, 0.1],
                               [0.8, 0.8, 0.7, 0.5, 0.1, 0.1, 0.1]],
                              columns=[2, 4, 6, 8, 10, 12, 14]).transpose()
        cdauc, _ = cdauc_uno(y_true, y_pred)
        # for same prediction for all patients, AUC should be 0.5
        self.assertAlmostEqual(0.5, cdauc)

    def test_cdauc_single_pred(self):
        y_true = np.array([[2, 0],
                           [13, 0],
                           [25, 1],
                           [31, 1],
                           [8, 0],
                           [4, 0],
                           [10, 0]])
        y_pred = pd.DataFrame([[0.9], [0.3], [0.4], [0.14], [0.8], [0.76], [0.35]],
                              columns=[-1]).transpose()
        cd_auc_uno_plot(y_true, y_pred)
        from matplotlib import pyplot as plt
        plt.show()

    def test_cdauc_plot(self):
        y_true = np.array([[9, 0],
                           [13, 0],
                           [10, 0]])
        y_pred = pd.DataFrame([[0.8, 0.8, 0.6, 0.5, 0.4, 0.4, 0.1],
                               [0.9, 0.85, 0.7, 0.6, 0.5, 0.2, 0.2],
                               [0.9, 0.85, 0.7, 0.6, 0.5, 0.2, 0.2]],
                              columns=[2, 4, 6, 8, 10, 12, 14]).transpose()
        cdauc, _ = cdauc_uno(y_true, y_pred, c_step_no=3)

        _, cdauc_plt = cd_auc_uno_plot(y_true, y_pred, c_step_no=3)

        self.assertEqual(cdauc, cdauc_plt)
        self.assertLess(cdauc, 1)
