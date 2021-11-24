import shutil
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from sacred.observers import FileStorageObserver
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from training_evaluation.evaluate_survival import create_survival_result_dataframe, evaluate_survival_model, \
                                                  surv_curve_and_risk_from_hazard
from unittests.unittest_helpers import create_image_dataset, create_image_label_dataset
from settings.sacred_experiment import ex


class TestEvaluateSurvival(unittest.TestCase):
    def test_surv_curve_and_risk_from_hazard_eqint(self):
        p_hazard = np.array([[0.1,0.1,0.1,0.1,0.5,0.4,0.1,0.1,0.1,0.1]])
        interval_limits = np.array([1,2,3,4,5,6,7,8,9,10], 'float32')
        surv_curve, risk = surv_curve_and_risk_from_hazard(p_hazard, interval_limits)
        expected_survival_curve = [0.9, 0.9**2, 0.9**3, 0.9**4,
                                   0.9**4*0.5, 0.9**4*0.5*0.6,
                                   0.9**5*0.5*0.6, 0.9**6*0.5*0.6, 0.9**7*0.5*0.6, 0.9**8*0.5*0.6]
        expected_risk = 1-(sum(expected_survival_curve)/10)
        [self.assertAlmostEqual(expected_survival_curve[i], list(surv_curve)[i], 3) for i in range(10)]
        self.assertAlmostEqual(expected_risk, risk, 5)

    def test_surv_curve_and_risk_from_hazard_unequal_intervals(self):
        p_hazard = np.array([[0.1,0.1,0.1,0.1,0.5,0.4,0.1,0.1,0.1,0.1]])
        interval_limits = np.array([1,2,3,4,5,6,7,8,9,12], 'float32')
        surv_curve, risk = surv_curve_and_risk_from_hazard(p_hazard, interval_limits)
        expected_survival_curve = [0.9, 0.9**2, 0.9**3, 0.9**4,
                                   0.9**4*0.5, 0.9**4*0.5*0.6,
                                   0.9**5*0.5*0.6, 0.9**6*0.5*0.6, 0.9**7*0.5*0.6, 0.9**8*0.5*0.6]
        expected_risk = 1-((sum(expected_survival_curve[:-1]) + expected_survival_curve[-1]*3)/12)
        [self.assertAlmostEqual(expected_survival_curve[i], list(surv_curve)[i], 3) for i in range(10)]
        self.assertAlmostEqual(expected_risk, risk)


    def test_create_survival_result_dataframe(self):
        model = tf.keras.applications.InceptionV3(include_top=True, weights=None, classes=10, input_shape=(128, 128, 3))
        dataset = create_image_dataset(3, True)
        dataset = dataset.map(lambda x: {'images': x,
                                         'labels': np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                                         'interval_limits': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                         'image_paths': 'path1',
                                         'original_label': 3.2
                                         })
        dataset = dataset.batch(1)
        df = create_survival_result_dataframe(model, dataset, [2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.iloc[0].img_path == 'path1')
        self.assertAlmostEqual(df.iloc[0].event_month, 3.2)
        self.assertAlmostEqual(df.iloc[0].is_censored, 0)
        self.assertTrue('hazard_1.00' in df.columns)
        self.assertTrue('surv_1.00' in df.columns)

    def test_evaluate_survival_model(self):
        model = tf.keras.applications.InceptionV3(include_top=True, weights=None, classes=10, input_shape=(128, 128, 3))
        dataset = create_image_label_dataset(3,
                                             [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
                                             True)

        dataset = dataset.map(lambda x, y: {'images': x,
                                            'labels': y,
                                            'interval_limits': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'float32'),
                                            'image_paths': 'path1',
                                            'original_label': tf.cast(tf.reduce_sum(y), tf.float32) + 0.2
                                            })
        dataset = dataset.batch(1)
        experiments_dir = '/opt/project/unittest_experiment'

        # don't store in default experiment folder, but in folder for unittests (will be created if not there yet)
        ex.observers.pop()
        ex.observers.append(FileStorageObserver(experiments_dir + '/'))

        @ex.main
        def main_fn(_run):
            result_metrics = evaluate_survival_model(model, dataset,
                                                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                     ['auc_uno', 'brier', 'c_index', 'd_calibration'],
                                                     experiments_dir,
                                                     _run._id)
            self.assertAlmostEqual(result_metrics['cd_auc_uno'], 0.5)   # same prediction per patient: auc=0.5
            self.assertGreater(result_metrics['brier_score'], 0.0)
            self.assertAlmostEqual(result_metrics['c_index'], 0.5)   # same prediction per patient: cindex=0.5

        ex.run()
        shutil.rmtree(experiments_dir)
