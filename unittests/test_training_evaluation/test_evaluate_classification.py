from sacred.observers import FileStorageObserver
import tensorflow as tf
import numpy as np
import unittest
import shutil
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


from training_evaluation.evaluate_classification import create_classification_result_dataframe, \
                                                        evaluate_classification_model
from unittests.unittest_helpers import create_image_dataset, create_image_label_dataset
from settings.sacred_experiment import ex


class TestEvaluateClassification(unittest.TestCase):
    def test_create_classification_result_dataframe(self):
        model = tf.keras.applications.InceptionV3(include_top=True, weights=None, classes=5, input_shape=(128, 128, 3))
        dataset = create_image_dataset(3, True)
        dataset = dataset.map(lambda x: {'images': x,
                                         'labels': np.array([1, 1, 1, 0, 0]),
                                         'image_paths': 'path1',
                                         })
        dataset = dataset.batch(1)
        df = create_classification_result_dataframe(model, dataset, [3, 0], 'isup', 5)
        self.assertTrue(df['img_path'][0] == 'path1')
        self.assertEqual(df['groundtruth_class'][0], 3)

    def test_evaluate_classification_model_isup(self):
        model = tf.keras.applications.InceptionV3(include_top=True, weights=None, classes=5, input_shape=(128, 128, 3))
        dataset = create_image_label_dataset(3,
                                             [[1, 1, 1, 1, 0],
                                              [1, 0, 0, 0, 0],
                                              [1, 1, 1, 1, 1]],
                                             True)
        dataset = dataset.map(lambda x, y: {'images': x,
                                            'labels': y,
                                            'image_paths': 'path1'
                                            })
        dataset = dataset.batch(1)
        experiments_dir = '/opt/project/unittest_experiment'
        ex.observers.pop()
        ex.observers.append(FileStorageObserver(experiments_dir + '/'))


        @ex.main
        def main_fn(_run):
            result_metrics = evaluate_classification_model(model, dataset, [0, 1, 0, 0, 1, 1], 'isup',
                                                           ['accuracy', 'kappa', 'f1_score'],
                                                           experiments_dir, _run._id)

            # model always predicts class 1, so one of the three predictions is correct
            self.assertAlmostEqual(result_metrics['accuracy'], 1/3)
            self.assertTrue(-1.0 <= result_metrics['cohens_kappa'] <= 1.0)
            self.assertTrue(result_metrics['f1_score'] <= 1.0)
            self.assertTrue(os.path.isfile(os.path.join(experiments_dir, _run._id, 'confusion_matrix.png')))
            self.assertTrue(os.path.isfile(os.path.join(experiments_dir, _run._id, 'confusion_matrix_relative.png')))

        ex.run()
        shutil.rmtree(experiments_dir)

    def test_evaluate_classification_model_bin(self):
        model = tf.keras.applications.InceptionV3(include_top=True, weights=None, classes=2,
                                                  input_shape=(128, 128, 3))
        dataset = create_image_label_dataset(3,
                                             [[1, 0],
                                              [0, 1],
                                              [0, 1]],
                                             True)
        dataset = dataset.map(lambda x, y: {'images': x,
                                            'labels': y,
                                            'image_paths': 'path1'
                                            })
        dataset = dataset.batch(1)
        experiments_dir = '/opt/project/unittest_experiment'
        ex.observers.pop()
        ex.observers.append(FileStorageObserver(experiments_dir + '/'))

        @ex.main
        def main_fn(_run):
            result_metrics = evaluate_classification_model(model, dataset, [1, 2], 'bin',
                                                           ['accuracy', 'kappa', 'f1_score'],
                                                           experiments_dir, _run._id)

            # model always predicts class 1, so one of the three predictions is correct
            self.assertTrue(0 <= result_metrics['accuracy'] <= 1)
            self.assertTrue(-1.0 <= result_metrics['cohens_kappa'] <= 1.0)
            self.assertTrue(result_metrics['f1_score'] <= 1.0)
            self.assertTrue(os.path.isfile(os.path.join(experiments_dir, _run._id, 'confusion_matrix.png')))
            self.assertTrue(
                os.path.isfile(os.path.join(experiments_dir, _run._id, 'confusion_matrix_relative.png')))

        ex.run()
        shutil.rmtree(experiments_dir)
