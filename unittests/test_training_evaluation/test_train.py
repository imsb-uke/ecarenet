from sacred.observers import FileStorageObserver
import tensorflow as tf
import numpy as np
import unittest
import shutil
import os

from training_evaluation.train import train_loop
from sacred import Experiment
from unittests.unittest_helpers import create_image_dataset
from settings.sacred_experiment import ex


class TestTraining(unittest.TestCase):
    def test_train_loop_bin(self):
        # in training mode, datasets don't seem exactly reproducible, but when iterated over all datapoints
        unittest_exp_path = '/opt/project/unittest_experiment'
        train_params = {'epochs': 2,
                        'initial_epoch': 0,
                        'callbacks': None,
                        'monitor_val': 'val_binary_accuracy',
                        'compile_metrics': ['binary_accuracy'],
                        'model_save_path': unittest_exp_path+'/'}
        ex.add_config(train_params)
        ex.command(train_loop)
        # don't store in default experiment folder, but in folder for unittests
        ex.observers.pop()
        ex.observers.append(FileStorageObserver(unittest_exp_path+'/'))


        @ex.main
        def main_fn(_run):
            _run._id = 1
            model = tf.keras.applications.InceptionV3(include_top=True, classes=2, input_shape=(128, 128, 3), weights=None)
            model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=tf.keras.losses.binary_crossentropy,
                          metrics=[tf.keras.metrics.BinaryAccuracy()])
            model._metrics = [tf.keras.metrics.BinaryAccuracy()]
            # fit needs to be called once in order to "activate" the metrics ???

            dataset = create_image_dataset(2, True)

            dataset = dataset.map(lambda x: {'images': x,
                                             'labels': np.array([1, 0])})
            dataset = dataset.batch(2)

            m, history = train_loop(model=model,
                                    train_dataset=dataset,
                                    valid_dataset=dataset,
                                    train_batch_size=2,
                                    valid_batch_size=2,
                                    train_params=train_params,
                                    train_class_distribution=np.array([2, 0]),
                                    valid_class_distribution=np.array([2, 0]),
                                    class_weights=None,
                                    label_type=11
                                    )
            # directory with experiment result should be created by sacred
            self.assertTrue(os.path.exists(unittest_exp_path))
            self.assertTrue(os.path.exists(os.path.join(unittest_exp_path, '1')))
            self.assertGreater(len(os.listdir(os.path.join(unittest_exp_path, '1'))), 0)

            self.assertTrue(isinstance(history, tf.keras.callbacks.History))
            self.assertTrue(isinstance(m, tf.keras.models.Model))
        ex.run()
        shutil.rmtree(unittest_exp_path)

