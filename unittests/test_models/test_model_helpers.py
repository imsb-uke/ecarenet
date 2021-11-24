import tensorflow as tf
import unittest

from models.m_bin import m_bin
from models.m_isup import m_isup
from models.ecare_net import ecare_net
from models.model_helpers import compile_model


class TestModelCompileMBin(unittest.TestCase):
    def test_mbin_compile(self):
        model_config = {'base_model': 'InceptionV3',
                        'n_classes': 2,
                        'keras_model_params': {'input_shape': [100, 100, 3],
                                               'weights': None},
                        'dense_layer_nodes': [2, 2]}
        model = m_bin(model_config)

        train_params = {'optimizer': {'name': 'Adam',
                                      'params': [{'learning_rate': 0.01}]},
                        'loss_fn': 'binary_crossentropy',
                        'compile_metrics': ['tf_categorical_accuracy'],   # needs to be the class name in CamelCase
                        'compile_attributes': {}}
        # fitting should not work without compiling
        with self.assertRaises(RuntimeError):
            model.fit(tf.zeros((1, 100, 100, 3)), tf.zeros((1, 2)))

        model = compile_model(model, train_params, None)

        self.assertEqual(model.loss.__name__, 'binary_crossentropy')
        self.assertTrue(model.metrics != [])   # only works for tf-gpu 2.1 :-(


class TestModelCompileMIsup(unittest.TestCase):
    def test_misup_compile(self):
        model_config = {'base_model': 'InceptionV3',
                        'n_classes': 6,
                        'keras_model_params': {'input_shape': [100, 100, 3],
                                               'weights': 'imagenet'},
                        'dense_layer_nodes': None}
        model = m_isup(model_config)
        train_params = {'optimizer': {'name': 'Adam',
                                      'params': [{'learning_rate': 0.01}]},
                        'loss_fn': 'categorical_crossentropy',
                        'compile_metrics': ['CategoricalAccuracy', 'tf_f1_score', 'cohens_kappa'],   # needs to be the class name in CamelCase
                        'compile_attributes': {}}
        model = compile_model(model, train_params, None)
        self.assertEqual(model.loss.__name__, 'categorical_crossentropy')


class TestModelCompileEcarenet(unittest.TestCase):
    def test_ecarenet_compile(self):
        model_config = {'base_model': 'InceptionV3',
                        'n_patches': 4,
                        'n_classes': 28,
                        'additional_input': ['bin'],
                        'rnn_layer_nodes': [256],
                        'dense_layer_nodes': [64],
                        'self_attention': True,
                        'mil_layer': True,
                        'keras_model_params': {'input_shape': [100, 100, 3],
                                               'weights': None}}
        model = ecare_net(model_config)
        train_params = {'optimizer': {'name': 'Adam',
                                      'params': [{'learning_rate': 0.01}]},
                        'loss_fn': 'ecarenet_loss',
                        'compile_metrics': None,
                        'compile_attributes': {}}
        model = compile_model(model, train_params, None)
        self.assertEqual(model.loss.__name__, 'ecarenet_loss_core')


