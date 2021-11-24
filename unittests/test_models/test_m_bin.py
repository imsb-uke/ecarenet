
import unittest
import os

from models.m_bin import m_bin
from unittests.unittest_helpers import get_model_directory


class TestMBin(unittest.TestCase):
    def test_m_bin(self):
        model_config = {'base_model': 'InceptionV3',
                        'n_classes': 2,
                        'keras_model_params': {'input_shape': [100, 100, 3],
                                               'weights': 'imagenet'},
                        'dense_layer_nodes': [16]}
        model = m_bin(model_config)
        self.assertListEqual(list(model.input.shape), [None, 100, 100, 3])
        self.assertListEqual(list(model.output.shape), [None, 2])
        # make sure imagenet weights are loaded
        self.assertAlmostEqual(model.weights[0][0][0][0][0], -0.45910555)

    def test_m_bin_load(self):
        # TODO: add json file to unittest_data
        model_directory = get_model_directory()
        model_config = {'base_model': model_directory,
                        'n_classes': 2,
                        'keras_model_params': {'input_shape': [100, 100, 3],
                                               'weights': 'imagenet'},
                        'cut_off_layer': 164,
                        'dense_layer_nodes':[4]}
        model = m_bin(model_config)
        # input shape is now defined by loaded model
        self.assertListEqual(list(model.input.shape), [None, 100, 100, 3])
        self.assertListEqual(list(model.output.shape), [None, 2])

