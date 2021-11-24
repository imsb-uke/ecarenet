
import unittest

from models.ecare_net import ecare_net


class TestMSurv(unittest.TestCase):
    def test_ecare_net(self):
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
        model = ecare_net( model_config)
        # input is list of [image and [binary_classification, time interval]]
        self.assertIsInstance(model.input, list)
        self.assertIsInstance(model.input[1], list)
        self.assertListEqual(list(model.input[0].shape), [None, 100, 100, 3])
        self.assertListEqual(list(model.input[1][0].shape), [None, 1])
        self.assertListEqual(list(model.input[1][1].shape), [None, 28])
        # output is batch_size x n_time_intervals
        self.assertListEqual(list(model.output.shape), [None, 28])

    def test_ecare_net_nopatch(self):
        model_config = {'base_model': 'InceptionV3',
                        'n_patches': 1,
                        'n_classes': 28,
                        'additional_input': ['bin'],
                        'rnn_layer_nodes': [256],
                        'dense_layer_nodes': [64],
                        'self_attention': False,
                        'mil_layer': False,
                        'keras_model_params': {'input_shape': [100, 100, 3],
                                               'weights': None}}
        model = ecare_net(model_config)
        # input is list of [image and [binary_classification, time interval]]
        self.assertIsInstance(model.input, list)
        self.assertIsInstance(model.input[1], list)
        self.assertListEqual(list(model.input[0].shape), [None, 100, 100, 3])
        self.assertListEqual(list(model.input[1][0].shape), [None, 1])
        self.assertListEqual(list(model.input[1][1].shape), [None, 28])
        # output is batch_size x n_time_intervals
        self.assertListEqual(list(model.output.shape), [None, 28])

