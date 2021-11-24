import unittest

from models.m_isup import m_isup


class TestMIsup(unittest.TestCase):
    def test_m_isup(self):
        model_config = {'base_model': 'InceptionV3',
                        'n_classes': 6,
                        'keras_model_params': {'input_shape': [100, 100, 3],
                                               'weights': 'imagenet'},
                        'dense_layer_nodes': None}
        model = m_isup(model_config)
        self.assertListEqual(list(model.input.shape), [None, 100, 100, 3])
        # for six classes, only return 5 output nodes because of ordinal regression
        self.assertListEqual(list(model.output.shape), [None, 5])
        # make sure imagenet weights are loaded
        # self.assertAlmostEqual(float(model.weights[0][0][0]), 0.80290633)

