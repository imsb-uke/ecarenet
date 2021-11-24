import pandas as pd
import numpy as np
import unittest

from dataset_creation.convert_to_tf_dataset import make_image_dataset


class TestConvertToTfDataset(unittest.TestCase):
    def test_make_image_dataset(self):
        """
        test if a set of strings and arrays (for image path and label) can be turned into a tf.dataset successfully
        """
        dataset = make_image_dataset(pd.Series(['img1.png', 'img2', 'img3']),
                                     np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]),
                                     [])
        for d in dataset.take(1):
            self.assertEqual(d['image_paths'], 'img1.png')

    def test_make_image_dataset_additional(self):
        """
        test if a set of strings and arrays (for image path and label) can be turned into a tf.dataset successfully
        """
        dataset = make_image_dataset(pd.Series(['img1.png', 'img2', 'img3']),
                                     np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]),
                                     {'censored': [0, 1, 0]})
        for d in dataset.take(1):
            self.assertEqual(d['image_paths'], 'img1.png')
            self.assertEqual(d['censored'], 0)
