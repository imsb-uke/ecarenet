import tensorflow as tf
import numpy as np
import unittest
from dataset_creation.image_transformations.normalization import normalize_image, tf_normalize_image


class TestNormalization(unittest.TestCase):
    # array of 4 images of shape 200x200 with 3 channels
    def test_normalization_rgb(self):
        image = np.ones((4, 200, 200, 3))*100
        image = normalize_image(image, 3)
        self.assertEqual(np.unique(image), 100/255)

    def test_normalization_4channels(self):
        # array of 4 images of shape 200x200 with 4 channels (only normalize first 3)
        image = np.ones((4, 200, 200, 4))*100
        image = normalize_image(image, 3)
        self.assertEqual(np.unique(image[..., :-1]), 100/255)
        self.assertEqual(np.unique(image[..., -1]), 100)

    # Tensor with 4 images of shape 200x200 with 3 channels
    def test_normalization_tf(self):
        image = np.ones((4, 200, 200, 3))*100
        image = tf.convert_to_tensor(image)
        image = tf_normalize_image(image, 3)
        self.assertEqual(np.unique(image), 100/255)

