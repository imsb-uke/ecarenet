from matplotlib import pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import unittest
import os
from dataset_creation.image_transformations.data_augmentation import get_random_params, tf_augment_image


def get_image_directory():
    return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'unittest_data', 'example_colors'))


class TestDataAugmentation(unittest.TestCase):
    def test_get_random_params_nochange(self):
        rotation, \
          width_shift, height_shift, \
          brightness, \
          horizontal_flip, vertical_flip = get_random_params(seed=None,
                                                             rotation_range=0.,
                                                             width_shift_range=0.,
                                                             height_shift_range=0.,
                                                             brightness_range=0.,
                                                             horizontal_flip=False,
                                                             vertical_flip=False)
        # no paramter should be set to something other than 0 or False
        self.assertEqual(rotation, 0)
        self.assertEqual(width_shift, 0)
        self.assertEqual(height_shift, 0)
        self.assertEqual(brightness, 0)
        self.assertEqual(horizontal_flip, False)
        self.assertEqual(vertical_flip, False)

    def test_get_random_params(self):
        rotation, \
          width_shift, height_shift, \
          brightness, \
          horizontal_flip, vertical_flip = get_random_params(seed=None,
                                                             rotation_range=90.,
                                                             width_shift_range=10.,
                                                             height_shift_range=20.,
                                                             brightness_range=0.9,
                                                             horizontal_flip=True,
                                                             vertical_flip=False)
        self.assertLess(rotation, 90)
        self.assertGreater(rotation, 0)
        self.assertLess(width_shift, 10)
        self.assertLess(height_shift, 20)
        self.assertLess(brightness, 0.9)
        self.assertGreater(brightness, -0.9)

    def test_augment_image(self):
        directory = get_image_directory()
        image = tf.io.read_file(os.path.join(directory, "img_red_0.png"))
        image = tf.image.decode_image(image, 3)
        image = tf.image.resize(image, (2048, 2048))
        image = tf.cast(image, dtype=tf.dtypes.float32)
        image = tf.expand_dims(image, 0)
        image = tf.concat((image, image, image, image), 0)
        image_aug = tf_augment_image(image, random_augmentation=False,
                                     intensity=0.1,
                                     rotation=45.,
                                     width_shift=100.,
                                     height_shift=500.,
                                     brightness=0.3,
                                     horizontal_flip=False,
                                     vertical_flip=False,
                                     fill_mode='constant',
                                     cval=255,
                                     )
        # visually check: image should be rotated by 45° to the left, shifted down (upper part is white),
        # shifted a bit to the right and it should be of lighter color than before
        plt.subplot(1, 2, 1)
        plt.imshow(image[0]/255)
        plt.subplot(1, 2, 2)
        plt.imshow(image_aug[0]/255)
        plt.show()
