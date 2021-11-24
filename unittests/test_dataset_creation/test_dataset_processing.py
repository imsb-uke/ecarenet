import tensorflow as tf
import numpy as np
import unittest
import os

from dataset_creation.dataset_processing import create_patched_dataset, get_class_distribution, squeeze_dataset
from unittests.unittest_helpers import create_image_dataset


class TestDatasetProcessing(unittest.TestCase):
    def test_create_patched_dataset(self):
        patch_size = 1024
        patching_config = {'patch_size': patch_size,
                           'overlap': 0,
                           'n_patches': 6,
                           'order': 'shuffle',
                           'keep_original': False}

        dataset = create_image_dataset(1)
        dataset = dataset.map(lambda x: {'images': x})
        dataset_patched = create_patched_dataset(dataset, patching_config, 3)
        for d in dataset_patched.take(1):
            self.assertTrue(d['images'].shape == [6, 1024, 1024, 3])

    def test_create_patched_dataset_keeporig(self):
        patch_size = 1024
        patching_config = {'patch_size': patch_size,
                           'overlap': 0,
                           'n_patches': 6,
                           'order': 'ranked',
                           'keep_original': 128}

        dataset = create_image_dataset(1)
        dataset = dataset.map(lambda x: {'images': x})
        dataset_patched = create_patched_dataset(dataset, patching_config, 3)
        for d in dataset_patched.take(1):
            self.assertTrue(d['images'].shape == [6, 1024, 1024, 3])
            self.assertTrue(d['img_original'].shape == [128, 128, 3])


class TestClassDistribution(unittest.TestCase):
    def test_get_class_distribution_isup(self):
        # 0:0x, 1:0x 2:2x 3:1x
        label_array = np.array([2, 2, 3])
        data_generation_config = {'number_of_classes': 6,
                                  'label_type': 'isup'}
        class_dist = get_class_distribution(label_array, data_generation_config)
        self.assertListEqual(list(class_dist), [0, 0, 2, 1, 0, 0])

    def test_get_class_distribution_bin(self):
        # 0:2x, 1:1x
        label_array = np.array([0, 0, 1])
        data_generation_config = {'number_of_classes': 2,
                                  'label_type': 'bin'}
        class_dist = get_class_distribution(label_array, data_generation_config)
        self.assertListEqual(list(class_dist), [2, 1])

    def test_get_class_distribution_survival(self):
        label_array = np.array([2, 2, 3, 8, 8, 5, 0])
        data_generation_config = {'number_of_classes': 9,
                                  'label_type': 'survival'}
        class_dist = get_class_distribution(label_array, data_generation_config)
        self.assertListEqual(list(class_dist), [1, 0, 2, 1, 0, 1, 0, 0, 2, 0])


class TestSqueezing(unittest.TestCase):
    def test_squeeze_dataset(self):
        # create an image patch of size 100x100x3
        image_patch = tf.convert_to_tensor(np.ones((100,100,3)))
        # create a tuple with patches of size 100x100x3
        image_patch = tf.expand_dims(image_patch, 0)
        # image_patches has 4 patches
        image_patches = tf.concat((image_patch, image_patch, image_patch, image_patch), 0)
        # batch with size 2
        image_patches_batched = tf.concat((tf.expand_dims(image_patches, 0), tf.expand_dims(image_patches, 0)), 0)
        images_patches_again = squeeze_dataset(image_patches_batched, 2, 'images')
        # now should be batchsize+n_patches(2*4=8) x 100 x 100 x 3
        self.assertListEqual(list(images_patches_again.shape), [8, 100, 100, 3])
