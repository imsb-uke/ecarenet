import unittest
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import numpy as np
from dataset_creation.image_transformations.patching import create_patches, advanced_patching, tf_create_patches
from unittests.unittest_helpers import create_image_label_dataset


class TestPatching(unittest.TestCase):

    def test_create_patches(self):
        """
        test that takes as input an image and cuts it into patches.
        Different sorting orders for patches are available and all tested here.
        Everywhere the resulting shape of the patches is known and tested for.
        """
        debug = True # False   # if set to True, will plot patches
        # read an image
        directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'unittest_data', 'example_colors'))
        image = tf.io.read_file(os.path.join(directory, "img_red_0.png"))
        image = tf.image.decode_image(image, 3)
        image = tf.cast(image, dtype=tf.dtypes.float32)
        image = tf.image.resize(image, (2048, 2048))
        image = image.numpy()

        # test with order = ranked
        patches, idx_v, idx_h, _ = create_patches(image, 512, 0, 0, n_patches=4, order='ranked')
        self.assertEqual(patches.shape, (4, 512, 512, 3))

        # test with order = original
        patches, idx_v, idx_h, _ = create_patches(image, 512, 0, 0, order='original')
        self.assertTrue(len(idx_v) == len(idx_h) == 16)
        self.assertEqual(patches.shape, (16, 512, 512, 3))
        self.assertFalse(np.all(patches[-1] == [33, 22, 17]))
        # the indices of patches that are returned are supposed to be in original order here
        self.assertTrue(np.all(idx_h == np.array([0, 512, 1024, 1536,
                                                  0, 512, 1024, 1536,
                                                  0, 512, 1024, 1536,
                                                  0, 512, 1024, 1536])))
        self.assertTrue(np.all(idx_v == np.array([0, 0, 0, 0,
                                                  512, 512, 512, 512,
                                                  1024, 1024, 1024, 1024,
                                                  1536, 1536, 1536, 1536])))
        if debug:
            for i in range(len(patches)):
                ax = plt.subplot(4, 4, i + 1)
                ax.axis('off')
                ax.imshow(patches[i] / 255)

            plt.show()

        # test with order = shuffle
        patches, idx_v, idx_h, _ = create_patches(image, 512, 0, 0, n_patches=9, order='shuffle')
        self.assertTrue(len(idx_v) == len(idx_h) == 9)

    def test_create_patches_overlap(self):
        """
        test that takes as input an image and cuts it into patches.
        Different sorting orders for patches are available and all tested here.
        Everywhere the resulting shape of the patches is known and tested for.
        """
        debug = False   # if set to True, plots patches
        # read an image
        directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'unittest_data', 'example_colors'))
        image = tf.io.read_file(os.path.join(directory, "img_red_0.png"))
        image = tf.image.decode_image(image, 3)
        image = tf.cast(image, dtype=tf.dtypes.float32)
        image = tf.image.resize(image, (2048, 2048))
        image = image.numpy()

        # test with order = ranked
        patches, idx_v, idx_h, _ = create_patches(image, 512, 16, 16, n_patches=16, order='ranked')
        self.assertEqual(patches.shape, (16, 512, 512, 3))

        # test with order = original
        patches, idx_v, idx_h, _ = create_patches(image, 512, 16, 16, n_patches=16, order='original')
        self.assertTrue(len(idx_v) == len(idx_h) == 16)
        self.assertEqual(patches.shape, (16, 512, 512, 3))
        if debug:
            for i in range(len(patches)):
                ax = plt.subplot(5, 5, i + 1)
                ax.axis('off')
                ax.imshow(patches[i] / 255)

            plt.show()

        # the indices of patches that are returned are supposed to be in original order here
        self.assertTrue(np.all(idx_h == np.array([0, 512-16, 1024-32, 1536-48, 2048-64,
                                                  0, 512-16, 1024-32, 1536-48, 2048-64,
                                                  0, 512-16, 1024-32, 1536-48, 2048-64,
                                                  0])))
        self.assertTrue(np.all(idx_v == np.array([0, 0, 0, 0, 0,
                                                  512-16, 512-16, 512-16, 512-16, 512-16,
                                                  1024-32, 1024-32, 1024-32, 1024-32, 1024-32,
                                                  1536-48])))

        # test with order = shuffle
        patches, idx_v, idx_h, _ = create_patches(image, 512, 16, 16, n_patches=9, order='shuffle')
        self.assertTrue(len(idx_v) == len(idx_h) == 9)
        # test with order = shuffle_ranked
        patches, idx_v, idx_h, _ = create_patches(image, 512, 16, 16, n_patches=9, order='shuffle_ranked')
        self.assertTrue(len(idx_v) == len(idx_h) == 9)
        # test with more patches than usually cut (extend with white patches)
        patches, idx_v, idx_h, _ = create_patches(image, 512, 16, 16, n_patches=30, order='ranked')
        self.assertTrue(len(idx_v) == len(idx_h) == 30)
        if debug:
            for i in range(30):
                plt.subplot(5, 6, i + 1)
                plt.imshow(patches[i] / 255)
                plt.axis('off')
            plt.show()

    def test_create_patches_map(self):
        """
        test if patching function also works with tensorflow mapping function, since this can cause some problems
        otherwise
        """
        # create a dataset with an image and an artificial label
        labelin = "3"
        patch_size = 512
        overlap = 0
        n_patches = 16
        order = 'original'
        dataset = create_image_label_dataset(1, labelin)

        # apply pathing function
        dataset = dataset.map(lambda x, y: (tf_create_patches(x, patch_size, overlap, overlap, n_patches, order, 3)[0], y))
        for d in dataset.take(1):
            self.assertEqual(d[0].shape, (n_patches, 512, 512, 3))

    def test_advanced_patching(self):
        """
        test the function advanced_patching separately
        """
        directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'unittest_data', 'example_colors'))
        image = tf.io.read_file(os.path.join(directory, "img_red_0.png"))
        image = tf.image.decode_image(image, 3)
        image = tf.cast(image, dtype=tf.dtypes.float32)
        image = image.numpy()
        patch_size = 500
        n_patches = 4
        patches, _, _, _ = advanced_patching(image, patch_size, n_patches, order='ranked')
        self.assertEqual(patches.shape, (4, 500, 500, 3))


