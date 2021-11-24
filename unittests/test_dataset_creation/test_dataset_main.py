import tensorflow as tf
import numpy as np
import unittest
import os

from dataset_creation.dataset_main import create_dataset, adjust_dataset_to_model
from unittests.unittest_helpers import create_image_dataset


class TestDatasetMain(unittest.TestCase):
    def test_create_dataset_survival(self):
        """
        test if dataset generation works for survival prediction: patches should be created here
        """
        directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'unittest_data/'))
        csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'unittest_data/train.csv'))
        patching_config = {'patch_size': 1024,
                           'overlap': 0,
                           'n_patches': 6,
                           'order': 'shuffle',
                           'keep_original': False}
        augmentation_config = {'rotation': 0.,
                               'brightness': 0.1}
        config = {"directory": 'unittest_data',
                  "data_generation": {'cache': None,
                                      'train_csv_file': csv_file,
                                      'train_batch_size': 2,
                                      'label_type': 'survival',
                                      'number_of_classes': 28,
                                      'annotation_column': 'relapse_time',
                                      'additional_columns': [{"censored": ["censored", 'label']}],
                                      'drop_cases': None,
                                      'patching': patching_config,
                                      'resize': [128, 128],
                                      'random_augmentation': False,
                                      'augmentation_config': augmentation_config,
                                      'seed': 234,
                                      'directory': directory}
                  }
        # the seed makes the random shuffling reproducible, so the image path can be compared to an expected value
        a, b = create_dataset(config['data_generation'], 'train')
        for d in a.take(1):
            self.assertListEqual(list(d['images'][0].shape), [12, 128, 128, 3])
            self.assertEqual(d['censored'][0], 0)
            self.assertEqual(d['image_paths'][0], 'example_colors/img_1.png')
            self.assertIsInstance(d['images'], tuple)
            np.testing.assert_array_equal(d['images'][-1][0], d['interval_limits'][0])
            self.assertLess(np.max(d['images'][0]), 10)

    def test_create_dataset_isup(self):
        """
        test if dataset generation works for isup label: no patches should be created here
        """
        directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'unittest_data/'))
        csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'unittest_data/train.csv'))
        patching_config = None
        config = {"directory": 'unittest_data',
                  "data_generation": {'cache':None,
                                      'valid_csv_file': csv_file,
                                      'valid_batch_size': 2,
                                      'label_type': 'isup',
                                      'number_of_classes': 6,
                                      'annotation_column': 'isup',
                                      'additional_columns': [{"censored": ["censored", 'label']}],
                                      'drop_cases': None,
                                      'patching': patching_config,
                                      'resize': [128, 128],
                                      'seed': 234,
                                      'directory': directory}
                  }
        a, b = create_dataset(config['data_generation'], 'valid')
        for d in a.take(1):
            self.assertEqual(d['censored'][0], 0)
            self.assertEqual(d['image_paths'][0], 'example_colors/img_0.png')
            self.assertIsInstance(d['images'], tf.Tensor)
            self.assertListEqual(list(d['images'].shape), [2, 128, 128, 3])
            self.assertRaises(KeyError, lambda: d['interval_limits'])
        print('done')


class TestAdditionalFunctions(unittest.TestCase):
    def test_adjust_dataset_to_model_noadditionalinput_isup(self):
        label = "3"

        intervals = [1, 2, 3, 4, 5]

        dataset = create_image_dataset(2)
        dataset = dataset.map(lambda x: {'images': x, 'labels': label, 'interval_limits': intervals})
        data_generation_config = {'label_type': 'isup',
                                  'patching': None,
                                  'additional_columns': None}

        dataset_adjusted, _ = adjust_dataset_to_model(dataset, data_generation_config)
        # nothing should have changed
        for a, b in zip(dataset.take(1), dataset_adjusted.take(1)):
            self.assertEqual(a['labels'], b['labels'])
            self.assertTrue(np.all(a['images'] == b['images']))

    def test_adjust_dataset_to_model_noadditionalinput_survival(self):
        label = "3"
        intervals = [1, 2, 3, 4, 5]

        dataset = create_image_dataset(2)
        dataset = dataset.map(lambda x: {'images': x, 'labels': label, 'interval_limits': intervals})
        data_generation_config = {'label_type': 'survival',
                                  'patching': None,
                                  'additional_columns': None}

        dataset_adjusted, _ = adjust_dataset_to_model(dataset, data_generation_config)
        # interval limits should be included into "images", so this can be read as one input later
        for a, b in zip(dataset.take(1), dataset_adjusted.take(1)):
            self.assertEqual(a['labels'], b['labels'])
            self.assertTrue(np.all(a['images'] == b['images'][0]))
            self.assertTrue(np.all(np.array(a['interval_limits'], 'float32') == b['images'][1]))

    def test_adjust_dataset_to_model_additionalinput(self):
        label = "3"

        intervals = [1, 2, 3, 4, 5]
        additional_inputs = [0, 1]

        dataset = create_image_dataset(2)
        dataset = dataset.map(lambda x1: {'images': x1, 'labels': label,
                                                      'interval_limits': intervals,
                                                      'additional_information': additional_inputs[0]})

        data_generation_config = {'label_type':'survival',
                                  'patching': None,
                                  'additional_columns': [{'additional_information': ['important_column', 'input']}]}

        dataset_adjusted, _ = adjust_dataset_to_model(dataset, data_generation_config)
        # interval limits should be included into "images", so this can be read as one input later
        # "additional_information" should be included into "images" label, too
        for a, b in zip(dataset.take(1), dataset_adjusted.take(1)):
            self.assertEqual(a['labels'], b['labels'])
            self.assertTrue(np.all(a['images'] == b['images'][0]))
            self.assertEqual(0, b['images'][1])
            self.assertTrue(np.all(np.array(a['interval_limits'], 'float32') == b['images'][2]))

    def test_adjust_dataset_to_model_additionallabel(self):
        label = [11.]
        directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'unittest_data', 'example_colors'))
        image = tf.io.read_file(os.path.join(directory, "img_red_0.png"))
        image = tf.image.decode_image(image, 3)
        image = tf.image.resize(image, (2048, 2048))
        image = tf.cast(image, dtype=tf.dtypes.float32)
        intervals = [1, 2, 3, 4, 5]
        additional_labels = [100, 10001]

        dataset = tf.data.Dataset.from_tensor_slices(([image, image],
                                                      [label, label],
                                                      [intervals, intervals],
                                                      [additional_labels[0], additional_labels[1]]))
        dataset = dataset.map(lambda x1, x2, x3, x4: {'images': x1, 'labels': x2,
                                                      'interval_limits': x3,
                                                      'name_of_patient': x4})

        data_generation_config = {'label_type':'survival',
                                  'patching': None,
                                  'additional_columns': [{'name_of_patient': ['name_column', 'label']}]}

        dataset_adjusted, _ = adjust_dataset_to_model(dataset, data_generation_config)
        # interval limits should be included into "images", so this can be read as one input later
        # "name_of_patient" should be included into "labels"
        for a, b in zip(dataset.take(1), dataset_adjusted.take(1)):

            self.assertTrue(np.all(a['images'] == b['images'][0]))
            self.assertTrue(np.all(np.array(a['interval_limits'], 'float32') == b['images'][1]))
            self.assertEqual(100, b['labels'][1])
