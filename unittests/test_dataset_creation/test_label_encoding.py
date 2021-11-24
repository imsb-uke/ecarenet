import numpy as np
import unittest

from dataset_creation.label_encoding import transform_labels, label_to_int, int_to_string_label


class TestLabelEncodingIsup(unittest.TestCase):
    """
    Test if label encoding works correct for isup classification
    """
    def test_transform_labels(self):
        config = {'label_type': 'isup',
                  'number_of_classes': 6}
        labels = ['0', '1', '2', '3', '4', '5']
        # labels = [0,1,2,3,4]
        labels_transformed, _, _ = transform_labels(labels, config)
        self.assertListEqual(list(labels_transformed[0]), [0, 0, 0, 0, 0])
        self.assertListEqual(list(labels_transformed[1]), [1, 0, 0, 0, 0])
        self.assertListEqual(list(labels_transformed[2]), [1, 1, 0, 0, 0])
        self.assertListEqual(list(labels_transformed[3]), [1, 1, 1, 0, 0])
        self.assertListEqual(list(labels_transformed[4]), [1, 1, 1, 1, 0])
        self.assertListEqual(list(labels_transformed[5]), [1, 1, 1, 1, 1])

        self.assertTrue(np.all(labels_transformed == np.array([[0, 0, 0, 0, 0],
                                                               [1, 0, 0, 0, 0],
                                                               [1, 1, 0, 0, 0],
                                                               [1, 1, 1, 0, 0],
                                                               [1, 1, 1, 1, 0],
                                                               [1, 1, 1, 1, 1]])))


class TestLabelEncodingBin(unittest.TestCase):
    """
    Test if label encoding works correct for binary classification
    """
    def test_transform_labels(self):
        config = {'label_type': 'bin',
                  'number_of_classes': 2}
        labels = ['0', '1']
        labels_transformed, _, _ = transform_labels(labels, config)
        self.assertTrue(np.all(labels_transformed == np.array([[1, 0],
                                                               [0, 1]])))


class TestLabelEncodingSurvival(unittest.TestCase):
    """
    Test if label encoding works correct for survival prediction
    """
    def test_transform_labels(self):
        config = {'label_type': 'survival',
                  'number_of_classes': 28}
        labels = ['0', '4', '13.4']
        labels_transformed, _, _ = transform_labels(labels, config)
        expected_labels = np.zeros(28)
        self.assertTrue(np.all(labels_transformed[0] == expected_labels))
        expected_labels[0] = 1
        self.assertTrue(np.all(labels_transformed[1] == expected_labels))
        expected_labels[1] = 1
        expected_labels[2] = 1
        expected_labels[3] = 1
        self.assertTrue(np.all(labels_transformed[2] == expected_labels))


class TestLabelToInt(unittest.TestCase):
    def test_label_to_int_bin(self):
        label = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        expected_output = [0, 0, 1, 1]
        output = label_to_int(label, 'bin')
        self.assertListEqual(expected_output, list(output))

    def test_label_to_int_isup(self):
        label = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        expected_output = [0, 1, 5]
        output = label_to_int(label, 'isup')
        self.assertListEqual(expected_output, list(output))

    def test_label_to_int_survival(self):
        label = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 0]])
        expected_output = [0, 1, 5, 7]
        output = label_to_int(label, 'survival')
        self.assertListEqual(expected_output, list(output))


class TestIntToString(unittest.TestCase):
    def test_int_to_string_isup(self):
        string_label = int_to_string_label(4, 'isup')
        self.assertEqual(string_label, '3+5/4+4/5+3 / 4')

        string_label = int_to_string_label(1, 'bin')
        self.assertEqual(string_label, '1')
