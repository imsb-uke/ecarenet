import pandas as pd
import unittest
import os

from dataset_creation.read_from_csv import create_img_and_label_series


def get_csv():
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'unittest_data/train.csv'))
    return csv_file


class TestReadFromCsv(unittest.TestCase):

    def test_create_img_and_label_series_isup_no_additional_columns(self):
        config = {"directory": 'unittest_data',
                  "data_generation": {'annotation_column': 'isup',
                                      'additional_columns': None,
                                      'drop_cases': None}
                  }
        csv_file = get_csv()
        images, labels, additional_columns = create_img_and_label_series(csv_file, config['data_generation'])
        self.assertIsInstance(images, pd.Series)
        self.assertIsInstance(labels, pd.Series)
        self.assertListEqual(additional_columns, [])
        self.assertEqual(images[0], 'example_colors/img_0.png')
        self.assertEqual(labels[0], 0)

    def test_create_img_and_label_series_relapse_time_no_additional_columns(self):
        config = {"directory": 'unittest_data',
                  "data_generation": {'annotation_column': 'relapse_time',
                                      'additional_columns': None,
                                      'drop_cases': None}
                  }
        csv_file = get_csv()
        images, labels, additional_columns = create_img_and_label_series(csv_file, config['data_generation'])
        self.assertIsInstance(images, pd.Series)
        self.assertIsInstance(labels, pd.Series)
        self.assertListEqual(additional_columns, [])
        self.assertEqual(images[0], 'example_colors/img_0.png')
        self.assertEqual(labels[0], 48.1)

    def test_create_img_and_label_series_relapse_time_additional_columns(self):
        config = {"directory": 'unittest_data',
                  "data_generation": {'annotation_column': 'relapse_time',
                                      'additional_columns': [{"censored": ["censored", 'label']}],
                                      'drop_cases': None}
                  }
        csv_file = get_csv()
        images, labels, additional_columns = create_img_and_label_series(csv_file, config['data_generation'])
        self.assertIsInstance(images, pd.Series)
        self.assertIsInstance(labels, pd.Series)
        self.assertIsInstance(additional_columns, dict)
        self.assertEqual(additional_columns['censored'][0], 0)

    def test_create_img_and_label_series_relapse_time_drop(self):
        # in this test, drop all uncensored cases, so only censored=1 is left
        config = {"directory": 'unittest_data',
                  "data_generation": {'annotation_column': 'relapse_time',
                                      'additional_columns': [{"censored": ["censored", 'label']}],
                                      'drop_cases': [['0', 'censored']]}
                  }
        csv_file = get_csv()
        images, labels, additional_columns = create_img_and_label_series(csv_file, config['data_generation'])
        self.assertIsInstance(images, pd.Series)
        self.assertIsInstance(labels, pd.Series)
        self.assertIsInstance(additional_columns, dict)
        self.assertEqual(additional_columns['censored'][0], 1)
        self.assertEqual(additional_columns['censored'].unique(), [1])
