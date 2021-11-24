import unittest
from matplotlib import pyplot as plt
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from additional_evaluations.attention import plot_attention_for_model
from unittests.unittest_helpers import get_model_directory


class TestAttention(unittest.TestCase):
    def test_attention(self):
        model_directory = get_model_directory()
        plot_attention_for_model(model_directory, run_id=4, n_examples=8, save_plot=True, mode='valid')
        plt.show()
