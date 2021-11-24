from matplotlib import pyplot as plt
import numpy as np
import unittest
import cv2
import os
import sys
sys.path.append('/opt/project')

from preparation.cut_center_pieces import show_cut_square, fit_circle
from unittests.unittest_helpers import get_data_directory


class TestCutCenterPiece(unittest.TestCase):
    def test_show_cut_square(self):
        # TODO somehow this is not printing the result : (
        df, f = show_cut_square(path=get_data_directory('circle'),
                                n_examples=3,
                                show_diff='both',
                                show_rect_or_cut='show',
                                df=None)
        print(f)
        plt.show()

    def test_fit_circle(self):
        img = cv2.imread(os.path.join(get_data_directory('circle'), 'img02.png'))
        im_circle, flag = fit_circle(img, show_rect_or_cut='cut')
        plt.imshow(im_circle)
        plt.show(block=False)
        self.assertListEqual([2048, 2048, 3], list(im_circle.shape))
