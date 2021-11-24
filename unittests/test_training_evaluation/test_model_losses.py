import numpy as np
import unittest
import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from training_evaluation.model_losses import ecarenet_loss_wrap


class TestLosses(unittest.TestCase):

    def test_ecarenet_loss(self):
        y_true = tf.constant([[1, 1, 0, 0], [1, 1, 1, 0]], dtype='float32')
        y_pred = tf.constant([[0.1, 0.4, 0.7, 0.3], [0.5, 0.4, 0.7, 0.3]], dtype='float32')
        censored = tf.constant([0, 0])
        true_int = [2, 3]

        # ### conservative way
        y_p = np.array(y_pred)
        # loss_z part 1
        loss_z1 = [0, 0]
        for idx, p in enumerate(y_p):
            if censored[idx] != 1:
                loss_z1[idx] = (np.log(p[true_int[idx]]))
        # loss_z part 2
        loss_z2 = [0, 0]
        for idx, p in enumerate(y_p):
            if censored[idx] != 1:
                for i in range(true_int[idx]):
                    loss_z2[idx] += np.log(1 - p[i])
        # loss_z
        loss_z = - np.array(loss_z1) - np.array(loss_z2)

        # loss_c
        surv = [1, 1]
        for idx, p in enumerate(y_p):
            for i in range(true_int[idx] + 1):
                surv[idx] = surv[idx] * (1 - p[i])
        loss_c = [0, 0]
        for idx, p in enumerate(y_p):
            loss_c[idx] = tf.cast(censored[idx], 'float32') * np.log(surv[idx]) + \
                          (1 - tf.cast(censored[idx], 'float32')) * np.log(1 - surv[idx])
        loss_c = -np.array(loss_c)

        loss_I = 0.5 * loss_z + 0.5 * loss_c
        loss_I = np.mean(loss_I)

        # my implementation
        y_true_cens = tf.concat((y_true, tf.cast(tf.expand_dims(censored, 1), tf.float32)), 1)
        censoring_DRSA_loss = ecarenet_loss_wrap([0.5, 1])
        loss_II = censoring_DRSA_loss(y_true_cens, y_pred)
        loss_II = np.mean(loss_II)

        self.assertAlmostEqual(loss_I, loss_II, places=4)


