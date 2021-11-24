import tensorflow as tf


def ecarenet_loss_wrap(ab=[0.5, 0], lz_mean=False):
    """
    maximum likelihood loss (similar to https://arxiv.org/abs/1809.02403)
    :param ab: alpha beta - for weighting Lz vs Lc, Lc_uncensored
                alpha * l_z + (1-alpha)*(l_c_censored + beta*l_c_uncensored)
    :param lz_mean: in Lz, use mean over all intervals without event
    :return:
    """
    @tf.function
    def ecarenet_loss_core(y_true, y_pred):
        """
        y_true is survival time and y_pred is hazard, the hazard should be highest when the survival is first time 0:
        survival: [1, 1, 1, 0, 0]
        hazard:   [0, 0, 0, 1, 0]
        :param y_true: [batch_size, n_intervals+1]
                       1 for each interval that was survived, 0 otherwise + last entry is censoring information
        :param y_pred: [batch_size, n_intervals] - hazard per interval (P[event happens | event did not happen yet])
        :return: single float
        """
        # alpha * l_z + (1-alpha)*(l_c_censored + beta*l_c_uncensored)
        alpha = ab[0]
        beta = ab[1]
        censored = y_true[:, -1]  # censored = \delta
        y_true = y_true[:, :-1]
        epsilon = 10e-7
        # modify: if no event in any interval, set censored to true, set event to last interval
        censored = tf.cast((tf.reduce_sum(tf.ones_like(y_true), 1) == tf.reduce_sum(y_true, 1)) | (censored == 1), 'float32')
        y_true = tf.concat((y_true[:, :-1], tf.zeros_like(y_true)[:, :1]), 1)
        # last interval with 1, add "1" in front of every batch to get interval when event happens
        # [0,0,0,0,0] - hazard [1,0,0,0,0] - event happens in int 0
        # [1,1,0,0,0] - hazard [0,0,1,0,0] - event happens in int 2
        y_true_no0 = tf.concat((tf.ones_like(y_true)[:, :1], y_true), 1)
        hazard_mask = (y_true_no0[:, :-1] - y_true_no0[:, 1:])

        loss_z_p1 = tf.reduce_sum(tf.math.log(y_pred+epsilon)*hazard_mask, 1)*(1-censored)
        loss_z_p2 = tf.reduce_sum(tf.math.log(1-y_pred+epsilon) * y_true, axis=1) * (1-censored)
        loss_z = -tf.reduce_sum((loss_z_p1, loss_z_p2), 0)   # [batchsize, 1]

        # loss1 = censoring_aware_loss_core(y_true, y_pred)
        y_pred_mask = tf.concat((tf.ones_like(y_true)[:, :1], y_true), 1)[:, :-1]   # y_true
        # S(t|x) = prod(1-output) prod up to true exit day
        y_pred_surv = tf.reduce_prod(1-y_pred*y_pred_mask, 1)
        loss_c = -tf.reduce_sum((censored * tf.math.log(y_pred_surv+epsilon),
                                 beta*(1-censored) * tf.math.log(1-y_pred_surv+epsilon)), 0)

        loss_all = tf.reduce_sum((alpha*loss_z, (1-alpha)*loss_c), 0)
        return loss_all
    return ecarenet_loss_core
