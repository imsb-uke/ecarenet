import tensorflow as tf
import pandas as pd
import numpy as np

from training_evaluation.model_metrics import cd_auc_uno_plot, concordance_td_core, \
                                              brier_score_censored_core, plot_dcal
from training_evaluation.evaluation_helpers import save_plot_to_artifacts, get_survival_curve
from settings.sacred_experiment import ex


def surv_curve_and_risk_from_hazard(p_hazard, interval_limits):
    """
    from hazard prediction per interval, calculate survival probability and risk score
    :param p_hazard: tf.Tensor/np.array [batch_size, n_intervals] hazard prediction per interval as float
    :param interval_limits: array [n_intervals, ] with limits of each interval
    :return: surv_curve as np.array [n_intervals,] and risk as single float
    """
    # ### from hazard to survival curve \prod(1-h)
    surv_curve = np.cumprod(1 - p_hazard)

    # ### from survival curve to risk
    # lim is vector of length per interval, e.g. [3,3,3,3,....] for equally spaced intervals (has to be float!)
    lim = tf.cast(interval_limits - tf.concat((tf.zeros_like(interval_limits)[:1], interval_limits[:-1]), 0), 'float32')

    # risk is now "1-relative area under the survival curve"
    risk = sum(lim * surv_curve)
    risk_max = sum(lim)
    p_risk = 1 - tf.expand_dims(tf.expand_dims(risk / risk_max, 0), 0)
    p_risk = float(tf.squeeze(p_risk))
    return surv_curve, p_risk


def create_survival_result_dataframe(model, dataset, class_distribution):
    """
    For each datapoint in the dataset, predict the outcome hazard. Convert hazard to survival and this to risk. Save all
    information for a single datapoint in a pandas dataframe. Batch size of dataset needs to be 1
    :param model: tensorflow/keras model
    :param dataset: tf.data Dateset in dictionary form with batch size 1 and at least
                    'images': tf.Tensor  [batch_size, h, w, 3]
                    'interval_limits': tf.Tensor [batch_size, n_intervals]
                    'image_paths': strings [batch_size, ]
                    'original_label': float  [batch_size, ] true month of event
                    'labels': tf.Tensor [batch_size, n_intervals+1] e.g. [1,1,1,1,1,0,0,0,1] last is censoring info
    :param class_distribution: array, how many examples per class are in dataset
    :return: pandas Dataframe with one row per example. Columns
             'img_path': string, path/to/image/file.png
             'event_month': float, month of event or censoring
             'is_censored': 1 if censored, 0 if not
             *hazard_limits: for each time interval, hazard prediction at time t_k
             *surv_limits: for each time interval, survival prediction at time t_k
             'risk': float, risk from survival curve
    """

    df_all = pd.DataFrame()
    for idx, d in enumerate(dataset.take(int(sum(class_distribution)))):
        if idx == 0:
            interval_limits = d['interval_limits'][0, :]
            surv_limits = ['surv_%.2f' % ll for ll in interval_limits]
            hazard_limits = ['hazard_%.2f' % ll for ll in interval_limits]
            # resulting dataframe should contain
            df_all = pd.DataFrame(columns=['img_path',   # string, path/to/image/file.png
                                           'event_month',   # float, month of event or censoring
                                           'is_censored',   # 1 if censored, 0 if not
                                           *hazard_limits,   # for each time interval, hazard prediction at time t_k
                                           *surv_limits,   # for each time interval, survival prediction at time t_k
                                           'risk'])   # float, risk from survival curve
        # ### run model to get prediction
        p_hazard = model(d['images'])
        surv_curve, p_risk = surv_curve_and_risk_from_hazard(p_hazard, d['interval_limits'][0, :])

        # add all information for this single image/patient to dataframe
        data_row = [d['image_paths'].numpy()[0].decode('utf8'),
                    *d['original_label'].numpy(),
                    *d['labels'][..., -1].numpy(),
                    *p_hazard[0].numpy(),
                    *surv_curve,
                    p_risk]
        df_all = df_all.append({x: y for x, y in zip(df_all.columns, data_row)}, ignore_index=True)
    return df_all


@ex.capture
def evaluate_survival_model(model, dataset, class_distribution, metrics, experiments_dir, run_id):
    """

    :param model:
    :param dataset:
    :param class_distribution:
    :param metrics:
    :param experiments_dir:
    :param run_id:
    :return: dictionary with resulting metrics. Possible are cd_auc_uno, brier_score
    """
    result_metrics = dict()
    results = create_survival_result_dataframe(model, dataset, class_distribution)
    results.to_csv(experiments_dir + '/' + str(run_id) + '/results.csv')

    y_true_survtime = get_y_true_survtime_censored(results)
    y_pred_survival, _ = get_survival_curve(results)

    # metric could be auc or auc_uno or aucuno, all are recognized
    if np.any(['auc' in m for m in metrics]):
        target = y_true_survtime
        prediction = y_pred_survival
        f, auc_u = cd_auc_uno_plot(target, prediction,
                                   tau_2=prediction.index[-1],
                                   event_time_col='event_month',
                                   censored_col='is_censored')
        print('cd auc uno:', auc_u)
        result_metrics['cd_auc_uno'] = float(auc_u)
        save_plot_to_artifacts(f, 'cd_auc_uno', experiments_dir, run_id)

    if np.any(['brier' in m for m in metrics]):
        brier = brier_score_censored_core(y_true_survtime, y_pred_survival)
        print('brier score: ', brier)
        result_metrics['brier_score'] = float(brier)

    if np.any(['c_index' in m for m in metrics]):
        cindex = concordance_td_core(y_true_survtime, y_pred_survival)
        print('c index: ', cindex)
        result_metrics['c_index'] = float(cindex)

    if np.any(['d_calibration' in m for m in metrics]):
        target = y_true_survtime
        f, mse = plot_dcal(target, y_pred_survival, 10)
        save_plot_to_artifacts(f, 'd_calibration', experiments_dir, run_id)
        print("d calibration: ", mse)
        result_metrics['d_calibration_error'] = float(mse)

    return result_metrics


def get_y_true_survtime_censored(survival_dataframe):
    """
    From dataframe, select only event month and censoring information
    :param survival_dataframe: dataframe that contains many information (output of create_survival_result_dataframe)
    :return: pandas Dataframe with two columns: 'event_month', 'is_censored'
    """
    y_true = survival_dataframe[['event_month', 'is_censored']]
    return y_true


# def get_y_pred_survival(survival_dataframe):
#     """
#     From survival dataframe, get the columns that contain the survival prediction information
#     :param survival_dataframe: dataframe that contains many information (output of create_survival_result_dataframe)
#     :return: pd.Dataframe with only survival predictions per interval, one row per interval, one column per patient
#     """
#     y_pred = survival_dataframe[[x for x in survival_dataframe.columns if x.startswith('surv')]]
#     limits = [float(x.split('_')[1]) for x in y_pred.columns]
#     y_pred = y_pred.transpose()
#     y_pred.index = limits
#     return y_pred


def get_y_pred_risk(survival_dataframe):
    """
    from survival dataframe, extract column "risk"
    :param survival_dataframe: dataframe that contains many information (output of create_survival_result_dataframe)
    :return: pandas Series with only column risk
    """
    return survival_dataframe['risk']
