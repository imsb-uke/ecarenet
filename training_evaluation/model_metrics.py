from sklearn.metrics import f1_score, cohen_kappa_score, auc
from lifelines import KaplanMeierFitter
from scipy.stats import chi2, chisquare
from pycox.evaluation import EvalSurv
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

from training_evaluation.training_helpers import split_y_true_censored, merge_y_true_to_df


# --- METRICS FOR ISUP CLASSIFICATION --- #

def tf_f1_score_wrap(class_weights):
    """
    F1 score for classification
    :param class_weights:
    :return:
    """

    def tf_f1_score(y_true, y_pred):
        # wrapper for numpy function
        return tf.numpy_function(f1_score_core, [y_true, y_pred], tf.double)

    def f1_score_core(y_true, y_pred):
        """
        :param y_true: np array that is encoded as ordinal regression, e.g. [[1,1,1,0,0]] for class [3]
        :param y_pred: np array that is encoded as ordinal regression, e.g. [[1,1,1,0,0]] for class [3]

        :return: single float
        """
        y_true = np.round(np.sum(y_true, axis=1))
        y_pred = np.round(np.sum(y_pred, axis=1))
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1
    return tf_f1_score


def cohens_kappa_wrap(class_weights):
    """
    cohens kappa
    :param class_weights: None or array with weight per class
    :return:
    """
    def cohens_kappa(y_true, y_pred):
        # wrapper for numpy function
        return tf.numpy_function(kappa_core, [y_true, y_pred], tf.double)

    def kappa_core(y_true, y_pred):
        if len(y_true.shape) is not 1:
            y_true = np.round(np.sum(y_true, axis=1))
            y_pred = np.round(np.sum(y_pred, axis=1))
        if class_weights is not None:
            sample_weight = [class_weights[y_true[i]] for i in range(y_true.shape[0])]
        else:
            sample_weight = None

        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic', sample_weight=sample_weight)
        return kappa

    return cohens_kappa


class TfCategoricalAccuracy(tf.keras.metrics.Metric):
    """
    Categorical accuracy needs to be defined new, because the resulting class in this case is not max(vector) but sum()
    """
    def __init__(self, name='tf_categorical_accuracy', **kwargs):
        super(TfCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.metr = 0

    def update_state(self, y_true, y_pred, class_weights=None):
        if (len(y_true.shape) is not 1):
            y_true = np.round(np.sum(y_true, axis=1))
            y_pred = np.round(np.sum(y_pred, axis=1))
        if class_weights is not None:
            sample_weight = [class_weights[int(y_true[i])] for i in range(y_true.shape[0])]
        else:
            sample_weight = None
        acc = np.average(tf.equal(y_true, y_pred), weights=sample_weight)
        # acc = np.mean(tf.equal(y_true, y_pred))
        self.metr = acc

    def result(self):
        return self.metr


def tf_categorical_accuracy_wrap(class_weights):
    def tf_categorical_accuracy(y_true, y_pred):
        # wrapper for numpy function
        return tf.numpy_function(categorical_accuracy_core, [y_true, y_pred], tf.double)

    def categorical_accuracy_core(y_true, y_pred):
        """

        :param y_true: shape [batch_size, n_output_nodes]
        :param y_pred:

        :return:

        """
        if (len(y_true.shape) is not 1):
            y_true = np.round(np.sum(y_true, axis=1))
            y_pred = np.round(np.sum(y_pred, axis=1))
        if class_weights is not None:
            sample_weight = [class_weights[int(y_true[i])] for i in range(y_true.shape[0])]
        else:
            sample_weight = None
        acc = np.average(tf.equal(y_true, y_pred), weights=sample_weight)
        # acc = np.mean(tf.equal(y_true, y_pred))
        return acc
    return tf_categorical_accuracy


#  --- METRICS FOR SURVIVAL MODEL --- #


def brier_score_censored_wrap(class_weights=None):
    # not needed because it will not be used during training, just for evaluation
    return brier_score_censored


def brier_score_censored(y_true, y_pred):
    return tf.numpy_function(brier_score_censored_core, [y_true, y_pred], tf.double)


def brier_score_censored_core(y_true, y_pred,
                              event_time_col='event_month', censored_col='is_censored'):
    """
    from pycox implementation
    see https://github.com/havakv/pycox/blob/master/examples/administrative_brier_score.ipynb
    :param y_true: array [2, n_examples] with survival time and censoring information (e.g.[[3.4, 0], [9.8,1]])
                   OR
                   dataframe with ['event_month', 'is_censored']
    :param y_pred: survival prediction dataframe with one row per interval, one column per example, index is interval limits!
    :param event_time_col: only used if y_true is dataframe (which column to get true event time from)
    :param censored_col: only used if y_true is dataframe (which column to get censoring time from)
    :return: brier score as single value
    """
    y_orig, censored = split_y_true_censored(y_true, event_time_col, censored_col)

    ev = EvalSurv(pd.DataFrame(y_pred),
                  y_orig,
                  1 - censored, 'km', steps='pre')
    time_grid = np.array(y_pred.index)
    brier = ev.integrated_brier_score(time_grid)
    return brier


def concordance_td_core(y_true, y_pred):
    """
    from pycox implementation
    :param y_true: array [n_examples, 2]] with survival time and censored
    :param y_pred: dataframe with one row per interval, one column per example, index is interval limits!
    :return: float cindex
    """
    from pycox.evaluation import EvalSurv
    y_orig, censored = split_y_true_censored(y_true)

    ev = EvalSurv(pd.DataFrame(y_pred),
                  y_orig,
                  1 - censored, 'km', steps='pre')
    cindex = ev.concordance_td()
    return cindex


def d_calibration(target, prediction, n_buckets, return_b=False, return_b_event_censored=False,
                  event_time_col='event_month', censored_col='is_censored'):
    """
    d-calibration score
    :param target: dataframe with is_censored and event_month as default columns
            (can be named differently, but this needs to be defined when calling the function)
    :param prediction: dataframe with one row per interval, one column per patient

    :return: mean squared error and p-value
    """
    target = merge_y_true_to_df(target, event_time_col=event_time_col, censored_col=censored_col)
    target = target.assign(has_event=lambda row: 1-row['is_censored']).astype('int')
    if len(target) != len(prediction):
        prediction = prediction.transpose()   # now prediction has one patient per row, one column per interval
    # for each patient, get survival curve at time of event
    bucket_survival_limits = prediction.columns   # e.g. 0,10,20,30,40
    # digitize: x <= [b for b in buckets],
    # leave out last bucket, so patients living longer than that still get assigned to last bucket
    bucket_each_patient = np.digitize(target['event_month'], bucket_survival_limits[:-1])
    #
    # surv_probability_each_patient = np.zeros(len(prediction))
    # for idx, b in enumerate(bucket_each_patient):
    #     surv_probability_each_patient[idx] = prediction.iloc[idx, b]

    surv_probability_each_patient = np.diag(prediction.iloc[:, bucket_each_patient])

    buckets = np.round(np.linspace(0, 1, n_buckets + 1), 3)
    buckets[-1] = 1.1
    b_event = np.zeros(n_buckets)
    b_censored = np.zeros(n_buckets)
    for k in range(len(buckets) - 1):   # 0 0.25 0.5 0.75 1
        p_k = buckets[k]
        p_k1 = buckets[k + 1]
        patients_in_bucket = ((surv_probability_each_patient >= p_k)
                              & (surv_probability_each_patient < p_k1))

        eq1 = sum(patients_in_bucket * target['has_event'])

        is_censored = (1 - target['has_event']).to_numpy()
        censored_patients_in_bucket = (patients_in_bucket * is_censored)

        eq2 = ((surv_probability_each_patient - p_k)
               / surv_probability_each_patient)[censored_patients_in_bucket == 1].sum()

        eq3 = ((p_k1 - p_k) / surv_probability_each_patient)[
            (surv_probability_each_patient >= p_k1) & (is_censored == 1)].sum()

        b_event[k] = eq1
        b_censored[k] = np.nansum((eq2, eq3))

    b = b_event + b_censored
    b_event_normalized = b_event / len(prediction)
    b_censored_normalized = b_censored / len(prediction)
    b_normalized = b / len(prediction)

    # chi squared test
    dof = n_buckets-1
    alpha = 0.05
    p = 1-alpha
    critical_value = chi2.ppf(p, dof)
    s, p = chisquare(b)
    if s >= critical_value:
        print('chi squared test failed with p value ', p)
    else:
        print('chi squared test passed with p value ', p)

    # error prediction
    expected_value = 1 / n_buckets
    mse = np.sum((expected_value - b_normalized)**2)
    if return_b_event_censored:
        return mse, b_event_normalized, b_censored_normalized
    elif return_b:
        return mse, b, b_normalized
    else:
        return mse, p


def plot_dcal(target, prediction, n_buckets,
                  event_time_col='event_month', censored_col='is_censored'):
    """

    :param target: pd.Dataframe with two columns: event_month (float) and has_event (0 or 1)
    :param prediction:
    :param n_buckets:
    :param event_time_col:
    :param censored_col:
    :return:
    """
    mse, b_event_normalized, b_censored_normalized = d_calibration(target, prediction, n_buckets,
                                                              return_b=False, return_b_event_censored=True,
                                                              event_time_col=event_time_col, censored_col=censored_col)
    f = plt.figure()
    plt.barh(np.arange(n_buckets), b_event_normalized)
    plt.barh(np.arange(n_buckets), b_censored_normalized, left=b_event_normalized)
    plt.legend(['event', 'censored'])
    expected_value = 1 / n_buckets
    plt.plot([expected_value, expected_value], [0, n_buckets - 1], 'k--')
    yt = [('%.2f - %.2f' % (i, i + expected_value)) for i in np.arange(0, 1, expected_value)]
    plt.yticks(np.arange(0, n_buckets, expected_value * n_buckets), yt)
    plt.tight_layout()
    return f, mse


def __get_predictions_at(predictions, t):
    """

    :param predictions: pd.Dataframe with one row per patient, one column per interval
    :param t: current time step
    :return: pd.Series - for each patient the current prediction is returned
    """
    bucket_of_t = np.digitize(t, predictions.columns[:-1])
    return predictions.iloc[:, bucket_of_t]


def specificity_uno(target, predictions, t, c=0.5, decimals=3):
    """
    How many of events that happen after time t are correctly predicted (surv probability smaller/equal to c)
    I(prediction <= threshold_c and target_time > time t) / I(target_time > time_t)
    :param target: pd.Dataframe with two columns: event_month (float) and has_event (0 or 1)
    :param predictions: pd.Dataframe with one row per patient, one column per interval
    :param t: current time step at which model is evaluated
    :param c: current threshold for prediction to count it as survival or relapse
    :param decimals: how many decimals to round the prediction to (e.g. to round 0.9999 to 1)
    :return: specificity as float
    """
    preds_at_t = __get_predictions_at(predictions, t)
    g_leq_c = np.round((1 - preds_at_t), decimals) <= c
    x_greater_t = target['event_month'] > t
    numerator = np.sum(g_leq_c & x_greater_t)
    denominator = np.sum(x_greater_t)
    return numerator / denominator


def sensitivity_uno(target, predictions, t, c=0.5, decimals=3,
                    censoring_km=None):
    """
    How many of events that happen before time t are correctly predicted (surv probability greater than threshold c)
    I(prediction > threshold_c and target_time <= time t) / I(target time >= time t)
    Plus, this is then weighted by ipcw (inverse probability of censoring weighting) to account for censored patients
    :param target: pd.Dataframe with two columns: event_month (float) and has_event (0 or 1)
    :param predictions: pd.Dataframe with one row per patient, one column per interval
    :param t: current time point to evaluate model at
    :param c: current threshold to decide if model predicts survival or not
    :param decimals: how many decimals to round prediction to
    :param censoring_km: None or KaplanMeierFitter if weighting by censoring weights if wanted
    :return: sensitivity as float
    """
    preds_at_t = __get_predictions_at(predictions, t)
    g = np.round(1 - preds_at_t, decimals)
    g_greater_c = g > c
    x_leq_t = target['event_month'] <= t

    # weight by ipcw

    numerator_is = target[target['has_event'] & x_leq_t & g_greater_c]
    denominator_is = target[target['has_event'] & x_leq_t]

    if censoring_km is not None:
        numerator = __weight_by_ipcw(numerator_is, censoring_km)
        denominator = __weight_by_ipcw(denominator_is, censoring_km)
    else:
        numerator = numerator_is.shape[0]
        denominator = denominator_is.shape[0]

    return numerator / denominator


def __weight_by_ipcw(target, censoring_km):
    """

    :param target: pd.Dataframe with two columns: event_month (float) and has_event (0 or 1)
    :param censoring_km: KaplanMeierFitter
    :return: float
    """
    return np.sum(
        np.reciprocal(
            censoring_km.survival_function_at_times(target['event_month']),  where=censoring_km.survival_function_at_times(target['event_month'])!=0))


def auc_uno_plot(target, predictions, t,
                 c_step_no=25, decimals=3, verbose=False, ax=None,
                 event_time_col='event_month', censored_col='is_censored'):
    """

    :param target: pd.Dataframe with two columns: event_month (float) and has_event (0 or 1)
    :param predictions: pd.Dataframe with one row per patient, one column per interval
    :param t:
    :param c_step_no:
    :param decimals:
    :param verbose:
    :param ax:
    :param event_time_col:
    :param censored_col:
    :return:
    """
    target = merge_y_true_to_df(target, event_time_col=event_time_col, censored_col=censored_col)
    target = target.assign(has_event=lambda row: 1-row['is_censored']).astype('int')
    target.drop(columns='is_censored', inplace=True)
    auc, spec, sens = auc_uno(target,
                              predictions,
                              t,
                              c_step_no,
                              decimals,
                              verbose,
                              return_spec_sens=True
                              )

    ax = ax or plt.gca()

    linewidth = 2
    ax.plot(1 - spec, sens, color='darkorange',
            lw=linewidth,
            label='ROC curve (auc = %0.2f)' % auc,
            drawstyle="steps-post")
    ax.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1-Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Receiver operating characteristic')
    ax.legend()
    return ax


def auc_uno(target, predictions, t, c_step_no=25, decimals=3, verbose=False,
            return_spec_sens=False):
    """
    AUC at a specific time point: how good is sensitivity and specificity at time t?
    :param target: pd.Dataframe with two columns: event_month (float) and has_event (0 or 1)
    :param predictions: pd.Dataframe with one row per patient, one column per interval
    :param t: float, current time point to evaluate the auc at
    :param c_step_no: for how many thresholds between 0 and 1 to evaluate the model
    :param decimals: how many decimals to round to
    :param verbose: whether or not to print some information in between
    :param return_spec_sens: True/False, return sensitivity and specificity or not
    :return: auc as float, maybe also sensitivity and specificity as array over thresholds
    """
    # min t should be equal to first (non-censored) event (sensitivity)
    # max t should be second to last event or censoring time (specificity)
    t = np.clip(t,
                target[target['has_event'] == 1]['event_month'].min(),
                target['event_month'].drop_duplicates().nlargest(2).iloc[1])

    if verbose:
        print(f"using t={t}")

    sens_c = np.array([])
    spec_c = np.array([])

    if ~target['has_event'].all():
        censoring_km = (KaplanMeierFitter(label="simulated_data")
                        .fit(target.loc[target['has_event'] == 0, 'event_month'])
                        )
    else:
        censoring_km = None

    for c in np.linspace(0, 1, c_step_no+1):
        if c == 0:
            c = -0.0001
        elif c == 1:
            c = 1.0001
        sens_c = np.append(sens_c,
                           sensitivity_uno(target,
                                           predictions,
                                           t=t,
                                           c=c,
                                           decimals=decimals,
                                           censoring_km=censoring_km))
        spec_c = np.append(spec_c,
                           specificity_uno(target,
                                           predictions,
                                           t=t,
                                           c=c,
                                           decimals=decimals))

    result = auc(1 - spec_c, sens_c)

    if not return_spec_sens:
        return result
    else:
        return result, spec_c, sens_c


def __get_all_auc_unos(target, predictions, tau_1=None, tau_2=None, decimals=3, c_step_no=25):
    """

    :param target: pd.Dataframe with two columns: event_month (float) and has_event (0 or 1)
    :param predictions: pd.Dataframe with one row per patient, one column per interval
    :param tau_1: None or float: first time point to evaluate AUC
    :param tau_2: None or float: last time point to evaluate AUC
    :param decimals: int, how many decimals to keep for calculation (to round 0.999999 to 1 for example)
    :param c_step_no: in how many steps to divide threshold scale between 0 and 1
    :return: pd.DataFrame with columns time of evaluation, S (values of KaplanMeier) and AUC
    """
    km_all = (KaplanMeierFitter(label='population')
              .fit(target['event_month'], target['has_event'])
              )
    if tau_2 is not None:
        last_interval = sum(km_all.timeline < tau_2)+1
        km_all.timeline = km_all.timeline[:last_interval]
        km_all_survival_function_values = km_all.survival_function_.values[:last_interval]
    else:
        km_all_survival_function_values = km_all.survival_function_.values
    my_auc = np.array([])
    for t in km_all.timeline:
        my_auc = np.append(my_auc, auc_uno(target, predictions, t, c_step_no, decimals))

    all_auc_unos = pd.DataFrame(data={
        't': km_all.timeline,
        'S_hat': km_all_survival_function_values[:, 0],
        'AUC': my_auc})

    # set [tau_1, tau_2] to largest possible interval
    if tau_1 is None:
        tau_1 = all_auc_unos['t'].min()

    if tau_2 is None:
        tau_2 = all_auc_unos['t'].max()

    return all_auc_unos[(all_auc_unos['t'] >= tau_1) & (all_auc_unos['t'] <= tau_2)]


def cdauc_uno(target, predictions, tau_1=None, tau_2=None, decimals=3, c_step_no=25,
              event_time_col='event_month', censored_col='is_censored'):
    """
    C/D AUC integrated over time, as suggested by Uno et al
    (https://www.jstor.org/stable/27639883?seq=1#metadata_info_tab_contents)
    Computes the sensitivity and specificity across thresholds for each timepoint and integrates over it
    :param target: [true survival time, censored] either as numpy array or as pandas DataFrame
    :param predictions: dataframe with one column per prediction, one row per time point (survival prediction)
    :param c_step_no: how many thresholds to split [0, 1] into for AUC
    :param tau_1: lower bound for t in integration
    :param tau_2: upper bound for t in integration
    :param decimals: to how many decimals to round predicted survival probability
    :param event_time_col: if target is dataframe, in which column is event found
    :param censored_col: if target is dataframe, in which column is censoring information found
    :return: auc as one scalar value and dataframe of all aucs per time interval
    """
    predictions = predictions.transpose()
    assert len(target) == len(predictions)
    target = merge_y_true_to_df(target, event_time_col=event_time_col, censored_col=censored_col)
    target = target.assign(has_event=lambda row: 1-row['is_censored']).astype('int')
    target.drop(columns='is_censored', inplace=True)

    all_auc_unos = __get_all_auc_unos(target=target,
                                      predictions=predictions,
                                      tau_1=tau_1,
                                      tau_2=tau_2,
                                      decimals=decimals,
                                      c_step_no=c_step_no)

    scaler = 1 / (all_auc_unos['S_hat'].iloc[0] - all_auc_unos['S_hat'].iloc[-1])
    return scaler * auc(all_auc_unos['S_hat'], all_auc_unos['AUC']), all_auc_unos


def cd_auc_uno_plot(target, predictions, tau_1=None, tau_2=None, decimals=3, c_step_no=25,
                    event_time_col='event_month', censored_col='is_censored', f=None):
    """

    :param target: [true survival time, censored] either as numpy array or as pandas DataFrame
    :param predictions: dataframe with one column per prediction, one row per time point (survival prediction)
    :param tau_1: lower bound for t in integration
    :param tau_2: upper bound for t in integration
    :param decimals: to how many decimals to round predicted survival probability
    :param c_step_no: how many thresholds to split [0, 1] into for AUC
    :param event_time_col: if target is dataframe, in which column is event found
    :param censored_col: if target is dataframe, in which column is censoring information found
    :param f: plt.figure or None (new figure will be created)
    :return:
    """
    predictions = predictions.transpose()
    assert len(target) == len(predictions)
    target = merge_y_true_to_df(target, event_time_col=event_time_col, censored_col=censored_col)
    target = target.assign(has_event=lambda row: 1-row['is_censored']).astype('int')
    target.drop(columns='is_censored', inplace=True)

    all_auc_unos = __get_all_auc_unos(target=target,
                                      predictions=predictions,
                                      tau_1=tau_1,
                                      tau_2=tau_2,
                                      decimals=decimals,
                                      c_step_no=c_step_no)
    scaler = 1 / (all_auc_unos['S_hat'].iloc[0] - all_auc_unos['S_hat'].iloc[-1])

    print('best AUCs: \n', all_auc_unos.sort_values(by='AUC', ascending=False)[:3])

    overall_auc = scaler * auc(all_auc_unos['S_hat'], all_auc_unos['AUC'])
    if f is None:
        f = plt.figure()
    plt.plot(all_auc_unos['t'], all_auc_unos['AUC'], drawstyle="steps-post")
    plt.xlabel('time t')
    plt.ylabel('AUC')
    plt.title('Integrated AUC: %s' % np.round(overall_auc, 4))
    plt.ylim([0.45, 0.9])
    plt.tight_layout()
    return f, overall_auc
