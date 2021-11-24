from lifelines.statistics import pairwise_logrank_test
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import os

from training_evaluation.evaluation_helpers import get_survival_curve


def find_best_risk_intervals(run_ids, experiments_path, mode, save_plot=False,
                             number_intervals=10, step_intervals=0.06,
                             logrank_limit=0.05, possible_intervals=False, log_weight=False):
    """

    :param run_ids: list of runs to evaluate
    :param experiments_path: /path/to/folder with run_id folders inside
    :param mode: 'valid' if run_id/results.csv should be used for evaluation, else 'test' or 'train'
    :param save_plot: False or True (save to experiments directory)
    :param number_intervals: how many intervals to find
    :param step_intervals: distance between intervals or minimum distance if intervals are given
    :param logrank_limit: p-value limit to tell if logrank test failed or passed
    :param possible_intervals: False, if only min risk to max risk in step interval distance should be evaluated
                               or np.array with possible interval limits
    :param log_weight: bool, use 'fleming-harrington' weights (for crossing survival curves) or not
    :return:
    """
    if len(run_ids) > 1:
        plot_width = 2
        plot_height = int(np.ceil(len(run_ids) / plot_width))
    else:
        plot_width = 1
        plot_height = 1
    fig = plt.figure(figsize=(15, 4 * plot_height))
    for idx, run_id in enumerate(run_ids):
        surv_curve, df_res = get_survival_curve(False, run_id, mode, experiments_path)

        if isinstance(possible_intervals, bool):
            possible_intervals = np.arange(df_res['risk'].sort_values().iloc[0], df_res['risk'].sort_values().iloc[-5],
                                           step_intervals)[1:]
        # best combination is the one with least failed tests, so store number of failed tests and according intervals
        least_fails = 999
        least_fail_intervals = [None]
        summary_p = pd.DataFrame()
        # all possible combinations of interval limits
        list_of_combinations = list(itertools.combinations(possible_intervals, number_intervals - 1))  # [::10]
        if len(list_of_combinations) > 600:
            print('reducing %i possible interval options to ...' % len(list_of_combinations))
            # reduce possible intervals, because often there are too many and this is too slow.
            # Use only interval combinations, where intervals next to each other are not too close .
            # Later, iteratively the best best combination can be found.
            list_of_combinations = [l for l in list_of_combinations
                                    if sum(np.array(l[1:]) - np.array(l[:-1]) > step_intervals) >= number_intervals / 2]
        print('evaluating %i possible interval options...' % len(list_of_combinations))
        for interval_idx, interval_limits in tqdm(enumerate(list_of_combinations)):
            # assign each data point to a risk group, according to the interval limits
            risk_groups = np.digitize(df_res['risk'], interval_limits)
            # only evaluate further if patients stratify in enough groups (not all patients in same group for example)
            if len(np.unique(risk_groups)) >= number_intervals :
                df_res['pred_surv_time_quant'] = risk_groups
                # evaluate logrank test for crossing or non-crossing survival curves
                # see (Li et al Statistical Inference Methods for Two Crossing Survival Curves: A Comparison of Methods)
                if log_weight:
                    res = pairwise_logrank_test(df_res['event_month'], risk_groups, 1 - df_res['is_censored'],
                                                weightings='fleming-harrington', p=1, q=0)
                else:
                    res = pairwise_logrank_test(df_res['event_month'], risk_groups, 1 - df_res['is_censored'])
                # how many tests failed
                failed = np.array(res.summary.p > logrank_limit)

                if sum(failed) < least_fails:
                    least_fails = sum(failed)
                    least_fail_intervals = [interval_limits]
                    summary_p = [res.summary.p]  # pd.DataFrame(res.summary.p, columns=[interval_idx])
                elif sum(failed) == least_fails:
                    least_fail_intervals.append(interval_limits)
                    summary_p.append(res.summary.p)
                if sum(failed) == 0:
                    pass  # break
        if save_plot:
            save_path = os.path.join(experiments_path, str(run_id), 'risk_strat.png')
        else:
            save_path = False

        # in the end, plot best interval combinations
        print('', end='\n')
        for lfi, sp in zip(least_fail_intervals, summary_p):
            print('BEST RESULT with %i fails: with ' % least_fails, lfi)
            df_res['risk_groups'] = np.digitize(df_res['risk'], lfi)
            plot_km_curves(df_res, sp, save_path)
        return least_fail_intervals


def plot_km_curves(df_res, sp, save_path):
    """

    :param df_res: pd Dataframe with at least columns event_month (float), risk_group (int), is_censored (0 or 1)
                                              one row per patient
    :param sp: pd Series with results from logrank test
    :param save_path: path/where/to/save/figure.png or False if should not be saved
    :return:
    """
    f = plt.figure(figsize=(20, 8))
    # plt.title(least_fail_intervals)
    spec = gridspec.GridSpec(ncols=2, nrows=1,
                             width_ratios=[2, 3])
    ax = f.add_subplot(spec[0])  # 121)
    kmf = KaplanMeierFitter()
    for i in range(10):
        survival_times = np.array(df_res['event_month'][df_res['risk_groups'] == i])
        censored = np.array(df_res['is_censored'][df_res['risk_groups'] == i])
        if len(censored) != 0:
            kmf.fit(survival_times, 1 - censored, label='%i' % i)
            kmf.plot(ax=ax, ci_alpha=0)

    df = pd.DataFrame()
    num_comparisons = sp.index.max()[1]
    for i in range(num_comparisons):
        df[i] = (sp[i])
    df = df.round(3)
    ax2 = f.add_subplot(spec[1])  # 122)
    font_size = 12
    bbox = [0, 0, 1, 1]
    ax2.axis('off')
    mpl_table = ax2.table(cellText=df.values, rowLabels=df.index, bbox=bbox, colLabels=df.columns)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def evaluate_risk_groups(run_id, experiments_path, mode, save_plot=False, interval_limits=False, log_weight=False):
    """

    :param run_id: id of folder where results are stored
    :param experiments_path: /path/to/experiments folder, where all experiments are stored
    :param mode: whether to load train, validation or test results
    :param save_plot: whether or not to save resulting plot
    :param interval_limits: limits to group patients to different risk groups
    :param log_weight: whether or not to use fleming-harrington weights
                       (recommended if survival curves are allowed to cross)
    :return: DataFrame with test statistic to show which tests passed or failed
    """
    _, df_res = get_survival_curve(False, run_id, mode, experiments_path)
    df_res['risk_groups'] = np.digitize(df_res['risk'], interval_limits)
    if log_weight:
        res = pairwise_logrank_test(df_res['event_month'], df_res['risk_groups'], 1 - df_res['is_censored'],
                                    weightings='fleming-harrington', p=1, q=0)
    else:
        res = pairwise_logrank_test(df_res['event_month'], df_res['risk_groups'], 1 - df_res['is_censored'])
    if save_plot:
        if isinstance(save_plot, str):
            if save_plot.endswith('.png'):
                save_path = save_plot
            else:
                save_path = os.path.join(save_plot, 'eval_risk_strat.png')
        else:
            save_path = os.path.join(experiments_path, str(run_id), 'eval_risk_strat.png')
    else:
        save_path = False
    plot_km_curves(df_res, res.summary.p, save_path)
    return res.summary
