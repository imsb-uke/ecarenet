from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

from training_evaluation.evaluation_helpers import get_survival_curve
from training_evaluation.model_metrics import cdauc_uno, brier_score_censored_core, concordance_td_core, d_calibration


def compare_models_barplot(run_ids, evaluation_methods, df_info, model_directory,
                           save_fig=False, limit=None, mode='valid'):
    """
    For all models in run_ids, calculate the metrics defined in the list evaluation_methods,
    use df_info for color and name information
    :param run_ids: list of integers [1,35,564]
    :param evaluation_methods: list of strings ['auc','brier','c_index','d_calibration']
    :param df_info: pandas dataframe with
                    index column of run_id,
                    column 'info' with string that gives information about run (for label in plot)
                    column 'color' with string that has information about how to plot
                    e.g.     |         info             |    color
                         ----|--------------------------|----------
                         923 | 'log input and loss xzf3 | 'steelblue'
    :param model_directory: path/to/folder where run_id folder is located
    :param save_fig: True/False save figure or not
    :param limit: None or list of lower and upper limit for plot, for comparability reasons
    :param mode: 'valid' or 'test': compare models on results.csv or on test_results.csv that is stored in folder
    :return: DataFrame with result per run
             DataFrame with mean result per model/experiment
             DataFrame with standard deviation of result per model/experiment
    """
    # array with one row per evaluation, one column per id
    eval_df = pd.DataFrame(index=evaluation_methods, columns=run_ids)

    plot_width = 2
    plot_height = int(np.ceil(len(run_ids) / plot_width))
    fig = plt.figure(figsize=(10, 2.5 * plot_height))

    # evaluate each run individually
    for run_idx, run_id in enumerate(run_ids):
        print(run_id, end=' ')
        surv_curve, df_results = get_survival_curve(False, run_id, mode=mode, experiments_path=model_directory)

        for method_idx, evaluation_method in enumerate(evaluation_methods):
            if 'auc' in evaluation_method:
                eval_df.rename(index={evaluation_method: 'auc'}, inplace=True)
                evaluation_method = 'auc'
                evaluation_methods[method_idx] = 'auc'
            if 'cindex' in evaluation_method:
                eval_df.rename(index={evaluation_method: 'c_index'}, inplace=True)
                evaluation_method = 'c_index'
                evaluation_methods[method_idx] = 'c_index'
            if 'dcal' in evaluation_method:
                eval_df.rename(index={evaluation_method: 'd_calibration'}, inplace=True)
                evaluation_method = 'd_calibration'
                evaluation_methods[method_idx] = 'd_calibration'
            f = getattr(evaluations, evaluation_method)
            mean_res = f(df_results, surv_curve)
            eval_df.loc[evaluation_method][run_idx] = mean_res

        plt.subplot(plot_height, plot_width, run_idx + 1)
        plt.subplots_adjust(left=0.1, right=0.35, bottom=0.5)
        plt.step(surv_curve.index, np.array(surv_curve), where='post')
        plt.title(' - '.join((run_id, df_info.loc[run_id]['info'])))
    plt.tight_layout()
    plt.show()

    eval_df = eval_df.astype('float32')

    # here, only name column after model/experiment type
    eval_results = eval_df.rename(columns={i: df_info.loc[i]['info'] for i in eval_df.columns})

    # to run_id add model information as column description
    eval_df.rename(columns={i: '---'.join((i, df_info.loc[i]['info'])) for i in eval_df.columns},
                   inplace=True)

    print(eval_results)
    # find mean and std per model/experiment setting over multiple runs
    mean = eval_results.groupby(lambda x: x, axis=1, sort=False).mean()
    std = eval_results.groupby(lambda x: x, axis=1, sort=False).std()

    # in case only one run is available for a setting, std would be nan, should be 0
    std.fillna(0, inplace=True)

    run_id_plus_info = list(eval_df.columns)

    # label for graph should be all run ids + information (e.g. '1+2+3---exp1' '4+5---exp2')
    ri = [run_id_plus_info[0]]
    for i in np.arange(1, len(run_id_plus_info)):
        if run_id_plus_info[i].split('---')[-1] == ri[-1].split('---')[-1]:
            ri[-1] = '+'.join((run_id_plus_info[i].split('---')[0], ri[-1]))
        else:
            ri.append(run_id_plus_info[i])
    run_ids = ri
    eval_mean = np.array(mean)
    eval_std = np.array(std)

    # loop over each metric: single bar plot per metric to compare all run_ids / experiments/models
    for metric_mean, idx_metric, evaluation_method, error in zip(eval_mean, range(len(eval_mean)), evaluation_methods,
                                                                 eval_std):
        #f = plt.figure(figsize=(int(np.min((len(run_ids) * 1.5, 12))), len(run_ids) * 0.25))
        f, ax = plt.subplots(1,1)
        f.set_tight_layout(True)
        #f.add_subplot(1,1,1)
        # bar plot
        plt.barh(range(len(run_ids) + 1, 1, -1), metric_mean,
                 color=[df_info.loc[i.split('---')[0].split('+')[0]]['color'] for i in run_ids], xerr=error)
        plt.yticks(range(len(run_ids) + 1, 1, -1),
                   run_ids)  # [' - '.join((r,df_info.loc[r]['info'])) for r in run_ids])
        plt.title(evaluation_method)
        if evaluation_method in ['cindex', 'brier', 'auc']:
            if limit is None:
                plt.xlim(np.min(metric_mean) - 0.005, np.max(metric_mean) + 0.005)
            else:
                plt.xlim(limit[0], limit[1])
        if save_fig:
            plt.savefig(os.path.join(save_fig, '%s.png' % evaluation_method), bbox_inches='tight')
        plt.tight_layout()
        f.set_figwidth(np.max((f.get_figwidth(), int(np.min((len(run_ids) * 1.5, 12))))))
        f.set_figheight( len(run_ids) * 0.25)
        plt.show()

    return eval_df, mean, std


class evaluations():
    """
    Helper class that has all metrics for survival model
    """
    def auc(df_results, surv_curve):
        tau_2 = surv_curve.columns[-1]
        overall_auc, all_aucs_unos = cdauc_uno(df_results[['event_month', 'is_censored']],
                       surv_curve,
                       tau_2=tau_2,
                       event_time_col='event_month',
                       censored_col='is_censored')
        return overall_auc

    def brier(df_results, surv_curve):
        b = brier_score_censored_core(np.concatenate(
                (np.array(df_results['event_month'])[:, np.newaxis],
                 np.array(df_results['is_censored'])[:, np.newaxis]), 1),
                surv_curve,
                event_time_col='event_month', censored_col='is_censored')#
        return b

    def c_index(df_results, surv_curve):
        c = concordance_td_core(
        np.concatenate(
            (np.array(df_results['event_month'])[:, np.newaxis],
             np.array(df_results['is_censored'])[:, np.newaxis]), 1),
        surv_curve)
        return c

    def d_calibration(df_results, surv_curve):
        dc = d_calibration(df_results[['event_month', 'is_censored']],
                              surv_curve,
                              10,
                              event_time_col='event_month',
                              censored_col='is_censored')[0]
        return dc
