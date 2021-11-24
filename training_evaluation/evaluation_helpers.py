from settings.sacred_experiment import ex
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

from dataset_creation.label_encoding import int_to_string_label


def save_plot_to_artifacts(fig, name, experiments_dir, run_id):
    """
    Save a plot to sacred artifacts and make it available to the mongo database
    :param fig:
    :param name:
    :param experiments_dir:
    :param run_id:
    :return:
    """
    plt.tight_layout()
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', experiments_dir, str(run_id)))
    fig.savefig(os.path.join(script_path, 'temp_'+name+'.png'))
    ex.add_artifact(os.path.join(script_path, 'temp_'+name+'.png'), name=name+'.png')
    os.remove(os.path.join(script_path, 'temp_'+name+'.png'))


def plot_confusion_matrix(cm, label_type, normalize=False, f=None):
    """

    :param cm: numpy array with prediction per column and true label per row
    :param label_type: string, 'isup' or 'bin' (needed for axis annotation)
    :param normalize:
    :param: f: None
    :return:
    """
    size = cm.shape[0]
    if normalize:
        precision = float
        for i in range(size):
            if sum(cm[i]) > 0:
                cm[i] = cm[i] / sum(cm[i])
            else:
                cm[i] = -1
    else:
        precision = int
    # Limits for the extent
    xy_start = 0.0
    xy_end = size
    extent = [xy_start, xy_end, xy_start, xy_end]
    # The normal figure
    if f is None:
        f = plt.figure(figsize=(16, 12))
    ax = f.add_subplot(111)
    im = ax.imshow(cm, extent=extent,  cmap='Blues')
    # Add the text
    jump_xy = (xy_end - xy_start) / (2.0 * size)
    xy_positions = np.linspace(start=xy_start, stop=xy_end, num=size, endpoint=False)
    # Change color of text to white above threshold
    max_value = np.sum(cm, axis=1).max()
    thresh = max_value / 2.
    for y_index, y in enumerate(reversed(xy_positions)):
        for x_index, x in enumerate(xy_positions):
            text_x = x + jump_xy
            text_y = y + jump_xy
            if cm[y_index, x_index] == -1:
                label = '-'
                ax.text(text_x, text_y, label, color="black", ha='center', va='center', fontdict={'size': 30})
            else:
                label = precision(round(cm[y_index, x_index]*100)/100)
                ax.text(text_x, text_y, label, color="white" if label > thresh else "black", ha='center', va='center', fontdict={'size': 30})
    if precision == float:
        color_range_ticks = np.linspace(0, max_value, 6, endpoint=True)
    else:
        color_range_ticks = np.linspace(0, max_value, 5, endpoint=True)
    #fig = plt.figure()
    im.set_clim(0, max_value)
    cbar = f.colorbar(im, ticks=color_range_ticks)
    cbar.ax.tick_params(labelsize=30)
    plt.xlabel('prediction', fontdict={'size': 30})
    plt.ylabel('ground truth', fontdict={'size': 30})

    ticklabels = [int_to_string_label(label_int, label_type) for label_int in range(len(cm))]

    plt.xticks(np.arange(xy_start + 0.5, xy_end), ticklabels, fontsize=30, rotation=20, horizontalalignment='right')
    plt.yticks(np.arange(xy_start + 0.5, xy_end), ticklabels[::-1], fontsize=30)
    # save figure
    # script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', experiments_dir, str(run_id)))
    plt.tight_layout()
    # save_plot_to_artifacts(fig, plot_name, experiments_dir, run_id)
    # plt.savefig(os.path.join(script_path, 'temp_'+plot_name+'.png'))
    # ex.add_artifact(os.path.join(script_path, 'temp_'+plot_name+'.png'), name=plot_name+'.png')
    # os.remove(os.path.join(script_path, 'temp_'+plot_name+'.png'))
    return f


def get_survival_curve(df=False, run_id=0, mode='valid', experiments_path='/opt/project/experiments/'):
    """
    From survival dataframe, get the columns that contain the survival prediction information

    either from dataframe directly, or from experiments directory and run id first get dataframe
    :param df: get survival curve directly from dataframe
    :param run_id: string, id of run to evaluate (only if df==False)
    :param mode: (only if df==False)
    :param experiments_path: /path/to/experiments/folder where folder 'run_id' is (only if df==False)
    :return: pandas dataframe with one row per patient, one column per interval and survival probability, results df
    """
    if not isinstance(df, pd.DataFrame):
        if mode == 'test':
            path = os.path.join(experiments_path, '%s' % run_id, 'test_results.csv')
        elif mode == 'valid':
            path = os.path.join(experiments_path, '%s' % run_id, 'results.csv')
        else:
            path = os.path.join(experiments_path, '%s' % run_id, 'train_results.csv')

        path = os.path.join(os.getcwd(), path)
        df = pd.read_csv(path)

    prediction_columns = [c for c in df.columns if c.startswith('surv')]
    y_pred = df[prediction_columns]
    surv_curve = y_pred.rename(columns={c: float(c.split('_')[-1]) for c in y_pred.columns})
    return surv_curve.transpose(), df