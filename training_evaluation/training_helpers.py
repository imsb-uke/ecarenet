import pandas as pd
import numpy as np


def split_y_true_censored(y_true, event_time_col='event_month', censored_col='is_censored'):
    if isinstance(y_true, pd.DataFrame):
        censored = np.array(y_true[censored_col], dtype='int64')
        y_orig = np.array(y_true[event_time_col])
    else:
        censored = (y_true[:, -1])[:, np.newaxis]
        y_orig = y_true[:, :-1]
    if len(y_orig.shape) > 1:
        y_orig = y_orig[:, 0]
    if len(censored.shape) > 1:
        censored = censored[:, 0]
    return y_orig, censored


def merge_y_true_to_df(y_true, event_time_col='event_month', censored_col='is_censored'):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.rename(columns={event_time_col: 'event_month', censored_col: 'is_censored'})
    else:
        y_true = pd.DataFrame({'event_month': y_true[:, 0], 'is_censored': y_true[:, -1]})
    return y_true

