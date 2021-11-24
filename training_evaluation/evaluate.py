from training_evaluation.evaluate_survival import evaluate_survival_model
from training_evaluation.evaluate_classification import evaluate_classification_model


def evaluate(model, dataset, label_type, class_distribution, metrics, experiments_dir, run_id):
    """
    for models that output hazard rates and risks, use this function to evaluate the metrics
    :param model: the trained model
    :param dataset: the test dataset as a tf.DataFrame
    :param label_type: string, for survival model 'survival' else 'isup' or 'bin'
    :param class_distribution: list, which class is how often present in dataset
    :param metrics: list of metrics to evaluate.
                    Choose from ['auc', 'brier', 'c_index', 'd_calibration',        - for survival
                                 'acc', 'kappa', 'f1_score']                        - for classification
    :param experiments_dir: folder where the results (and plots) should be saved
    :param run_id: id of current run (also for saving reasons) - usually set by sacred automatically
    :return:
    """

    if label_type == 'survival':
        result_metrics = evaluate_survival_model(model, dataset, class_distribution,
                                                 metrics, experiments_dir, run_id)
    else:
        result_metrics = evaluate_classification_model(model, dataset, class_distribution,
                                                       label_type,
                                                       metrics, experiments_dir, run_id)
    return result_metrics
