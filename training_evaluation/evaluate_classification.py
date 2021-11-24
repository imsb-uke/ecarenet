from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from tensorflow import math
import pandas as pd
import numpy as np

from dataset_creation.label_encoding import label_to_int
from training_evaluation.evaluation_helpers import plot_confusion_matrix, save_plot_to_artifacts


def create_classification_result_dataframe(model, dataset, class_distribution, label_type, n_output_nodes):
    """

    :param model: tensorflow/keras model
    :param dataset: tf.data.Dataset with entries like dictionary, needs to include
                    'images': tf.Tensor [batch_size(1), w, h, 3]
                    'labels': tf.Tensor [batch_size(1), n_output_nodes]
    :param class_distribution: list with length n_classes, and elements how many examples in dataset per class
    :param label_type: string 'isup' or 'bin'
    :param n_output_nodes: int, how many output nodes the model has
    :return: results as pandas Dataframe, with columns
             img_path: string
             groundtruth_class: int
             predicted_class: int
             pred_class_x: one column per output node, returned prediction (float) per output node
             any additional labels that were defined during dataset creation
    """
    assert dataset.element_spec['labels'].shape[-1] == n_output_nodes == model.layers[-1].output_shape[-1]

    # go through all examples and find predicted value
    n_examples = int(sum(class_distribution))

    targets = np.zeros((n_examples, n_output_nodes))
    predictions = np.zeros((n_examples, n_output_nodes))
    img_names = [''] * n_examples
    for idx, d in enumerate(dataset.take(n_examples)):
        if idx == 0:
            additional_labels = {k: [] for k in d if k not in ['images', 'labels']}
        img = d['images']
        label = d['labels']

        # run model to get prediction
        pred = model(img, training=False).numpy()
        # only use first element, because one element with batch size 1 is used
        pred = pred[0]

        predictions[idx, :] = pred
        targets[idx, :] = label[0]
        img_names[idx] = d['image_paths'].numpy()[0].decode('utf-8')
        [additional_labels[k].append(np.array(d[k]).squeeze()) for k in d if k not in ['images', 'labels']]

    results = pd.DataFrame()
    results['img_path'] = img_names
    results['groundtruth_class'] = label_to_int(targets, label_type)
    results['predicted_class'] = label_to_int(predictions, label_type)
    for pred_class in range(predictions.shape[1]):
        results['pred_class_'+str(pred_class)] = predictions[:, pred_class]
    for k in additional_labels:
        results[k] = list(additional_labels[k])

    return results


def evaluate_classification_model(model, dataset, class_distribution, label_type, metrics, experiments_dir, run_id):
    """

    :param model: tensorflow/keras model
    :param dataset: tf.data.Dataset with entries like dictionary, needs to include
                    'images': tf.Tensor [batch_size(1), w, h, 3]
                    'labels': tf.Tensor [batch_size(1), n_output_nodes]
    :param class_distribution: list with length n_classes, and elements how many examples in dataset per class
    :param label_type: string 'isup' or 'bin'
    :param metrics: list of strings with metrics ['acc','f1','kappa']
    :param experiments_dir: path to where to store results, e.g. /path/to/experiments
    :param run_id: folder number where to store results, usually comes from sacred experiment
    :return: resulting metrics as dictionary, depending on which were specified can include
             'accuracy': float between 0 and 1
             'cohens_kappa: float between -1 and 1
             'f1_score': float between 0 and 1
    """
    print("Begin testing of model")
    result_metrics = dict()
    n_classes = int(len(class_distribution))
    if label_type == 'isup':
        n_output_nodes = n_classes - 1
    else:
        n_output_nodes = n_classes

    results = create_classification_result_dataframe(model, dataset, class_distribution, label_type, n_output_nodes)
    results.to_csv(experiments_dir + '/' + str(run_id) + '/results.csv')

    if np.any(['acc' in m for m in metrics]):
        acc = accuracy_score(results['groundtruth_class'], results['predicted_class'])
        print("accuracy: ", acc)
        result_metrics['accuracy'] = float(acc)

    if np.any(["kappa" in m for m in metrics]):
        kappa = cohen_kappa_score(results['groundtruth_class'], results['predicted_class'], weights='quadratic')
        print("kappa: ", kappa)
        result_metrics['cohens_kappa'] = float(kappa)

    if np.any(['f1' in m for m in metrics]):
        f1 = f1_score(results['groundtruth_class'], results['predicted_class'], average='macro')
        print("F1 score: ", f1)
        result_metrics['f1_score'] = float(f1)

    # CONFUSION MATRIX
    mat = math.confusion_matrix(results['groundtruth_class'], results['predicted_class'], num_classes=n_classes)
    f = plot_confusion_matrix(np.array(mat, dtype='float32'), label_type, True)
    save_plot_to_artifacts(f, 'confusion_matrix_relative', experiments_dir, run_id)
    f = plot_confusion_matrix(np.array(mat), label_type, False)
    save_plot_to_artifacts(f, 'confusion_matrix', experiments_dir, run_id)
    result_metrics['cm'] = mat.numpy().astype(float).tolist()

    return result_metrics
