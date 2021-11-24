from tensorflow.python.client import device_lib
from sacred.observers import MongoObserver
import tensorflow as tf
import numpy as np
import importlib
import json
import os
import sys
sys.path.append('/opt/project')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from dataset_creation.dataset_main import create_dataset
from models.model_helpers import compile_model
from settings.sacred_experiment import ex
from training_evaluation.train import train_loop
from training_evaluation.evaluate import evaluate
from run_files.run_helpers import check_config


def print_system_summary(_log):
    # system summary
    _log.debug('PATH: {}'.format(sys.path))
    _log.debug('current working directory: {}'.format(os.getcwd()))
    _log.info('Num GPUs Available: {}'.format(len(tf.config.experimental.list_physical_devices('GPU'))))
    _log.debug('devices:{} '.format(device_lib.list_local_devices()))
    _log.info('tensorflow version {}'.format(tf.__version__))


def run(_config, _log, _run):
    """
    this is the main function that runs training and test of the model once
    Returns: best result, based on monitoring values in config
    """
    # define threads used
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    print_system_summary(_log)

    if _config['general']['use_mongo_db'] is not None:
        ex.observers.append(MongoObserver(url=_config['general']['use_mongo_db']))

    # checks on the config
    check_config(_config)
    np.random.seed(_config['data_generation']['seed'])
    tf.random.set_seed(_config['data_generation']['seed'])
    experiments_dir = _config['training']['model_save_path']

    # DATA GENERATION TRAINING AND VALIDATION
    train_data, train_class_distribution = create_dataset(data_generation_config=_config['data_generation'],
                                                          usage_mode='train'
                                                          )

    valid_data, valid_class_distribution = create_dataset(data_generation_config=_config['data_generation'],
                                                          usage_mode='valid'
                                                          )
    # for d in train_data.take(1):
    #     pass
    # for d in valid_data.take(1):
    #     pass
    number_of_traindata = sum(train_class_distribution)
    print('number of records in training data set: {}'.format(number_of_traindata))
    valid_number_of_records = sum(valid_class_distribution)
    print('number of records in validation dataset: {}'.format(str(valid_number_of_records)))
    # MODEL GENERATION
    number_of_classes = _config['data_generation']['number_of_classes']
    mod = importlib.import_module('models.' + _config['model']['name'])
    model = getattr(mod, _config['model']['name'])(_config['model'])
    print(model.input[0].shape)
    
    class_weights, metric_class_weights = get_class_weights(_config['training']['class_weight'],
                                                            _config['training']['weighted_metrics'],
                                                            train_class_distribution,
                                                            _log)
    model = compile_model(model, _config['training'], metric_class_weights)
    model_json = model.to_json()
    with open(os.path.join(experiments_dir, str(_run._id), 'model_json.json'), 'w') as json_file:
        json_file.write(model_json)
    model.summary()
    # MODEL TRAINING
    model, history = train_loop(model, train_data, valid_data,
                                _config['data_generation']['train_batch_size'], _config['data_generation']['valid_batch_size'],
                                _config['training'],
                                train_class_distribution, valid_class_distribution,
                                class_weights, _config['data_generation']['label_type'],
                                _run, _log=None)

    # MODEL EVALUATION (during tuning, use usage_mode 'valid', for real testing, use usage_mode 'test'
    test_data, test_class_distribution = create_dataset(data_generation_config=_config['data_generation'],
                                                        usage_mode='test'
                                                        )
    test_results = evaluate(model, test_data, _config['data_generation']['label_type'],
                            test_class_distribution, _config['evaluation']['metrics'], experiments_dir, _run._id)

    # STORE RESULTS
    train_history = {}
    for k, v in history.history.items():
        train_history[k] = np.array(v).astype(float).tolist()

    allresults = {'train_results': train_history, 'test_results': test_results}
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', experiments_dir, str(_run._id)))
    with open(os.path.join(script_path, 'temp_results.json'), 'w') as resultsfile:
        json.dump(allresults, resultsfile)
    ex.add_artifact(os.path.join(script_path, 'temp_results.json'), 'results.json')
    os.remove(os.path.join(script_path, 'temp_results.json'))


def get_class_weights(class_weight, metric_weight, class_distribution, _log):
    if class_weight:
        class_weights = {i: sum(class_distribution)/class_distribution[i] if class_distribution[i] != 0 else 0
                         for i in range(len(class_distribution))}
        max_weight = np.max([class_weights[k] for k in class_weights])
        class_weights = {k: class_weights[k]/max_weight for k in class_weights}
        _log.debug('class weights: ')
        _log.debug(class_weights)
    else:
        class_weights = None
    if metric_weight:
        metric_class_weights = class_weights
    else:
        metric_class_weights = None
    return class_weights, metric_class_weights


@ex.automain
def main(_config, _log, _run):
    run(_config, _log, _run)


