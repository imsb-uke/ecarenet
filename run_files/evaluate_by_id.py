from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
import json
import os
import sys

sys.path.append('/opt/project')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from dataset_creation.dataset_main import create_dataset
from models.model_helpers import compile_model, load_base_model
from settings.sacred_experiment import ex
from training_evaluation.evaluate import evaluate


def print_system_summary(_log):
    # system summary
    _log.debug('PATH: {}'.format(sys.path))
    _log.debug('current working directory: {}'.format(os.getcwd()))
    _log.info('Num GPUs Available: {}'.format(len(tf.config.experimental.list_physical_devices('GPU'))))
    _log.debug('devices:{} '.format(device_lib.list_local_devices()))
    _log.info('tensorflow version {}'.format(tf.__version__))


def run(_config, _log, _run):
    """
    this is the main function that runs evaluation of the model best model for a given run_id once

    """
    # checks on the config
    # check_config(_config)
    np.random.seed(_config['data_generation']['seed'])
    tf.random.set_seed(_config['data_generation']['seed'])
    experiments_dir = _config['training']['model_save_path']
    id = _config['inference']['inference_id']
    config = json.load(open(os.path.join(experiments_dir, str(id), 'config.json')))

    if _config['inference']['eval_csv'] is not None:
        config['data_generation']['test_csv_file'] = _config['explain']['test_path']
    # MODEL EVALUATION (during tuning, use usage_mode 'valid', for real testing, use usage_mode 'test'
    test_data, test_class_distribution = create_dataset(data_generation_config=_config['data_generation'],
                                                        usage_mode='test'
                                                        )
    model, _ = load_base_model(os.path.join(experiments_dir, str(id)),
                               {'keras_model_params':
                                    {'input_shape': config['model']['keras_model_params']['input_shape']}})
    model = compile_model(model, config['training'])
    # model_json = model.to_json()
    # with open(os.path.join(experiments_dir, str(_run._id), 'model_json.json'), 'w') as json_file:
    #     json_file.write(model_json)
    model.summary()

    test_results = evaluate(model, test_data, config['data_generation']['label_type'],
                            test_class_distribution, _config['inference']['metrics'], experiments_dir, _run._id)

    allresults = {'test_results': test_results}
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', experiments_dir, str(_run._id)))
    with open(os.path.join(script_path, 'temp_results.json'), 'w') as resultsfile:
        json.dump(allresults, resultsfile)
    ex.add_artifact(os.path.join(script_path, 'temp_results.json'), 'test_results.json')
    os.remove(os.path.join(script_path, 'temp_results.json'))


@ex.automain
def main(_config, _log, _run):
    run(_config, _log, _run)
