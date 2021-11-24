import re
import os


def check_config(config):
    # GENERAL
    check_config_general(config)

    # DATA GENERATION

    # the value to monitor (decide which model is best), needs to be one of the metrics or loss
    if config['train']['compile_metrics'] is not None:
        assert config['train']['monitor_val'][4:] in (
            *[re.sub(r'(?<!^)(?=[A-Z])', '_', m).lower() for m in config['train']['compile_metrics']],
            *['loss'], *config['train']['loss_fn']), \
            'monitor_value is not in watchlist'
    else:
        assert config['train']['monitor_val'][4:] in (
            *['loss'], *config['train']['loss_fn']), \
            'monitor_value is not in watchlist'



    assert 'ModelCheckpoint' not in [m['name'] for m in config['train']['callbacks']], \
        'model is automatically saved, please do not use ModelCheckpoint'


def check_config_general(config):
    if config['general']['label_type'] == 'isup':
        assert ('CategoricalAccuracy' not in config['training']['compile_metrics']), \
            'use tf_categorical_accuracy instead of CategoricalAccuracy or categorical_accuracy'
        assert ('tf_f1_score' not in config['train']['compile_metrics']), 'f1 not valid for fillin encoding'
        assert (config['model']['name'] == 'm_isup'), 'For ISUP classification, please use model m_isup'

    if config['general']['label_type'] == 'bin':
        assert (config['model']['name'] == 'm_bin'), 'For binary classification, please use model m_bin'
        assert (config['model']['n_classes'] == 2), 'For binary classification, use 2 classes'

    assert config['general']['number_of_classes'] is not None, 'Please specify number of classes in config.general'
    assert isinstance(config['general']['number_of_classes'], int), \
        'Please specify number_of_classes in config.general as int'
    assert config['general']['number_of_classes'] > 0, \
        'Please specify number_of_classes in config.general greater than 0'

    # data directory should not be empty
    assert os.listdir(config['image_directory']) != [], \
        "Image directory {} is empty".format(config['directory'])

    assert (config['general']['image_channels'] == 3), 'currently, only RGB implemented, so set img_channels to 3'


def check_data_generation_config(config):
    assert os.path.isfile(config['data_generation']['train_csv_file']), 'train csv file not found'
    assert os.path.isfile(config['data_generation']['valid_csv_file']), 'valid csv file not found'
    assert os.path.isfile(config['data_generation']['test_csv_file']), 'test csv file not found'

    assert config['data_generation']['train_batch_size'] is not None, 'please specify a training batch size'
    assert config['data_generation']['valid_batch_size'] is not None, 'please specify a validation batch size'

    if 'ecarenet' in config['train']['loss_fn']:
        additional_labels = {k: d[k] for d in config['dataset_generation']['additional_labels'] for k in d}
        assert additional_labels['censored'] is not None, \
            'You have no censoring information but use a eCaReNet loss that needs censoring information'


def check_model_config(config):
    assert config['model'] in ['m_bin', 'm_isup', 'ecare_net'], 'Please choose as model m_bin, m_isup or ecare_net'
    assert isinstance(config['model']['dense_layer_nodes'], list), 'Provide number of dense layers as list'

    if config['model'] == 'mbin':
        assert config['general']['number_of_classes'] == 2, 'For m_bin, only using 2 classes if valid'
    if config['model'] == 'ecare_net':
        assert config['training']['loss_fn'] == 'ecarenet_loss', 'For eCaReNet use ecarenet_loss'


def check_training_config(config):
    assert config['epochs'] is not None, 'Must provide number of epochs as int'
    assert isinstance(config['epochs'], int), 'Must provide number of epochs as int'




