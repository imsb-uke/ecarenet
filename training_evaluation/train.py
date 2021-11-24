from tensorflow.keras.callbacks import Callback
from settings.sacred_experiment import ex
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import logging
import time
import re
import os
from dataset_creation.label_encoding import label_to_int


class LogPerformance(Callback):
    """
    Self defined Callback function, which at the moment calls an internal function at the end of each training epoch to
    log the metrics, save each model and delete old model if there was a better one afterwards
    """
    def __init__(self, train_params, run_id):
        """
        initialization
        :param train_params: dictionary with 'initial_epoch': int
                                             'monitor_val': string (which metric/loss to use to decide for best model)
                                             'model_save_path': string, where to save model
        :param run_id:
        """
        self.train_params = train_params
        self.prev_metric_value = - 9999.9
        self.curr_epoch = int(train_params['initial_epoch'])
        self.best_epoch = self.curr_epoch
        self.monitor = train_params['monitor_val']
        self.save_path = os.path.join(train_params['model_save_path'], str(run_id))

        super().__init__()
    @ex.capture
    def on_epoch_end(self, epoch, logs, _run):
        """
        at end of each epoch, it should be evaluated whether the model is better now (based on monitor_val) and should
        be saved or not
        :param epoch: current epoch, needed to save model
        :param logs: current metrics and losses
        :return:
        """
        self.log_performance(train_params=self.train_params, logs=logs, _run=_run)
        model_save_name = self.monitor.split('_')[-1][:5]
        self.model.save(os.path.join(self.save_path,
                        'temp_weights{:02d}_{:s}{:.3f}.hdf5'.format(epoch, model_save_name, logs.get(self.monitor))))

    def better_than_best_epoch(self, logs):
        """
        compare the current model to the previous ones
        for deleting the weights which do not belong to the best model/epoch
        own function b/c accuracy needs to be higher to be better, but loss needs to be lower

        :param logs: current metric/loss
        :return: True or False whether current model is best
        """
        if ('accuracy' in self.monitor) or ('f1' in self.monitor) or ('kappa' in self.monitor):
            if logs.get(self.monitor) > self.prev_metric_value:
                return True
            else:
                return False
        elif 'loss' in self.monitor:
            if logs.get(self.monitor) < abs(self.prev_metric_value):
                return True
            else:
                return False

    @ex.capture
    def log_performance(self, train_params, _run, logs):
        """
        this logs the loss and all defined metrics from config.yaml, so they are saved and plotted in the mongoDB

        :param _run: parameter from sacred experiment
        :param logs: keras Callback logs

        Returns:

        """
        if train_params["compile_metrics"] is not None:
            for metric in train_params["compile_metrics"]:
                metrics = re.sub(r'(?<!^)(?=[A-Z])', '_', metric).lower()
                _run.log_scalar(metrics, float(logs.get(metrics)))
                _run.log_scalar("val_"+metrics, float(logs.get('val_'+metrics)))
        _run.log_scalar("loss", float(logs.get('loss')))
        _run.log_scalar("val_loss", float(logs.get('val_loss')))
        # only keep last and best weights - delete the rest to avoid wasting memory
        if self.better_than_best_epoch(logs):
            self.prev_metric_value = float(logs.get(self.monitor))
            self.best_epoch = self.curr_epoch
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
                                                   train_params['model_save_path'], str(_run._id)))
        file_to_remove = [f for f in os.listdir(script_path) if
                          'temp_weights' in f and int(f.split('temp_weights')[1].split('_')[0]) != self.best_epoch]
        if any(file_to_remove):
            for f in file_to_remove:
                os.remove(os.path.join(script_path, f))
        self.curr_epoch += 1


@tf.function
def fiveepochlower(epoch, lr):
    """
    halve learning rate every five epochs
    :param epoch: int, current epoch
    :param lr: float, current learning rate
    :return: float, updated learning rate
    """
    if (epoch % 5 == 0) and epoch != 0:
        lr = lr/2
    return lr


def tenepochlower(epoch, lr):
    """
    halve learning rate every ten epochs
    :param epoch: int, current epoch
    :param lr: float, current learning rate
    :return: float, updated learning rate
    """
    if (epoch % 10 == 0) and epoch != 0:
        lr = lr/2
    return lr


@ex.capture
def list_callbacks(train_params, _run, _log):
    """
    This function returns a customized callback function (which logs the metrics) and can also add more standard
    keras callbacks

    :param train_params: _config["train"] or a dictionary with info about epochs, callbacks, ...
    :param _run: the _id of _run is important to be able to save results in correct folder

    :return:

    """
    callbacks = []
    callbacks.append(LogPerformance(train_params, _run._id))
    valid_schedulers = {'tenepochlower': tenepochlower, 'fiveepochlower': fiveepochlower}

    if train_params['callbacks'] is not None:
        for t in train_params['callbacks']:
            if t['name'] == 'LearningRateScheduler':
                callbacks.append(getattr(tf.keras.callbacks, t['name'])(schedule=valid_schedulers[t['params']['schedule']]))
            else:
                callbacks.append(getattr(tf.keras.callbacks, t['name'])(**t['params']))
            _log.debug(type(callbacks[-1]))
            _log.debug(t['params'])
    return callbacks



@tf.function
def training_step(model, datapoint, class_weights, label_type):
    """
    run model on input data, compute the loss and update the weights

    :param model: tensorflow model
    :param datapoint: datapoint as (tf.data) dict with labels and images
    :param class_weights: None or list of how many examples of each class exist, in order to weight samples
    :param label_type: 'bin' 'isup' or 'survival'
    :return:
    """
    label = datapoint['labels']
    img = datapoint['images']
    if class_weights is not None:
        class_weights = tf.convert_to_tensor([class_weights[k] for k in class_weights])
        int_of_class_weights = tf.cast([label_to_int(label, label_type)[i] for i in range(label.shape[0])], 'int32')
        sample_weights = tf.cast(tf.gather(class_weights, int_of_class_weights), 'float32')
    else:
        sample_weights = None

    with tf.GradientTape() as tape:
        prediction = model(img, training=True)
        loss = model.loss(y_true=label, y_pred=prediction)
        if sample_weights is not None:
            loss = loss * sample_weights
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return prediction, loss


@tf.function
def valid_step(model, datapoint, class_weights, label_encoding):
    """
    validation step: run model on batch and evaluate loss, no weight update
    :param model: tensorflow/keras model
    :param datapoint: dictionary with
                      'images' tf.Tensor [batchsize, h, w, 3] and
                     'labels' tf.Tensor [batch_size, n_classes (maybe+1 for censoring information)]
    :param class_weights: None or array with weight per class
    :param label_encoding: 'bin' or 'isup' or 'survival'

    :return: prediction tf.Tensor and loss tf.Tensor (scalar value though)
    """
    label = datapoint['labels']
    img = datapoint['images']
    if class_weights is not None:   # not np.all([class_weights[k] == 1 for k in class_weights]):
        class_weights = tf.convert_to_tensor([class_weights[k] for k in class_weights])
        int_of_class_weights = tf.cast([label_to_int(label, label_encoding)[i] for i in range(label.shape[0])], 'int32')
        sample_weights = tf.cast(tf.gather(class_weights, int_of_class_weights), 'float32')
    else:
        sample_weights = None
    prediction = model(img, training=False)

    loss = model.loss(y_true=label, y_pred=prediction)
    # loss = model.loss(y_pred=prediction, y_true=label)
    if sample_weights is not None:
        loss = loss * sample_weights
    loss = tf.reduce_mean(loss)
    return prediction, loss


def setup(num_classes, model):
    """
    for training and validation loop, the loss, error, data and prediction need to be (re)set each epoch
    :param num_classes: int, number of classes, needed for confusion matrix in classification
    :param model: tensorflow model, for which the metrics should be reset
    :return:
    """
    loss = 0
    error = 999
    data_storage = {'labels': list(), 'censored': list()}
    prediction_storage = list()
    cm = np.zeros((num_classes, num_classes))
    for metric in model.metrics:
        metric.reset_states()
    return loss, error, data_storage, prediction_storage, cm, model

#@tf.function
def update_standard_metrics(model, data_batch, prediction_batch, batch_size):
    """
    most metrics can be updated here, only some are left out (the ones that require more than a few data points)
    :param model: tensorflow / keras model
    :param data_batch: dictionary with 'images': tf.Tensor and 'labels': tf.Tensor
    :param prediction_batch: tf.Tensor (batch_size, n_classes)
    :param batch_size: TODO: remove b/c can be read from prediction shape?
    :return:
    """
    batch_size = prediction_batch.shape[0]
    for metric in model.metrics:
        label_batch = data_batch['labels']
        if metric.name in ['cohens_kappa'] and batch_size == 1:
            pass  # only calculate in the end
        elif metric.name in ['c_index_censor']:
            pass
        else:
            if 'censor' in metric.name:
                censored_batch = data_batch['censored']
                label_batch = tf.concat((np.array(label_batch, 'float32'), tf.expand_dims(np.array(censored_batch, 'float32'), 1)), 1)
            metric.update_state(label_batch, prediction_batch)

@tf.function
def update_other_metrics(model, data_gathered, prediction_gathered):
    """
    update cindex and cohens kappa metrics, since they require more datapoints than just a single batch for calculation
    :param model: tensorflow /keras model
    :param data_gathered:
    :param prediction_gathered:
    :return:
    """
    label_gathered = data_gathered['labels']
    for metric in model.metrics:
        if metric.name in ['cohens_kappa', 'c_index_censor']:
            if 'censor' in metric.name:
                censored_gathered = data_gathered['censored']
                label_gathered = tf.concat((tf.expand_dims(np.array(label_gathered, 'float32'), 1),
                                         tf.expand_dims(np.array(censored_gathered, 'float32'), 1)), 1)
            metric.update_state(label_gathered, prediction_gathered)


def handle_additional_metrics(pred_storage, data_storage, prediction_batch, datapoint_batch,
                              model, dpt_idx, train_steps_per_epoch, label_type):
    """
    metrics cohens kappa and cindex cannot be calculated on single batches during training, therefore information over
    multiple batches needs to be stored and the evaluation is only done every 10 epochs
    :param pred_storage: former predictions
    :param data_storage: former data
    :param prediction_batch: current prediction
    :param datapoint_batch: current data
    :param model: tensorflow model
    :param dpt_idx: current epoch
    :param train_steps_per_epoch: int, how many steps per epoch (to make sure evaluation is done in last epoch for sure)
    :param label_type: to turn label into integer, needs to know if it is binary, isup_classification or survival
    :return: model (maybe with updated metrics), updated or reset prediction storage and updated or reset data storage
    """
    # TODO: is it necessary to use the label encoding here for cohens kappa?

    label_batch = datapoint_batch['labels']
    pred_storage.extend(label_to_int(prediction_batch, label_type))
    data_storage['labels'].extend(label_to_int(label_batch, label_type))
    try:
        data_storage['censored'].extend(datapoint_batch['censored'])
    except:
        pass
    if ((dpt_idx % 10 == 0) and (dpt_idx != 0)) or (dpt_idx >= train_steps_per_epoch):
        update_other_metrics(model, data_gathered=data_storage, prediction_gathered=pred_storage)
        pred_storage = list()
        data_storage = {'labels': list(), 'censored': list()}
    return model, pred_storage, data_storage


@ex.capture
def train_loop(model, train_dataset, valid_dataset, train_batch_size, valid_batch_size, train_params,
               train_class_distribution, valid_class_distribution, class_weights, label_type,
               _run, _log=None):
    """

    :param model: tensorflow / keras model
    :param train_dataset: tf.data.Dataset with a dictionary structure, so ['images'] and ['labels'] can be accessed
    :param valid_dataset: tf.data.Dataset with a dictionary structure, so ['images'] and ['labels'] can be accessed
    :param train_batch_size: int
    :param valid_batch_size: int
    :param train_params: dictionary with at least
                         epochs: int, how many epochs to train in total
                         initial_epoch: int, 0 if it is a new training, higher if training is resumed
    :param train_class_distribution: array with size [n_classes,1], for each class the number indicates how many
                                     examples of this class are present in the dataset
    :param valid_class_distribution: array with size [n_classes,1], for each class the number indicates how many
                                     examples of this class are present in the dataset
    :param class_weights: None or array with class weights to be applied to each class (depends on class distribution)
    :param label_type: 'bin' or 'isup' or 'survival' - needed to calculate integer class from array
    :param experiments_dir: path to where the results should be saved (best model, intermediate loss, metrics, ...)
    :param _run: current run (id is needed for saving path) - defined by sacred
    :param _log: to print some intermediate results and the progress
    :return:
    """
    # INITIALIZATION
    if _log is None:
        _log = logging.getLogger('logger')
    initial_epoch = train_params['initial_epoch']
    epochs = train_params['epochs']
    train_steps_per_epoch = int(sum(train_class_distribution) / train_batch_size)
    if train_steps_per_epoch == 0:
        train_steps_per_epoch = 1
    valid_steps_per_epoch = int(sum(valid_class_distribution)/valid_batch_size)
    if valid_steps_per_epoch == 0:
        valid_steps_per_epoch = 1

    callbacks = list_callbacks(train_params, _run=_run)
    # at least, History() should be used as callback
    callbacks.append(tf.keras.callbacks.History())
    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin({m.name: m.result() for m in model.metrics})
        callback.on_epoch_begin(initial_epoch)

    #######################################
    # loop over epochs
    #######################################
    for epoch in np.arange(initial_epoch, epochs):
        # update callbacks
        for callback in callbacks:
            callback.on_epoch_begin(epoch)
        s = time.time()
        train_loss, train_error, data_storage, pred_storage, cm, model = \
            setup(train_class_distribution.shape[0], model)

        #######################################################################################
        # TRAINING
        #######################################################################################
        pbar = tqdm(total=train_steps_per_epoch)

        # iteration over train batches
        for dpt_idx, datapoint_batch in enumerate(train_dataset.take(train_steps_per_epoch)):
            # make prediction and calculate loss for this batch
            prediction_batch, loss_batch = training_step(model, datapoint_batch, class_weights, label_type)
            train_loss = train_loss + np.mean(loss_batch)

            update_standard_metrics(model, datapoint_batch, prediction_batch, train_batch_size)

            if (len([metric.name for metric in model.metrics if metric.name in ['cohens_kappa', 'c_index_censor']])) > 0:
                handle_additional_metrics(pred_storage, data_storage, prediction_batch, datapoint_batch,
                                          model, dpt_idx, train_steps_per_epoch, label_type)
            if dpt_idx % 20 == 0:
                pbar.update(20)
            if train_steps_per_epoch < 20:
                pbar.update(1)
        pbar.close()

        train_metrics = {**{m.name: m.result() for m in model.metrics}, 'loss': train_loss/(dpt_idx+1)}

        #######################################################################################
        # VALIDATION
        #######################################################################################

        valid_loss, valid_error, valid_label_storage, valid_pred_storage, valid_cm, model = \
            setup(valid_class_distribution.shape[0], model)

        pbar = tqdm(total=valid_steps_per_epoch)
        for dpt_idx, datapoint_batch in enumerate(valid_dataset.take(valid_steps_per_epoch)):
            prediction_batch, loss_batch = valid_step(model, datapoint_batch, class_weights, label_type)
            valid_loss += np.mean(loss_batch)

            update_standard_metrics(model, datapoint_batch, prediction_batch, valid_batch_size)

            if (len([metric.name for metric in model.metrics if metric.name in ['cohens_kappa', 'c_index_censor']])) > 0:
                handle_additional_metrics(pred_storage, data_storage, prediction_batch, datapoint_batch,
                                          model, dpt_idx, train_steps_per_epoch, label_type)

            if dpt_idx % 20 == 0:
                pbar.update(20)
            if valid_steps_per_epoch < 20:
                pbar.update(1)
        pbar.close()

        valid_metrics = {**{'_'.join(('val', m.name)): m.result() for m in model.metrics}, 'val_loss': valid_loss/(dpt_idx+1)}

        # update callbacks
        for callback in callbacks:
            callback.on_epoch_end(epoch, {**train_metrics, **valid_metrics})
        print(time.time()-s)
        print('Epoch {:03d}  -   '.format(epoch), end='')
        for v in train_metrics:
            print('    {:s}: {:.4f}   '.format(v, train_metrics[v]), end='')
        print('')

        print('Validation -   ', end='')
        for v in valid_metrics:
            print('{:s}: {:.4f}   '.format(v, valid_metrics[v]), end='')
        print('')
        print(model.optimizer.learning_rate)

    #######################################
    # AT END OF TRAINING - LOAD BEST MODEL
    #######################################

    _log.debug('Finished training -> evaluation')
    # find best epoch (minimum loss or maximum accuracy)
    if 'loss' in train_params['monitor_val']:
        best_epoch = np.argmin(callbacks[-1].history[train_params['monitor_val']]) + initial_epoch
    else:
        best_epoch = np.argmax(callbacks[-1].history[train_params['monitor_val']]) + initial_epoch

    _log.debug('Best epoch: {}'.format(best_epoch))

    best_model_path = [f for f in os.listdir(os.path.join(train_params['model_save_path'], str(_run._id)))
                       if 'weights' in f and int(f.split('weights')[1].split('_')[0]) == best_epoch][0]
    model.load_weights(os.path.abspath(os.path.join(train_params['model_save_path'], str(_run._id), best_model_path)))
    os.rename(os.path.join(train_params['model_save_path'], str(_run._id), best_model_path),
              os.path.join(train_params['model_save_path'], str(_run._id), best_model_path.replace('temp', 'temp_best')))

    # add best model to sacred
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', train_params['model_save_path']))
    files = [f for f in os.listdir(os.path.abspath(os.path.join(script_path, str(_run._id)))) if 'weights' in f]

    for f in files:
        if 'best' in f:
            ex.add_artifact(os.path.join(script_path, str(_run._id), f), name=f[5:],
                            content_type="application/octet-stream", )
        os.remove(os.path.join(script_path, str(_run._id), f))


    best_result = np.max(callbacks[-1].history[train_params['monitor_val']])
    _log.debug('best {monitor_val}: {result}'.format(monitor_val=train_params['monitor_val'], result=best_result))

    return model, callbacks[-1]


