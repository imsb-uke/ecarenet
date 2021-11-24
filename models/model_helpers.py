import training_evaluation.model_metrics as own_metrics
import training_evaluation.model_losses as own_losses
from tensorflow.keras import applications
import tensorflow as tf
import importlib
import logging
import json
import os


def load_base_model(base_model, model_config):
    """
    either load a model predefined in tf.keras (pretrained or not) or load an own model
    :param base_model:
    :param model_config:
    :return:
    """
    try:
        # load standard (pretrained) model, which can be InceptionV3 but also any other model defined in tf.keras
        bmodel = getattr(applications, base_model)(include_top=False, **model_config["keras_model_params"])
    except AttributeError:
        # load own pretrained model
        best_model_json = 'model_json.json'
        best_model_hdf = [f for f in os.listdir(base_model) if 'best_weights' in f][0]
        try:
            omodel = tf.keras.models.model_from_json(open(os.path.join(base_model, best_model_json)).read())
        except ValueError:
            config = json.load(open(os.path.join(base_model, 'config.json')))
            mod = importlib.import_module('models.' + config['model']['name'])
            omodel = getattr(mod, config['model']['name'])(config['model'])
        omodel.load_weights(os.path.join(base_model, best_model_hdf))
        if isinstance(omodel.input, list):
            inp_shape = omodel.input[0].shape[1:]
        else:
            inp_shape = omodel.input.shape[1:]
        if inp_shape != model_config['keras_model_params']['input_shape']:
            omodel._layers[1]._batch_input_shape = (None,
                                                    model_config['keras_model_params']['input_shape'][1],
                                                    model_config['keras_model_params']['input_shape'][1], 3)
            omodel._layers[0]._batch_input_shape = (None,
                                                    model_config['keras_model_params']['input_shape'][1],
                                                    model_config['keras_model_params']['input_shape'][1], 3)
            omodel._layers[0].__input_shape = (None,
                                               model_config['keras_model_params']['input_shape'][1],
                                               model_config['keras_model_params']['input_shape'][1], 3)
    #
            bmodel = tf.keras.models.model_from_json(omodel.to_json())
        else:
            bmodel = omodel
        for layer in bmodel.layers:
            try:
                layer.set_weights(omodel.get_layer(name=layer.name).get_weights())
                layer._name = layer._name + str('_base')
            except:
                print("Could not transfer weights for layer {}".format(layer.name))

    if 'cut_off_layer' in model_config:
        cut_off_after_layer = model_config['cut_off_layer']
    else:
        cut_off_after_layer = False
    if isinstance(cut_off_after_layer, int) and not isinstance(cut_off_after_layer, bool):
        x = bmodel.layers[cut_off_after_layer].output
    elif isinstance(cut_off_after_layer, str):
        x = bmodel.get_layer(cut_off_after_layer).output
    else:
        x = bmodel.layers[-1].output
    return bmodel, x


def compile_model(model, train_params, metric_class_weights=None, _log=logging):
    """

    :param model: tensorflow/keras model
    :param train_params: dictionary with 'optimizer.name'
    :param metric_class_weights:
    :param _log:
    :return:
    """
    try:
        # parameters are a list, so not unintentionally overwritten by main config
        optimizer = getattr(tf.keras.optimizers, train_params["optimizer"]['name'])(
                **{k: v for d in train_params['optimizer']['params'] for k, v in d.items()})
    except AttributeError:
        raise NotImplementedError("only optimizers available at tf.keras.optimizers are implemented at the moment")

    try:
        loss = getattr(tf.keras.losses,  train_params["loss_fn"])#(reduction=tf.keras.losses.Reduction.NONE)
    except AttributeError:
        try:
            loss = getattr(own_losses, train_params['loss_fn']+'_wrap')()
        except AttributeError:
            raise NotImplementedError("only losses "
                                      "available at tf.keras.losses or "
                                      "cdor, deepconvsurv and ecarenet_loss "
                                      "are implemented at the moment")
    except TypeError:
        print('loss_information', *train_params['loss_fn'])
        loss = getattr(own_losses, train_params['loss_fn'][0]+'_wrap')(*train_params['loss_fn'][1:])

    # read all metrics from the list in config.yaml

    metrics = []
    if train_params['compile_metrics'] is not None:
        for metric in train_params["compile_metrics"]:
            try:
                metrics.append(getattr(tf.keras.metrics, metric)())
            except AttributeError:
                try:
                    metrics.append(getattr(own_metrics, metric + '_wrap')(metric_class_weights))
                except AttributeError:
                    raise NotImplementedError("the given metric {} is not implemented!".format(metric))

    compile_attributes = train_params["compile_attributes"]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        **compile_attributes
    )

    if _log is not None:
        _log.info("model successfully compiled with optimizer %s %s" % (train_params["optimizer"], optimizer))
    else:
        print("model successfully compiled with optimizer %s %s" % (train_params["optimizer"], optimizer))
    return model
