import tensorflow as tf

from models.model_helpers import load_base_model


def m_isup(model_config):
    """
    M_ISUP to predict one out of 5 ISUP classes or benign from image, output is encoded with ordinal regression,
        therefore, number of output nodes is one less than number of classes

    :param n_classes: how many classes to use (one more than output nodes actually)
    :param model_config: dictionary with
                         base_model: e.g. string that is either a model that can be loaded from tf.keras.applications
                                    (like "InceptionV3") or it is the path to a folder with a self-pretrained model
                         for loading the model from tf.keras (input_shape: [m,n,3],
                                                              weights:'imagenet' / None)
    :return: tensorflow model
    """
    # first, load the base model, which can be InceptionV3 but also any other model defined in tf.keras
    bmodel, x = load_base_model(model_config['base_model'], model_config)
    n_classes = model_config['n_classes']

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if (model_config['dense_layer_nodes'] is not None) and (model_config['dense_layer_nodes'] != False):
        for n_nodes in model_config['dense_layer_nodes']:
            x = tf.keras.layers.Dense(units=n_nodes)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(units=n_classes-1, activation='sigmoid')(x)
    model = tf.keras.models.Model(bmodel.input, x)
    return model


