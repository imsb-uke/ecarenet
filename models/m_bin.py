import tensorflow as tf

from models.model_helpers import load_base_model


def m_bin(model_config):
    """
    M_Bin model for binary classification of relapse before/after 2 years
    :param model_config: dictionary with
                         base_model: e.g. string that is either a model that can be loaded from tf.keras.applications
                                    (like "InceptionV3") or it is the path to a folder with a self-pretrained model
                         for loading the model from tf.keras (input_shape: [m,n,3],
                                                              weights:'imagenet' / None)
                         n_classes: two for binary prediction
    :return: tensorflow model
    """
    bmodel, x = load_base_model(model_config['base_model'], model_config)
    n_classes = model_config['n_classes']

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if (model_config['dense_layer_nodes'] is not None) and (model_config['dense_layer_nodes'] is not False):
        for n_nodes in model_config['dense_layer_nodes']:
            x = tf.keras.layers.Dense(units=n_nodes)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(units=n_classes, activation='softmax')(x)
    model = tf.keras.models.Model(bmodel.input, x)
    return model


