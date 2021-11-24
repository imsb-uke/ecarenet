import tensorflow as tf
import numpy as np

from models.model_helpers import load_base_model


def ecare_net(model_config):
    """
    model for survival prediction
    :param model_config: dictionary, with at least
                         base_model: e.g. string that is either a model that can be loaded from tf.keras.applications
                                    (like "InceptionV3") or it is the path to a folder with a self-pretrained model
                         n_patches: int (1 if no patches are cut, but image is used as a whole)
                         additional_input: list of strings, for each string, another input besides the image is expected
                                           additional inputs have shape 1
                         n_time_intervals: how many time intervals are modeled, equals number_of_classes
                         rnn_layer_nodes: list of integers, how many nodes per rnn layer
                         dense_layer_nodes: list of integers: after RNN, another dense layer is needed for final
                                            prediction, so specify here how many layers to include
                         mil_layer: True/False, include MIL layer or not (use patches or not
                         self_attention: True/False, include self-attention or not (only if MIL=True)
                         n_classes: number of output intervals
    :return:
    """
    # read parameters from config file
    n_patches = model_config['n_patches']
    if n_patches == 0:
        n_patches = 1
    additional_input = model_config['additional_input']
    rnn_layer_nodes = model_config['rnn_layer_nodes']
    dense_layer_nodes = model_config['dense_layer_nodes']
    use_mil_layer = model_config['mil_layer']
    use_self_attention = model_config['self_attention']
    n_time_intervals = model_config['n_classes']

    bmodel, x = load_base_model(model_config['base_model'], model_config)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)   # [b_s*n_patches, g]

    if use_self_attention:
        # model dependencies in between patches
        x = tf.reshape(x, [-1, n_patches, x.shape[-1]])   # [b_s, n_patches, g]
        x = self_attention(x, x.shape[-1])
        x = tf.reshape(x, [-1, x.shape[-1]])   # shape [b_s*n_patches, g]
    x = tf.keras.layers.RepeatVector(n_time_intervals, name='add_time_dimension')(x)   # [b_s*n_patches, time_dim, g]
    x2_in = []
    # in case binary classification should be added, can be extended to other parameters
    if additional_input:
        for i in additional_input:
            # dimension of additional input is 1
            x2_in.append(tf.keras.layers.Input((1), name=i))
            x2 = tf.reshape(x2_in[-1], (-1, 1))
            # needs to be repeated n_time_intervals times, to model the time dimension
            x2 = tf.repeat(x2, n_time_intervals, axis=1)
            # needs to be repeated n_patches times, to have one additional input per patch
            # e.g. originally img1-bin1
            # but now, with patches, we need patch_1a-bin1 patch_1b-bin1 patch_1c-bin1
            x2 = tf.repeat(x2, n_patches, 0)
            x2 = tf.expand_dims(x2, -1)
            # concat additional input with representation of image
            x = tf.concat((x2, x), 2)

    # as another input to the survival model, time intervals are needed
    x2_in.append(tf.keras.layers.Input(n_time_intervals, name='input_time'))

    # need to be repeated n_patches times, to have one time interval array per patch
    x2 = tf.repeat(x2_in[-1], n_patches, 0)
    x2 = tf.expand_dims(x2, -1)
    x = tf.concat((x2, x), 2)   # shape [b_s*patches, time_dim, g+additional_input]

    # model time dependency with RNN (here GRU)
    x = rnn_block(x, len(rnn_layer_nodes), rnn_layer_nodes, np.repeat(False, len(rnn_layer_nodes)), 'GRU')

    # output of RNN is of length rnn_layer_nodes, needs to be reduced to 1 node per time step
    if dense_layer_nodes[0] != 0:
        for n_nodes in dense_layer_nodes:
            x = tf.keras.layers.Dense(n_nodes, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    x = tf.keras.layers.Flatten()(x)

    if use_mil_layer:
        # if patches are used, the result needs to be one prediction per image, so mil is used here
        x = tf.reshape(x, [-1, n_patches, n_time_intervals])
        x = mil_layer(x, k=1)
    model = tf.keras.Model([bmodel.input, x2_in], x)
    return model


def self_attention(x, shape_out):
    """
    nice tutorial here: https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
    code also here: https://github.com/gmum/Kernel_SA-AbMILP/blob/daf4de6fd07924f1303887cc55244a4ff0bbfadf/model.py#L80
    paper: Kernel Self-Attention in Deep Multiple Instance Learning, Rymarczyk et al. 2020 - https://arxiv.org/abs/2005.12991
    :param x: input tensor
    :param shape_out: int, equals input shape, is needed to define size of key, value and query
    :return:
    """
    # first, define key, value and query
    key = tf.keras.layers.Conv1D(shape_out // 8, 1, name='key')(x)
    value = tf.keras.layers.Conv1D(shape_out, 1, name='value', activation='sigmoid')(x)
    query = tf.keras.layers.Conv1D(shape_out // 8, 1, name='query')(x)

    # query ° key = score (scalar product)
    score = tf.einsum('ijk,ikl->ijl', query, tf.transpose(key, [0, 2, 1]))   # same as tf.matmul

    # softmax(score)
    score = tf.keras.activations.softmax(score, axis=-1)

    # score ° value = multipl
    out = tf.einsum('ijk,ikl->ijl', score, value)   # score*value == (valueT*scoreT)T

    out = ConstMultiplierLayer()(out)
    out = out+x
    return out


def mil_layer(x, d=128, k=1):
    """
    see also https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
    paper: Attention-based Deep Multiple Instance Learning, Ilse et al. 2018 - https://arxiv.org/abs/1802.04712
    MIL for a batch size greater than 1
    :param x: input vector of shape (batch_size, n_patches, output_size) (one entry per patch from multiple batches)
              batch_size is treated as time dimension here, to handle a batch_size > 1
    :param d:
    :param k: 1 if one prediction for all patches is wanted, with n_patches it could return one prediction per patch
    :param gated: gated or not gated attention, according to the paper

    :return: a tensor, that now has one prediction per image, not per patch anymore (batch_size, output_size)
    -------

    """
    alpha = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d, name='attention_dense1'))(x)   # NxD
    alpha = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh', name='attention_tanh'))(alpha)  # N x D
    beta = 1
    alpha = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(k, name='attention_dense2'))(alpha*beta)  # NxK
    alpha = tf.keras.layers.Permute((2, 1))(alpha)   # K x N
    alpha = tf.keras.layers.TimeDistributed(tf.keras.layers.Softmax(), name='attention_softmax')(alpha)   # K x L

    x = tf.einsum('ijk,ikl->il', alpha, x)
    return x


class ConstMultiplierLayer(tf.keras.layers.Layer):
    """
    Layer to multiply by a constant value
    https://github.com/keras-team/keras/issues/10204
    """
    def __init__(self, **kwargs):
        super(ConstMultiplierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer='zeros',
            trainable=True,
            #constraint=tf.keras.constraints.NonNeg()
        )

        super(ConstMultiplierLayer, self).build(input_shape)

    def call(self, x):
        return tf.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape


def rnn_block(x, rnn_layer_num, rnn_node_num, dropout, layer_type):
    """
    add rnn layers (to the end of a 'basic' network)

    :param x: input tensor of shape (time_dimension(n_time_intervals), batch_size, g)
    :param rnn_layer_num: int how many RNN layers
    :param rnn_node_num: list of ints, how many nodes in each layer
    :param dropout: how much dropout (or False) for each layer
    :param layer_type: which type, LSTM or GRU or ...

    :return: output tensor

    """
    for i in range(rnn_layer_num):
        x = getattr(tf.keras.layers, layer_type)(rnn_node_num[i], dropout=dropout[i], return_sequences=True)(x)
    return x
