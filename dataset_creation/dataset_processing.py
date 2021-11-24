import tensorflow as tf
import numpy as np
import os

from dataset_creation.image_transformations.patching import tf_create_patches


def read_image(filename, directory='', channels=3):
    """
    read image,
    can be applied/mapped to tf.data.Dataset!
    :param filename: string of image to be read
    :param directory: where to read image from
    :param channels: 3 channels for RGB

    :return: image as EagerTensor
    """
    if isinstance(filename, bytes):
        filename = filename.decode("utf-8")

    if isinstance(directory, bytes):
        directory = directory.decode("utf-8")

    image = tf.io.read_file(os.path.join(directory, filename))
    try:
        image = tf.image.decode_png(image, channels=channels)
        image = tf.cast(image, dtype=tf.dtypes.float32)
    except:
        print('file not readable:', filename)

    return image


def tf_read_image(filename, directory="", channels=3, original_image_shape=2048):
    """
    tensorflow wrapper to read image
    """
    image = tf.numpy_function(read_image, (filename, directory, channels), tf.float32)
    # image = tf.reshape(image, [original_image_shape, original_image_shape, channels])

    return image


def create_patched_dataset(dataset, patching_config, n_channels):
    """
    Use a dataset with images as input and return a dataset with multiple patches extracted from each image,
    the patches can be kept in their original order, shuffled or ranked by color intensity
    it is possible to keep the original image along with the patches

    :param dataset: dataset with 'images', labels, ...
    :param patching_config: dict with the following options:
        patch_size: int, size of one patch (width and height) to be cut from original image
        overlap: int, overlap between patches
        n_patches: &n_patches int, how many patches should be used in total per image
        order: original, ranked or shuffle
    :param n_channels: int, 3 if rgb

    :return: dataset, that now includes patches instead of whole images
    -------

    """
    patch_size = patching_config["patch_size"]
    if 'order' in patching_config:
        order = patching_config['order']
    else:
        print("no order specified, shuffling")
        order = 'shuffle'
    if "keep_original" in patching_config:
        keep_original = patching_config['keep_original'] if patching_config['keep_original'] is not None else False
    else:
        keep_original = False
    # multiple patches per image are required
    if "n_patches" in patching_config:
        n_patches = patching_config['n_patches']
    else:
        n_patches = 0
    overlap = patching_config['overlap']

    dataset = dataset.map(lambda x: ({'images_sums': tf_create_patches(x['images'],
                                                          patch_size,
                                                          overlap,
                                                          overlap,
                                                          n_patches=n_patches,
                                                          order=order,
                                                          n_channels=n_channels),
                                      **{key: x[key] for key in x if ((key not in ['images']) | keep_original)}}),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if keep_original:
        print('CAUTION: if image is used as a whole, all images need to have the same shape. '\
              'This is hard coded to be 2048 in dataset.py!')
        dataset = dataset.map(lambda x: ({'img_original': tf.image.resize(tf.reshape(x['images'], [2048, 2048, 3]),
                                                                          (keep_original, keep_original)),
                                          **{key: x[key] for key in x if key not in ['images']}}),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: ({'images': x['images_sums'][0],
                                      'patch_sums': x['images_sums'][1],
                                      **{key: x[key] for key in x if
                                         (((key not in ['images']) | keep_original) and
                                          (key not in ['images_sums']))}}),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def get_class_distribution(array_of_labels, data_generation_config):
    """
    return how many instances of which class exist, can be used to weight loss later for example
    :param array_of_labels: np.array but is kind of a list of interval/class per patient (like [3,2,2,6,1,0,0,2,3,4]
    :param data_generation_config: must include label_type and number_of_classes
    :return:
    """
    if (data_generation_config['label_type'] == 'isup'):
        class_distribution = np.array(
            [sum(array_of_labels == c) for c in range(data_generation_config['number_of_classes'])])
    elif (data_generation_config['label_type'] == 'survival'):
        class_distribution = np.array(
            [sum(array_of_labels == c) for c in range(data_generation_config['number_of_classes']+1)])
    elif data_generation_config['label_type'] == 'bin':
        class_distribution = np.array(
            [sum(array_of_labels == c) for c in range(data_generation_config['number_of_classes'])])
    else:
        raise NotImplementedError('only label_type isup, bin or survival implemented')
    return class_distribution


def squeeze_dataset(image_dataset, batch_size, info):
    """
    This function is used for MIL mainly, to remove the batchsize dimension from the dataset and have n_patches as first dimension
    input: dataset with size batch_size x n_patches x width x height x channels
    output: dataset with size batch_size*n_patches x width x height x channels

    batch_size needs to be passed because it is None usually
    :param image_dataset:
    :param batch_size:
    :param info:
    :return:
    """
    if info in ['images']:
        shape = image_dataset.shape
        #image_dataset = tf.cond(tf.equal(image_dataset.shape[0], 1), lambda: image_dataset[0], lambda: image_dataset) #tf.squeeze(image_dataset,0)
        image_dataset = tf.reshape(image_dataset, [shape[1]*batch_size, *shape[2:]])
    return image_dataset #tf.squeeze(image_dataset)
