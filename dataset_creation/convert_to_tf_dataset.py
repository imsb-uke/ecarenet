import tensorflow as tf


def make_image_dataset(images, encoded_labels, additional_info):
    """
    :param images: pd.Series with one image path per row
    :param encoded_labels: array with one row per example and as many columns as output nodes
    :param additional_info: dictionary with pd.Series or list of additional information like censoring information
    :return: tf.Data.dataset containing image_paths, labels and maybe further fields for additional information
    """

    dataset = tf.data.Dataset.from_tensor_slices((images, encoded_labels))
    dataset = dataset.map(lambda x, la: ({'image_paths': x, 'labels': la}))
    if len(additional_info) != 0:
        dataset_add = tf.data.Dataset.from_tensor_slices(({key: additional_info[key] for key in additional_info}))
        dataset = dataset.zip((dataset_add, dataset)).map(lambda x, y: {**{k: x[k] for k in x}, **{k: y[k] for k in y}})

    return dataset
