import tensorflow as tf
import os


def get_data_directory(examples='color'):
    if 'color' in examples:
        e = 'example_colors'
    elif 'circle' in examples:
        e = 'example_circles'
    directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'unittest_data', e))
    return directory


def get_model_directory(run_id='parent'):
    if run_id == 'parent':
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'unittest_data', 'test_models'))
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'unittest_data', 'test_models', run_id))


def create_image_dataset(number_of_examples, resize=False):
    directory = get_data_directory()
    image = tf.io.read_file(os.path.join(directory, "img_red_0.png"))
    image = tf.image.decode_image(image, 3)
    if resize:
        image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, dtype=tf.dtypes.float32)

    dataset = tf.data.Dataset.from_tensor_slices(([image for i in range(number_of_examples)]))
    return dataset


def create_image_label_dataset(number_of_examples, label, resize=False):
    directory = get_data_directory()
    image = tf.io.read_file(os.path.join(directory, "img_red_0.png"))
    image = tf.image.decode_image(image, 3)
    if resize:
        image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, dtype=tf.dtypes.float32)
    if not isinstance(label, list):
        label = [label for i in range(number_of_examples)]
    dataset = tf.data.Dataset.from_tensor_slices(([image for i in range(number_of_examples)], [label[i] for i in range(number_of_examples)]))
    return dataset