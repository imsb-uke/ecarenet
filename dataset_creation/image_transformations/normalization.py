import tensorflow as tf


def normalize_image(image_np, image_channels):
    """
        normalize image to have values between 0 and 1
    :param image_np: np array: can be an image with width, height, channels or
                         multiple images (patches) with n_img, height, width, image_channels
    :param image_channels: integer, usually 3, e.g. needed if a mask is added to the image, the image has four
                                                        channels but should be normalized only on first three
    :return: np array with values only between 0 and 1
    """
    image_np[..., :image_channels] /= 255.0  # normalize to [0,1] range
    image_np = image_np.astype('float32')
    return image_np


@tf.function
def tf_normalize_image(image, image_channels):
    """
    tensorflow wrapper for normalization
    """
    shape = image.shape
    image = tf.numpy_function(normalize_image, (image, image_channels), tf.float32)
    image = tf.reshape(image, shape)
    return image
