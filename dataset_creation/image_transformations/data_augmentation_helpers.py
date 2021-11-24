from numpy.core.multiarray import normalize_axis_index
from scipy import ndimage
import tensorflow as tf
import numpy as np


def random_rotation(
    image,
    rotation_range,
    channel_axis=2,
    fill_mode='nearest',
    cval=255,
    interpolation_order=1
):
    """

    :param image: image as numpy array, needs to have three dimensions
    :param rotation_range: rotation range in degrees, image will be shifted some value from -range to +range
    :param channel_axis: index of axis for channels in the input tensor.
    :param fill_mode: str, points outside the boundaries of the input are filled according to the given mode
                      (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param cval: int or float, value used for points outside the boundaries of the input if `mode='constant'`
    :param interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`
    :return: rotated image as numpy
    """
    theta = np.random.uniform(-rotation_range, rotation_range)
    image = apply_affine_transform(
        image, theta=theta, channel_axis=channel_axis,
        fill_mode=fill_mode, cval=cval, order=interpolation_order
    )
    return image


@tf.function
def tf_rotate_image_numpy(image, rotation, fill_mode, cval):
    """
    :param image:
    :param rotation:
    :param fill_mode:
    :param cval:
    :return:
    """
    img_shape = image.shape
    if len(img_shape) == 4:
        idx = 0
        imgs = tf.zeros_like(image)
        for img in image:
            img = tf.numpy_function(random_rotation, [img, rotation, 2, fill_mode,
    cval, 1], tf.float32)
            imgs = tf.concat([imgs, tf.expand_dims(img, 0)], 0)[1:, ...]
            imgs = tf.reshape(imgs, img_shape)
        img = imgs

    else:
        img = tf.numpy_function(random_rotation, [image, rotation, 0, 1, 2, fill_mode,
    cval, 1], tf.float32)

    return tf.reshape(img, img_shape)


def tf_rollaxis(matrix, axis, start):
    """
    this is a copy of numpy.rollaxis, because it cannot be used with a tensor
    :param matrix:
    :param axis:
    :param start:
    :return:
    """
    n = matrix.ndim
    axis = normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise np.AxisError(msg % ('start', -n, 'start', n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        return matrix[...]
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return tf.transpose(matrix, axes)


def transform_matrix_offset_center(matrix, x, y):
    """

    :param matrix:
    :param x:
    :param y:
    :return:
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform(
    x,
    theta=0,
    row_axis=0,
    col_axis=1,
    channel_axis=2,
    fill_mode='constant',
    cval=255,
    order=1
):
    """

    :param x: 2D numpy array, single image.
    :param theta: Rotation angle in degrees.
    :param row_axis: Index of axis for rows in the input image.
    :param col_axis: Index of axis for columns in the input image.
    :param channel_axis: Index of axis for channels in the input image.
    :param fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
    :param cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order (int): int, order of interpolation
    :param order:
    :return: transformed (np.ndarray): The transformed version of the input.
    """

    transform_matrix = None
    if theta is not None and theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [            0,              0, 1]])

        transform_matrix = rotation_matrix

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        # added for tf.numpy_function
        h = tf.cast(h, tf.float32)
        w = tf.cast(w, tf.float32)
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        # put the channel in the first dimension (to iterate over 3 channels later?)
        x = tf_rollaxis(x, channel_axis, 0)   # np.rollaxis(x, channel_axis, 0)       # does not work with tf.Tensor
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        if not isinstance(fill_mode, str):
            fill_mode = fill_mode.decode("utf-8")
        channel_images = [
            ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=order,
                mode=fill_mode,
                cval=cval
            ) for x_channel in x
        ]

        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)   # works now, because x is an ndarray, not a tensor anymore!

    return x


@tf.function
def tf_shift_image_numpy(image, width_shift, height_shift, fill_mode, cval):
    """

    :param image:
    :param width_shift:
    :param height_shift:
    :param fill_mode:
    :param cval:
    :return:
    """
    img_shape = image.shape
    width_shift = int(img_shape[-2] * width_shift)
    height_shift = int(img_shape[-3] * height_shift)
    img = tf.numpy_function(shift_numpy, [image, width_shift, height_shift, cval], tf.float32)
    return tf.reshape(img, img_shape)


def shift_numpy(image, width_shift, height_shift, cval):
    """

    :param image:
    :param width_shift:
    :param height_shift:
    :param cval:
    :return:
    """
    image_new = np.ones_like(image)*cval
    if height_shift > 0:
        if width_shift > 0:
            image_new[..., height_shift:, width_shift:, :] = image[..., :-height_shift, :-width_shift, :]
        elif width_shift < 0:
            image_new[..., height_shift:, :width_shift, :] = image[..., :-height_shift, -width_shift:, :]
        else:
            image_new[..., height_shift:, :, :] = image[..., :-height_shift, :, :]
    elif height_shift < 0:
        if width_shift > 0:
            image_new[..., :height_shift, width_shift:, :] = image[..., -height_shift:, :-width_shift, :]
        elif width_shift < 0:
            image_new[..., :height_shift, :width_shift, :] = image[..., -height_shift:, -width_shift:, :]
        else:
            image_new[..., :height_shift, :, :] = image[..., -height_shift:, :, :]
    else:
        if width_shift > 0:
            image_new[..., width_shift:, :] = image[..., :-width_shift, :]
        elif width_shift < 0:
            image_new[..., :width_shift, :] = image[..., -width_shift:, :]
        else:
            image_new = image
    return image_new
