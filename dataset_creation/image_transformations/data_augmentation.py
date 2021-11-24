from dataset_creation.image_transformations.data_augmentation_helpers import *


def get_one_uniform_value(value_range_or_upper_bound, lower_bound=0):
    """
    return a random value from a specified interval or [0, 1]
    :param value_range_or_upper_bound: list or upper and lower limits or float of upper bound
    :param lower_bound: float, lower limit
    :return: single float value
    """
    if isinstance(value_range_or_upper_bound, list):
        random_value = np.array(np.random.uniform(value_range_or_upper_bound[0], value_range_or_upper_bound[1], 1), dtype='float32')[0]
    else:
        random_value = np.array(np.random.uniform(lower_bound, value_range_or_upper_bound, 1), dtype='float32')[0]
    return random_value


def get_random_params(
                      seed=None,
                      rotation_range=0.,
                      width_shift_range=0.,
                      height_shift_range=0.,
                      brightness_range=0.,
                      horizontal_flip=False,
                      vertical_flip=False):
    """
    return a random value per parameter. Disable parameter with default values 0 or False
    :param seed: integer, for reproducibility
    :param rotation_range: rotation in degree,
                           value will be chosen from 0 to rotation_range or from rotation_range[0] to rotation_range[1]
    :param width_shift_range: width shift in pixels
                           value will be chosen from -width_shift_range to +width_shift_range or
                           from width_shift_range[0] to width_shift_range[1]
    :param height_shift_range: height shift in pixels
                               value will be chosen from -height_shift_range to +height_shift_range or
                               from height_shift_range[0] to height_shift_range[1]
    :param brightness_range: brightness in percent
                             value will be chosen from -brightness_range to +brightness_range or from ...[0] to ...[1]
                             brightness*255 will be added to all image pixel values
    :param horizontal_flip: if True, randomly returns True or False
    :param vertical_flip: if True, randomly returns True or False

    :return: rotation (float), width_shift (float), height_shift (float), brightness (float),
             horizontal_flip (bool), vertical_flip (bool)
    """

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    # tf.random.set_seed(9999)
    # np.random.seed(999)

    if brightness_range and (brightness_range != 0):
        brightness = get_one_uniform_value(brightness_range, 0)
        if np.random.randint(2) == 0:
            brightness = - brightness
    else:
        brightness = False
    if rotation_range and (rotation_range != 0):
        rotation = get_one_uniform_value(rotation_range, 0)
    else:
        rotation = False
    if width_shift_range and (width_shift_range != 0):
        width_shift = get_one_uniform_value(width_shift_range, 0)
        if np.random.randint(2) == 0:
            width_shift = - width_shift
    else:
        width_shift = False
    if height_shift_range and (height_shift_range != 0):
        height_shift = get_one_uniform_value(height_shift_range, 0)
        if np.random.randint(2) == 0:
            height_shift = -height_shift
    else:
        height_shift = False
    if horizontal_flip:
        horizontal_flip = tf.cast(tf.random.uniform([1], minval=0, maxval=2, dtype=tf.dtypes.int32), tf.bool)
    if vertical_flip:
        vertical_flip = tf.cast(tf.random.uniform([1], minval=0, maxval=2, dtype=tf.dtypes.int32), tf.bool)
    return rotation, width_shift, height_shift, brightness, horizontal_flip, vertical_flip


@tf.function
def tf_augment_image(image, random_augmentation=False,
                     intensity=0,
                     rotation=0,
                     width_shift=0.,
                     height_shift=0.,
                     brightness=None,
                     horizontal_flip=False,
                     vertical_flip=False,
                     fill_mode='constant',
                     cval=255,
                     ):
    """
    Augments an array of images
    :param image: Tensor of images (batch_size x m x n x 3)
    :param random_augmentation: if True, augmentation parameters will be chosen randomly, bound by given values
                                if False, augmentation will be deterministic with given values
    :param intensity: adapt intensity
    :param rotation: in degree, either single value or list of lower and upper bound
    :param width_shift: shift image to left/right, either single value or list of lower and upper bound
    :param height_shift: shift image up/down, either single value or list of lower and upper bound
    :param brightness: percentage*255 will be added to pixel values of image,
                       either single value or list of lower and upper bound
    :param horizontal_flip: True/False, flip image horizontally or not
    :param vertical_flip: True/False, flip image vertically or not
    :param fill_mode: string, how to fill image parts that are "empty" after augmentation. Defaults to a constant color
    :param cval: which value to use to fill "empty" parts if fill_mode==constant. Defaults to 255 (white)

    :return: augmented image array
    """

    if random_augmentation:
        rotation, \
          width_shift, height_shift, \
          brightness, \
          horizontal_flip, vertical_flip = get_random_params(
                                            seed=None,
                                            rotation_range=rotation,
                                            width_shift_range=width_shift,
                                            height_shift_range=height_shift,
                                            brightness_range=brightness,
                                            horizontal_flip=horizontal_flip,
                                            vertical_flip=vertical_flip)

    if intensity:
        if intensity != 0:
            image = tf.image.adjust_saturation(image, intensity)

    if rotation:
        if rotation != 0:
            if rotation == 90:
                image = tf.image.rot90(image, 1)
            else:
                rotation = rotation/180*np.pi
                image = tf_rotate_image_numpy(image, rotation, fill_mode, cval)

    width_shift = width_shift if width_shift else 0
    height_shift = height_shift if height_shift else 0
    if width_shift or height_shift:
        image = tf_shift_image_numpy(image, width_shift=width_shift, height_shift=height_shift, fill_mode=fill_mode, cval=cval)

    if brightness:
        if brightness != 0:
            image = tf.image.adjust_brightness(image, brightness*255.)

    if horizontal_flip:
        image = tf.image.flip_left_right(image)
    if vertical_flip:
        image = tf.image.flip_up_down(image)
    # image = tf.clip_by_value(image, 0.0, 255.0)
    return image


