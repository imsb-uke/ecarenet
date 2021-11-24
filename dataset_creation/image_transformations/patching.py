import tensorflow as tf
import numpy as np
import random


def create_patches(image, patch_size, overlap_horizontal, overlap_vertical, n_patches=0, order='ranked'):
    """
    This function takes an image (whole spot of prostate cancer biopsy for example) and cuts it into a variable number
    of patches, which may or may not overlap

    :param image: a numpy array of an image [m x n x 3]
    :param patch_size: int - the size that the patches should have in the end
    :param overlap_horizontal: either a percentage (for example 0.3) or a pixel amount of how much overlap is desired
    :param overlap_vertical: either a percentage (for example 0.3) or a pixel amount of how much overlap is desired
    :param n_patches: 0 if all patches should be used, otherwise any integer
                      (if more patches are cut than available, white patches are returned as well)
    :param order: 'original', 'shuffle', 'ranked', 'shuffle_ranked'

    :return: an array of patches [n_patches x patch_size x patch_size x 3],
             array of pixels in vertical direction where image patches start,
             array of pixels in horizontal direction where image patches start,
             array of per patch sums
    """

    img_shape = image.shape
    if isinstance(order, bytes):
        order = order.decode("utf-8")
    if n_patches == 0:
        assert img_shape[0] == img_shape[1] == 2048, \
            "right now, image size 2048x2048 is hard coded in tf_create_patches, " \
            "if different shape is used, please change source code to that shape or define n_patches in config"
    # pad image in case the image cannot be cut into patches of patchsize without having "leftovers"
    pad0 = (patch_size - img_shape[0] % (patch_size - overlap_vertical)) % (patch_size - overlap_vertical)
    pad1 = (patch_size - img_shape[1] % (patch_size - overlap_horizontal)) % (patch_size - overlap_horizontal)

    image = np.pad(image, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], constant_values=255)
    if n_patches == 0:
        n_patches = int(np.floor((image.shape[0] - overlap_horizontal) / (overlap_horizontal - patch_size)) *
                        np.floor((image.shape[1] - overlap_vertical) / (overlap_vertical - patch_size)))

    # in case the overlap is given as a ratio, the pixel value is calculated through the patch size
    if overlap_horizontal < 1:
        overlap_horizontal *= patch_size
    if overlap_vertical < 1:
        overlap_vertical *= patch_size

    if overlap_horizontal == overlap_vertical == 0:
        return advanced_patching(image, patch_size, n_patches, order)

    # make sure the overlap is in whole pixels
    overlap_vertical = int(overlap_vertical)
    overlap_horizontal = int(overlap_horizontal)

    # get the dimension of the image
    image_size = image.shape[0]

    first_indices_horizontal = np.arange(0, image_size - patch_size + 1, patch_size - overlap_horizontal)
    first_indices_vertical = np.arange(0, image_size - patch_size + 1, patch_size - overlap_vertical)

    number_resulting_patches = first_indices_horizontal.size * first_indices_vertical.size
    if number_resulting_patches < n_patches:
        extend_idx = n_patches - number_resulting_patches
        number_resulting_patches = n_patches
    else:
        extend_idx = False
    patches = np.ones((number_resulting_patches, patch_size, patch_size, 3)) * 255

    j = 0
    idx_v = []
    idx_h = []
    for idx_ver in first_indices_vertical:
        for idx_hor in first_indices_horizontal:
            patches[j, ...] = np.array(image[idx_ver:idx_ver + patch_size, idx_hor:idx_hor + patch_size, :])
            j += 1
            idx_v.append(idx_ver)
            idx_h.append(idx_hor)
    if extend_idx:
        idx_v = np.pad(idx_v, [0, extend_idx], constant_values=0)
        idx_h = np.pad(idx_h, [0, extend_idx], constant_values=0)
    idx_v = np.array(idx_v)
    idx_h = np.array(idx_h)
    if (order == 'shuffle_ranked') or (order == 'ranked') or (order == 'shuffle'):
        if (order == 'shuffle_ranked') or (order == 'ranked'):
            # rank patches according to highest color information
            idxs = np.argsort(patches.reshape(patches.shape[0], -1).sum(-1))[:n_patches]
        else:   # order == 'shuffle'
            idxs = [i for i in range(number_resulting_patches)]
        if (order == 'shuffle') or (order == 'shuffle_ranked'):
            random.shuffle(idxs)
            idxs = idxs[:n_patches]
        patches = patches[idxs]
        idx_v = idx_v[idxs]
        idx_h = idx_h[idxs]

    elif order == 'original':
        if len(patches) > n_patches:
            patches = patches[:n_patches]
            idx_h = idx_h[:n_patches]
            idx_v = idx_v[:n_patches]
    else:
        raise Exception('order needs to be one of shuffle, ranked or original')
    patch_sum = np.array(patches.reshape(patches.shape[0], -1).sum(-1), 'float32')
    idx_v = np.array(idx_v, dtype='int64')
    idx_h = np.array(idx_h, dtype='int64')
    return np.array(patches, 'float32'), np.array(idx_v), np.array(idx_h), patch_sum


def advanced_patching(image, patch_size, n_patches, order):
    """
    This should be faster, because the image is just reshaped, then tiled and sorted by color value,
    without for loops
    inspired by https://www.kaggle.com/iafoss/panda-16x128x128-tiles
    :param image: a numpy array of an image [m x n x 3]
    :param patch_size: int
    :param n_patches: int
    :param order: shuffle or ranked or original or shuffle_ranked (if not all possible patches are cut,
                                                                   shuffle the patches with most image information)
    :return:
    """
    img_shape = image.shape
    pad0 = (patch_size - img_shape[0] % (patch_size)) % (patch_size)
    pad1 = (patch_size - img_shape[1] % (patch_size)) % (patch_size)

    image = np.pad(image, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], constant_values=255)
    patches = image.reshape(image.shape[0] // patch_size, patch_size, image.shape[1] // patch_size, patch_size, 3)
    first_indices_horizontal = np.arange(0, image.shape[1], patch_size)
    first_indices_vertical = np.arange(0, image.shape[0], patch_size)
    idx_v = first_indices_vertical.repeat(first_indices_horizontal.shape[0])
    idx_h = np.tile(first_indices_horizontal, first_indices_vertical.shape[0])
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, 3)
    # in case more patches are needed than we actually have when cutting all, append white patches
    if len(patches) < n_patches:
        patches = np.pad(patches, [[0, n_patches - len(patches)], [0, 0], [0, 0], [0, 0]], constant_values=255)
        idx_h = np.pad(idx_h, [[0, n_patches-len(idx_h)]], constant_values=255)
        idx_v = np.pad(idx_v, [[0, n_patches-len(idx_v)]], constant_values=255)
    if (order == 'shuffle') or (order == 'ranked') or (order == 'shuffle_ranked'):
        patch_sum = patches.reshape(patches.shape[0], -1).sum(-1)
        if (order == 'ranked') or (order == 'shuffle_ranked'):
            idxs = np.argsort(patch_sum)[:n_patches]
            patch_sum = patch_sum[idxs]
            patches = patches[idxs]
            idx_h = idx_h[idxs]
            idx_v = idx_v[idxs]
        if (order == 'shuffle') or (order == 'shuffle_ranked'):
            idxs = [i for i in range(n_patches)]
            random.shuffle(idxs)
            patches = patches[idxs]
            patch_sum = patch_sum[idxs]
            idx_h = idx_h[idxs]
            idx_v = idx_v[idxs]
    elif order == 'original':
        patches = patches[:n_patches]
        patch_sum = patches.reshape(patches.shape[0], -1).sum(-1)[:n_patches]
        idx_h = idx_h[:n_patches]
        idx_v = idx_v[:n_patches]
    else:
        raise Exception("only order shuffle, ranked, 'shuffle_ranked' or original are valid in patching")
    return patches, np.array(idx_v, dtype='int64'), np.array(idx_h, dtype='int64'), patch_sum #np.array(patches.reshape(patches.shape[0], -1).sum(-1), 'float32')


@tf.function
def tf_create_patches(image, patch_size, overlap_horizontal, overlap_vertical, n_patches, order, n_channels=3):
    """
    tensorflow wrapper for patching
    """
    shape = image.shape
    image, _, _, patch_sums = tf.numpy_function(create_patches,
                                    (image, patch_size, overlap_horizontal, overlap_vertical, n_patches, order),
                                    (tf.float32, tf.int64, tf.int64, tf.float32))
    if shape[0] is None:
        shape0 = 2048
    else:
        shape0 = shape[0]
    if shape[1] is None:
        shape1 = 2048
    else:
        shape1 = shape[1]
    # if number of patches is not specified, the maximum number of patches is used, which needs to be computed here
    if n_patches == 0:
        if shape0 % (patch_size - overlap_vertical) == 0:
            image_number1 = int(shape0 / (patch_size - overlap_vertical))
        else:
            image_number1 = int(np.ceil(shape0 / (patch_size - overlap_vertical)))
        if shape1 % (patch_size - overlap_horizontal) == 0:
            image_number2 = int(shape1 / (patch_size - overlap_horizontal))
        else:
            image_number2 = int(np.ceil(shape1 / (patch_size - overlap_horizontal)))
        image_number = image_number1 * image_number2
    else:
        image_number = n_patches
    image = tf.reshape(image, [image_number, patch_size, patch_size, n_channels])
    return image, patch_sums
