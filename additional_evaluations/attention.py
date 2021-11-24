from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import json
import os

from dataset_creation.dataset_main import create_dataset
from models.model_helpers import load_base_model, compile_model


def get_mil_alpha(model, image, save_path, show_steps=False):
    """
    evaluate model on image, find attention weights per patch and plot an overlay of image and attention weights
    :param model: tensorflow/keras model
    :param image:  tuple of tf.Tensor (image as Tensor +  interval limits as Tensor)
    :param save_path: False or complete file path (with filename) where to save attention image
    :param show_steps:
    :return:
    """
    # alpha has the output of softmax layer, so the attention per input patch
    alpha = model(image)
    # squeeze to remove batch size 1
    alpha = tf.squeeze(alpha)
    print(alpha)
    # find out if input was just patches or an image plus patches (and if additional input that is no image was used)
    try:
        # only input are MIL patches
        img_shape = image.shape[0]
        multi = False
    except AttributeError:
        # more than just an image or MIL
        img_shape = image[0].shape[0]
        if len(image[1].shape) > 2:
            # multi MIL: patches + image
            img_w = image[1]
            multi = True
        else:
            # patches + additional input
            multi = False
        image = image[0]

    # to plot the patches in the shape of the original image, find out how many rows and columns are needed
    # e.g. for 16 patches, plot them as 4x4
    n_plots_1 = int(np.round(np.ceil(np.sqrt(img_shape))))
    n_plots_0 = int(img_shape/n_plots_1)

    # size of one patch
    patch_shape0 = image.shape[1]
    patch_shape1 = image.shape[2]
    # size of original image, if patches are stitched
    original_shape0 = patch_shape0 * n_plots_0
    original_shape1 = patch_shape1 * n_plots_1

    # stitch patches together to one larger image
    p_idx = 0
    image_whole = np.ones((original_shape0, original_shape1, 3))
    for i in np.arange(0, original_shape0, patch_shape0):
        for j in np.arange(0, original_shape1, patch_shape1):
            if (p_idx == 0) & multi & False:   # TODO: what's this supposed to mean?
                image_whole[i:i + patch_shape0, j:j + patch_shape1] = tf.image.resize(img_w, (patch_shape0, patch_shape1))
            else:
                image_whole[i:i + patch_shape0, j:j + patch_shape1] = image[p_idx]
            p_idx += 1

    # attention was just a scalar per patch, now blast it, so it is same size as image
    attention_whole = np.ones((image_whole.shape[0], image_whole.shape[1]))
    alpha_idx = 0
    for i in np.arange(0, original_shape0, patch_shape0):
        for j in np.arange(0, original_shape1, patch_shape1):
            if (alpha_idx == 0) & (multi) & False:   # TODO: what's this supposed to mean?
                attention_whole[i:i+patch_shape0, j:j+patch_shape1] *= alpha[-1]
            else:
                attention_whole[i:i+patch_shape0, j:j+patch_shape1] *= alpha[alpha_idx]
            alpha_idx += 1

    if multi:
        image_whole = np.concatenate((tf.image.resize(image_whole, (img_w[0].shape[0], img_w[0].shape[1])), img_w[0]), 1)
        attention_add = np.ones_like(img_w[0]) * alpha[-1]
        attention_whole = np.repeat(attention_whole[..., np.newaxis],3,-1)
        attention_whole = np.concatenate((tf.image.resize(attention_whole, (img_w[0].shape[0], img_w[0].shape[1])), attention_add),1)
        attention_whole = attention_whole[..., 0]

    # ### plot patches ### #
    if show_steps:
        for j in range(img_shape):
            plt.subplot(n_plots_0, n_plots_1, j + 1)
            plt.imshow(image[j])
            plt.axis('off')
            plt.title(alpha[j].numpy())
        # plt.title('original patches')
        plt.show()

        plt.imshow(attention_whole, cmap='OrRd')
        plt.title('attention')
        plt.colorbar()
        plt.show()
        f = plt.figure()
        plt.imshow(image_whole)
        plt.axis('off')

    if show_steps:
        f = plt.figure()
        plt.imshow(image_whole)
        plt.imshow(attention_whole, cmap='gray', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
    if save_path:
        aw = attention_whole[..., np.newaxis].repeat(3, 2)
        aw = aw - np.min(aw)
        aw = aw / np.max(aw)
        img_x = image_whole / 2 + aw / 2
        img_x = Image.fromarray(np.array(img_x * 255, dtype='uint8'))
        img_x = img_x.resize((512, 512))
        img_x.save(save_path)

    plt.show()
    plt.close('all')
    return alpha


def attention_per_image(model, dataset, n_examples, save_plot):
    """
    iterate over all (n_example) images in dataset, find attention weights per patch and maybe save image as png
    along with resulting weights in csv file
    :param model: tensorflow/keras model
    :param dataset:tf.data.Dataset in a dictionary style with at least
                   'image_paths': string - path to image
                   'images': tf.Tensor with image
                   'censored':  tf.Tensor with 0 or 1
    :param n_examples: how many examples to evaluate
    :param save_plot: False or path to where to save model as string
    :return:
    """

    mil_output = [l for l in model.layers if 'attention_softmax' in l.name][0].output
    model = tf.keras.models.Model(model.input, mil_output)
    list_of_attentions = []
    for idx, d in enumerate(dataset.take(n_examples)):
        if save_plot:
            c = int(np.array(d['censored']))
            save_path = save_plot + '/attention_%s_c%i.png' % (
            d['image_paths'].numpy()[0].decode('utf8').split('/')[1][:-4], c)
        else:
            save_path = False

        alpha = get_mil_alpha(model, d['images'], save_path)
        alpha = list(alpha)
        alpha = [float(a) for a in alpha]
        alpha.insert(0, d['image_paths'].numpy()[0].decode('utf8'))
        list_of_attentions.append(alpha)
    if save_plot:
        pd.DataFrame(list_of_attentions).to_csv(save_plot + '/attentions.csv')


def plot_attention_for_model(model_directory, run_id, n_examples=4, save_plot=False, mode='valid'):
    """
    create the dataset and start the attention calculation process with this function

    :param model_directory: /path/to/folder where run_id is located
    :param run_id: folder where model is stored
    :param n_examples: how many examples to read from dataset
    :param save_plot: bool, save resulting plot in model_directory/run_id or not
    :param mode: which dataset to use (defined in model_directory/run_id/config.json)
    :return:
    """
    if save_plot:
        save_plot = os.path.join(model_directory, str(run_id))
    config = json.load(open(os.path.join(model_directory, str(run_id), 'config.json')))
    config['data_generation']['valid_batch_size'] = 1
    config['data_generation']['train_batch_size'] = 1
    config['data_generation']['cache'] = None

    dataset, class_distribution = create_dataset(data_generation_config=config['data_generation'],
                                                 usage_mode=mode,
                                                 _config=config
                                                 )

    model, _ = load_base_model(os.path.join(model_directory, str(run_id)), {'keras_model_params':
                                    {'input_shape': config['model']['keras_model_params']['input_shape']}})

    # compile model
    model = compile_model(model, config['training'], None)

    attention_per_image(model, dataset, n_examples, save_plot)
