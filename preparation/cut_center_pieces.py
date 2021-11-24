from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import cv2
import os
sys.path.append('/opt/project')


def fit_circle(img, show_rect_or_cut='show'):
    """
    fit an ellipse to the contour in the image and find the overlaying square.
    Either cut the center square or just plot the resulting square

    Code partly taken from here:
    https://stackoverflow.com/questions/55621959/opencv-fitting-a-single-circle-to-an-image-in-python

    :param img: numpy array with width, height,3
    :param show_rect_or_cut: string 'show' or 'cut'
    :return: image, either cut center piece or with drawn square
             flag, whether algorithm thinks this image is difficult (if the circle is too small or narrow
    """
    # convert image to grayscale and use otsu threshold to binarize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)

    # fill holes
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(15, 15))
    morph_img = thresh.copy()
    cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)

    # find contours in image and use the biggest found contour
    contours, _ = cv2.findContours(morph_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)

    cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour

    if len(cnt) < 10:
        return img, 'Diff'

    # fit ellipse and use found center as center for square
    ellipse = cv2.fitEllipse(cnt)
    if np.min((ellipse[1][0], ellipse[1][1])) < 900:
        flag = 'Diff'
    else:
        flag = False

    r_center_x = int(ellipse[0][0])
    r_center_y = int(ellipse[0][1])

    r_center_x = np.max((r_center_x, 1024))
    r_center_x = np.min((r_center_x, img.shape[0] - 1024))

    r_center_y = np.max((r_center_y, 1024))
    r_center_y = np.min((r_center_y, img.shape[1] - 1024))

    if show_rect_or_cut == 'show':
        half_width = 1024
        cv2.rectangle(img,
                      (r_center_x - half_width, r_center_y - half_width),
                      (r_center_x + half_width, r_center_y + half_width),
                      (0, 150, 0), 40)
    elif show_rect_or_cut == 'cut':
        img = img[r_center_y - 1024:r_center_y + 1024,
                  r_center_x - 1024:r_center_x + 1024, :]

    return img, flag


def show_cut_square(path='/data',
                    n_examples=99999,
                    show_rect_or_cut='show',
                    show_diff='both',
                    df=None):
    """
    cut a square from all images by first fitting a circle/ellipse and then using the square around its center

    :param path: /path/to/folder where the original images are stored
    :param n_examples: how many examples to plot, 0 if all should be used
    :param show_diff: if difficult images should be plotted (True, False, 'both') -
                      used to quickly see if difficult flag is correct
    :param show_rect_or_cut:
    :param df: pandas DataFrame, in which column 'difficult' can be filled with True/False
    :return: pandas DataFrame (or None if none passed) and figure with resulting plots
    """
    """

    :param path: 
    :return: shows image

    """
    # center_point = (2525/2, 2525/2)
    # side_length = int(np.round(np.sqrt(diameter**2/2)))
    # start = int(np.round(img_size/2 - (side_length/2))) - shift_xy0
    n_examples = 99999 if n_examples == 0 else n_examples
    if n_examples >= 8:
        n_col = 8
        n_row = 10  # int(np.ceil(n_examples/n_col))
    else:
        n_col = n_examples
        n_row = 1
    idx = 1
    idx_e = 0
    f = plt.figure(figsize=(60, 60))

    for root, dirs, files in os.walk(path):
        for file in files:
            if df is not None:
                if len(df[[file in p for p in df.img_path]]) == 0:
                    continue
            if idx % (n_col+1) == 0:
                f = plt.figure(figsize=(60, 60))
                idx = 1
            if idx_e < n_examples:
                plt.subplot(n_col, n_row, idx)
                img = cv2.imread(os.path.join(root, file))
                img_circle, flag = fit_circle(img, show_rect_or_cut)
                if show_diff == 'both':
                    # plot all images
                    plt.imshow(img_circle)
                    if flag:
                        plt.imshow(np.ones_like(img_circle) * (255, 0, 0), alpha=0.1)
                        plt.title(file + ' - ' + flag, fontsize=24)
                    else:
                        plt.title(file, fontsize=24)
                    plt.axis('off')
                    idx += 1
                    idx_e += 1
                elif show_diff:
                    # plot only the difficult images
                    if flag:
                        plt.subplot(n_col, n_row, idx)
                        plt.imshow(img_circle)
                        plt.title(file)
                        plt.axis('off')
                        idx += 1
                        idx_e += 1
                else:
                    # plot only the working images
                    if not flag:
                        plt.subplot(n_col, n_row, idx)
                        plt.imshow(img_circle)
                        plt.title(file)
                        plt.axis('off')
                        idx += 1
                        idx_e += 1
                if df is not None:
                    if flag:
                        df['difficult'] = df.apply(lambda row: row['difficult'] | (file in row['img_path']), axis=1)
    return df, f


def save_cut_square(path='/data/old',
                    save_directory='/data/new'
                    ):
    """
    cut a square from all images by first fitting a circle/ellipse and then using the square around its center

    :param path: where the original images are stored
    :param save_directory: path to where to save the resulting images
    :return: shows image

    """

    for root, dirs, files in os.walk(path):
        print(path)
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            img_center, flag = fit_circle(img, 'cut')
            cv2.imwrite(save_directory + file, img_center)


if __name__ == '__main__':
    """ run this script to either cut center pieces or see what cut center pieces would look like """
    source_folder = '/data/original_images'
    dest_folder = '/data/center_piece_images'
    mode = 'train'

    df = pd.read_csv('/data/%s.csv' % mode)
    if not 'difficult' in df.columns:
        df['difficult'] = False

    df_a = show_cut_square(source_folder,
                           n_examples=0,
                           show_rect_or_cut='rect', show_diff='both', df=df)
