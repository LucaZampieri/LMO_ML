# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 10, 2016

author: jakeret

Modified in Dec 2017 by O.Chanon, M.Ciprian and L.Zampieri
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import cv2
import os
from skimage.transform import (rotate, hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.io import imshow, show
from skimage import feature
import pandas as pd


def plot_prediction(x_test, y_test, prediction, save=False):
    import matplotlib
    import matplotlib.pyplot as plt

    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12,12), sharey=True, sharex=True)

    x_test = crop_to_shape(x_test, prediction.shape)
    y_test = crop_to_shape(y_test, prediction.shape)

    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        plt.colorbar(cax, ax=ax[i,0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i,1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i,2])
        if i==0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()

    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()

def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    if offset0 == offset1 == 0:
        return data
    else:
        return data[:, offset0:(-offset0), offset1:(-offset1)]

def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor

    :returns img: the concatenated rgb image
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)),
                          to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)),
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    return img

def save_image(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)

def rotate_img(img, angle, rgb):
    rows, cols = img.shape[0:2]
    if rgb:
        id = 1
    else:
        id = 0
    rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, id)
    return cv2.warpAffine(img, rot_M, (cols, rows))

def flip_img(img, border_id):
    return cv2.flip(img, border_id)

# Luca added image functions ---------------------------------------------------
def resize_img(img, size=256):
    return cv2.resize(img, (size,size), interpolation = cv2.INTER_AREA)

def rotate_my_img(img, random=True, angle=45):
    if random == True:
        rotations = [30,60]
        angle = np.random.choice(rotations)
        return rotate(img, angle, resize=False, mode='reflect'), angle
    else:
        return rotate(img, angle, resize=False, mode='reflect')

def crop_my_img(img, random=False):
    return 0

def img_divide_in_4(img, size):
    """ Divide the image in 4 squares lrtb (left-right-top-bottom)
    im: input image
    size: size of wanted squares
    """
    if img.shape[0]<=size or img.shape[1]<=size:
        return [img,img,img,img]
    list_imgs= []
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    list_imgs.append(img[0:size, 0:size])
    list_imgs.append(img[0:size,  imgheight-size:imgheight])
    list_imgs.append(img[imgwidth-size:imgwidth, 0:size])
    list_imgs.append(img[imgwidth-size:imgwidth, imgheight-size:imgheight])
    return list_imgs

# end luca added image functions -----------------------------------------------

# Assign a label to a patch v
def value_to_class(v):
    # you can remark the hot encoding
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Functions from the old code ----------------------------------------
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8) # if pixel_depth = 255
    return rimg


def value_to_class_2(v,threshold):
    # you can remark the hot encoding
    foreground_threshold = threshold # percentage of pixels > 1 required to assign a foreground label to a patch TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

def img_crop(im, w, h):
    """ Extract a list of patches from a given image
    im: input image
    w: width of input image
    h: height of input image
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255 # if pixel depth is 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.3)
    return new_img

def RGB_to_grey(image):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    return 0.299* R + 0.587* G + 0.114* B

def find_angle(img):
    img=RGB_to_grey(img)
    edges = feature.canny(img,sigma=1.5)
    lines = probabilistic_hough_line(edges, threshold=5, line_length=25,
                                 line_gap=3)
    angles = []
    for x in lines:
        a= x[0]
        b= x[1]
        angle =  np.arccos( (b[0]-a[0])/np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2) )*180/3.1415
        if angle>90:
            angle = angle - 90
        angle = angle//1*1 # one can then be changed to other int numbers
        angles.append(angle)

    dominant_angle = pd.Series(angles).value_counts().index[0]
    return - dominant_angle
