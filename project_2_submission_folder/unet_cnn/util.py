""" File util.py containing all helper functions to perform the road
segmentation algorithm with a U-net.
# Authors: Ondine Chanon, Matteo Ciprian, Luca Zampieri """

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


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    Input:
        img: the array to convert [nx, ny, channels]
    Ouput:
        img: the rgb image [nx, ny, 3]
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

    Input:
        data: the array to crop
        shape: the target shape (tuple)
    Output: cropped image
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

    Input:
        data: the data tensor
        gt: the ground thruth tensor
        pred: the prediction tensor
    Output:
        img: the concatenated rgb image
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

    Input:
        img: the rgb image to save
        path: the target path
    """
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)

def rotate_img(img, angle, rgb):
    """
        Rotates an image with a certain given angle.
        Input:
            img (array): original image
            angle (int): rotation angle in degree
            rgb (bool): True if the image is given in RGB
        Output:
            rotated image (no boundary conditions)
    """
    rows, cols = img.shape[0:2]
    if rgb:
        id = 1
    else:
        id = 0
    rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, id)
    return cv2.warpAffine(img, rot_M, (cols, rows))

def flip_img(img, border_id):
    """ Flips (symmetry) the image img with respect to the border border_id """
    return cv2.flip(img, border_id)

def resize_img(img, size=256):
    """ Resizes the image img to the dimension size x size """
    return cv2.resize(img, (size,size), interpolation = cv2.INTER_AREA)

def rotate_my_img(img, random=True, angle=45):
    """
        Rotates an image with a certain given angle, possibly chosen randomly,
        applying mirror boundary conditions ('reflect').
        Input:
            img (array): original image
            random (bool): True if the choice of the angle is randomly chosen
            angle (int): rotation angle in degree
        Output:
            rotated image (no boundary conditions)
    """
    if random == True:
        rotations = [30,60]
        angle = np.random.choice(rotations)
        return rotate(img, angle, resize=False, mode='reflect'), angle
    else:
        return rotate(img, angle, resize=False, mode='reflect')

def img_divide_in_4(img, size):
    """
        Divide the image in 4 squares lrtb (left-right-top-bottom)
        Input:
            img: input image
            size: size of wanted squares
        Ouput:
            list_imgs: List of arrays representing the four squares
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

def binary_to_uint8(img):
    """ Converts the image img with pixels described between 0 and 1 into pixels
        described between 0 and 255 (standard definition for colors RGB) """
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def img_float_to_uint8(img):
    """ Converts the image img with pixels described between two float values into
        integer values of pixels described between 0 and 255 (standard definition
        for colors RGB) """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8) # if pixel_depth = 255
    return rimg

def value_to_class(v, threshold=0.25):
    """
        Computes the label (hot encoding) to assign to each set of pixels v by
        computing the mean of the values on the set and comparing it to a given threshold.
        Input:
            v (array): set of pixel values
            threshold (float): threshold value over which
                the set of pixel is considered as foreground (ie street).
        Ouput:
            [0, 1] if the set of pixels is considered as foreground, [1, 0] else.
    """
    df = np.mean(v)
    if df > threshold:
        return [0, 1]
    else:
        return [1, 0]

def img_crop(im, w, h):
    """ Extract a list of patches from a given image
        Input:
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
    """
        Creates an array of labels of dimension imgwidth x imgheight such that
        each label corresponds to a patch of dimension w x h.
        Input:
            imgwidth (int): number of horizontal pixels of the full image
            imgheight (int): number of vertical pixels of the full image
            w (int): width of each patch
            h (int): height of each patch
            labels (array): array of labels (one for each patch, from left to
                            right, from top to bottom)
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    """
        Creates an image that superposes the original image img and the prediction
        called predicted_img to visually asset the quality of the prediction.
    """
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
    """
        Transforms an image array defined in RGB to an image array defined in
        shades of grey.
    """
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    return 0.299* R + 0.587* G + 0.114* B

def find_angle(img):
    """
        For a given image img, finds the main orientation of the image and
        returns the angle such that the image rotated with this angle has main
        direction horizontal.
    """
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
