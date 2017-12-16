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

from skimage.transform import rotate
from skimage.io import imshow, show
# from padding import mirror_padding

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
def resize_img(img, size):
    return cv2.resize(img, (size,size), interpolation = cv2.INTER_AREA)

def rotate_my_img(img,random=True):
    if random == True:
        rotations = [30,60]
        angle = np.random.choice(rotations)
    else:
        angle = 45
    return rotate(img,angle,resize=False,mode='reflect')

def img_divide_in_4(img, size):
    """ Divide the image in 4 squares lrtb (left-right-top-bottom)
    im: input image
    size: size of wanted squares
    """
    if img.shape[0]<=size or img.shape[1]<=size:
        return [img,img,img,img]
    list_imgs= []
    list_imgs.append(img[:size, :size])
    list_imgs.append(img[:size,  -size:])
    list_imgs.append(img[-size:, :size])
    list_imgs.append(img[-size:, -size:])
    return list_imgs

def postprocess_labels_test(labels, size): #, pixels_bc=0):
    labels = np.array([resize_img(labels[i], 400) for i in range(labels.shape[0])])
    print('heeeeee')
    print(labels.shape)

    out_shape = int(labels.shape[0]/4.0)
    init_size = labels.shape[1]
    offset = size-init_size
    offset_artifact = 10

    if offset <= 0:
        return labels

    out = np.empty([out_shape, size, size, labels.shape[3]])
    temp = np.zeros([4, size, size, labels.shape[3]])
    for i in range(out_shape):
        temp[0, :init_size-offset_artifact, :init_size-offset_artifact, :] = labels[4*i, :-offset_artifact, :-offset_artifact]
        temp[1, :init_size-offset_artifact, -init_size+offset_artifact:, :] = labels[4*i+1, :-offset_artifact, offset_artifact:]
        temp[2, -init_size+offset_artifact:, :init_size-offset_artifact, :] = labels[4*i+2, offset_artifact:, :-offset_artifact]
        temp[3, -init_size+offset_artifact:, -init_size+offset_artifact:, :] = labels[4*i+3, offset_artifact:, offset_artifact:]

        out[i] = np.sum(temp, axis=0)
        out[i, :offset+offset_artifact, offset+offset_artifact:-offset-offset_artifact, :] /= 2.0
        out[i, -offset-offset_artifact:, offset+offset_artifact:-offset-offset_artifact, :] /= 2.0
        out[i, offset+offset_artifact:-offset-offset_artifact, :offset+offset_artifact, :] /= 2.0
        out[i, offset+offset_artifact:-offset-offset_artifact, -offset-offset_artifact:, :] /= 2.0
        out[i, offset+offset_artifact:-offset-offset_artifact, offset+offset_artifact:-offset-offset_artifact, :] /= 4.0
    return out

def postprocess_imgs_test(data, size): #, pixels_bc):
    data = np.array([resize_img(data[i], 400) for i in range(data.shape[0])])

    out_shape = int(data.shape[0]/4.0)
    init_size = data.shape[1]
    offset = size-init_size

    if offset <= 0:
        return data

    out = np.empty([out_shape, size, size, data.shape[3]])
    for i in range(out_shape):
        out[i, :init_size, :init_size, :] = data[4*i]
        out[i, :init_size, offset:, :] = data[4*i+1]
        out[i, offset:, :init_size, :] = data[4*i+2]
        out[i, offset:, offset:, :] = data[4*i+3]
    return out

# end luca added image functions -----------------------------------------------

# added resize flag
def extract_data(filename, num_images, augmentation=False, train=False, resize=False):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting data...')
    imgs = []
    for i in range(1, num_images+1):
        if i%10==0:
            print('Extract original images... i=',i)
        if train:
            imageid = "satImage_%.3d" % i
        else:
            imageid = "test_%.1d" % i  + "/test_%.1d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            if resize == True:
                img = resize_img(img, 256)
            imgs.append(img)

            if augmentation:
                if resize == True:
                    imgs.append(rotate_my_img(img,random=False))
                    imgs.append(rotate_my_img(img,random=True))
                else :
                    img_cv2 = cv2.imread(image_filename)
                    img_flip = np.flip(flip_img(img_cv2, 1),2)/255
                    imgs.append(img_flip)

                    imgs.append(np.flip(rotate_img(img_cv2, 90, True),2)/255)
                    imgs.append(np.flip(rotate_img(img_cv2, 180, True),2)/255)
                    imgs.append(np.flip(rotate_img(img_cv2, 270, True),2)/255)
        else:
            print ('File ' + image_filename + ' does not exist')

    img_size = imgs[0].shape[0]
    img_height = imgs[0].shape[1]
    if img_size != img_height:
        print('Error!! The images should have their height equal to their width.')

    return np.asarray(imgs).astype(np.float32)

# Assign a label to a patch v
def value_to_class(v):
    # you can remark the hot encoding
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images, augmentation=False, resize=False):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting labels...')
    gt_imgs = []
    for i in range(1, num_images+1):
        if i%10==0:
            print('Extract groundtruth images... i=',i)
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            if resize == True:
                img = resize_img(img, 256)
            gt_imgs.append(img)

            if augmentation:
                if resize == True:
                    gt_imgs.append(rotate_my_img(img,random=False))
                    gt_imgs.append(rotate_my_img(img,random=True))
                else :
                    img_cv2 = cv2.imread(image_filename,0)
                    gt_img_flip = flip_img(img_cv2, 1)/255
                    gt_imgs.append(gt_img_flip)

                    gt_imgs.append(rotate_img(img_cv2, 90, True)/255)
                    gt_imgs.append(rotate_img(img_cv2, 180, True)/255)
                    gt_imgs.append(rotate_img(img_cv2, 270, True)/255)

        else:
            print ('File ' + image_filename + ' does not exist')

    data = np.asarray(gt_imgs)
    out_lab = [[[value_to_class(data[i][j][k]) for k in range(data.shape[2])] for j in range(data.shape[1])] for i in range(data.shape[0])]

    # Convert to dense 1-hot representation.
    return np.asarray(out_lab).astype(np.float32)

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


# new function: Extract test for when we need to cut the test images:
def extract_test(filename, num_images, augmentation=False, train=False, resize=True):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting data...')
    imgs = []
    for i in range(1, num_images+1):
        if i%10==0:
            print('Extract original images... i=',i)
        if train:
            imageid = "satImage_%.3d" % i
        else:
            imageid = "test_%.1d" % i  + "/test_%.1d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            if resize == True:
                original_size = img.shape[0]
                croped_imgs = img_divide_in_4(img, 400) # divide in 4 400x400 img
                for x in croped_imgs:
                    x = resize_img(x, 256)
                    imgs.append(x)
            else:
                imgs.append(img)

            if augmentation:
                imgs.append(rotate_my_img(img,random=False))
                imgs.append(rotate_my_img(img,random=True))

        else:
            print ('File ' + image_filename + ' does not exist')

    img_size = imgs[0].shape[0]
    img_height = imgs[0].shape[1]
    if img_size != img_height:
        print('Error!! The images should have their height equal to their width.')

    return np.asarray(imgs).astype(np.float32), original_size
