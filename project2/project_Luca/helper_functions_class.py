""" Helper functions for the TF"""


import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import numpy as np

import cv2


# ################################ Variables ##################
NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2  # 0 or 1
CONSIDER_PATCHES = False # True if we split the images patch by path, False if we consider the whole images and pixel by pixel

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

########################## functions ###########################################
# balance the data
def balance_classes_in_data(train_data,train_labels):
    # Count the number of data points on each class
    c0 = np.sum((train_labels[:,0]==1)*1)
    c1 = train_labels.shape[0]-c0
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # Balance to take the same number of patches with c0 and c1 classes
    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c] # Concatenate lists
    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]
    train_size = train_labels.shape[0]
    print ('train_data.shape after balancing: ',train_data.shape)

    c0 = np.sum((train_labels[:,0]==1)*1)
    c1 = train_labels.shape[0]-c0
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    return train_data, train_labels, train_size


# Extract patches from a given image
def img_crop(im, w, h):
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



def extract_data(filename, num_images, patches=True, mytype='train'):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting data...')
    imgs = []
    for i in range(1, num_images+1):
        if mytype == 'train':
            imageid = "satImage_%.3d" % i
        else:
            imageid = "test_%.1d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            # uncomment next line if needed
            #img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    img_size = imgs[0].shape[0]
    img_height = imgs[0].shape[1]
    if img_size != img_height:
        print('Error!! The images should have their height equal to their width.')
    elif patches:
        N_PATCHES_PER_IMAGE = (img_size/IMG_PATCH_SIZE)**2

        num_images = len(imgs) # necessary if an image (or more) has not been found
        img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
        imgs = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
        return np.asarray(imgs)

    return np.asarray(imgs), img_size


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
def extract_labels(filename, num_images, patches=True):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting labels...')
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    if patches:
        num_images = len(gt_imgs) # necessary if an image (or more) has not been found
        gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
        data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
        out_lab = [value_to_class(np.mean(data[i])) for i in range(len(data))]
    else:
        data = np.asarray(gt_imgs)
        out_lab = [[[value_to_class(data[i][j][k]) for k in range(data.shape[2])] for j in range(data.shape[1])] for i in range(data.shape[0])]

    # Convert to dense 1-hot representation.
    return np.asarray(out_lab).astype(np.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    if CONSIDER_PATCHES:
        return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) ==  np.argmax(labels, 1)) / predictions.shape[0])
    else:
        return 100.0 - (100.0 * np.sum(np.argmax(predictions, 3) ==  np.argmax(labels, 3)) / np.prod(predictions.shape[0:3]))

# Write predictions from neural network to a file TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO FOR NON PATCHES!!!!!!! NEVER CALLED
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO NEVER CALLED!
def print_predictions(predictions, labels):
    if CONSIDER_PATCHES:
        indx = 1
    else:
        indx = 3
    max_labels = np.argmax(labels, indx)
    max_predictions = np.argmax(predictions, indx)
    print (str(max_labels) + ' ' + str(max_predictions))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels, patches=True):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if patches:
                lab = labels[idx][0]
            else:
                lab = labels[0,j,i,0]

            if lab > 0.5: # TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img): # TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO NEVER CALLED!!
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def False_concatenate_images(gt_img): # just to make predictions
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]

    gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
    gt_img8 = img_float_to_uint8(gt_img)
    gt_img_3c[:,:,0] = gt_img8
    gt_img_3c[:,:,1] = gt_img8
    gt_img_3c[:,:,2] = gt_img8
    return gt_img_3c

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

# Make an image summary for 4d tensor image with index idx
def get_image_summary(img, idx = 0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    min_value = tf.reduce_min(V)
    V = V - min_value
    max_value = tf.reduce_max(V)
    V = V / (max_value*PIXEL_DEPTH)
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V

# Make an image summary for 3d tensor image with index idx
def get_image_summary_3d(img):
    V = tf.slice(img, (0, 0, 0), (1, -1, -1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V
