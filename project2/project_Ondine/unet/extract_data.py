import numpy as np
import cv2

from util import *


def extract_data(filename, num_images, augmentation=False, train=False, resize=False):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting data...')
    imgs = []
    angles = []
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
                    imgs.append(rotate_my_img(img, random=False, angle=45))
                    imgs.append(rotate_my_img(img, random=False, angle=30))
                    imgs.append(rotate_my_img(img, random=False, angle=60))
                    """img_rand_rot, angle = rotate_my_img(img, random=True)
                    imgs.append(img_rand_rot)
                    angles.append(angle)"""
                    imgs.append(rotate_my_img(img, random=False, angle=90))
                    imgs.append(rotate_my_img(img, random=False, angle=180))
                    # imgs.append(rotate_my_img(img, random=False, angle=270))
                    # imgs.append(img[:, ::-1])
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

    return np.asarray(imgs).astype(np.float32), np.asarray(angles).astype(np.int8)



# Extract label images
def extract_labels(filename, num_images, angles, augmentation=False, resize=False):
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
                    gt_imgs.append(rotate_my_img(img,random=False, angle=45))
                    gt_imgs.append(rotate_my_img(img, random=False, angle=30))
                    gt_imgs.append(rotate_my_img(img, random=False, angle=60))
                    # gt_imgs.append(rotate_my_img(img,random=False,angle=angles[i-1]))
                    gt_imgs.append(rotate_my_img(img,random=False,angle=90))
                    gt_imgs.append(rotate_my_img(img,random=False,angle=180))
#                     gt_imgs.append(rotate_my_img(img,random=False,angle=270))
                    # gt_imgs.append(img[:, ::-1])
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



# Extract test for when we need to cut the test images:
def extract_test(filename, num_images, augmentation=False, train=False, resize=True):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting test data...')
    imgs = []
    angles = []
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
            original_test_pixels = img.shape[0]
            croped_imgs = img_divide_in_4(img, 400) # divide in 4 400x400 img
            if resize == True:
                for x in croped_imgs:
                    x = resize_img(x, 256)
                    imgs.append(x)
                    if augmentation:
                        imgs.append(rotate_my_img(x,random=False,angle=45))
                        img_rand_rot, angle = rotate_my_img(x, random=True)
                        imgs.append(img_rand_rot)
                        angles.append(angle)
                        prefered_angle = find_angle(x)
                        imgs.append(rotate_my_img(x,random=False,angle=prefered_angle))
                        angles.append(prefered_angle)
                        #imgs.append(rotate_my_img(x,random=False,angle=90))
                        #imgs.append(rotate_my_img(x,random=False,angle=180))
                        #imgs.append(rotate_my_img(x,random=False,angle=270))
                        imgs.append(x[:, ::-1])
            else:
                imgs.append(img)
                if augmentation:
                    imgs.append(rotate_my_img(img,random=False,angle=45))
                    img_rand_rot, angle = rotate_my_img(img, random=True)
                    imgs.append(img_rand_rot)
                    angles.append(angle)
                    prefered_angle = find_angle(img)
                    imgs.append(rotate_my_img(img,random=False,angle=prefered_angle))
                    angles.append(prefered_angle)
                    #imgs.append(rotate_my_img(img,random=False,angle=90))
                    #imgs.append(rotate_my_img(img,random=False,angle=180))
                    #imgs.append(rotate_my_img(img,random=False,angle=270))
                    imgs.append(img[:, ::-1])

        else:
            print ('File ' + image_filename + ' does not exist')

    img_size = imgs[0].shape[0]
    img_height = imgs[0].shape[1]
    if img_size != img_height:
        print('Error!! The images should have their height equal to their width.')

    return np.asarray(imgs).astype(np.float32), original_test_pixels, angles
