import numpy as np
import cv2

from util import *


def extract_data(filename, num_images, train=False, resize=False, \
                 angles=np.empty(0), flip=False, imgType='data', \
                 divide_test_in_4=False, train_pixel_nb=400, \
                 resize_pixel_nb=256, pref_rotation=False):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    original_pixel_nb = 0
    pref_angles = []
    for i in range(1, num_images+1):
        if i%10==0:
            if imgType=='data':
                print('Extract original images... i =',i)
            elif imgType=='label':
                print('Extract groundtruth images... i =',i)
            else:
                print('extract_data:Error! imgType should be either data or label.')

        if train:
            imageid = "satImage_%.3d" % i
        else:
            imageid = "test_%.1d" % i  + "/test_%.1d" % i
        image_filename = filename + imageid + ".png"

        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            original_pixel_nb = img.shape[0]

            if divide_test_in_4:
                croped_imgs = img_divide_in_4(img, train_pixel_nb) # divide in 4 400x400 img
            else:
                croped_imgs = [img]

            for x in croped_imgs:
                if resize == True:
                    x = resize_img(x, resize_pixel_nb)
                imgs.append(x)

                if angles.shape[0]!=0:
                    for angle in angles:
                        imgs.append(rotate_my_img(x, random=False, angle=angle))

                if pref_rotation:
                    pref_angle = find_angle(x)
                    if len(np.where(angles==pref_angle)[0]) == 0:
                        imgs.append(rotate_my_img(x, random=False, angle=pref_angle))
                        pref_angles.append(pref_angle)
                    else:
                        imgs.append(x) # it will not be used, but we need to fulfill this entry to keep track of the index
                        pref_angles.append(0)

                if flip:
                    imgs.append(x[:, ::-1])
        else:
            print ('extract_data:Warning! File ' + image_filename + ' does not exist')

    if imgType=='label':
        data = np.asarray(imgs)
        imgs = [[[value_to_class(data[i][j][k]) for k in range(data.shape[2])]
                                                   for j in range(data.shape[1])]
                                                   for i in range(data.shape[0])]

    return np.asarray(imgs).astype(np.float32), original_pixel_nb, \
           np.asarray(pref_angles).astype(np.int8)
