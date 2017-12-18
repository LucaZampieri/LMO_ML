import numpy as np

from util import resize_img, rotate_my_img

def postprocess_test(img, resize=False, train_pixel_nb=400, divide_test_in_4=False, \
                     nb_imgs_per_img_test=1, imgType='data', angles=np.empty(0), \
                     flip=False, test_pixel_nb=608):
    if resize:
        img = np.array([resize_img(img[i], train_pixel_nb) for i in range(img.shape[0])])

    if divide_test_in_4:
        nb_subimg = 4
    else:
        nb_subimg = 1
    out_data_size = int(img.shape[0]/(nb_imgs_per_img_test*nb_subimg))

    if imgType=='data':
        temp = img[0::nb_imgs_per_img_test]
    elif imgType=='label':
        temp = np.empty([nb_subimg*out_data_size, img.shape[1], img.shape[2], img.shape[3]])
        for i in range(nb_subimg*out_data_size):
            temp[i] = img[nb_imgs_per_img_test*i]
            for j in range(angles.shape[0]):
                temp[i] = temp[i] \
                        + rotate_my_img(img[nb_imgs_per_img_test*i+j+1], \
                                        random=False, angle=-angles[j])
            if flip:
                temp[i] = temp[i] + img[nb_augmented*(i+1)-1,:,::-1]
            temp[i] /= nb_augmented
    else:
        print('postprocess_test:ERROR! imgType should be either data or label.')

    if divide_test_in_4:
        mid = int(test_pixel_nb/2.0)
        out = np.empty([out_data_size, test_pixel_nb, test_pixel_nb, img.shape[3]])
        for i in range(out_data_size):
            out[i, :mid, :mid, :] = temp[nb_subimg*i, :mid, :mid, :]
            out[i, :mid, mid:, :] = temp[nb_subimg*i+1, :mid, -mid:, :]
            out[i, mid:, :mid, :] = temp[nb_subimg*i+2, -mid:, :mid, :]
            out[i, mid:, mid:, :] = temp[nb_subimg*i+3, -mid:, -mid:, :]
        return out
    else:
        return temp
