import numpy as np

from util import resize_img, rotate_my_img

def postprocess_test(img, size, augmentation=False, angles=[], imgType='data'):
    img = np.array([resize_img(img[i], 400) for i in range(img.shape[0])])

    out_data_size = int(img.shape[0]/20.0) #28.0)
    mid = int(size/2.0)

    if augmentation:
        if imgType=='data':
            temp = img[0::5] # 8
        elif imgType=='label':
            temp = np.empty([4*out_data_size, img.shape[1], img.shape[2], img.shape[3]])
            for i in range(4*out_data_size):
                temp[i] = img[5*i] \
                          + rotate_my_img(img[5*i+1],random=False,angle=-45) \
                          + rotate_my_img(img[5*i+2],random=False,angle=-angles[2*i]) \
                          + rotate_my_img(img[5*i+3],random=False,angle=-angles[2*i+1]) \
                          + img[5*i+4,:,::-1] # 8*i+7
                          #+ rotate_my_img(img[8*i+3],random=False,angle=-90) \
                          #+ rotate_my_img(img[8*i+4],random=False,angle=-180) \
                          #+ rotate_my_img(img[8*i+5],random=False,angle=-270) \
                temp[i] /= 5.0 # 8.0
        else:
            print('ERROR! imgType should be either data or label to postprocess the test.')
    else:
        temp = img

    out = np.empty([out_data_size, size, size, img.shape[3]])
    for i in range(out_data_size):
        out[i, :mid, :mid, :] = temp[4*i, :mid, :mid, :]
        out[i, :mid, mid:, :] = temp[4*i+1, :mid, -mid:, :]
        out[i, mid:, :mid, :] = temp[4*i+2, -mid:, :mid, :]
        out[i, mid:, mid:, :] = temp[4*i+3, -mid:, -mid:, :]

    return out
