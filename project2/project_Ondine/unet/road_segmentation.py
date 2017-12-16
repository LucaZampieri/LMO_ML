# coding: utf-8
# Implementation and run with U-Net architecture

from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from PIL import Image

import unet
from util import rotate_img, flip_img, extract_data, extract_labels, combine_img_prediction, save_image, binary_to_uint8, img_float_to_uint8
from util import make_img_overlay, label_to_img, img_crop, value_to_class_2, extract_test, postprocess_labels_test, postprocess_imgs_test, resize_img
import matplotlib.image as mpimg
import image_util
from mask_to_submission import masks_to_submission

# --- Define parameters and input/output files ---
# Input directories
data_dir = '../training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
test_data_dir = '../test_set_images/'

# Ouput directory
saving_path = 'trialOndine/'
model_path = 'afterLargeCorrections/'
Re_run = False # True if we want to do the training; False if the model already exists
submission_filename = 'output/'+saving_path+'submission.csv'

# Training and testing parameters
optimizer = "adam"
dropout = 1.0
display_step = 5
nb_layers = 5
features_root = 4

augmentation = True
resize = True
TRAINING_SIZE = 10
TESTING_SIZE = 5
batch_size = 8
training_iters = 2
epochs = 3

foreground_threshold = 0.25


# --- Data extraction ---
data = extract_data(train_data_filename, TRAINING_SIZE, augmentation=augmentation, train=True, resize=resize)
labels = extract_labels(train_labels_filename, TRAINING_SIZE, augmentation=augmentation, resize=resize)
img_provider = image_util.SimpleDataProvider(data=data, label=labels, channels=3, n_class=2)

print(' ')
print('Shape data:', data.shape)
print('Shape labels:', labels.shape)
print(' ')

# ----------------------------
# --------- TRAINING ---------
# ----------------------------

print('Creating the net...')
net = unet.Unet(channels=3, n_class=2, layers=nb_layers, features_root=features_root)
    #, cost_kwargs={'regularizer':1e-4}) # class_weights
# Optimizer = "momentum" or "adam"
print('Creating the trainer...')
trainer = unet.Trainer(net, batch_size=batch_size, optimizer=optimizer)
    #, opt_kwargs=dict(momentum=0.2)), learning_rate, decay_rate
if Re_run == True:
    print('Beginning of the training...')
    trained_model_path = trainer.train(data_provider=img_provider, output_path="./unet_trained/"+model_path,
                         training_iters=training_iters, epochs=epochs, dropout=dropout,
                         display_step=display_step, prediction_path='./prediction/'+saving_path) # 20, 20

if augmentation:
    if resize == True:
        data = data[range(0,data.shape[0],3)]
        labels = labels[range(0,labels.shape[0],3)]
    else:
        data = data[range(0,data.shape[0],5)]
        labels = labels[range(0,labels.shape[0],5)]

print('Making predictions...')

if Re_run == True:
    prediction = net.predict(trained_model_path, data)
else:
    prediction = net.predict("./unet_trained/"+model_path+"model.cpkt", data)

if resize:
    data = np.array([resize_img(data[i], 400) for i in range(data.shape[0])])
    labels = np.array([resize_img(labels[i], 400) for i in range(labels.shape[0])])
    prediction = np.array([resize_img(prediction[i], 400) for i in range(prediction.shape[0])])


# Plot results -------------------------
path_saved_pred = "output/"+saving_path
for num in range(0,TRAINING_SIZE):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
    ax[0].imshow(data[num], aspect="auto")
    ax[1].imshow(labels[num,:,:,1], aspect="auto")
    #mask = prediction[num,:,:,1] > 0.3
    #ax[2].imshow(mask, aspect="auto")
    ax[2].imshow(prediction[num,:,:,1], aspect="auto")
    ax[0].set_title("Input")
    ax[1].set_title("Ground truth")
    ax[2].set_title("Prediction")
    fig.tight_layout()
    if not os.path.exists(path_saved_pred):
        os.makedirs(path_saved_pred)
    fig.savefig(path_saved_pred+"roadSegmentationTrain"+str(num)+".png")
    plt.close(fig)

imggg = combine_img_prediction(data, labels, prediction)
save_image(imggg, "%s.jpg"%(path_saved_pred+"allPred"))


# ----------------------------
# --------- TESTING ---------
# ----------------------------
if resize == True:
    test_data, original_size = extract_test(test_data_dir, TESTING_SIZE,augmentation=False, train=False, resize=resize)
else:
    test_data = extract_data(test_data_dir, TESTING_SIZE, train=False, resize=resize)

# choose whether rerunning the whole model or loading the model
if Re_run == True:
    test_prediction = net.predict(trained_model_path, test_data)
else :
    test_prediction = net.predict("./unet_trained/"+model_path+"model.cpkt", test_data)

if resize:
    test_data = postprocess_imgs_test(test_data, original_size)
    test_prediction = postprocess_labels_test(test_prediction, original_size)

# Plot results -------------------------
for num in range(0,TESTING_SIZE):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,6))
    ax[0].imshow(test_data[num], aspect="auto")
    #mask = test_prediction[num,:,:,1] > 0.2
    #ax[1].imshow(mask, aspect="auto")
    ax[1].imshow(test_prediction[num,:,:,1], aspect="auto")
    ax[0].set_title("Input")
    ax[1].set_title("Prediction")
    fig.tight_layout()
    fig.savefig(path_saved_pred+"roadSegmentationTest"+str(num)+".png")
    plt.close(fig)

# Save results in apropriate folder -------------------------------------------
for num in range(0,TESTING_SIZE):
    mask_ori = test_prediction[num,:,:,1]
    #mask = test_prediction[num,:,:,1] > 0.2
    mask = img_float_to_uint8(mask_ori) # raw predictions
    Image.fromarray(mask).save(path_saved_pred+"raw_pred"+str(num+1)+".png")

    pred_patches = img_crop(mask_ori, 16, 16)
    data = np.asarray([pred_patches[i] for i in range(len(pred_patches))])
    out_lab = [value_to_class_2(np.mean(data[i]), foreground_threshold) for i in range(len(data))]
    out_lab = np.asarray(out_lab).astype(np.float32)
    pred = out_lab[:,1]

    pred = label_to_img(mask.shape[0], mask.shape[1], 16, 16, pred)
    pred_to_show = binary_to_uint8(pred)
    Image.fromarray(pred_to_show).save(path_saved_pred+"patches_pred"+str(num+1)+".png")

    if resize == True:
        img = test_data[num,:,:,:]
        oimg = make_img_overlay(img, mask_ori)
        #oimg.save(path_saved_pred+"overlay_pred"+str(num)+".png")
        oimg2 = make_img_overlay(img, pred)
        #oimg2.save(path_saved_pred+"overlay_patches_pred"+str(num)+".png")
    else:
        # get original test image
        image_filename = test_data_dir+"test_%.1d" % (num+1)  + "/test_%.1d" % (num+1) + ".png"
        img = mpimg.imread(image_filename)#.astype(np.float32)
        oimg = make_img_overlay(img, mask_ori)
        #oimg.save(path_saved_pred+"overlay_pred"+str(num)+".png")
        oimg2 = make_img_overlay(img, pred)
        #oimg2.save(path_saved_pred+"overlay_patches_pred"+str(num)+".png")

    widths, heights = zip(*(i.size for i in [oimg, oimg2]))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = -1
    for im in [oimg, oimg2]:
        new_im.paste(im, (x_offset+1,0))
        x_offset += im.size[0]
    new_im.save(path_saved_pred+"overlay_pred"+str(num)+".png")

# Make submission -------------------------
image_filenames = []
for i in range(1, 51):
    #image_filename = test_data_dir+"test_%.1d" % (num+1)  + "/test_%.1d" % (num+1) + ".png"
    image_filename = path_saved_pred+"patches_pred"+str(i)+".png"
    image_filenames.append(image_filename)
print(image_filenames)
masks_to_submission(submission_filename, *image_filenames, foreground_threshold=foreground_threshold)
