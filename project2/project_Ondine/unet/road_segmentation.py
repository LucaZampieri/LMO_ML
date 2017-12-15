# coding: utf-8
# Implementation and run with U-Net architecture

from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

import unet
from util import rotate_img, flip_img, extract_data, extract_labels, combine_img_prediction, save_image
from mask_to_submission import masks_to_submission

# --- Define parameters and input/output files ---
# Input directories
data_dir = '../training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
test_data_dir = '../test_set_images/'

# Ouput directory
saving_path = 'afterCorrections/'
submission_filename = 'output/'+saving_path+'submission.csv'

# Training and testing parameters
optimizer = "adam"
dropout = 1.0
display_step = 5
nb_layers = 5
features_root = 4

augmentation = False
TRAINING_SIZE = 10 #0
TESTING_SIZE = 5 #0
batch_size = 16
training_iters = 20
epochs = 2 #13

foreground_threshold = 0.25

# --- Data extraction ---
data = extract_data(train_data_filename, TRAINING_SIZE, augmentation=augmentation, train=True)
labels = extract_labels(train_labels_filename, TRAINING_SIZE, augmentation=augmentation)

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
print('Beginning of the training...')
trained_model_path = trainer.train(data=data, labels=labels, output_path="./unet_trained/"+saving_path,
                     training_iters=training_iters, epochs=epochs, dropout=dropout,
                     display_step=display_step, prediction_path='./prediction/'+saving_path) # 20, 20

if augmentation:
    data = data[range(0,data.shape[0],5)]
    labels = labels[range(0,labels.shape[0],5)]

print('Making predictions...')
#prediction = net.predict("./unet_trained/"+saving_path+"model.cpkt", data) # HERE trained_model_path
prediction = net.predict(trained_model_path, data) # HERE trained_model_path

# Plot results -------------------------
path_saved_pred = "output/"+saving_path
for num in range(0,TRAINING_SIZE):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
    ax[0].imshow(data[num])
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
test_data = extract_data(test_data_dir, TESTING_SIZE, train=False)
#test_prediction = net.predict("./unet_trained/"+saving_path+"model.cpkt", test_data) # HERE trained_model_path
test_prediction = net.predict(trained_model_path, test_data) # HERE trained_model_path

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

# Make submission -------------------------
image_filenames = []
for i in range(1, 51):
    image_filename = test_data_dir+"test_%.1d" % i  + "/test_%.1d" % i + ".png"
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames, foreground_threshold=foreground_threshold)
