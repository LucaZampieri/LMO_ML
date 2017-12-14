# coding: utf-8
# Implementation and run with U-Net architecture

from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import unet
from util import rotate_img, flip_img, extract_data, extract_labels


# --- Define parameters and input/output files ---
# Input directories
data_dir = '../training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
test_data_dir = '../test_set_images/'

# Ouput directory
saving_path = 'trial1/'

# Training and testing parameters
optimizer = "adam"
dropout = 1.0
display_step = 5
nb_layers = 5
features_root = 4

augmentation = False
TRAINING_SIZE = 5
TESTING_SIZE = 5
batch_size = 2 #16
training_iters = 5
epochs = 5

# --- Data extraction ---
data = extract_data(train_data_filename, TRAINING_SIZE, augmentation=augmentation, train=True)
labels = extract_labels(train_labels_filename, TRAINING_SIZE, augmentation=augmentation)

if augmentation:
    initial_data = data[range(0,data.shape[0],5)]
    initial_labels = labels[range(0,data.shape[0],5)]
else:
    initial_data = data.copy()
    initial_labels = labels.copy()

print(' ')
print('Shape data:', data.shape)
print('Shape labels:', labels.shape)
print('Shape initial data:', initial_data.shape)
print('Shape initial labels:', initial_labels.shape)
print(' ')

# ----------------------------
# --------- TRAINING ---------
# ----------------------------
net = unet.Unet(channels=3, n_class=2, layers=nb_layers, features_root=features_root)
    #, cost_kwargs={'regularizer':1e-4}) # class_weights
# Optimizer = "momentum" or "adam"
trainer = unet.Trainer(net, batch_size=batch_size, optimizer=optimizer)
    #, opt_kwargs=dict(momentum=0.2)), learning_rate, decay_rate
path = trainer.train(data=data, labels=labels, output_path="./unet_trained/"+saving_path,
                     training_iters=training_iters, epochs=epochs, dropout=dropout,
                     display_step=display_step, prediction_path='./prediction/'+saving_path) # 20, 20
prediction = net.predict("./unet_trained/"+saving_path+"model.cpkt", initial_data)

# Plot results -------------------------
for num in range(0,TRAINING_SIZE):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
    ax[0].imshow(initial_data[num])
    ax[1].imshow(initial_labels[num,:,:,1], aspect="auto")
    #mask = prediction[num,:,:,1] > 0.3
    #ax[2].imshow(mask, aspect="auto")
    ax[2].imshow(prediction[num,:,:,1], aspect="auto")
    ax[0].set_title("Input")
    ax[1].set_title("Ground truth")
    ax[2].set_title("Prediction")
    fig.tight_layout()
    fig.savefig("output/"+saving_path+"roadSegmentationTrain"+str(num)+".png")

# ----------------------------
# --------- TESTING ---------
# ----------------------------
test_data = extract_data(test_data_dir, TESTING_SIZE, train=False)
test_prediction = net.predict("./unet_trained/"+saving_path+"model.cpkt", test_data)

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
    fig.savefig("output/"+saving_path+"roadSegmentationTest"+str(num)+".png")
