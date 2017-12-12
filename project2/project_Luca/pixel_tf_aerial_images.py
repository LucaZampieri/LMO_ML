"""
In this file I try to define classes to use the same session for multiple purposes
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import tensorflow.python.platform

import numpy
import tensorflow as tf

## functions for image processing
from helper_functions_pixels import *


########################## Saving directory  ###################################
tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist/test5',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

########################### Parameters #########################################
TRAINING_SIZE = 10
# VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 50  # Set to None for random seed.
BATCH_SIZE = 10 # 16 # 64 (< train_size, and train_size = TRAINING_SIZE if CONSIDER_PATCHES = False)
NUM_EPOCHS = 3 # how many as you like
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 10
TEST = False  # if we want to predict test image as well
TESTING_SIZE = 50 # number of test images i.e. 50

# parameters:
print ('TRAINING_SIZE: ', TRAINING_SIZE )
print ('SEED: ', SEED)
print ('BATCH_SIZE: ', BATCH_SIZE )
print ('NUM_EPOCHS: ', NUM_EPOCHS)

print ('RESTORE_MODEL: ', RESTORE_MODEL )
print ('RECORDING_STEP: ', RECORDING_STEP)
print ('TEST: ', TEST )
print ('TESTING_SIZE: ', TESTING_SIZE)

print ('NUM_CHANNELS: ', NUM_CHANNELS )    # 3?
print ('PIXEL_DEPTH: ', PIXEL_DEPTH)       # 255?
print ('NUM_LABELS: ', NUM_LABELS )        # 2?
print ('IMG_PATCH_SIZE: ', IMG_PATCH_SIZE) # 16?

print('CONSIDER PATCHES?  ', CONSIDER_PATCHES)



# further functions ############################################################
def weight_variable(shape, stddev_ = 0.05):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=stddev_, seed=SEED)
    return tf.Variable(initial, name="W")

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="B")

########################## internal functions ##############################

def flatten_layer(layer):
    """Function to flatten the layers"""
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def fc_layer(input,channels_in,channels_out,relu=True, name="fc"):
    """Function to define a fully connected layer.
     Relu is applied by default"""
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([channels_in,channels_out]),name="W")
        b = tf.Variable(tf.zeros([channels_out]),name="B")
        layer = tf.matmul(input,w)+b
        if relu == True:
            layer = tf.nn.relu(layer)
        return layer,w,b

def conv_layer(input,w,b, name='conv'): # channels_in,channels_out
	with tf.name_scope(name):
		#w = weight_variable([5,5,channels_in,channels_out])
		#b = tf.Variable(tf.zeros([channels_out]),name="B")
		conv = tf.nn.conv2d(input,w,strides=[1,1,1,1],padding="SAME")
		layer = tf.nn.relu(conv + b)
		#####tf.nn.relu(tf.nn.bias_add(conv, b))
		tf.summary.histogram("weights", w)
		tf.summary.histogram("biases", b)
		#tf.summary.histogram("activations", act)
		return layer

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def upconv2d(x, W, shape):
    """upconv2d returns a 2d transpose convolution layer with full stride."""
    return tf.nn.conv2d_transpose(x, W, output_shape=shape, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

########### define directory of the training images ############################
data_dir = 'training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'

# Extract it into numpy arrays.
if CONSIDER_PATCHES:
    train_data = extract_data(train_data_filename, TRAINING_SIZE, CONSIDER_PATCHES)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE, CONSIDER_PATCHES)
    train_data, train_labels, train_size = balance_classes_in_data(train_data,train_labels)
else:
    train_data, train_img_size = extract_data(train_data_filename, TRAINING_SIZE, patches = CONSIDER_PATCHES)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE, CONSIDER_PATCHES)

train_size = train_data.shape[0]
print('Size of the training dataset =', train_data.shape)



with tf.name_scope('input'):
    if CONSIDER_PATCHES:
        train_data_node = tf.placeholder(
			tf.float32,
			shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
        train_labels_node = tf.placeholder(tf.float32,
										   shape=(BATCH_SIZE, NUM_LABELS),name = 'train_labels_nodes')
    else:
        train_data_node = tf.placeholder(tf.float32, shape=[BATCH_SIZE, train_img_size, train_img_size, NUM_CHANNELS])
        train_labels_node = tf.placeholder(tf.float32, shape=[BATCH_SIZE, train_img_size, train_img_size, NUM_LABELS])

    train_all_data_node = tf.constant(train_data)


class NN(object):
    """Our Neural network"""
    def __init__(self, options, session):
        #self._options = options
        self._session = session

        # Here one can add weights:
        if CONSIDER_PATCHES:
            with tf.name_scope("conv1"):
                self.conv1_weights = weight_variable([5, 5, NUM_CHANNELS, 32]) # 5x5 filter, depth 32.
                self.conv1_biases = tf.Variable(tf.zeros([32]), name = "B")

            with tf.name_scope("conv2"):
                self.conv2_weights = weight_variable([5, 5, 32, 64])
                self.conv2_biases = bias_variable(shape=[64])

            with tf.name_scope("fc1"):
                self.fc1_weights = weight_variable(shape= [int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                                            stddev_ = 0.1) # fully connected, depth 512.
                self.fc1_biases = bias_variable(shape=[512])

            with tf.name_scope("fc2"):
                self.fc2_weights = weight_variable(shape= [512, NUM_LABELS],stddev_ = 0.1)
                self.fc2_biases = bias_variable(shape=[NUM_LABELS])

        else:
            self.w1 = 16
            self.w2 = self.w1*2
            self.w3 = self.w2*2
            self.w4 = self.w3*2
            self.w5 = self.w4*2
            self.filter_size1 = 3
            self.upfilter_size = 3

            # First convolutional layers
            with tf.name_scope('conv1a'):
                self.W_conv1 = weight_variable([self.filter_size1, self.filter_size1, NUM_CHANNELS, self.w1])
                self.b_conv1 = bias_variable([self.w1])
            with tf.name_scope('conv1b'):
                self.W_conv1b = weight_variable([self.filter_size1, self.filter_size1, self.w1, self.w1])
                self.b_conv1b = bias_variable([self.w1])

            # Second convolutional layers
            with tf.name_scope('conv2a'):
                self.W_conv2 = weight_variable([self.filter_size1, self.filter_size1, self.w1, self.w2])
                self.b_conv2 = bias_variable([self.w2])
            with tf.name_scope('conv2b'):
                self.W_conv2b = weight_variable([self.filter_size1, self.filter_size1, self.w2, self.w2])
                self.b_conv2b = bias_variable([self.w2])

            # Third convolutional layers
            with tf.name_scope('conv3a'):
                self.W_conv3 = weight_variable([self.filter_size1, self.filter_size1, self.w2, self.w3])
                self.b_conv3 = bias_variable([self.w3])
            with tf.name_scope('conv3b'):
                self.W_conv3b = weight_variable([self.filter_size1, self.filter_size1, self.w3, self.w3])
                self.b_conv3b = bias_variable([self.w3])

            # Fourth convolutional layers
            with tf.name_scope('conv4a'):
                self.W_conv4 = weight_variable([self.filter_size1, self.filter_size1, self.w3, self.w4])
                self.b_conv4 = bias_variable([self.w4])
            with tf.name_scope('conv4b'):
                self.W_conv4b = weight_variable([self.filter_size1, self.filter_size1, self.w4, self.w4])
                self.b_conv4b = bias_variable([self.w4])

            # Fifth (and last down) convolutional layers
            with tf.name_scope('conv5a'):
                self.W_conv5 = weight_variable([self.filter_size1, self.filter_size1, self.w4, self.w5])
                self.b_conv5 = bias_variable([self.w5])
            with tf.name_scope('conv5b'):
                self.W_conv5b = weight_variable([self.filter_size1, self.filter_size1, self.w5, self.w5])
                self.b_conv5b = bias_variable([self.w5])

            # First up-convolution layer
            with tf.name_scope('upconv1'):
                self.W_upconv1 = weight_variable([2, 2, self.w4, self.w5])
                self.b_upconv1 = bias_variable([self.w4])

            # Sixth convolutional layers
            with tf.name_scope('conv6a'):
                self.W_conv6 = weight_variable([self.filter_size1, self.filter_size1, self.w5, self.w4])
                self.b_conv6 = bias_variable([self.w4])
            with tf.name_scope('conv6b'):
                self.W_conv6b = weight_variable([self.filter_size1, self.filter_size1, self.w4, self.w4])
                self.b_conv6b = bias_variable([self.w4])

            # Second up-convolution layer
            with tf.name_scope('upconv2'):
                self.W_upconv2 = weight_variable([self.upfilter_size, self.upfilter_size, self.w3, self.w4])
                self.b_upconv2 = bias_variable([self.w3])

            # Seventh convolutional layers
            with tf.name_scope('conv7a'):
                self.W_conv7 = weight_variable([self.filter_size1, self.filter_size1, self.w4, self.w3])
                self.b_conv7 = bias_variable([self.w3])
            with tf.name_scope('conv7b'):
                self.W_conv7b = weight_variable([self.filter_size1, self.filter_size1, self.w3, self.w3])
                self.b_conv7b = bias_variable([self.w3])

            # Third up-convolution layer
            with tf.name_scope('upconv3'):
                self.W_upconv3 = weight_variable([self.upfilter_size, self.upfilter_size, self.w2, self.w3])
                self.b_upconv3 = bias_variable([self.w2])

            # Eigth convolutional layers
            with tf.name_scope('conv8a'):
                self.W_conv8 = weight_variable([self.filter_size1, self.filter_size1, self.w3, self.w2])
                self.b_conv8 = bias_variable([self.w2])
            with tf.name_scope('conv8b'):
                self.W_conv8b = weight_variable([self.filter_size1, self.filter_size1, self.w2, self.w2])
                self.b_conv8b = bias_variable([self.w2])

            # Fourth up-convolution layer
            with tf.name_scope('upconv4'):
                self.W_upconv4 = weight_variable([self.upfilter_size, self.upfilter_size, self.w1, self.w2])
                self.b_upconv4 = bias_variable([self.w1])

            # Nineth convolutional layers
            with tf.name_scope('conv9a'):
                self.W_conv9 = weight_variable([self.filter_size1, self.filter_size1, self.w2, self.w1])
                self.b_conv9 = bias_variable([self.w1])
            with tf.name_scope('conv9b'):
                self.W_conv9b = weight_variable([self.filter_size1, self.filter_size1, self.w1, self.w1])
                self.b_conv9b = bias_variable([self.w1])

            # Last 1x1 convolutional layer
            with tf.name_scope('lastconv'):
                self.W_convlast = weight_variable([1, 1, self.w1, NUM_LABELS])
                self.b_convlast = bias_variable([NUM_LABELS])


    def model(self, data, train=False):
        """The Model definition."""

        if CONSIDER_PATCHES==False:
            initial_number = data.get_shape().as_list()[0]

            reshape_size = data.get_shape().as_list()[1] #400 # TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO MODIFY???

            with tf.name_scope('reshape'):
                x_image = data #tf.reshape(data, [int(float(data.get_shape().as_list()[0]*data.get_shape().as_list()[1]*data.get_shape().as_list()[2])/reshape_size**2), reshape_size, reshape_size, NUM_CHANNELS])

            # First convolutional layers
            with tf.name_scope('conv1a'):
                h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)

            with tf.name_scope('conv1b'):
                h_conv1b = tf.nn.relu(conv2d(h_conv1, self.W_conv1b) + self.b_conv1b)

            # Pooling layer - downsamples by 2X.
            with tf.name_scope('pool1'):
                h_pool1 = max_pool_2x2(h_conv1b)

            # Second convolutional layers
            with tf.name_scope('conv2a'):
                h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + self.b_conv2)
            with tf.name_scope('conv2b'):
                h_conv2b = tf.nn.relu(conv2d(h_conv2, self.W_conv2b) + self.b_conv2b) # h_conv2b

            # Second pooling layer.
            with tf.name_scope('pool2'):
                h_pool2 = max_pool_2x2(h_conv2b)

            # Third convolutional layers
            with tf.name_scope('conv3a'):
                h_conv3 = tf.nn.relu(conv2d(h_pool2, self.W_conv3) + self.b_conv3)
            with tf.name_scope('conv3b'):
                h_conv3b = tf.nn.relu(conv2d(h_conv3, self.W_conv3b) + self.b_conv3b)

            # Third pooling layer.
            with tf.name_scope('pool3'):
                h_pool3 = max_pool_2x2(h_conv3b)

            # Fourth convolutional layers
            with tf.name_scope('conv4a'):
                h_conv4 = tf.nn.relu(conv2d(h_pool3, self.W_conv4) + self.b_conv4)
            with tf.name_scope('conv4b'):
                h_conv4b = tf.nn.relu(conv2d(h_conv4, self.W_conv4b) + self.b_conv4b)

            # Fourth pooling layer.
            with tf.name_scope('pool4'):
                h_pool4 = max_pool_2x2(h_conv4b)

            # Fifth (and last down) convolutional layers
            with tf.name_scope('conv5a'):
                h_conv5 = tf.nn.relu(conv2d(h_pool4, self.W_conv5) + self.b_conv5)
            with tf.name_scope('conv5b'):
                h_conv5b = tf.nn.relu(conv2d(h_conv5, self.W_conv5b) + self.b_conv5b)

            # First up-convolution layer
            with tf.name_scope('upconv1'):
                h_upconv1 = upconv2d(h_conv5b, self.W_upconv1, [h_conv5b.get_shape().as_list()[0], 2*h_conv5b.get_shape().as_list()[1], 2*h_conv5b.get_shape().as_list()[2], self.w4]) + self.b_upconv1
                h_upconv1ext = tf.concat([h_conv4b, h_upconv1], 3)

            # Sixth convolutional layers
            with tf.name_scope('conv6a'):
                h_conv6 = tf.nn.relu(conv2d(h_upconv1ext, self.W_conv6) + self.b_conv6)
            with tf.name_scope('conv6b'):
                h_conv6b = tf.nn.relu(conv2d(h_conv6, self.W_conv6b) + self.b_conv6b)

            # Second up-convolution layer
            with tf.name_scope('upconv2'):
                h_upconv2 = upconv2d(h_conv6b, self.W_upconv2, [h_conv6b.get_shape().as_list()[0], 2*h_conv6b.get_shape().as_list()[1], 2*h_conv6b.get_shape().as_list()[2], self.w3]) + self.b_upconv2
                h_upconv2ext = tf.concat([h_conv3b, h_upconv2], 3)

            # Seventh convolutional layers
            with tf.name_scope('conv7a'):
                h_conv7 = tf.nn.relu(conv2d(h_upconv2ext, self.W_conv7) + self.b_conv7)
            with tf.name_scope('conv7b'):
                h_conv7b = tf.nn.relu(conv2d(h_conv7, self.W_conv7b) + self.b_conv7b)

            # Third up-convolution layer
            with tf.name_scope('upconv3'):
                h_upconv3 = upconv2d(h_conv7b, self.W_upconv3, [h_conv7b.get_shape().as_list()[0], 2*h_conv7b.get_shape().as_list()[1], 2*h_conv7b.get_shape().as_list()[2], self.w2]) + self.b_upconv3
                h_upconv3ext = tf.concat([h_conv2b, h_upconv3], 3)

            # Eigth convolutional layers
            with tf.name_scope('conv8a'):
                h_conv8 = tf.nn.relu(conv2d(h_upconv3ext, self.W_conv8) + self.b_conv8)
            with tf.name_scope('conv8b'):
                h_conv8b = tf.nn.relu(conv2d(h_conv8, self.W_conv8b) + self.b_conv8b)

            # Fourth up-convolution layer
            with tf.name_scope('upconv4'):
                h_upconv4 = upconv2d(h_conv8b, self.W_upconv4,\
                    [h_conv8b.get_shape().as_list()[0], 2*h_conv8b.get_shape().as_list()[1],\
                        2*h_conv8b.get_shape().as_list()[2], self.w1]) + self.b_upconv4
                h_upconv4ext = tf.concat([h_conv1b, h_upconv4], 3)


            # Nineth convolutional layers
            with tf.name_scope('conv9a'):
                h_conv9 = tf.nn.relu(conv2d(h_upconv4ext, self.W_conv9) + self.b_conv9)
            with tf.name_scope('conv9b'):
                h_conv9b = tf.nn.relu(conv2d(h_conv9, self.W_conv9b) + self.b_conv9b)

            # Last 1x1 convolutional layer
            with tf.name_scope('lastconv'):
                out = conv2d(h_conv9b, self.W_convlast) + self.b_convlast

            with tf.name_scope('reshape2'):
                out = tf.reshape(out, [initial_number, int(np.sqrt(np.prod(out.get_shape().as_list()[0:3])/initial_number)), int(np.sqrt(np.prod(out.get_shape().as_list()[0:3])/initial_number)), NUM_LABELS])

        else:
            layer = conv_layer(input=data,w=self.conv1_weights,b=self.conv1_biases, name='conv1')
            layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            layer = conv_layer(input=layer,w=self.conv2_weights,b=self.conv2_biases, name='conv2')
            layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            # add here conv layers
            reshaped, _ = flatten_layer(layer)
            hidden = tf.nn.relu(tf.matmul(reshaped, self.fc1_weights) + self.fc1_biases)
            out = tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

        return out

    def set_stuff(self):
        # Training computation: logits + cross-entropy loss.
        logits = self.model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
        # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits = logits, labels = train_labels_node),name="my_xent")

        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = tf.train.exponential_decay(
            0.01,                # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,          # Decay step.
            0.95,                # Decay rate.
            staircase=True, name="learning_rate")

        # Use simple momentum for the optimization.
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                               0.0).minimize(self.loss,
                                                             global_step=batch)

        self.train_prediction = tf.nn.softmax(logits)
        # We'll compute them only once in a while by calling their {eval()} method.
        train_all_prediction = tf.nn.softmax(self.model(train_all_data_node))

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def optimize(self):
        tf.global_variables_initializer().run()
        #self.saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
        summary_writer.add_graph(self._session.graph)

        print ('Initialized!')
        # Loop through training steps.
        print ('Total number of iterations = ' + str(int(NUM_EPOCHS * train_size / BATCH_SIZE)))
        training_indices = range(train_size)

        for iepoch in range(NUM_EPOCHS):
            print('Epoch = ', iepoch)

            perm_indices = training_indices # numpy.random.permutation(training_indices)

            for step in range (int(train_size / BATCH_SIZE)):
                #print('Step =', step)

                if train_size != BATCH_SIZE:
                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                else:
                    offset = 0

                batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]
                batch_data = train_data[batch_indices, :, :, :]
                batch_labels = train_labels[batch_indices]
                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels}

                #print('Init training network')
                _, l, lr, predictions = self._session.run(
                    [self.optimizer, self.loss, self.learning_rate, self.train_prediction],
                    feed_dict=feed_dict)

                #print('end training network, begin recording step')

                if step % RECORDING_STEP == 0:
                    # summary_str = self._session.run(summary_op, feed_dict=feed_dict) TODOOOOOOOOOOOOOOOOOO put it back!
                    # summary_writer.add_summary(summary_str, step)
                    # summary_writer.flush()

                    print ('global step:', iepoch*int(train_size / BATCH_SIZE)+step,\
                            ' over ',NUM_EPOCHS*int(train_size / BATCH_SIZE),\
                            '       i.e. : ',\
                            int((iepoch*int(train_size / BATCH_SIZE)+step)/(NUM_EPOCHS*int(train_size / BATCH_SIZE))*100),'%')
                    print ('Epoch: ', iepoch, '   || Step',float(step))
                    print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                 batch_labels))
                    sys.stdout.flush()

        # Print the time-usage.
        print("End optimisation")
        # Save the variables to disk.
        save_path = self.saver.save(self._session, FLAGS.train_dir + "/model.ckpt")
        print("Model saved in file: %s" % save_path)



    ############################# methods for predictions ######################
    # Get prediction for given input image
    def get_prediction(self, img):
        if CONSIDER_PATCHES:
            data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
            data_node = tf.constant(data)
        else:
            data_node = tf.constant(img.reshape([1,img.shape[0],img.shape[1],img.shape[2]]))

        output = tf.nn.softmax(self.model(data_node))
        output_prediction = self._session.run(output)

        if CONSIDER_PATCHES == True:
            img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction, patches=True)
        else:
            img_prediction = label_to_img(img.shape[0], img.shape[1], 1, 1, output_prediction, CONSIDER_PATCHES)

        return img_prediction

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(self, filename, image_idx,testing=False):
        if testing == True:
            imageid = "test_" + str(image_idx)
            subdir = filename+str(image_idx)+"/"
            image_filename = subdir + imageid + ".png"
        else:
            imageid = "satImage_%.3d" % image_idx
            image_filename = filename + imageid + ".png"

        img = mpimg.imread(image_filename)
        img_prediction = self.get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)
        return oimg

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_predicted_groundtruth(self, filename, image_idx, testing=False):
        if testing == True:
            imageid = "test_" + str(image_idx)
            subdir = filename+str(image_idx)+"/"
            image_filename = subdir + imageid + ".png"
        else:
            imageid = "satImage_%.3d" % image_idx
            image_filename = filename + imageid + ".png"

        img = mpimg.imread(image_filename)
        img_prediction = self.get_prediction(img)
        cimg = False_concatenate_images(img_prediction)
        return cimg


    def eval(self):
        print ("Running prediction on training set")
        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)

        for i in range(1, TRAINING_SIZE+1):
            pred_img = self.get_predicted_groundtruth(train_data_filename, i)
            Image.fromarray(pred_img).save(prediction_training_dir + "predicted_groundtruth_" + str(i) + ".png")
            #pimg = get_prediction_with_groundtruth(train_data_filename, i)
            #Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")

            oimg = self.get_prediction_with_overlay(train_data_filename, i)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
        print('Finished predicting training set')

    def test(self):
        print ("Running prediction TEST! (yeppa!)")
        data_dir = 'test_set_images/'
        test_subdir_filename = data_dir + 'test_'
        prediction_testing_dir = "predictions_for_TEST/"
        if not os.path.isdir(prediction_testing_dir):
            os.mkdir(prediction_testing_dir)
        for i in range(1, TESTING_SIZE+1):
            pred_img = self.get_predicted_groundtruth(test_subdir_filename, i, testing = True)
            Image.fromarray(pred_img).save(prediction_testing_dir + "predicted_groundtruth_" + str(i) + ".png")

            oimg = self.get_prediction_with_overlay(test_subdir_filename, i,testing = True)
            oimg.save(prediction_testing_dir + "overlay_" + str(i) + ".png")
        print('Finished predicting test set ! Humpa Lumpa!')


####################### END CLASS ##############################################





##### MAIN RUNNING FUNCTION ####################################################
def main(argv=None):  # pylint: disable=unused-argument

    opts = ''
    with tf.Session() as session:
        my_NN = NN(opts, session)
        # Run all the initializers to prepare the trainable parameters.
        my_NN.set_stuff()
        if RESTORE_MODEL:
            my_NN.saver.restore(my_NN._session, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")
        else:
            my_NN.optimize() # train the neural network

        my_NN.eval() # evaluate on training set

        if TEST == True:
            my_NN.test()
    print('End of main')


if __name__ == '__main__':
    tf.app.run()
