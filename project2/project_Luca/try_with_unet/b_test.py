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
from helper_functions import *
################################################################################


tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist/test4',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

################################################################################
TRAINING_SIZE = 10
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 50  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 3 # how many as you like
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 50
TEST = False  # if we want to predict test image as well
TESTING_SIZE = 50 # number of test images i.e. 50

#variables imported from helper functions:
print ('NUM_CHANNELS: ', NUM_CHANNELS )    # 3?
print ('PIXEL_DEPTH: ', PIXEL_DEPTH)       # 255?
print ('NUM_LABELS: ', NUM_LABELS )        # 2?
print ('IMG_PATCH_SIZE: ', IMG_PATCH_SIZE) # 16?


# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.





# further functions ############################################################
def new_weights(shape,stddev_ = 0.05):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev_,seed=SEED),name="W")

def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=[length]),name="B")

########################## internal functions ##############################
def conv_layer(self, input,w,b, name='conv'): # channels_in,channels_out
    with tf.name_scope(name):
        #w = new_weights([5,5,channels_in,channels_out])
        #b = tf.Variable(tf.zeros([channels_out]),name="B")
        conv = tf.nn.conv2d(input,w,strides=[1,1,1,1],padding="SAME")
        act = tf.nn.relu(conv + b)
        #####tf.nn.relu(tf.nn.bias_add(conv, b))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        #tf.summary.histogram("activations", act)
        layer = tf.nn.max_pool(act,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        return layer

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def fc_layer(input,channels_in,channels_out,name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([channels_in,channels_out]),name="W")
        b = tf.Variable(tf.zeros([channels_out]),name="B")
        layer = tf.nn.relu(tf.matmul(input,w)+b)
        return layer,w,b

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



########### define directory of the training images ############################
data_dir = '../training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'

# Extract it into numpy arrays.
train_data = extract_data(train_data_filename, TRAINING_SIZE)
train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

num_epochs = NUM_EPOCHS

################ Balance the data ##############################################
train_data, train_labels, train_size = balance_classes_in_data(train_data,train_labels)
# END of balancing #############################################################

with tf.name_scope('input'):
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS),name = 'train_labels_nodes') # name put by luca
    train_all_data_node = tf.constant(train_data)




class NN(object):
    """Our Neural network"""
    def __init__(self, options, session):
        #self._options = options
        self._session = session

        with tf.name_scope("conv1"):
            self.conv1_weights = new_weights([5, 5, NUM_CHANNELS, 32]) # 5x5 filter, depth 32.
            self.conv1_biases = tf.Variable(tf.zeros([32]), name = "B")

        with tf.name_scope("conv2"):
            self.conv2_weights = new_weights([5, 5, 32, 64])
            self.conv2_biases = new_biases(length=64)

        with tf.name_scope("fc1"):
            self.fc1_weights = new_weights(shape= [int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                                        stddev_ = 0.1) # fully connected, depth 512.
            self.fc1_biases = new_biases(length=512)

        with tf.name_scope("fc2"):
            self.fc2_weights = new_weights(shape= [512, NUM_LABELS],stddev_ = 0.1)
            self.fc2_biases = new_biases(length=NUM_LABELS)

    def conv_layer(self, input,w,b, name='conv'): # channels_in,channels_out
        with tf.name_scope(name):
            #w = new_weights([5,5,channels_in,channels_out])
            #b = tf.Variable(tf.zeros([channels_out]),name="B")
            conv = tf.nn.conv2d(input,w,strides=[1,1,1,1],padding="SAME")
            act = tf.nn.relu(conv + b)
            #####tf.nn.relu(tf.nn.bias_add(conv, b))
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            #tf.summary.histogram("activations", act)
            layer = tf.nn.max_pool(act,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            return layer

    def model(self, data, train=False):
        """The Model definition."""
        pool = self.conv_layer(input=data,w=self.conv1_weights,b=self.conv1_biases, name='conv1')
        pool2 = self.conv_layer(input=pool,w=self.conv2_weights,b=self.conv2_biases, name='conv2')
        '''  # jsut a test
        pool, self.conv1_weights, self.conv1_biases\
                = self.conv_layer(input=data,channels_in=NUM_CHANNELS,channels_out=32, name='conv1')

        pool2, self.conv2_weights, self.conv2_biases\
                = self.conv_layer(input=pool,channels_in=32,channels_out=64, name='conv2')'''

        reshape, _ = flatten_layer(pool2)
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        out = tf.matmul(hidden, self.fc2_weights) + self.fc2_biases
        return out

    def optimize(self):
        tf.global_variables_initializer().run()
        #self.saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
        summary_writer.add_graph(self._session.graph)

        print ('Initialized!')
        # Loop through training steps.
        print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))
        training_indices = range(train_size)

        for iepoch in range(num_epochs):

            perm_indices = numpy.random.permutation(training_indices)

            for step in range (int(train_size / BATCH_SIZE)):

                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                batch_data = train_data[batch_indices, :, :, :]
                batch_labels = train_labels[batch_indices]

                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels}

                _, l, lr, predictions = self._session.run(
                    [self.optimizer, self.loss, self.learning_rate, self.train_prediction],
                    feed_dict=feed_dict)
                
                if step % RECORDING_STEP == 0:
                    summary_str = self._session.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    print ('global step:', iepoch*int(train_size / BATCH_SIZE)+step,\
                            ' over ',num_epochs*int(train_size / BATCH_SIZE),\
                            '       i.e. : ',\
                            int((iepoch*int(train_size / BATCH_SIZE)+step)/(num_epochs*int(train_size / BATCH_SIZE))*100),'%')
                    print ('Epoch: ', iepoch, '   || Step',float(step))
                    print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                 batch_labels))
                    sys.stdout.flush()
                '''else:
                    # Run the graph and fetch some of the nodes.
                    _, l, lr, predictions = self._session.run(
                        [self.optimizer, self.loss, self.learning_rate, self.train_prediction],
                        feed_dict=feed_dict)'''
        # Print the time-usage.
        print("End optimisation")
        # Save the variables to disk.
        save_path = self.saver.save(self._session, FLAGS.train_dir + "/model.ckpt")
        print("Model saved in file: %s" % save_path)

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
            staircase=True,name="learning_rate")


        # Use simple momentum for the optimization.
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                               0.0).minimize(self.loss,
                                                             global_step=batch)

        self.train_prediction = tf.nn.softmax(logits)
        # We'll compute them only once in a while by calling their {eval()} method.
        train_all_prediction = tf.nn.softmax(self.model(train_all_data_node))

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    ############################# methods for predictions ######################
    # Get prediction for given input image
    def get_prediction(self, img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        data_node = tf.constant(data)
        output = tf.nn.softmax(self.model(data_node))
        output_prediction = self._session.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

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
        #for i in range(TRAINING_SIZE, TRAINING_SIZE+11):
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
