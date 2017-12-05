"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
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

###########################################################################
###############################################################################
########################## internal functions ##############################
def conv_layer(input,w,b, name='conv'): # channels_in,channels_out
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

'''def conv_layer(input,channels_in,channels_out, name='conv'): # channels_in,channels_out
    with tf.name_scope(name):
        w = new_weights([5,5,channels_in,channels_out])
        b = tf.Variable(tf.zeros([channels_out]),name="B")
        conv = tf.nn.conv2d(input,w,strides=[1,1,1,1],padding="SAME")
        act = tf.nn.relu(conv + b)
        #####tf.nn.relu(tf.nn.bias_add(conv, b))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        #tf.summary.histogram("activations", act)
        layer = tf.nn.max_pool(act,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        return layer'''

def fc_layer(input,channels_in,channels_out,name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([channels_in,channels_out]),name="W")
        b = tf.Variable(tf.zeros([channels_out]),name="B")
        layer = tf.nn.relu(tf.matmul(input,w)+b)
        return layer,w,b


#x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
#  x_image = tf.reshape(x, [-1, 28, 28, 1])
#tf.summary.image('input', x_image, 3)
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True,name="conv"):  # Use 2x2 max-pooling.
    with tf.name_scope(name):
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = new_weights(shape=shape)
        biases = new_biases(length=num_filters)
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        layer += biases
        # Use pooling to down-sample the image resolution?
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        layer = tf.nn.relu(layer)
    return layer, weights



data_dir = 'training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'

# Extract it into numpy arrays.
train_data = extract_data(train_data_filename, TRAINING_SIZE)
train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

num_epochs = NUM_EPOCHS
# Now check the size of both classes and balance ###############################
c0 = 0
c1 = 0
for i in range(len(train_labels)):
    if train_labels[i][0] == 1:
        c0 = c0 + 1
    else:
        c1 = c1 + 1
print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
# balance to take the same number of patches with c0 and c1 classes
print ('Balancing training data...')
min_c = min(c0, c1)
idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
new_indices = idx0[0:min_c] + idx1[0:min_c]
print ('len(new_indices): ',len(new_indices))
print ('train_data.shape: ',train_data.shape)
train_data = train_data[new_indices,:,:,:]
train_labels = train_labels[new_indices]

train_size = train_labels.shape[0]

c0 = 0
c1 = 0
for i in range(len(train_labels)):
    if train_labels[i][0] == 1:
        c0 = c0 + 1
    else:
        c1 = c1 + 1
print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
# END of balancing #############################################################

with tf.name_scope('input'):
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS),name = 'train_labels_nodes') # name put by luca
    train_all_data_node = tf.constant(train_data)


##### MAIN RUNNING FUNCTION ####################################################
def main(argv=None):  # pylint: disable=unused-argument




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

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg







    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.


    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}


    with tf.name_scope("conv1"):
        conv1_weights = new_weights([5, 5, NUM_CHANNELS, 32]) # 5x5 filter, depth 32.
        conv1_biases = tf.Variable(tf.zeros([32]), name = "B")

    with tf.name_scope("conv2"):
        conv2_weights = new_weights([5, 5, 32, 64])
        conv2_biases = new_biases(length=64)

    with tf.name_scope("fc1"):
        fc1_weights = new_weights(shape= [int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                                    stddev_ = 0.1) # fully connected, depth 512.
        fc1_biases = new_biases(length=512)

    with tf.name_scope("fc2"):
        fc2_weights = new_weights(shape= [512, NUM_LABELS],stddev_ = 0.1)
        fc2_biases = new_biases(length=NUM_LABELS)



    ########################## OTHER functions #################################
    # Get prediction for given input image
    def get_prediction(img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction


    # Get a concatenation of the prediction and groundtruth for given input file
    def get_predicted_groundtruth(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        cimg = False_concatenate_images(img_prediction)
        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg
    ################### end other functions ####################################


    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        #with tf.name_scope("conv1"):
             #conv = tf.nn.conv2d(data,conv1_weights,strides=[1, 1, 1, 1],padding='SAME')
            # Bias and rectified linear non-linearity.
             #relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
            # Max pooling. The kernel size spec {ksize} also follows the layout of
            # the data. Here we have a pooling window of 2, and a stride of 2.
             #pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
             #tf.summary.histogram("weights", conv1_weights)
             #tf.summary.histogram("biases", conv1_biases)
            #tf.summary.histogram("pooling", pool)

        # giving the weights
        pool = conv_layer(input=data,w=conv1_weights,b=conv1_biases, name='conv1')
        pool2 = conv_layer(input=pool,w=conv2_weights,b=conv2_biases, name='conv2')

        # taking channels_in channels_out
        #pool = conv_layer(input=data,channels_in=[5, 5, NUM_CHANNELS, 32],\
        #                            channels_out=[32], name='conv1')
        #pool2 = conv_layer(input=pool,channels_in=[5, 5, 32, 64],channels_out=[64], name='conv2')

        # taking the third function:
        # conv 1
        '''pool, weights_conv1 = \
            new_conv_layer(input=data,
                           num_input_channels=NUM_CHANNELS,
                           filter_size=5,
                           num_filters=32,
                           use_pooling=True)
        # conv 2
        pool2, weights_conv2 = \
            new_conv_layer(input=pool,
                           num_input_channels=32,
                           filter_size=5,
                           num_filters=64,
                           use_pooling=True)'''
        '''# flatten
        layer_flat, num_features = flatten_layer(layer_conv2)

        # fully connected 1
        layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)
        # dropout?
        if train:
            layer_fc1 = tf.nn.dropout(layer_fc1, 0.5, seed=SEED)

        # fully connected 2
        layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=NUM_LABELS,
                             use_relu=False)


        out = layer_fc2'''



        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        if train == True:
            summary_id = '_0'
            s_data = get_image_summary(data)
            filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)

            #s_conv = get_image_summary(conv)
            #filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
            s_pool = get_image_summary(pool)
            filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool)
            #s_conv2 = get_image_summary(conv2)
            #filter_summary4 = tf.summary.image('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            filter_summary5 = tf.summary.image('summary_pool2' + summary_id, s_pool2)

        return out
    ############################# end model ####################################


    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    with tf.name_scope("xent"): # xent is basically our loss
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits = logits, labels = train_labels_node),name="xent")
        tf.summary.scalar('xent', loss)

    '''all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)'''

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True,name="learning_rate")
    tf.summary.scalar('learning_rate', learning_rate)

    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.0).minimize(loss,
                                                         global_step=batch)
    '''optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                        global_step=batch)'''

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:


        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
            summary_writer.add_graph(s.graph)

            print ('Initialized!')
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)

                for step in range (int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step % RECORDING_STEP == 0:

                        summary_str, _, l, lr, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                        summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        # print_predictions(predictions, batch_labels)

                        #print ('Epoch', (float(step) * BATCH_SIZE / train_size))
                        print ('Epoch: ', iepoch, '   || Step',float(step))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

            # Save the variables to disk.
            save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
            print("Model saved in file: %s" % save_path)


        print ("Running prediction on training set")
        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        for i in range(1, TRAINING_SIZE+1):
        #for i in range(TRAINING_SIZE, TRAINING_SIZE+11):
            pred_img = get_predicted_groundtruth(train_data_filename, i)
            Image.fromarray(pred_img).save(prediction_training_dir + "predicted_groundtruth_" + str(i) + ".png")

            #pimg = get_prediction_with_groundtruth(train_data_filename, i)
            #Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
            oimg = get_prediction_with_overlay(train_data_filename, i)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")



        if TEST:
            # Get a concatenation of the prediction and groundtruth for given input file
            def get_predicted_groundtruth(filename, image_idx):

                imageid = "test_" + str(image_idx)
                subdir = filename+str(image_idx)+"/"
                image_filename = subdir + imageid + ".png"
                img = mpimg.imread(image_filename)

                img_prediction = get_prediction(img)
                cimg = False_concatenate_images(img_prediction)
                return cimg

            # Get prediction overlaid on the original image for given input file
            def get_prediction_with_overlay(filename, image_idx):

                imageid = "test_" + str(image_idx)
                subdir = filename+str(image_idx)+"/"
                image_filename = subdir + imageid + ".png"
                img = mpimg.imread(image_filename)

                img_prediction = get_prediction(img)
                oimg = make_img_overlay(img, img_prediction)

                return oimg
            print ("Running on TEST")
            data_dir = 'test_set_images/'
            test_subdir_filename = data_dir + 'test_'

            prediction_testing_dir = "predictions_for_TEST/"
            if not os.path.isdir(prediction_testing_dir):
                os.mkdir(prediction_testing_dir)
            for i in range(1, TESTING_SIZE+1):
                pred_img = get_predicted_groundtruth(test_subdir_filename, i)
                Image.fromarray(pred_img).save(prediction_testing_dir + "predicted_groundtruth_" + str(i) + ".png")

                oimg = get_prediction_with_overlay(test_subdir_filename, i)
                oimg.save(prediction_testing_dir + "overlay_" + str(i) + ".png")



if __name__ == '__main__':
    tf.app.run()
