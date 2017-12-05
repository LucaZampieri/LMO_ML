import numpy as np
import tensorflow as tf

from helper_functions import *

################################################################################
tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist/testO',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS
################################################################################

def balance_data(data, labels):
    # Count the number of data points on each class
    c0 = np.sum((labels[:,0]==1)*1)
    c1 = labels.shape[0]-c0
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # Balance to take the same number of data points with c0 and c1 classes
    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c] # Concatenate lists
    data = data[new_indices,:,:,:]
    labels = labels[new_indices]
    
    c0 = np.sum((labels[:,0]==1)*1)
    c1 = labels.shape[0]-c0
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    return data, labels

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

def neuralnetwork(x):
    """neuralnetwork builds the graph for a net for classifying roads and background.
    
    Args:
        x: an input tensor with the dimensions (N_examples, BATCH_SIZE), where BATCH_SIZE is the
        number of pixels in a patch image.
    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, NUM_LABELS), with values
        equal to 1 or 0 (road or not). keep_prob is a scalar placeholder for the probability of
        dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" -- 1 if a greyscale image, 3 for RGB, 4 for RGBA, etc.
#with tf.name_scope('reshape'):
#        x_image = tf.reshape(x, [-1, 28, 28, NUM_CHANNELS])
                    
    # First convolutional layers
    with tf.name_scope('conv1a'):
        W_conv1 = weight_variable([3, 3, NUM_CHANNELS, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('conv1b'):
        W_conv1b = weight_variable([3, 3, 64, 64])
        b_conv1b = bias_variable([64])
        h_conv1b = tf.nn.relu(conv2d(h_conv1, W_conv1b) + b_conv1b)
            
    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1b)
                
    # Second convolutional layers
    with tf.name_scope('conv2a'):
        W_conv2 = weight_variable([3, 3, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope('conv2b'):
        W_conv2b = weight_variable([3, 3, 128, 128])
        b_conv2b = bias_variable([128])
        h_conv2b = tf.nn.relu(conv2d(h_conv2, W_conv2b) + b_conv2b)
    
    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2b)

    # Third convolutional layers
    with tf.name_scope('conv3a'):
        W_conv3 = weight_variable([3, 3, 128, 256])
        b_conv3 = bias_variable([256])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    with tf.name_scope('conv3b'):
        W_conv3b = weight_variable([3, 3, 256, 256])
        b_conv3b = bias_variable([256])
        h_conv3b = tf.nn.relu(conv2d(h_conv3, W_conv3b) + b_conv3b)

    # Third pooling layer.
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3b)
    
    # Fourth convolutional layers
    with tf.name_scope('conv4a'):
        W_conv4 = weight_variable([3, 3, 256, 512])
        b_conv4 = bias_variable([512])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    with tf.name_scope('conv4b'):
        W_conv4b = weight_variable([3, 3, 512, 512])
        b_conv4b = bias_variable([512])
        h_conv4b = tf.nn.relu(conv2d(h_conv4, W_conv4b) + b_conv4b)
    
    # Fourth pooling layer.
    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv4b)

    # Fifth and last convolutional layers
    with tf.name_scope('conv5a'):
        W_conv5 = weight_variable([3, 3, 512, 1024])
        b_conv5 = bias_variable([1024])
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    with tf.name_scope('conv5b'):
        W_conv5b = weight_variable([3, 3, 1024, 1024])
        b_conv5b = bias_variable([1024])
        h_conv5b = tf.nn.relu(conv2d(h_conv5, W_conv5b) + b_conv5b)





    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
            
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
                                                                                        
    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                                                                                                    
    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
                                                                                                                
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def main(argv=None):
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    # Check and balance the size of each class in the training set
    train_data, train_labels = balance_data(train_data, train_labels)
    
    num_epochs = NUM_EPOCHS
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, BATCH_SIZE])
        
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])

    print('END')


if __name__ == '__main__':
    tf.app.run()
