import numpy as np
import tensorflow as tf

import skimage.transform

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
    data_size = labels.shape[0]
    
    c0 = np.sum((labels[:,0]==1)*1)
    c1 = labels.shape[0]-c0
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    return data, labels, data_size


def get_prediction(img, s):
    #data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    data_node = tf.constant(img.reshape([1,img.shape[0],img.shape[1],img.shape[2]]))
    output = tf.nn.softmax(neuralnetwork(data_node))
    output_prediction = s.run(output)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    
    return img_prediction


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

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1, seed=SEED)
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
    reshape_size = 16
    
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, reshape_size, reshape_size, NUM_CHANNELS])

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

    # Fifth (and last down) convolutional layers
    with tf.name_scope('conv5a'):
        W_conv5 = weight_variable([3, 3, 512, 1024])
        b_conv5 = bias_variable([1024])
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    with tf.name_scope('conv5b'):
        W_conv5b = weight_variable([3, 3, 1024, 1024])
        b_conv5b = bias_variable([1024])
        h_conv5b = tf.nn.relu(conv2d(h_conv5, W_conv5b) + b_conv5b)

    # First up-convolution layer
    with tf.name_scope('upconv1'):
        W_upconv1 = weight_variable([2, 2, 512, 1024])
        b_upconv1 = bias_variable([512])
        print('--------------------')
        print(W_upconv1)
        print(b_upconv1)
        h_upconv1 = upconv2d(h_conv5b, W_upconv1, [-1, 2*h_conv5b.get_shape().as_list()[1], 2*h_conv5b.get_shape().as_list()[2], 512]) + b_upconv1
        h_upconv1ext = tf.concat([h_conv4b, h_upconv1], 3)

    # Sixth convolutional layers
    with tf.name_scope('conv6a'):
        W_conv6 = weight_variable([3, 3, 1024, 512])
        b_conv6 = bias_variable([512])
        h_conv6 = tf.nn.relu(conv2d(h_upconv1ext, W_conv6) + b_conv6)
    with tf.name_scope('conv6b'):
        W_conv6b = weight_variable([3, 3, 512, 512])
        b_conv6b = bias_variable([512])
        h_conv6b = tf.nn.relu(conv2d(h_conv6, W_conv6b) + b_conv6b)

    # Second up-convolution layer
    with tf.name_scope('upconv2'):
        W_upconv2 = weight_variable([2, 2, 256, 512])
        b_upconv2 = bias_variable([256])
        print('--------------------')
        print(W_upconv2)
        print(b_upconv2)
        h_upconv2 = upconv2d(h_conv6b, W_upconv2, [-1, 2*h_conv6b.get_shape().as_list()[1], 2*h_conv6b.get_shape().as_list()[2], 256]) + b_upconv2
        h_upconv2ext = tf.concat([h_conv3b, h_upconv2], 3)
    
    # Seventh convolutional layers
    with tf.name_scope('conv7a'):
        W_conv7 = weight_variable([3, 3, 512, 256])
        b_conv7 = bias_variable([256])
        h_conv7 = tf.nn.relu(conv2d(h_upconv2ext, W_conv7) + b_conv7)
    with tf.name_scope('conv7b'):
        W_conv7b = weight_variable([3, 3, 256, 256])
        b_conv7b = bias_variable([256])
        h_conv7b = tf.nn.relu(conv2d(h_conv7, W_conv7b) + b_conv7b)

    # Third up-convolution layer
    with tf.name_scope('upconv3'):
        W_upconv3 = weight_variable([2, 2, 128, 256])
        b_upconv3 = bias_variable([128])
        h_upconv3 = upconv2d(h_conv7b, W_upconv3, [-1, 2*h_conv7b.get_shape().as_list()[1], 2*h_conv7b.get_shape().as_list()[2], 128]) + b_upconv3
        h_upconv3ext = tf.concat([h_conv2b, h_upconv3], 3)
    
    # Eigth convolutional layers
    with tf.name_scope('conv8a'):
        W_conv8 = weight_variable([3, 3, 256, 128])
        b_conv8 = bias_variable([128])
        h_conv8 = tf.nn.relu(conv2d(h_upconv3ext, W_conv8) + b_conv8)
    with tf.name_scope('conv8b'):
        W_conv8b = weight_variable([3, 3, 128, 128])
        b_conv8b = bias_variable([128])
        h_conv8b = tf.nn.relu(conv2d(h_conv8, W_conv8b) + b_conv8b)
    
    # Fourth up-convolution layer
    with tf.name_scope('upconv4'):
        W_upconv4 = weight_variable([2, 2, 64, 128])
        b_upconv4 = bias_variable([64])
        h_upconv4 = upconv2d(h_conv8b, W_upconv4, [-1, 2*h_conv8b.get_shape().as_list()[1], 2*h_conv8b.get_shape().as_list()[2], 64]) + b_upconv4
        h_upconv4ext = tf.concat([h_conv1b, h_upconv4], 3)
    
    # Nineth convolutional layers
    with tf.name_scope('conv9a'):
        W_conv9 = weight_variable([3, 3, 128, 64])
        b_conv9 = bias_variable([64])
        h_conv9 = tf.nn.relu(conv2d(h_upconv4ext, W_conv9) + b_conv9)
    with tf.name_scope('conv9b'):
        W_conv9b = weight_variable([3, 3, 64, 64])
        b_conv9b = bias_variable([64])
        h_conv9b = tf.nn.relu(conv2d(h_conv9, W_conv9b) + b_conv9b)

    # Last 1x1 convolutional layer
    with tf.name_scope('lastconv'):
        W_convlast = weight_variable([1, 1, 64, NUM_LABELS])
        b_convlast = bias_variable([NUM_LABELS])
        print('--------------------')
        print(W_convlast)
        print(b_convlast)
        y_conv = conv2d(h_conv9b, W_convlast) + b_convlast

    return y_conv


def main(argv=None):
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract images into numpy arrays.
    train_data, train_img_size = extract_data(train_data_filename, TRAINING_SIZE, patches=False)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE, patches=False)
    train_size = train_data.shape[0]
    print('Size of the training dataset =', train_data.shape)

# !!!!!! TODO !!!!!! Check and balance the size of each class in the training set ----------------------------- WHAT DOES IT MEAN TO BALANCE DATA WHEN CONSIDERING PIXEL BY PIXEL??!
# train_data, train_labels, train_size = balance_data(train_data, train_labels)

    # Create the model
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, train_img_size, train_img_size, NUM_CHANNELS])
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, train_img_size, train_img_size, NUM_LABELS])

    # Build the graph for the deep net
    y_conv = neuralnetwork(x)
    
    # Compute the loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_), name="loss")
    
    with tf.name_scope('optimizer'):
        # Optimizer: set up a variable that is incremented once per batch and controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(0.01,                # Base learning rate.
                                                   batch * BATCH_SIZE,  # Current index into the dataset.
                                                   train_size,       # Decay step.
                                                   0.95,                # Decay rate.
                                                   staircase=True,
                                                   name="learning_rate")

        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               0.0).minimize(loss,
                                               global_step=batch)
        '''optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                                global_step=batch)'''
# with tf.name_scope('accuracy'):
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# correct_prediction = tf.cast(correct_prediction, tf.float32)
# accuracy = tf.reduce_mean(correct_prediction)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as s:
        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")
        
        else:
            tf.global_variables_initializer().run()
            
            for iepoch in range(NUM_EPOCHS):
                print('Learning process:', iepoch/NUM_EPOCHS*100, '%')
                # Permute training indices
                perm_indices = np.random.permutation(range(train_size))

                for step in range(int(train_size/BATCH_SIZE)):
                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]
                        
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {x: batch_data,
                                 y_: batch_labels}

#if step % 100 == 0:
#    train_accuracy = accuracy.eval(feed_dict=feed_dict)
#    print('Epoch %d, Step %d, training accuracy %g' % (iepoch, ste, train_accuracy))

# Run the graph
#optimizer.run(feed_dict=feed_dict)

# print('--> Test accuracy %g' % accuracy.eval(feed_dict ={x: , y_: }))

            # Save the variables to disk.
            save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
            print("Model saved in file: %s" % save_path)

        print ("Running predictions on training set...")
        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)

        for i in range(1, TRAINING_SIZE+1):
            imageid = "satImage_%.3d" % i
            image_filename = train_data_filename + imageid + ".png"
            img = mpimg.imread(image_filename)
            
            img_prediction = get_prediction(img, s)
            
            pred_img = False_concatenate_images(img_prediction)
            Image.fromarray(pred_img).save(prediction_training_dir + "predicted_groundtruth_" + str(i) + ".png")
            
            oimg = make_img_overlay(img, img_prediction)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")


    print('END')


if __name__ == '__main__':
    tf.app.run()
