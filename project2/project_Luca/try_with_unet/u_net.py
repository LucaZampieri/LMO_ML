from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import gzip
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


from helper_functions import * # ???????????????????????????????????

################################################################################
SEED = 50


# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.
######################### FLAGS ################################################


flags = tf.app.flags

flags.DEFINE_string('save_path', '/tmp/mnist/test4',
                    """Directory where to write event logs """
                    """and checkpoint.""")
flags.DEFINE_string("train_data__", 'training/images/', "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", 'training/groundtruth/', "File consisting of analogies of four tokens."
    "embedding 2 - embedding 1 + embedding 3 should be close "
    "to embedding 4."
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 5,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 16,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 500,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")

FLAGS = flags.FLAGS
################################# END OF FLAGS #################################


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
print ('new train_data.shape: ',train_data.shape)
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


train_data_node = tf.placeholder(
   tf.float32,
   shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS),
   name='train_data_node')
train_labels_node = tf.placeholder(tf.float32,
                                  shape=(BATCH_SIZE, NUM_LABELS))
train_all_data_node = tf.constant(train_data)



################ beginning of the classes ######################################
class Options(object):
    """Options used by our Sat_ model."""

    def __init__(self):
        # Model options:
        # Embedding dimension.
        self.emb_dim = FLAGS.embedding_size

        # Training options.
        # The training text file.
        self.train_data__ = FLAGS.train_data__

        # Number of negative samples per example.
        self.num_samples = FLAGS.num_neg_samples

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = FLAGS.epochs_to_train

        # Concurrent training steps.
        self.concurrent_steps = FLAGS.concurrent_steps

        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size

        # The number of words to predict to the left and right of the target word.
        self.window_size = FLAGS.window_size

        # The minimum number of word occurrences for it to be included in the
        # vocabulary.
        self.min_count = FLAGS.min_count

        # Subsampling threshold for word occurrence.
        self.subsample = FLAGS.subsample

        # How often to print statistics.
        self.statistics_interval = FLAGS.statistics_interval

        # How often to write to the summary file (rounds up to the nearest
        # statistics_interval).
        self.summary_interval = FLAGS.summary_interval

        # How often to write checkpoints (rounds up to the nearest statistics
        # interval).
        self.checkpoint_interval = FLAGS.checkpoint_interval

        # Where to write out summaries.
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Eval options.
        # The text file for eval.
        self.eval_data = FLAGS.eval_data

def new_weights(shape,stddev_ = 0.05):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev_,seed=SEED),name="W")


def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=[length]),name="B")

def new_conv_layer(input,              # The previous layer.
              num_input_channels, # Num. channels in prev. layer.
              filter_size,        # Width and height of each filter.
              num_filters,        # Number of filters.
              use_pooling=True,name="conv"):  # Use 2x2 max-pooling.
    with tf.name_scope(name):
       # Shape of the filter-weights for the convolution.
       # This format is determined by the TensorFlow API.
       shape = [filter_size, filter_size, num_input_channels, num_filters]
       weights = new_weights(shape=shape)
       biases = new_biases(length=num_filters)
       layer = tf.nn.conv2d(input=input,
                            filter=weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
       layer += biases
       if use_pooling:
           layer = tf.nn.max_pool(value=layer,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
       layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]
    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
            num_inputs,     # Num. inputs from prev. layer.
            num_outputs,    # Num. outputs.
            use_relu=True,name="fc"): # Use Rectified Linear Unit (ReLU)?
    with tf.name_scope(name):
       # Create new weights and biases.
       weights = new_weights(shape=[num_inputs, num_outputs])
       biases = new_biases(length=num_outputs)
       # Calculate the layer as the matrix multiplication of
       # the input and weights, and then add the bias-values.
       layer = tf.matmul(input, weights) + biases
       # Use ReLU?
       if use_relu:
           layer = tf.nn.relu(layer)
    return layer


class Sat_ (object):
        """Sat_  model."""
        def __init__(self, options, session):
            self._options = options
            self._session = session
            self.build_graph()
            self.seed = SEED
            #self.build_eval_graph()

        def forward (self, data,labels, train=True):
            """Build the graph for the forward pass."""

            true_logits = labels

            # Global step: scalar, i.e., shape [].
            self.global_step = tf.Variable(0, name="global_step")

            # conv 1
            layer_conv1, weights_conv1 = \
               new_conv_layer(input=data,
                              num_input_channels=NUM_CHANNELS,
                              filter_size=filter_size1,
                              num_filters=num_filters1,
                              use_pooling=True)
            # conv 2
            layer_conv2, weights_conv2 = \
               new_conv_layer(input=layer_conv1,
                              num_input_channels=num_filters1,
                              filter_size=filter_size2,
                              num_filters=num_filters2,
                              use_pooling=True)
            # flatten
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


            logits = layer_fc2
            return logits , true_logits


        def nce_loss(self, logits, labels):
            """Build the graph for the NCE loss."""

            # cross-entropy(logits, labels)
            opts = self._options
            xent = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits(
                 logits = logits, labels = labels),name="xent")

            nce_loss_tensor = (tf.reduce_sum(xent)) / opts.batch_size
            return nce_loss_tensor

        def optimize(self, loss):
            """Build the graph to optimize the loss function."""

            # Optimizer nodes.
            # Linear learning rate decay.
            opts = self._options
            batch = self.global_step #tf.Variable(0) # or opts.somethong
            # base learning rate should be opts.something
            lr = tf.train.exponential_decay(
               learning_rate = 0.01,                # Base learning rate.
               global_step = batch * BATCH_SIZE,  # Current index into the dataset.
               decay_steps = train_size,          # Decay step.
               decay_rate = 0.95,                # Decay rate.
               staircase = True, name = "learning_rate")
            tf.summary.scalar('learning_rate', lr)

            self._lr = lr
            #optimizer = tf.train.GradientDescentOptimizer(lr)
            #train = optimizer.minimize(loss,
            #                        global_step=self.global_step,
            #                           gate_gradients=optimizer.GATE_NONE)

            # Use simple momentum for the optimization.
            optimizer = tf.train.MomentumOptimizer(self._lr,0.0)
            # batch = self.global_step ?
            train = optimizer.minimize(loss,global_step=batch)

            self._train = train


        def build_graph(self):
            """Build the graph for the full model."""
            opts = self._options
            # The training data. A text file.
            #self._epoch =
            #self._words =

            self._train_data_node = train_data_node
            self._train_labels_node = train_labels_node
            self._train_all_data_node = train_all_data_node

            examples = self._train_data_node
            labels = self._train_labels_node

            print("Data file: ", opts.train_data__)



            logits,true_logits = self.forward(examples, labels)
            #train_prediction = tf.nn.softmax(logits)
            self.train_prediction = tf.nn.softmax(logits)

            loss = self.nce_loss(logits, true_logits)
            tf.summary.scalar("xent loss", loss)
            self._loss = loss
            self.optimize(loss)

            # Properly initialize all variables.
            tf.global_variables_initializer().run()

            self.saver = tf.train.Saver()




        def train(self):
            """Train the model."""
            opts = self._options

            # old line
            #initial_epoch, initial_words = self._session.run([self._epoch, self._words])

            #tf.global_variables_initializer().run() # not sure if needed

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(opts.save_path, self._session.graph)

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
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    print(len(batch_indices))
                    print(batch_data.shape)
                    print(batch_data.dtype)
                    print(batch_labels.shape)
                    print(batch_labels.dtype)
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {self._train_data_node: batch_data,
                                self._train_labels_node: batch_labels}

                    _, loss, lr, predictions = self._session.run(
                        [summary_op, self._loss, self._lr, self.train_prediction],
                        feed_dict=feed_dict)
                    # removed optimizer... should we put it back?
                    # or these [self._epoch, self.global_step, self._loss, self._words, self._lr]

                    if step % RECORDING_STEP == 0:
                        summary_str = self._session.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                        #summary_writer.flush()
                        print ('global step:', iepoch*int(train_size / BATCH_SIZE)+step,\
                              ' over ',num_epochs*int(train_size / BATCH_SIZE))
                        print ('Epoch: ', iepoch, '   || Step',float(step))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (loss, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                  batch_labels))
                        sys.stdout.flush()

            return num_epochs






def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
      user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
    """Train a word2vec model."""
    #if not FLAGS.train_data__ or not FLAGS.eval_data or not FLAGS.save_path:
    #  print("--train_data --eval_data and --save_path must be specified.")
    #  sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = Sat_(opts, session)
        #for _ in xrange(opts.epochs_to_train):
        model.train()  # Process one epoch
        #model.eval()  # Eval analogies.
        # Perform a final save.
        model.saver.save(session,
                       os.path.join(opts.save_path, "model.ckpt"),
                       global_step=model.global_step)
        if FLAGS.interactive:
        # E.g.,
        # [0]: model.analogy(b'france', b'paris', b'russia')
        # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
            _start_shell(locals())

if __name__ == "__main__":
    tf.app.run()
