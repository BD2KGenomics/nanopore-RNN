#!/usr/bin/env python
"""Create BLSTM networks with various layers using TensorFlow"""
########################################################################
# File: networks.py
#  executable: network.py
# separate model class with a bulid model function
# keep objects as class variables
# and a separate class for running the model
# use json for hyperparameters


# Author: Andrew Bailey
# History: 5/20/17 Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
import numpy as np
from datetime import datetime
from utils import project_folder
import tensorflow as tf
from tensorflow.contrib import rnn


class BuildGraph():
    """Build a tensorflow network graph."""
    def __init__(self, x):
        self.x = x

    def create_graph(self, num_layers, n_steps=1):
        """Create a graph using a functions for different layers"""
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        input1 = tf.unstack(self.x, n_steps, 1)
        for number in range(1, num_layers+1):
            input1 = self.blstm(input1, layer_name="layer"+str(number))
        return input1

    def blstm(self, input_vector, layer_name="layer1", n_hidden=128, forget_bias=5.0):
        """Create a bidirectional LSTM using code from the example at https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py"""
        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias)
        # Backward direction cell
        lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias)
        # try:
        with tf.variable_scope(layer_name):
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, \
            input_vector, dtype=tf.float32)
        # except Exception: # Old TensorFlow version only returns outputs not states
        #     with tf.variable_scope(layer_name):
        #         outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, \
        # x, dtype=tf.float32)
        return outputs

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    source: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def main():
    """Control the flow of the program"""
    start = timer()
    # load data
    training = np.load(project_folder()+"/testing.npy")
    # grab labels
    labels = training[:, 1]
    # grab feature vectors
    features = training[:, 0]
    # find the length of the label vector
    label_len = len(labels[0])
    feature_len = len(features[0])
    # convert the inputs into numpy arrays
    features2 = np.asarray([np.asarray(features[x]) for x in range(len(features))])
    labels2 = np.asarray([np.asarray(labels[x]) for x in range(len(labels))])


    # Parameters
    learning_rate = 0.001
    training_iters = 10000
    batch_size = 100
    display_step = 10
    # Network Parameters
    n_input = feature_len
    n_steps = 1 # one vector per timestep
    n_hidden = 128 # hidden layer num of features
    n_classes = label_len


    # TODO build factoring by batch size into a method
    # Should I make a data set class?
    batch1_x = features2[:batch_size]
    # batch1_x = batch1_x.reshape((batch_size, n_steps, n_input))
    batch1_y = labels2[:batch_size]
    batch1_x = batch1_x.reshape((batch_size, n_steps, n_input))


    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input], name='x')
    y = tf.placeholder("float", [None, n_classes], name='y')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Define weights
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    graph = BuildGraph(x)
    outputs = graph.create_graph(4)
    # Linear activation, using rnn inner loop last output
    with tf.name_scope("predition"):
        pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
        variable_summaries(pred)

    # Define loss and optimizer
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        variable_summaries(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, \
     global_step=global_step)

    # Evaluate model
    with tf.name_scope("correct_pred"):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        variable_summaries(cost)
    with tf.name_scope("correct_pred"):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        variable_summaries(accuracy)

    merged_summaries = tf.summary.merge_all()
    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        test_writer = tf.summary.FileWriter('/Users/andrewbailey/nanopore-RNN/logs'+\
        datetime.now().strftime("%m%b-%d-%Hh-%Mm"), sess.graph)
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch1_x, y: batch1_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                run_metadata = tf.RunMetadata()
                acc, summary = sess.run([accuracy, merged_summaries], \
                feed_dict={x: batch1_x, y: batch1_y}, run_metadata=run_metadata)

                test_writer.add_summary(summary, step)
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch1_x, y: batch1_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        # Calculate accuracy for 128 mnist test images
        # test_len = 128
        print("Testing Accuracy: {}".format(sess.run(accuracy, \
        feed_dict={x: batch1_x, y: batch1_y})))
        test_writer.close()
    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
