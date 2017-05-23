#!/usr/bin/env python
"""Create BLSTM networks with various layers using TensorFlow"""
########################################################################
# File: networks.py
#  executable: network.py
#
# Author: Andrew Bailey
# History: 5/20/17 Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
import numpy as np
from utils import project_folder
import tensorflow as tf
from tensorflow.contrib import rnn


def blstm(x, weights, biases, n_steps=1, n_hidden=128, forget_bias=1.0):
    """Create a bidirectional LSTM using code from the example at https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py"""
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=forget_bias)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=forget_bias)
    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']




def main():
    """Main docstring"""
    start = timer()
    training = np.load(project_folder()+"/testing.npy")
    classes = len(training[:,1][0])
    # 1024
    labels = training[:,1]
    training = training[:,0]
    # Parameters
    learning_rate = 0.001
    training_iters = 100000
    batch_size = 128
    display_step = 10

    # Network Parameters
    # n_input = 28 # MNIST data input (img shape: 28*28)
    n_input = 4
    # n_steps = 28 # timesteps
    n_steps = 1 # one vector per timestep
    n_hidden = 128 # hidden layer num of features
    # n_classes = 10 # MNIST total classes (0-9 digits)
    n_classes = 1024 # MNIST total classes (0-9 digits)

    training2 = np.asarray([np.asarray(training[x]) for x in range(len(training))])
    batch1_x = training2[:128]

    batch1_x = batch1_x.reshape((batch_size, n_steps, n_input))

    labels = np.asarray([np.asarray(labels[x]) for x in range(len(labels))])
    batch1_y = labels[:128]

    # tf Graph input
    # x = tf.placeholder("float", [None, n_input])
    x = tf.placeholder("float", [None, n_steps, n_input])

    y = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = blstm(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
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
                acc = sess.run(accuracy, feed_dict={x: batch1_x, y: batch1_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch1_x, y: batch1_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        # Calculate accuracy for 128 mnist test images
        test_len = 128
        print("Testing Accuracy: {}".format(sess.run(accuracy, \
        feed_dict={x: batch1_x, y: batch1_y})))

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
