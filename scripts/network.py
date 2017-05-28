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
import os
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
from utils import project_folder
import tensorflow as tf
from tensorflow.contrib import rnn


class BuildGraph():
    """Build a tensorflow network graph."""
    #TODO make it possible to change the prediciton function, cost and optimizer
    #   by creating functions for each of those
    def __init__(self, n_steps, n_input, n_classes, learning_rate, n_layers=1,\
    layer_sizes=tuple([128])):
        self.x = tf.placeholder("float", [None, n_steps, n_input], name='x')
        self.y = tf.placeholder("float", [None, n_classes], name='y')
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Define weights and bias of last layer
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([2*layer_sizes[-1], n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        self.lstm_layer1 = tf.placeholder("float", [None, layer_sizes[0]], name="layer1")
        self.lstm_layer2 = tf.placeholder("float", [None, layer_sizes[0]], name="layer2")
        self.lstm_layer3 = tf.placeholder("float", [None, layer_sizes[0]], name="layer3")
        self.lstm_layer4 = tf.placeholder("float", [None, layer_sizes[0]], name="layer4")
        self.seq_len = tf.placeholder(tf.int32, [None], name="batch_size")

        forward_state = rnn.LSTMStateTuple(self.lstm_layer1, self.lstm_layer2)
        backward_state = rnn.LSTMStateTuple(self.lstm_layer3, self.lstm_layer4)
        # print(backward_state)
        # print(backward_state.c)
        def fulconn_layer(input_data, output_dim, activation_func=None):
            input_dim = int(input_data.get_shape()[1])
            W = tf.Variable(tf.random_normal([input_dim, output_dim]))
            b = tf.Variable(tf.random_normal([output_dim]))
            if activation_func:
                return activation_func(tf.matmul(input_data, W) + b)
            else:
                return tf.matmul(input_data, W) + b
        # clear_state = tf.group(
            # tf.assign(forward_state, tf.zeros([layer_sizes[0]])),
            # tf.assign(backward_state, tf.zeros([layer_sizes[0]])))

        # outputs = self.create_graph(n_layers, forward_state, backward_state, layer_sizes=layer_sizes, n_steps=n_steps)
        # input1 = tf.unstack(self.x, n_steps, 1)
        print("self.x.shape", self.x.shape)
        print("self.y.shape", self.y.shape)

        # print("input1.shape", input1.shape)
        outputs = self.blstm(self.x, forward_state, backward_state, layer_name="layer1", n_hidden=layer_sizes[0], forget_bias=5.0)
        print(outputs)
        # print("outputs[0]", outputs[0])
        print("outputs[:,0:]", outputs[:,0,:])
        last_output = outputs[:,0,:]
        # yhat = fulconn_layer(last_output, n_classes)

        # Linear activation, using rnn inner loop last output
        with tf.name_scope("predition"):
            pred = tf.matmul(outputs[:,0,:], weights['out']) + biases['out']
            print("pred shape", pred.shape)
            self.variable_summaries(pred)

        # Define loss and optimizer
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,\
             labels=self.y))
            self.variable_summaries(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost, \
         global_step=global_step)

        # Evaluate model
        with tf.name_scope("correct_pred"):
            self.correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
            # self.variable_summaries(self.correct_pred)
        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.variable_summaries(self.accuracy)

        self.merged_summaries = tf.summary.merge_all()
        # Initializing the variables
        self.init = tf.global_variables_initializer()



    def create_graph(self, num_layers, forward_state, backward_state, layer_sizes=tuple([128]), n_steps=1):
        """Create a graph using a functions for different layers"""
        # TODO create a graph with variable number of nodes in each blstm layer
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        assert len(layer_sizes) == num_layers
        input1 = tf.unstack(self.x, n_steps, 1)
        for number in range(num_layers):
            input1 = self.blstm(input1, forward_state, backward_state, layer_name="layer"+str(number),\
            n_hidden=layer_sizes[number])
        return input1

    def blstm(self, input_vector, forward_state, backward_state, layer_name="layer1", n_hidden=128, forget_bias=5.0):
        """Create a bidirectional LSTM using code from the example at
         https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py"""
        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
        # print(lstm_bw_cell.state_size)
        # try:
        with tf.variable_scope(layer_name):
            outputs, self.output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, \
            input_vector, dtype=tf.float32, initial_state_fw=forward_state, initial_state_bw=backward_state)
            print("output_states = ", self.output_states[0])
            print("outputs = ", outputs)

        # print(f.state_size)
        # print(f.shape)
        output = tf.concat(outputs, 2)
        return output

    @staticmethod
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
    print(label_len)
    feature_len = len(features[0])
    # convert the inputs into numpy arrays
    features2 = np.asarray([np.asarray(features[x]) for x in range(len(features))])
    labels2 = np.asarray([np.asarray(labels[x]) for x in range(len(labels))])

    # TODO make hyperparameters a json file
    # Parameters
    learning_rate = 0.001
    training_iters = 10000
    batch_size = 100
    display_step = 10
    # Network Parameters
    n_input = feature_len
    n_steps = 1 # one vector per timestep
    layer_sizes = tuple([100]) # hidden layer num of features
    n_classes = label_len
    n_layers = 1

    # TODO build factoring by batch size into a method
    # Should I make a data set class?
    batch1_x = features2[:batch_size]
    # batch1_x = batch1_x.reshape((batch_size, n_steps, n_input))
    batch1_y = labels2[:batch_size]
    print(batch1_y.shape)
    batch1_x = batch1_x.reshape((batch_size, n_steps, n_input))
    # batch1_y = batch1_y.reshape((batch_size, n_steps, n_classes))
    # train_seq_len = np.ones(batch_size) * seq_length

    model = BuildGraph(n_steps, n_input, n_classes, learning_rate, n_layers, layer_sizes)
    x = model.x
    y = model.y
    cost = model.cost
    accuracy = model.accuracy
    merged_summaries = model.merged_summaries
    optimizer = model.optimizer
    # backward_state = model.backward_state
    # forward_state = model.forward_state

    # forward_state1 = rnn.LSTMStateTuple(tf.Variable(tf.zeros([128, layer_sizes[0]])), tf.Variable(tf.zeros([128, layer_sizes[0]])))
    # backward_state1 = rnn.LSTMStateTuple(tf.Variable(tf.zeros([128, layer_sizes[0]])), tf.Variable(tf.zeros([128, layer_sizes[0]])))
    init = model.init
    forward = model.output_states[0]
    backward = model.output_states[1]
    print(backward, forward)
    # Launch the graph
    with tf.Session() as sess:
        # TODO make directory for each new run
        logfolder_path = os.path.join(project_folder(), 'logs/', datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
        test_writer = tf.summary.FileWriter((logfolder_path), sess.graph)
        sess.run(init)
        step = 1
        lstm_layer1 = np.zeros([batch_size, layer_sizes[0]])
        lstm_layer2 = np.zeros([batch_size, layer_sizes[0]])
        lstm_layer3 = np.zeros([batch_size, layer_sizes[0]])
        lstm_layer4 = np.zeros([batch_size, layer_sizes[0]])
        print(type(lstm_layer1), lstm_layer1)
        print(type(model.lstm_layer3), model.lstm_layer3)
        print(type(batch1_x), batch1_x)
        # print("Batch shape = " , batch1_x.shape)
        print("Batch shape input= " , batch1_x.shape)
        print("lstm_layer size = ", lstm_layer1.shape)
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            # Run optimization op (backprop)
            # print(model.lstm_layer1)
            print(lstm_layer1[0][0])
            _, forward1, backward1 = sess.run([optimizer, forward, backward], feed_dict={x: batch1_x, y: batch1_y, model.lstm_layer1: lstm_layer1, model.lstm_layer2: lstm_layer2, model.lstm_layer3: lstm_layer3, model.lstm_layer4: lstm_layer4})
            # sess.run(optimizer, feed_dict={x: batch1_x, y: batch1_y})
            lstm_layer1, lstm_layer2 = forward1
            lstm_layer3, lstm_layer4 = backward1
            print(lstm_layer1[0][0])
            # print(backward)
            # print(model.lstm_layer1)
            # print(lstm_layer1)
            # sess.run([optimizer, model.assign_state], feed_dict={x: batch1_x, y: batch1_y})
            # print("WEGOTHERE!!!")
            #
            # if next_read:
            #     sess.run(clear_state)

            if step % display_step == 0:
                # Calculate batch accuracy
                run_metadata = tf.RunMetadata()
                acc, summary = sess.run([accuracy, merged_summaries], \
                feed_dict={x: batch1_x, y: batch1_y, model.lstm_layer1: lstm_layer1, model.lstm_layer2: lstm_layer2, model.lstm_layer3: lstm_layer3, model.lstm_layer4: lstm_layer4}, run_metadata=run_metadata)

                test_writer.add_summary(summary, step)
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch1_x, y: batch1_y, model.lstm_layer1: lstm_layer1, model.lstm_layer2: lstm_layer2, model.lstm_layer3: lstm_layer3, model.lstm_layer4: lstm_layer4})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        # Calculate accuracy for 128 mnist test images
        # test_len = 128
        lstm_layer1 = np.zeros([batch_size, layer_sizes[0]])
        lstm_layer2 = np.zeros([batch_size, layer_sizes[0]])
        lstm_layer3 = np.zeros([batch_size, layer_sizes[0]])
        lstm_layer4 = np.zeros([batch_size, layer_sizes[0]])

        print("Testing Accuracy: {}".format(sess.run(accuracy, \
        feed_dict={x: batch1_x, y: batch1_y,model.lstm_layer1: lstm_layer1, model.lstm_layer2: lstm_layer2, model.lstm_layer3: lstm_layer3, model.lstm_layer4: lstm_layer4})))
        test_writer.close()
    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
