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
import itertools
import numpy as np
from utils import project_folder, merge_two_dicts
import tensorflow as tf
from tensorflow.contrib import rnn

class BuildGraph():
    """Build a tensorflow network graph."""
    #TODO make it possible to change the prediciton function, cost and optimizer
    #   by creating functions for each of those
    def __init__(self, n_input, n_classes, learning_rate, n_steps=1,\
    layer_sizes=tuple([100]), forget_bias=5.0):
        self.x = tf.placeholder("float", [None, n_steps, n_input], name='x')
        self.y = tf.placeholder("float", [None, n_classes], name='y')
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.n_input = n_input
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.forget_bias = forget_bias
        # output states
        self.output_states = []
        # layer placeholders
        self.layers = []

        outputs = self.create_deep_blstm()
        # outputs = self.blstm(self.x, layer_name="layer1", n_hidden=layer_sizes[0], forget_bias=5.0)
        # outputs = self.blstm(outputs, layer_name="layer2", n_hidden=layer_sizes[0], forget_bias=5.0)


        self.last_output = outputs[:, 0, :]
        # Linear activation, using rnn inner loop last output
        self.pred = self.create_prediction_layer()
        # Define loss and optimizer
        self.cost = self.cost_function()
        self.optimizer = self.optimizer_function()
        # Evaluate model
        self.correct_pred = self.prediction_function()
        self.accuracy = self.accuracy_function()
        # merge summary information
        self.merged_summaries = tf.summary.merge_all()
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        # clear_state = tf.group(
        #     tf.assign(forward_state, tf.zeros([layer_sizes[0]])),
        #     tf.assign(backward_state, tf.zeros([layer_sizes[0]])))

    def create_prediction_layer(self):
        """Create a prediction layer from output of blstm layers"""
        with tf.name_scope("predition"):
            pred = self.fulconn_layer(self.last_output, self.n_classes)
            # print("pred shape", pred.shape)
            self.variable_summaries(pred)
        return pred

    def prediction_function(self):
        """Compare predicions with label to calculate number correct"""
        with tf.name_scope("correct_pred"):
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            # self.variable_summaries(correct_pred)
        return correct_pred

    def cost_function(self):
        """Create a cost function for optimizer"""
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred,\
             labels=self.y))
            self.variable_summaries(cost)
        return cost

    def optimizer_function(self):
        """Create optimizer function"""
        global_step = tf.Variable(0, name='global_step', trainable=False)
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, \
         global_step=global_step)

    def accuracy_function(self):
        """Create accuracy function to calculate accuracy of the prediction"""
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.variable_summaries(accuracy)
        return accuracy

    def create_deep_blstm(self):
        """Create a multi-layer blstm neural network"""
        # make sure we are making as many layers as we have layer sizes
        input1 = self.x
        for number in range(self.n_layers):
            input1 = self.blstm(input1, layer_name="blstm_layer"+str(number),\
            n_hidden=self.layer_sizes[number], forget_bias=self.forget_bias)
        return input1

    def blstm(self, input_vector, layer_name="blstm_layer1", n_hidden=128, forget_bias=5.0):
        """Create a bidirectional LSTM using code from the example at
         https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py"""
        with tf.variable_scope(layer_name):
            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
            # Backward direction cell
            lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
            # print(lstm_bw_cell.state_size)
            lstm_layer1 = tf.placeholder("float", [None, n_hidden], name="forward.C")
            lstm_layer2 = tf.placeholder("float", [None, n_hidden], name="forward.H")
            lstm_layer3 = tf.placeholder("float", [None, n_hidden], name="backward.C")
            lstm_layer4 = tf.placeholder("float", [None, n_hidden], name="backward.H")
            self.seq_len = tf.placeholder(tf.int32, [None], name="batch_size")
            self.layers.extend([lstm_layer1, lstm_layer2, lstm_layer3, lstm_layer4])

            forward_state = rnn.LSTMStateTuple(lstm_layer1, lstm_layer2)
            backward_state = rnn.LSTMStateTuple(lstm_layer3, lstm_layer4)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, \
            lstm_bw_cell, input_vector, dtype=tf.float32, initial_state_fw=forward_state, \
            initial_state_bw=backward_state)
            self.output_states.extend(output_states)
            # concat two output layers so we can treat as single output layer
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

    @staticmethod
    def fulconn_layer(input_data, output_dim, activation_func=None):
        """Create a fully connected layer.
        source: https://stackoverflow.com/questions/39808336/tensorflow-bidirectional-dynamic-rnn-none-values-error/40305673
        """
        input_dim = int(input_data.get_shape()[1])
        weight = tf.Variable(tf.random_normal([input_dim, output_dim]))
        bais = tf.Variable(tf.random_normal([output_dim]))
        if activation_func:
            output = activation_func(tf.matmul(input_data, weight) + bais)
        else:
            output = tf.matmul(input_data, weight) + bais
        return output

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

    batch1_x = features2[:batch_size]
    batch1_y = labels2[:batch_size]
    batch1_x = batch1_x.reshape((batch_size, n_steps, n_input))
    # train_seq_len = np.ones(batch_size) * seq_length

    model = BuildGraph(n_input, n_classes, learning_rate)
    x = model.x
    y = model.y
    cost = model.cost
    accuracy = model.accuracy
    merged_summaries = model.merged_summaries
    optimizer = model.optimizer
    init = model.init
    run_optimizer = [optimizer]+ model.output_states

    # Launch the graph
    with tf.Session() as sess:
        logfolder_path = os.path.join(project_folder(), 'logs/', datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
        test_writer = tf.summary.FileWriter((logfolder_path), sess.graph)
        sess.run(init)
        step = 1
        kwargs = dict(zip(model.layers, [np.zeros([batch_size, layer_sizes[0]])]*len(model.layers)))
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            # Run optimization op (backprop)
            inputs1 = {x: batch1_x, y: batch1_y}
            feed_dict = merge_two_dicts(kwargs, inputs1)

            layers = sess.run(run_optimizer, feed_dict=feed_dict)

            layers2 = list(itertools.chain.from_iterable(layers[1:]))
            kwargs = dict(zip(model.layers, layers2))
            print("layer1", layers2[0][0][0])
            print("layer2", layers2[1][0][0])

            # if next_read:
            #     sess.run(clear_state)

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                run_metadata = tf.RunMetadata()
                acc, summary, loss = sess.run([accuracy, merged_summaries, cost], \
                feed_dict=feed_dict, run_metadata=run_metadata)
                # add summary statistics
                test_writer.add_summary(summary, step)
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        # Calculate accuracy for 128 mnist test images
        # test_len = 128
        kwargs = dict(zip(model.layers, [np.zeros([batch_size, layer_sizes[0]])]*len(model.layers)))
        inputs1 = {x: batch1_x, y: batch1_y}
        feed_dict = merge_two_dicts(kwargs, inputs1)

        print("Testing Accuracy: {}".format(sess.run(accuracy, feed_dict=feed_dict)))
        test_writer.close()
    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
