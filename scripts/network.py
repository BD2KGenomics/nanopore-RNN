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
import traceback
import numpy as np
from utils import project_folder, merge_two_dicts, list_dir
from data import Data
import tensorflow as tf
from tensorflow.contrib import rnn


class BuildGraph():
    """Build a tensorflow network graph."""
    def __init__(self, n_input, n_classes, learning_rate, n_steps=1,\
        layer_sizes=tuple([100]), forget_bias=5.0, batch_size=100, y=None, x=None):
        # TODO get variable batch size

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.y_flat = tf.reshape(self.y, [-1, n_classes])
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.n_input = n_input
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.forget_bias = forget_bias
        self.output_states = []
        self.layers = []
        # self.seq_len = tf.placeholder(tf.int32, [None], name="batch_size")
        # self.batch_size = tf.Variable()
        # outputs = self.create_deep_blstm()
        outputs = self.blstm(self.x, layer_name="layer1", n_hidden=layer_sizes[0], forget_bias=self.forget_bias)

        # clear_state = tf.group(
        #     tf.assign(forward_state, tf.zeros([layer_sizes[0]])),
        #     tf.assign(backward_state, tf.zeros([layer_sizes[0]])))
        self.rnn_outputs_flat = tf.reshape(outputs, [-1, 2*layer_sizes[-1]])
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # Linear activation, using rnn inner loop last output
        self.pred = self.create_prediction_layer()
        # Define loss and optimizer
        self.cost = self.cost_function_prob()
        self.optimizer = self.optimizer_function()
        # Evaluate model
        self.correct_pred = self.prediction_function()
        self.accuracy = self.accuracy_function()
        # merge summary information
        self.merged_summaries = tf.summary.merge_all()
        # Initializing the variables
        self.init = tf.global_variables_initializer()

    def create_prediction_layer(self):
        """Create a prediction layer from output of blstm layers"""
        with tf.name_scope("predition"):
            pred = self.fulconn_layer(self.rnn_outputs_flat, self.n_classes)
            # print("pred shape", pred.shape)
            self.variable_summaries(pred)
        return pred

    def prediction_function(self):
        """Compare predicions with label to calculate number correct"""
        with tf.name_scope("correct_pred"):
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_flat, 1))
            # self.variable_summaries(correct_pred)
        return correct_pred

    def cost_function_prob(self):
        """Create a cost function for optimizer"""
        with tf.name_scope("cost"):
            loss = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred,\
                labels=self.y_flat), [self.batch_size, self.n_steps])
            cost = tf.reduce_mean(loss)
            self.variable_summaries(cost)
        return cost

    def cost_function_binary(self):
        """Create a cost function for optimizer"""
        with tf.name_scope("cost"):
            loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred,\
                labels=self.y_flat), [self.batch_size, self.n_steps])
            cost = tf.reduce_mean(loss)
            self.variable_summaries(cost)
        return cost

    def cost_function_with_mask(self):
        """Create a cost function for optimizer"""
        # TODO create a cost function with a defined mask for padded input sequences
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred,\
                labels=self.y_flat))
            # mask = tf.sign(tf.to_float(self.y_flat))
            # masked_losses = mask * losses
            self.variable_summaries(cost)
        return cost

    def optimizer_function(self):
        """Create optimizer function"""
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, \
         global_step=self.global_step)

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

    def get_state_update_op(self, state_variables, new_states):
        """Update the state with new values"""
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                               state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)

    def blstm(self, input_vector, layer_name="blstm_layer1", n_hidden=128, forget_bias=5.0):
        """Create a bidirectional LSTM using code from the example at
         https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py"""
        with tf.variable_scope(layer_name):
            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
            # Backward direction cell
            lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)

            # forward states
            fw_state_c, fw_state_h = lstm_fw_cell.zero_state(self.batch_size, tf.float32)
            lstm_fw_cell_states = rnn.LSTMStateTuple(
                tf.Variable(fw_state_c, trainable=False, name="forward_c"),
                tf.Variable(fw_state_h, trainable=False, name="forward_h"))

            # backward states
            bw_state_c, bw_state_h = lstm_bw_cell.zero_state(self.batch_size, tf.float32)
            lstm_bw_cell_states = rnn.LSTMStateTuple(
                tf.Variable(bw_state_c, trainable=False, name="backward_c"),
                tf.Variable(bw_state_h, trainable=False, name="backward_h"))

            # self.reset_fw = self.get_state_update_op(lstm_fw_cell_states, (fw_state_c, fw_state_h))
            # self.reset_bw = self.get_state_update_op(lstm_bw_cell_states, (bw_state_c, bw_state_h))

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, \
            lstm_bw_cell, input_vector, dtype=tf.float32, initial_state_fw=lstm_fw_cell_states, \
            initial_state_bw=lstm_bw_cell_states)

            self.update_op = self.get_state_update_op((lstm_fw_cell_states, lstm_bw_cell_states), output_states)

            # concat two output layers so we can treat as single output layer
            output = tf.concat(outputs, 2)
        return output

    @staticmethod
    def variable_summaries(var):
        # pylint: disable=C0301
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
        # pylint: disable=C0301
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
    # if "CUDA_HOME" in os.environ:
    #     utilization = re.findall(r"Utilization.*?Gpu.*?(\d+).*?Memory.*?(\d+)",
    #                              subprocess.check_output(["nvidia-smi", "-q"]),
    #                              flags=re.MULTILINE | re.DOTALL)
    #     print("GPU Utilization", utilization)
    #
    #     if ('0', '0') in utilization:
    #         print("Using GPU Device:", utilization.index(('0', '0')))
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(utilization.index(('0', '0')))
    #         os.environ["CUDA_DEVICE_ORDER"]  = "PCI_BUS_ID"  # To ensure the index matches
    #     else:
    #         print("All GPUs in Use")
    #         exit
    # else:
    #     print("Running using CPU, NOT GPU")

    # TODO make hyperparameters a json file
    # Parameters
    learning_rate = 0.001
    training_iters = 100
    batch_size = 2
    queue_size = 10
    display_step = 10
    n_steps = 300 # one vector per timestep
    layer_sizes = tuple([100]) # hidden layer num of features
    training_dir = project_folder()+"/training2"
    training_files = list_dir(training_dir, ext="npy")

    # continually load data on the CPU
    with tf.device("/cpu:0"):
        data = Data(training_files, batch_size, queue_size=queue_size, verbose=False, \
                pad=0, trim=True, n_steps=n_steps)
        images_batch, labels_batch = data.get_inputs()

    # build model
    model = BuildGraph(data.n_input, data.n_classes, learning_rate, n_steps=n_steps, \
            layer_sizes=layer_sizes, batch_size=batch_size, x=images_batch, y=labels_batch)
    cost = model.cost
    accuracy = model.accuracy
    merged_summaries = model.merged_summaries
    optimizer = model.optimizer
    # define what we want from the optimizer run
    init = model.init
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
    # Launch the graph
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
        # create logs
        logfolder_path = os.path.join(project_folder(), 'logs/', \
                        datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
        writer = tf.summary.FileWriter((logfolder_path), sess.graph)
        # initialize
        sess.run(init)
        step = 1
        # start queue
        tf.train.start_queue_runners(sess=sess)
        data.start_threads(sess)
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            # Run optimization and update layers
            output_states = sess.run([optimizer, model.update_op])
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                run_metadata = tf.RunMetadata()
                acc, summary, loss = sess.run([accuracy, merged_summaries, cost], run_metadata=run_metadata)
                # add summary statistics
                writer.add_summary(summary, step)
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1

        print("Optimization Finished!")
        # Calculate accuracy for a bunch of test data

        saver.save(sess, project_folder()+'/testing/my_test_model.ckpt', global_step=model.global_step)

        print("Testing Accuracy: {}".format(sess.run(accuracy)))
        writer.close()




    with tf.Session() as sess:
        # To initialize values with saved data
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        data.start_threads(sess)

        saver.restore(sess, '/Users/andrewbailey/nanopore-RNN/testing/my_test_model.ckpt-49')
        # new_saver.restore(sess, tf.train.latest_checkpoint('/Users/andrewbailey/nanopore-RNN/testing/'))
        print(sess.run(accuracy)) # returns 1000



    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
