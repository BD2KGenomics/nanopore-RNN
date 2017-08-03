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
from nanotensor.utils import project_folder, list_dir, load_json, save_json
from nanotensor.data import DataQueue
import tensorflow as tf
from tensorflow.contrib import rnn


class BuildGraph():
    """Build a tensorflow network graph."""
    def __init__(self, n_input, n_classes, learning_rate, n_steps=1,\
                forget_bias=5.0, y=None, x=None, network=None, binary_cost=True):
        # self.x = x
        self.y = tf.placeholder_with_default(y, shape=[None, n_steps, n_classes])
        self.x = tf.placeholder_with_default(x, shape=[None, n_steps, n_input])

        self.batch_size = tf.shape(self.x)[0]

        # self.batch_size = tf.shape(self.y)[0]

        # self.batch_size = batch_size
        self.y_flat = tf.reshape(self.y, [-1, n_classes])

        self.n_input = n_input
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.forget_bias = forget_bias
        self.output_states = []
        self.layers = []
        # list of operations to reset states of each blstm layer
        # self.zero_states = []
        # self.reset_fws = []
        # self.reset_bws = []
        # Summay Information
        self.training_summaries = []
        self.testing_summaries = []
        assert network != None, "Must specify network structure. [{'type': 'blstm', 'name': 'blstm_layer1', 'size': 128}, ...]"
        self.network = network
        outputs, final_layer_size = self.create_model(self.network)
        self.rnn_outputs_flat = tf.reshape(outputs, [-1, final_layer_size])

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.zero_state = self.combine_arguments(self.zero_states, "zero_states")
        # self.fw_reset = self.combine_arguments(self.reset_fws, "reset_fws")
        # self.bw_reset = self.combine_arguments(self.reset_bws, "reset_bws")


        # Linear activation, using rnn inner loop last output
        self.pred = self.create_prediction_layer()
        tf.add_to_collection("predicton", self.pred)
        self.evaluate_pred = tf.argmax(self.pred, 1, name="evaluate_pred")
        # Define loss and optimizer
        if binary_cost:
            self.cost = self.cost_function_binary()
        else:
            self.cost = self.cost_function_prob()

        self.optimizer = self.optimizer_function()
        # Evaluate model
        tf.add_to_collection("optimizer", self.optimizer)

        self.correct_pred = self.prediction_function()
        self.accuracy = self.accuracy_function()
        # merge summary information

        self.test_summary = tf.summary.merge(self.testing_summaries)
        self.train_summary = tf.summary.merge(self.training_summaries)



    def create_model(self, network_model=None):
        """Create a model from a list of dictironaries with "name", "type", and "size" keys"""
        assert network_model != None, "Must specify network structure. [{'type': 'blstm', 'name': 'blstm_layer1', 'size': 128}, ...]"
        ref_types = {"tanh": tf.tanh, "relu":tf.nn.relu, "sigmoid":tf.sigmoid,\
                    "softplus":tf.nn.softplus, "none": None}
        input_vector = self.x
        prevlayer_size = 0
        for layer in network_model:
            if layer["type"] == "blstm":
                input_vector = self.blstm(input_vector=input_vector, n_hidden=layer["size"], \
                                layer_name=layer["name"], forget_bias=layer["bias"])
                prevlayer_size = layer["size"]*2
            else:
                # reshape matrix to fit into a single activation function
                input_vector = tf.reshape(input_vector, [-1, prevlayer_size*self.n_steps])
                input_vector = self.fulconn_layer(input_data=input_vector, output_dim=layer["size"],\
                                seq_len=self.n_steps, activation_func=ref_types[layer["type"]])
                # reshape matrix to correct shape from output of
                input_vector = tf.reshape(input_vector, [-1, self.n_steps, layer["size"]])
                prevlayer_size = layer["size"]

        return input_vector, prevlayer_size


    def create_prediction_layer(self):
        """Create a prediction layer from output of blstm layers"""
        with tf.name_scope("prediction"):
            pred = self.fulconn_layer(self.rnn_outputs_flat, self.n_classes)
            print("pred shape = ", pred.get_shape())
        return pred


    def prediction_function(self):
        """Compare predicions with label to calculate number correct"""
        with tf.name_scope("correct_pred"):
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_flat, 1))
        return correct_pred

    def cost_function_prob(self):
        """Create a cost function for optimizer"""
        with tf.name_scope("cost"):
            loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.pred,\
                labels=self.y_flat)
            loss = tf.reshape(loss1, [self.batch_size, self.n_steps])

            cost = tf.reduce_mean(loss)
            self.variable_summaries(cost)
        return cost

    def cost_function_binary(self):
        """Create a cost function for optimizer"""
        with tf.name_scope("cost"):
            y_label_indices = tf.argmax(self.y_flat, 1, name="y_label_indices")
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred,\
                labels=y_label_indices)
            loss = tf.reshape(loss1, [self.batch_size, self.n_steps])

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

    @staticmethod
    def combine_arguments(tf_tensor_list, name):
        """Create a single operation that can be passed to the run function"""
        return tf.tuple(tf_tensor_list, name=name)

    @staticmethod
    def get_state_update_op(state_variables, new_states):
        """Update the state with new values"""
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                               state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return update_ops

    @staticmethod
    def blstm(input_vector, layer_name="blstm_layer1", n_hidden=128, forget_bias=5.0):
        """Create a bidirectional LSTM using code from the example at
         https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py"""
        with tf.variable_scope(layer_name):
            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
            # Backward direction cell
            lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)

            # forward states
            # fw_state_c, fw_state_h = lstm_fw_cell.zero_state(self.batch_size, tf.float32)
            # lstm_fw_cell_states = rnn.LSTMStateTuple(
            #     tf.Variable(fw_state_c, trainable=False, name="forward_c"),
            #     tf.Variable(fw_state_h, trainable=False, name="forward_h"))
            #
            # # backward states
            # bw_state_c, bw_state_h = lstm_bw_cell.zero_state(self.batch_size, tf.float32)
            # lstm_bw_cell_states = rnn.LSTMStateTuple(
            #     tf.Variable(bw_state_c, trainable=False, name="backward_c"),
            #     tf.Variable(bw_state_h, trainable=False, name="backward_h"))

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, \
            lstm_bw_cell, input_vector, dtype=tf.float32)#, initial_state_fw=lstm_fw_cell_states, \
            #initial_state_bw=lstm_bw_cell_states)

            # create operations for resetting both states and only the forward or backward states
            # self.zero_states.extend(self.get_state_update_op((lstm_fw_cell_states, \
            #                         lstm_bw_cell_states), output_states))
            # self.reset_fws.extend(self.get_state_update_op(lstm_fw_cell_states, output_states[0]))
            # self.reset_bws.extend(self.get_state_update_op(lstm_bw_cell_states, output_states[1]))

            # concat two output layers so we can treat as single output layer
            output = tf.concat(outputs, 2)
        return output

    def variable_summaries(self, var):
        # pylint: disable=C0301
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        source: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
        """
        with tf.name_scope("testing"):
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                summary = tf.summary.scalar('mean', mean)
                self.testing_summaries.append(summary)
        with tf.name_scope("training"):
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                summary = tf.summary.scalar('mean', mean)
                self.training_summaries.append(summary)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))

    @staticmethod
    def fulconn_layer(input_data, output_dim, seq_len=1, activation_func=None):
        # pylint: disable=C0301
        """Create a fully connected layer.
        source: https://stackoverflow.com/questions/39808336/tensorflow-bidirectional-dynamic-rnn-none-values-error/40305673
        """
        # get input dimentions
        input_dim = int(input_data.get_shape()[1])
        weight = tf.Variable(tf.random_normal([input_dim, output_dim*seq_len]))
        bais = tf.Variable(tf.random_normal([output_dim*seq_len]))
        if activation_func:
            output = activation_func(tf.matmul(input_data, weight) + bais)
        else:
            output = tf.matmul(input_data, weight) + bais
        return output

def main():
    """Control the flow of the program"""
    # start = timer()
    #
    # #
    # # input1 = self.x
    # # for layer_size in range(self.n_layers):
    # #     input1 = self.blstm(input1, layer_name="blstm_layer"+str(layer_size),\
    # #     n_hidden=self.layer_sizes[layer_size], forget_bias=self.forget_bias)
    # # return input1
    #
    # # TODO make hyperparameters a json file
    # # Parameters
    # learning_rate = 0.001
    # training_iters = 100
    # batch_size = 2
    # queue_size = 10
    # display_step = 10
    # n_steps = 300 # one vector per timestep
    # layer_sizes = tuple([100]) # hidden layer num of features
    # training_dir = project_folder()+"/training2"
    # training_files = list_dir(training_dir, ext="npy")
    #
    # # continually load data on the CPU
    # with tf.device("/cpu:0"):
    #     data = DataQueue(training_files, batch_size, queue_size=queue_size, verbose=False, \
    #             pad=0, trim=True, n_steps=n_steps)
    #     images_batch, labels_batch = data.get_inputs()
    # # build model
    # model = BuildGraph(data.n_input, data.n_classes, learning_rate, n_steps=n_steps, \
    #         layer_sizes=layer_sizes, batch_size=batch_size, x=images_batch, y=labels_batch)
    # cost = model.cost
    # accuracy = model.accuracy
    # merged_summaries = model.merged_summaries
    # optimizer = model.optimizer
    # # define what we want from the optimizer run
    # init = tf.global_variables_initializer()
    #
    # saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
    # # Launch the graph
    # with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
    #     # create logs
    #     logfolder_path = os.path.join(project_folder(), 'logs/', \
    #                     datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
    #     writer = tf.summary.FileWriter((logfolder_path), sess.graph)
    #     # initialize
    #     sess.run(init)
    #     step = 1
    #     # start queue
    #     tf.train.start_queue_runners(sess=sess)
    #     data.start_threads(sess)
    #     # Keep training until reach max iterations
    #     while step * batch_size < training_iters:
    #         # Run optimization and update layers
    #         output_states = sess.run([optimizer, model.zero_state])
    #         if step % display_step == 0:
    #             # Calculate batch loss and accuracy
    #             run_metadata = tf.RunMetadata()
    #             acc, summary, loss = sess.run([accuracy, merged_summaries, cost], run_metadata=run_metadata)
    #             # add summary statistics
    #             writer.add_summary(summary, step)
    #             print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
    #                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
    #                   "{:.5f}".format(acc))
    #         step += 1
    #
    #     print("Optimization Finished!")
    #     # Calculate accuracy for a bunch of test data
    #
    #     saver.save(sess, project_folder()+'/testing/my_test_model.ckpt', global_step=model.global_step)
    #
    #     print("Testing Accuracy: {}".format(sess.run(accuracy)))
    #     writer.close()
    # #
    #
    # stop = timer()
    # print("Running Time = {} seconds".format(stop-start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
