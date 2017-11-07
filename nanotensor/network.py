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
from nanotensor.queue import DataQueue
from nanotensor.error import Usage
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class BuildGraph:
    """Build a tensorflow network graph."""

    def __init__(self, y=None, x=None, seq_len=None, network=None, n_input=0, n_classes=0, learning_rate=0, n_steps=0,
                 forget_bias=5.0, cost_function="cross_entropy", reuse=None, optimizer=True):
        # initialize placeholders or take in tensors from a queue or Dataset object
        self.sequence_length = tf.placeholder_with_default(seq_len, shape=[None])
        self.cost_function = cost_function
        if self.cost_function == "ctc_loss":
            self.n_classes = n_classes+1
            self.x = tf.placeholder_with_default(x, shape=[None, n_steps])
            self.x = tf.reshape(self.x, shape=[-1, n_steps, 1])
            if y:
                self.y = tf.SparseTensor(y[0], y[1], y[2])
            else:
                y_indexs = tf.placeholder(tf.int64)
                y_values = tf.placeholder(tf.int32)
                y_shape = tf.placeholder(tf.int64)
                self.y = tf.SparseTensor(y_indexs, y_values, y_shape)
        else:
            self.x = tf.placeholder_with_default(x, shape=[None, n_steps, n_input])
            self.y = tf.placeholder_with_default(y, shape=[None, n_steps, n_classes])
            self.y_flat = tf.reshape(self.y, [-1, n_classes])
            self.n_classes = n_classes
        # bool for multi-gpu usage
        self.reuse = reuse
        # length of input tensor, step number and number of classes for output layer
        self.n_input = n_input
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.batch_size = tf.shape(self.x)[0]
        self.forget_bias = forget_bias

        # TODO if retrain grab old global step
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Summary Information
        self.training_summaries = []
        self.testing_summaries = []
        assert network is not None, "Must specify network structure. [{'type': 'blstm', 'name': 'blstm_layer1', " \
                                    "'size': 128}, ...] "
        # self.network = network
        output, final_layer_size = self.create_model(network)
        self.output_shape = output.get_shape().as_list()
        flat_outputs = tf.reshape(output, [-1, final_layer_size])
        # Linear activation, using rnn inner loop last output
        with tf.name_scope("output_layer"):
            self.output_layer, _, _ = self.fulconn_layer(flat_outputs, self.n_classes)
            if self.cost_function == 'ctc_loss':
                self.output_layer = tf.reshape(self.output_layer, shape=[-1, self.output_shape[1], self.n_classes])
            tf.add_to_collection("output_layer", self.output_layer)

        # create prediction and accuracy stats
        self.correct_pred = self.prediction_function()
        self.accuracy = self.accuracy_function()

        if not self.cost_function == "ctc_loss":
            self.evaluate_pred = tf.argmax(self.output_layer, 1, name="evaluate_pred")

        # Define loss and optimizer
        self.cost = self.create_cost_function()
        if optimizer:
            self.optimizer = self.optimizer_function()
        # # Evaluate model
        # merge summary information
        self.test_summary = tf.summary.merge(self.testing_summaries)
        self.train_summary = tf.summary.merge(self.training_summaries)
        # list of operations to reset states of each blstm layer

    def create_model(self, network_model=None):
        """Create a model from a list of dictionaries with "name", "type", and "size" keys"""
        assert network_model is not None, "Must specify network structure. [{'type': 'blstm', 'name': 'blstm_layer1', " \
                                          "'size': 128}, ...] "
        ref_types = {"tanh": tf.tanh, "relu": tf.nn.relu, "sigmoid": tf.sigmoid,
                     "softplus": tf.nn.softplus, "none": None, }
        input_vector = self.x
        prevlayer_size = 0
        for layer in network_model:
            inshape = input_vector.get_shape().as_list()
            if layer["type"] == "blstm":
                ratio = self.n_steps / inshape[1]
                # print("ratio = {}".format(ratio))
                input_vector = self.blstm(input_vector=input_vector, n_hidden=layer["size"],
                                          layer_name=layer["name"], forget_bias=layer["bias"], reuse=self.reuse,
                                          concat=layer["concat"], sequence_length=self.sequence_length/ratio)
                prevlayer_size = layer["size"] * 2
            elif layer["type"] == "residual_layer":
                with tf.variable_scope(layer["name"]):
                    if len(inshape) != 4:
                        input_vector = tf.reshape(input_vector, [tf.shape(input_vector)[0], 1,
                                                                 inshape[1], inshape[2]])
                    input_vector = residual_layer(input_vector,
                                                  out_channel=layer["out_channel"],
                                                  i_bn=layer["batchnorm"])
                    inshape = input_vector.get_shape().as_list()
                    input_vector = tf.reshape(input_vector, [tf.shape(input_vector)[0], inshape[2],
                                                             inshape[3]], name='residual_features')
            elif layer["type"] == "chiron_fnn":
                with tf.variable_scope(layer["name"]):
                    ### split blstm output and put fully con layer on each
                    lasth_rs = tf.reshape(input_vector, [tf.shape(input_vector)[0], inshape[1], 2, inshape[2] / 2],
                                          name='lasth_rs')
                    weight_out = tf.get_variable(name="weights", shape=[2, inshape[2] / 2],
                                                 initializer=tf.truncated_normal_initializer(
                                                     stddev=np.sqrt(2.0 / (inshape[2]))))
                    biases_out = tf.get_variable(name="bias", shape=[inshape[2] / 2], initializer=tf.zeros_initializer)
                    input_vector = tf.nn.bias_add(tf.reduce_sum(tf.multiply(lasth_rs, weight_out), axis=2), biases_out,
                                                  name='lasth_bias_add')
                    prevlayer_size = inshape[2] / 2
            else:
                with tf.variable_scope(layer["name"]):
                    # reshape matrix to fit into a single activation function
                    input_vector = tf.reshape(input_vector, [-1, prevlayer_size * self.n_steps])
                    input_vector = self.fulconn_layer(input_data=input_vector, output_dim=layer["size"],
                                                      seq_len=self.n_steps, activation_func=ref_types[layer["type"]])[0]
                    # reshape matrix to correct shape from output of
                    input_vector = tf.reshape(input_vector, [-1, self.n_steps, layer["size"]])
                    prevlayer_size = layer["size"]

        return input_vector, prevlayer_size

    def create_cost_function(self):
        """Create cost function depending on cost function options"""
        with tf.name_scope("cost"):
            if self.cost_function == "cross_entropy":
                # standard cross entropy implementation
                loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer,
                                                                labels=self.y_flat)
            elif self.cost_function == "binary_cross_entropy":
                # cross entropy with binary labels.
                y_label_indices = tf.argmax(self.y_flat, 1, name="y_label_indices")
                loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output_layer,
                                                                       labels=y_label_indices)
            elif self.cost_function == "ctc_loss":
                loss1 = tf.nn.ctc_loss(inputs=self.output_layer, labels=self.y,
                                       sequence_length=self.sequence_length, ctc_merge_repeated=True,
                                       time_major=False)
            else:
                raise Usage("Cost function in config file is not one of the options: "
                            "[ctc_loss, binary_cross_entropy, cross_entropy]")
            cost = tf.reduce_mean(loss1)
            self.variable_summaries(cost)
        return cost

    def optimizer_function(self):
        """Create optimizer function"""
        # opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.gradients = opt.compute_gradients(self.cost)
        # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
        #                                            100000, 0.96, staircase=True)
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,
                                                                                 global_step=self.global_step)

    def prediction_function(self):
        """Compare predicions with label to calculate number correct or edit distance"""
        with tf.name_scope("prediction"):
            if self.cost_function == 'ctc_loss':
                logits = tf.transpose(self.output_layer, perm=[1, 0, 2])
                predict = tf.nn.ctc_greedy_decoder(logits, self.sequence_length, merge_repeated=True)[0]
            else:
                predict = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.y_flat, 1))

        return predict

    def accuracy_function(self):
        """Create accuracy function to calculate accuracy of the prediction"""
        with tf.name_scope("accuracy"):
            if self.cost_function == 'ctc_loss':
                # get edit distance and return mean as accuracy
                edit_d = tf.edit_distance(tf.to_int32(self.correct_pred[0]), self.y, normalize=True)
                accuracy = tf.reduce_mean(edit_d, axis=0)
            else:
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
    def blstm(input_vector, sequence_length, layer_name="blstm_layer1", n_hidden=128, forget_bias=5.0,
              reuse=None, concat=True):
        """Create a bidirectional LSTM using code from the example at
        https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
        """
        with tf.variable_scope(layer_name):
            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True, reuse=reuse)
            # Backward direction cell
            lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True, reuse=reuse)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                     lstm_bw_cell, input_vector,
                                                                     dtype=tf.float32,
                                                                     sequence_length=sequence_length)
            # concat two output layers so we can treat as single output layer
            if concat:
                output = tf.concat(outputs, 2)
            else:
                output = outputs
        return output

    def variable_summaries(self, var):
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
        """Create a fully connected layer.
        source:
        https://stackoverflow.com/questions/39808336/tensorflow-bidirectional-dynamic-rnn-none-values-error/40305673
        """
        # get input dimensions
        input_dim = int(input_data.get_shape()[1])
        weight = tf.get_variable(name="weights", shape=[input_dim, output_dim * seq_len],
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / (2 * output_dim))))
        bias = tf.get_variable(name="bias", shape=[output_dim * seq_len],
                               initializer=tf.zeros_initializer)

        # weight = tf.Variable(tf.random_normal([input_dim, output_dim * seq_len]), name="weights")
        # bias = tf.Variable(tf.random_normal([output_dim * seq_len]), name="bias")
        if activation_func:
            output = activation_func(tf.nn.bias_add(tf.matmul(input_data, weight), bias))
        else:
            output = tf.nn.bias_add(tf.matmul(input_data, weight), bias)
        return output, weight, bias


def dense_to_sparse(dense_tensor):
    """Convert dense tensor to sparse tensor"""
    where_dense_non_zero = tf.where(tf.not_equal(dense_tensor, 0))
    indices = where_dense_non_zero
    values = tf.gather_nd(dense_tensor, where_dense_non_zero)
    shape = dense_tensor.get_shape()

    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=shape)


def sparse_tensor_merge(indices, values, shape):
    """Creates a SparseTensor from batched indices, values, and shapes.

    Args:
      indices: A [batch_size, N, D] integer Tensor.
      values: A [batch_size, N] Tensor of any dtype.
      shape: A [batch_size, D] Integer Tensor.
    Returns:
      A SparseTensor of dimension D + 1 with batch_size as its first dimension.
    """
    # print(indices.get)
    merged_shape = tf.reduce_max(shape, axis=0)
    batch_size, elements, shape_dim = tf.unstack(tf.shape(indices))
    index_range_tiled = tf.tile(tf.range(batch_size)[..., None],
                                tf.stack([1, elements]))[..., None]
    merged_indices = tf.reshape(
        tf.concat([tf.cast(index_range_tiled, tf.int64), indices], axis=2),
        [-1, 1 + tf.size(merged_shape)])
    merged_values = tf.reshape(values, [-1])
    return tf.SparseTensor(
        merged_indices, merged_values,
        tf.concat([[tf.cast(batch_size, tf.int64)], merged_shape], axis=0))


### The following funtions were originally taken from chiron https://github.com/haotianteng/Chiron
def conv_layer(indata, ksize, padding, name, dilate=1, strides=[1, 1, 1, 1], bias_term=False, active=True,
               BN=True):
    """A standard convlotional layer"""
    with tf.variable_scope(name):
        W = tf.get_variable("weights", dtype=tf.float32, shape=ksize,
                            initializer=tf.contrib.layers.xavier_initializer())
        if bias_term:
            b = tf.get_variable("bias", dtype=tf.float32, shape=[ksize[-1]])
        if dilate > 1:
            if bias_term:
                conv_out = b + tf.nn.atrous_conv2d(indata, W, rate=dilate, padding=padding, name=name)
            else:
                conv_out = tf.nn.atrous_conv2d(indata, W, rate=dilate, padding=padding, name=name)
        else:
            if bias_term:
                conv_out = b + tf.nn.conv2d(indata, W, strides=strides, padding=padding, name=name)
            else:
                conv_out = tf.nn.conv2d(indata, W, strides=strides, padding=padding, name=name)
    if BN:
        with tf.variable_scope(name + '_bn') as scope:
            #            conv_out = batchnorm(conv_out,scope=scope,training = training)
            conv_out = simple_global_bn(conv_out, name=name + '_bn')
    if active:
        with tf.variable_scope(name + '_relu'):
            conv_out = tf.nn.relu(conv_out, name='relu')
    return conv_out


def simple_global_bn(inp, name):
    ksize = inp.get_shape().as_list()
    ksize = [ksize[-1]]
    mean, variance = tf.nn.moments(inp, [0, 1, 2], name=name + '_moments')
    scale = tf.get_variable(name + "_scale",
                            shape=ksize, initializer=tf.contrib.layers.variance_scaling_initializer())
    offset = tf.get_variable(name + "_offset",
                             shape=ksize, initializer=tf.contrib.layers.variance_scaling_initializer())
    return tf.nn.batch_normalization(inp, mean=mean, variance=variance, scale=scale, offset=offset,
                                     variance_epsilon=1e-5)


def inception_layer(indata, times=16):
    """Inception module with dilate conv layer from http://arxiv.org/abs/1512.00567"""
    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1_AvgPooling'):
        avg_pool = tf.nn.avg_pool(indata, ksize=(1, 1, 3, 1), strides=(1, 1, 1, 1), padding='SAME',
                                  name='avg_pool0a1x3')
        conv1a = conv_layer(avg_pool, ksize=[1, 1, in_channel, times * 3], padding='SAME',
                            name='conv1a_1x1')
    with tf.variable_scope('branch2_1x1'):
        conv0b = conv_layer(indata, ksize=[1, 1, in_channel, times * 3], padding='SAME',
                            name='conv0b_1x1')
    with tf.variable_scope('branch3_1x3'):
        conv0c = conv_layer(indata, ksize=[1, 1, in_channel, times * 2], padding='SAME',
                            name='conv0c_1x1')
        conv1c = conv_layer(conv0c, ksize=[1, 3, times * 2, times * 3], padding='SAME',
                            name='conv1c_1x3')
    with tf.variable_scope('branch4_1x5'):
        conv0d = conv_layer(indata, ksize=[1, 1, in_channel, times * 2], padding='SAME',
                            name='conv0d_1x1')
        conv1d = conv_layer(conv0d, ksize=[1, 5, times * 2, times * 3], padding='SAME',
                            name='conv1d_1x5')
    with tf.variable_scope('branch5_1x3_dilate_2'):
        conv0e = conv_layer(indata, ksize=[1, 1, in_channel, times * 2], padding='SAME',
                            name='conv0e_1x1')
        conv1e = conv_layer(conv0e, ksize=[1, 3, times * 2, times * 3], padding='SAME',
                            name='conv1e_1x3_d2', dilate=2)
    with tf.variable_scope('branch6_1x3_dilate_3'):
        conv0f = conv_layer(indata, ksize=[1, 1, in_channel, times * 2], padding='SAME',
                            name='conv0f_1x1')
        conv1f = conv_layer(conv0f, ksize=[1, 3, times * 2, times * 3], padding='SAME',
                            name='conv1f_1x3_d3', dilate=3)
    return (tf.concat([conv1a, conv0b, conv1c, conv1d, conv1e, conv1f], axis=-1, name='concat'))


def residual_layer(indata, out_channel, i_bn=False):
    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1'):
        indata_cp = conv_layer(indata, ksize=[1, 1, in_channel, out_channel], padding='SAME',
                               name='conv1', BN=i_bn, active=False)
    with tf.variable_scope('branch2'):
        conv_out1 = conv_layer(indata, ksize=[1, 1, in_channel, out_channel], padding='SAME',
                               name='conv2a', bias_term=False)
        conv_out2 = conv_layer(conv_out1, ksize=[1, 3, out_channel, out_channel], padding='SAME',
                               name='conv2b', bias_term=False)
        conv_out3 = conv_layer(conv_out2, ksize=[1, 1, out_channel, out_channel], padding='SAME',
                               name='conv2c', bias_term=False, active=False)
    with tf.variable_scope('plus'):
        relu_out = tf.nn.relu(indata_cp + conv_out3, name='final_relu')
    return relu_out


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
