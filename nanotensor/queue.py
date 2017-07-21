#!/usr/bin/env python
"""Create inputs to a tensorflow graph using tf operations and queues"""
########################################################################
# File: data.py
#  executable: data.py

# Author: Andrew Bailey
# History: 06/05/17 Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
import threading
import numpy as np
from nanotensor.utils import project_folder, list_dir
import tensorflow as tf


class DataQueue:
    """Parses data and feeds inputs to the tf graph"""

    def __init__(self, file_list, batch_size=10, queue_size=100, verbose=False, pad=0, trim=True, n_steps=1):

        # test if inputs are correct types
        assert type(file_list) is list, "file_list is not list: type(file_list) = {}".format(type(file_list))
        assert len(file_list) >= 1, "file_list is empty: len(file_list) = {}".format(len(file_list))
        assert type(batch_size) is int, "batch_size is not int: type(batch_size) = {}".format(type(batch_size))
        assert type(queue_size) is int, "queue_size is not int: type(queue_size) = {}".format(type(queue_size))
        assert type(verbose) is bool, "verbose is not bool: type(verbose) = {}".format(type(verbose))
        assert type(pad) is int, "pad is not int: type(pad) = {}".format(type(pad))
        assert type(trim) is bool, "trim is not bool: type(trim) = {}".format(type(trim))
        assert type(n_steps) is int, "n_steps is not int: type(n_steps) = {}".format(type(n_steps))
        assert queue_size / 3 >= batch_size, "Batch size is larger than 1/3 of the queue size"

        # assign class objects
        self.file_list = file_list
        self.num_files = len(self.file_list)
        self.file_index = 0
        self.batch_size = batch_size
        self.verbose = verbose
        self.trim = trim
        self.seq_len = n_steps

        # TODO implement padding
        self.pad = 0

        # get size of inputs and classes
        data = np.load(self.file_list[0])
        self.n_input = len(data[0][0])
        self.n_classes = len(data[0][1])
        if self.verbose:
            print("Size of input vector = {}".format(self.n_input), file=sys.stderr)
            print("Size of label vector = {}".format(self.n_classes), file=sys.stderr)

        self.dataX = tf.placeholder("float", [n_steps, self.n_input], name='Input')
        self.dataY = tf.placeholder("float", [n_steps, self.n_classes], name='Label')
        # The actual queue of data.
        self.queue = tf.RandomShuffleQueue(shapes=[[n_steps, self.n_input], [n_steps, self.n_classes]],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=queue_size,
                                           min_after_dequeue=queue_size / 2)
        # define the enqueue operation
        self.enqueue_op = self.queue.enqueue([self.dataX, self.dataY])
        # True if there are files that have not been read into the queue
        self.files_left = True

    def shuffle(self):
        """Shuffle the input file order"""
        if self.verbose:
            print("Shuffle data files", file=sys.stderr)
        np.random.shuffle(self.file_list)
        self.files_left = False
        return True

    def add_to_queue(self, batch, sess, pad=0):
        """Add a batch to the queue"""
        if pad > 0:
            # if we want to pad sequences with zeros
            batch = self.pad_with_zeros(batch, pad=pad)
        features = batch[:, 0]
        labels = batch[:, 1]
        features = np.asarray([np.asarray(features[n]) for n in range(len(features))])
        labels = np.asarray([np.asarray(labels[n]) for n in range(len(labels))])
        sess.run(self.enqueue_op, feed_dict={self.dataX: features, self.dataY: labels})

    @staticmethod
    def pad_with_zeros(matrix, pad=0):
        """Pad an array with zeros so it has the correct shape for the batch"""
        column1 = len(matrix[0][0])
        column2 = len(matrix[0][1])
        one_row = np.array([[np.zeros([column1]), np.zeros([column2])]])
        new_rows = np.repeat(one_row, pad, axis=0)
        # print(new_rows.shape)
        return np.append(matrix, new_rows, axis=0)

    def create_batches(self, data, sess):
        """Create batches from input data array"""
        num_batches = (len(data) // self.seq_len)
        pad = self.seq_len - (len(data) % self.seq_len)
        if self.verbose:
            print("{} batches in this file".format(num_batches), file=sys.stderr)
        batch_number = 0
        index_1 = 0
        index_2 = self.seq_len
        while batch_number < num_batches:
            next_in = data[index_1:index_2]
            self.add_to_queue(next_in, sess)
            batch_number += 1
            index_1 += self.seq_len
            index_2 += self.seq_len

        if not self.trim:
            # moved this down because we don't care about connecting between reads right now
            self.add_to_queue(np.array([[str(pad), str(pad)]]), sess)
            next_in = data[index_1:index_2]
            # print(np.array([pad]))
            self.add_to_queue(next_in, sess, pad=pad)

        return True

    def read_in_file(self, sess):
        """Read in file from file list"""
        data = np.load(self.file_list[self.file_index])
        self.file_index += 1
        self.create_batches(data, sess)
        return True

    def load_data(self, sess):
        """Create infinite loop of adding to queue and shuffling files"""
        counter = 0
        while counter <= 10:
            self.read_in_file(sess)
            if self.verbose:
                print("File Index = {}".format(self.file_index), file=sys.stderr)
            if self.file_index == self.num_files:
                self.shuffle()
                self.file_index = 0
        return True

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        # batch_size = np.random.randint(1, self.batch_size)
        reads_batch, labels_batch = self.queue.dequeue_many(self.batch_size, name="dequeue")
        tf.add_to_collection("read_batch", reads_batch)
        tf.add_to_collection("labels_batch", labels_batch)
        # print(tf.shape(reads_batch))
        return reads_batch, labels_batch

    def thread_main(self, sess, data_obj):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for x_data, y_data in data_obj.batch_generator():
            sess.run(self.enqueue_op, feed_dict={self.dataX: x_data, self.dataY: y_data})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.load_data, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


def main():
    """Main docstring"""
    start = timer()

    tf.set_random_seed(1234)

    training_dir = "/Users/andrewbailey/nanopore-RNN/test_files/create_training_files/07Jul-20-11h-28m"
    training_files = list_dir(training_dir, ext="npy")

    for file1 in training_files:
        data = np.load(file1)
        data2 = data[:500]
        print(len(data2))
        assert (data2 == data).all()

        # np.save(file1, data2)

    # Doing anything with data on the CPU is generally a good idea.
    # data = DataQueue(training_files, batch_size=2, queue_size=10, verbose=False, pad=0, trim=True, n_steps=10)
    # images_batch, labels_batch = data.get_inputs()
    # images_batch1 = tf.reshape(images_batch, [-1, data.n_input])
    # labels_batch1 = tf.reshape(labels_batch, [-1, data.n_classes])
    #
    # # simple model
    # input_dim = int(images_batch1.get_shape()[1])
    # weight = tf.Variable(tf.random_normal([input_dim, data.n_classes]))
    # bias = tf.Variable(tf.random_normal([data.n_classes]))
    # prediction = tf.matmul(images_batch1, weight) + bias
    #
    # print(tf.shape(prediction))
    # print(tf.shape(labels_batch))
    # print(tf.shape(labels_batch1))
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_batch1)
    #
    # train_op = tf.train.AdamOptimizer().minimize(loss)
    #
    # sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
    # init = tf.global_variables_initializer()
    # sess.run(init)
    #
    # # start the tensorflow QueueRunner's
    # tf.train.start_queue_runners(sess=sess)
    # # start our custom queue runner's threads
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # data.start_threads(sess)
    #
    # _, loss_val = sess.run([train_op, loss])
    # print(loss_val)
    #
    # coord.request_stop()
    # coord.join(threads)
    # sess.close()

    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
