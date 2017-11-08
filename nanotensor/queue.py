#!/usr/bin/env python
"""Create inputs to a tensorflow graph using tf operations and queues"""
########################################################################
# File: queue.py
#  executable: queue.py

# Author: Andrew Bailey
# History: 06/05/17 Created
########################################################################

from __future__ import print_function
import sys
import os
from timeit import default_timer as timer
import threading
import numpy as np
import collections
from chiron.chiron_input import read_signal, read_label, read_raw
from nanotensor.trim_signal import SignalLabel

try:
    import Queue as queue
except ImportError:
    import queue
from nanotensor.utils import list_dir
import tensorflow as tf

training_labels = collections.namedtuple('training_data', ['input', 'seq_len', 'label'])
inference_labels = collections.namedtuple('inference_data', ['input', 'seq_len'])


class CreateDataset:
    """Create Dataset object for tensorflow imput pipeline"""
    def __init__(self, file_list, training=True, sparse=True, batch_size=10, verbose=False, pad=0, seq_len=1,
                 n_epochs=5, datatype='signal'):
        """
        import_function, x_shape, y_shape,
        :param file_list: list of data files (numpy files for now)
        :param import_function: function to import a single file from list of files
        :param x_shape: input shape in form of list
        :param y_shape: label shape in form of list
        :param training: bool to indicate to create repeat and shuffle batches
        :param sparse: bool option to create sparse tensor for label representation
        :param batch_size: integer representing number of elements in a batch
        :param verbose: bool option to print more information
        :param pad: pad input sequences
        :param seq_len: estimated sequence length
        :param n_epochs: number of looping through training data
        """
        # test if inputs are correct types
        assert type(file_list) is list, "file_list is not list: type(file_list) = {}".format(type(file_list))
        assert len(file_list) >= 1, "file_list is empty: len(file_list) = {}".format(len(file_list))
        assert type(batch_size) is int, "batch_size is not int: type(batch_size) = {}".format(type(batch_size))
        assert type(verbose) is bool, "verbose is not bool: type(verbose) = {}".format(type(verbose))
        assert type(pad) is int, "pad is not int: type(pad) = {}".format(type(pad))
        assert type(seq_len) is int, "seq_len is not int: type(n_steps) = {}".format(type(seq_len))

        # assign class objects
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        # self.import_function = import_function
        # TODO implement padding
        self.pad = 0

        #TODO implement TFrecords data
        if datatype == 'signal':
            self.n_classes = 3
            self.n_input = 1
            self.file_list = file_list
            x_shape = [None, self.seq_len]
        else:
            self.file_list, self.bad_files = self.test_numpy_files(file_list)
            self.num_files = len(self.file_list)
            print("Not using {} files".format(len(self.bad_files)), file=sys.stderr)
            assert self.num_files >= 1, "There are no passing npy files to read into queue"
            if self.verbose and self.bad_files:
                print(self.bad_files, file=sys.stderr)

            # get size of inputs and classes
            data = np.load(self.file_list[0])
            self.n_input = len(data[0][0])
            self.n_classes = len(data[0][1])
            x_shape = [None, self.seq_len, self.n_input]
        # test if files can be read
        if self.verbose:
            print("Size of input vector = {}".format(self.n_input), file=sys.stderr)
            print("Size of label vector = {}".format(self.n_classes), file=sys.stderr)

        self.dataX = tf.placeholder(tf.float32, shape=x_shape, name='Input')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='Sequence_Length')

        # self.dataX = tf.sparse_placeholder(tf.float32, shape=None, name='Input')
        X = tf.data.Dataset.from_tensor_slices(self.dataX)
        seq_length = tf.data.Dataset.from_tensor_slices(self.seq_length)
        X = X.batch(self.batch_size)
        seq_length = seq_length.batch(self.batch_size)
        if sparse:
            self.dataY = tf.placeholder(tf.int32, [None, None], name='Label')
            Y = tf.data.Dataset.from_tensor_slices(self.dataY)
            def remove_padding(y):
                """Remove padding from Y labels"""
                # return tf.equal(y, -1)
                return tf.boolean_mask(y, tf.not_equal(y, -1))
                # return tf.boolean_mask(y, tf.cond(y == -1))
                # return y[:index]
            # print(dataset)
            dataset = Y.map(remove_padding, num_parallel_calls=10)
            # print(dataset)
            # Y = dataset.batch(self.batch_size)
            Y = dataset.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=self.batch_size,
                                                                    row_shape=[self.seq_len]))
        else:
            self.dataY = tf.placeholder(tf.int32, [None, self.seq_len, self.n_classes], name='Label')
            Y = tf.data.Dataset.from_tensor_slices(self.dataY)
            Y = Y.batch(self.batch_size)
        if training:
            dataset = tf.data.Dataset.zip((X, seq_length, Y))
            dataset = dataset.repeat(self.n_epochs)
            dataset = dataset.shuffle(buffer_size=10000)
        else:
            dataset = tf.data.Dataset.zip((X, seq_length))
        dataset = dataset.prefetch(buffer_size=10)
        self.iterator = dataset.make_initializable_iterator()
        self.data = self.signal_label_reader(motif=True)

    def test(self):
        """Test to make sure the data was loaded correctly"""
        in_1, seq, out = self.iterator.get_next()
        with tf.Session() as sess:
            sess.run(self.iterator.initializer,
                     feed_dict={self.dataX: self.data.input,
                                self.seq_length: self.data.seq_len,
                                self.dataY: self.data.label})
            test1, test2, test3 = sess.run([in_1, seq, out])
            print(len(test1[0]))
            # print(test1, test2, test3)
        return True

    def numpy_train_dataloader(self):
        """Load data from numpy files"""
        X = []
        Y = []
        sequence_length = []
        for np_file in self.file_list:
            data = np.load(np_file)
            features = []
            labels = []
            seq_len = []
            num_batches = (len(data) // self.seq_len)
            # pad = self.seq_len - (len(data) % self.seq_len)
            batch_number = 0
            index_1 = 0
            index_2 = self.seq_len
            while batch_number < num_batches:
                next_in = data[index_1:index_2]
                features.append(np.vstack(next_in[:, 0]))
                labels.append([2, 2, 1])
                # labels.append(np.vstack(next_in[:, 1]))
                sequence_length.append(self.seq_len)
                batch_number += 1
                index_1 += self.seq_len
                index_2 += self.seq_len
            X.extend(features)
            Y.extend(labels)
        features = np.asarray(X)
        labels = np.asanyarray(Y)
        print(labels.shape)
        seq_len = np.asarray(sequence_length)
        return training_labels(input=features, seq_len=seq_len, label=labels)

    def signal_label_reader(self, k_mer=1, motif=True):
        ###Read from raw data
        event = list()
        event_length = list()
        label = list()
        label_length = list()
        count = 0
        file_count = 0
        for name in self.file_list:
            if name.endswith(".signal"):
                try:
                    file_pre = os.path.splitext(name)[0]
                    f_signal = read_signal(name)
                    label_name = file_pre + '.label'
                    if motif:
                        trim_signal = SignalLabel(name, label_name)
                        motif_generator = trim_signal.trim_to_motif(["CCAGG", "CCTGG", "CEAGG", "CETGG"],
                                                                    prefix_length=0,
                                                                    suffix_length=0,
                                                                    methyl_index=1,
                                                                    blank=True)
                        for motif in motif_generator:
                            tmp_event, tmp_event_length, tmp_label, tmp_label_length = read_raw(f_signal, motif, self.seq_len)
                            event += tmp_event
                            event_length += tmp_event_length
                            label += tmp_label
                            label_length += tmp_label_length
                            count = len(event)
                        if file_count % 10 == 0:
                            sys.stdout.write("%d lines read.   \n" % (count))
                        file_count += 1
                    else:
                        f_label = read_label(label_name, skip_start=10, window_n=(k_mer - 1) / 2,
                                             alphabet=self.n_classes)
                        tmp_event, tmp_event_length, tmp_label, tmp_label_length = read_raw(f_signal, f_label, self.seq_len)
                        event += tmp_event
                        event_length += tmp_event_length
                        label += tmp_label
                        label_length += tmp_label_length
                        count = len(event)
                        if file_count % 10 == 0:
                            sys.stdout.write("%d lines read.   \n" % (count))
                        file_count += 1
                except ValueError:
                    print("Error Reading Data from file {}".format(name))
                    continue
        padded_labels = []
        pad_len = max(label_length)
        for i in range(len(label)):
            padded_labels.append(np.lib.pad(label[i], (0, pad_len-label_length[i]), 'constant', constant_values=-1))
        return training_labels(input=np.asarray(event), seq_len=np.asarray(event_length),
                               label=padded_labels)

    @staticmethod
    def test_numpy_files(path_list):
        """Test if files in list are readable by numpy"""
        assert type(path_list) is list, "path_list is not list: type(path_list) = {}".format(type(path_list))
        # passing files and files which threw error
        good = []
        bad = []
        for file1 in path_list:
            try:
                np.load(file1)
                good.append(file1)
            except IOError:
                bad.append(file1)

        return good, bad


def batch2sparse(label_batch):
    """Transfer a batch of label to a sparse tensor"""
    values = []
    indices = []
    for batch_i, label_list in enumerate(label_batch[:, 0]):
        for indx, label in enumerate(label_list):
            indices.append([batch_i, indx])
            values.append(label)
    shape = [len(label_batch), max(label_batch[:, 1])]
    return (indices, values, shape)




if __name__ == "__main__":
    print("This file is just a library", file=sys.stderr)
    raise SystemExit
