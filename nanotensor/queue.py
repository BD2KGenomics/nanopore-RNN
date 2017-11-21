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
from nanotensor.utils import debug
import abc
import logging as log

try:
    import Queue as queue
except ImportError:
    import queue
from nanotensor.utils import list_dir
import tensorflow as tf


class CreateDataset(object):
    """Create Dataset object for tensorflow imput pipeline"""

    def __init__(self, mode=0, x_shape=list(), y_shape=list(), sequence_shape=list(), batch_size=10,
                 seq_len=10, len_y=0, len_x=0, n_epochs=5, verbose=False,
                 shuffle_buffer_size=10000, prefetch_buffer_size=100):
        """
        :param x_shape: input shape in form of list
        :param y_shape: label shape in form of list
        :param sequence_shape: sequence length shape
        :param training: bool to indicate to create repeat and shuffle batches
        :param batch_size: integer representing number of elements in a batch
        :param verbose: bool option to print more information
        :param seq_len: estimated sequence length
        :param n_epochs: number of looping through training data
        :param len_y: length of label vector
        :param len_x: length of input vector
        :param shuffle_buffer_size: size of buffer for shuffle option when training
        :param prefetch_buffer_size: size of buffer for prefetch option

        """
        # test if inputs are correct types
        assert type(mode) is int, "mode option is not int: type(mode) = {}".format(type(mode))
        assert type(x_shape) is list, "x_shape is not list: type(x_shape) = {}".format(type(x_shape))
        assert type(y_shape) is list, "y_shape is not list: type(y_shape) = {}".format(type(y_shape))
        assert type(sequence_shape) is list, \
            "sequence_shape is not list: type(sequence_shape) = {}".format(type(sequence_shape))
        assert type(batch_size) is int, "batch_size is not int: type(batch_size) = {}".format(type(batch_size))
        assert type(seq_len) is int, "seq_len is not int: type(seq_len) = {}".format(type(seq_len))
        assert type(len_y) is int, "len_y is not int: type(len_y) = {}".format(type(len_y))
        assert type(len_x) is int, "len_x is not int: type(len_x) = {}".format(type(len_x))
        assert type(verbose) is bool, "verbose is not bool: type(verbose) = {}".format(type(verbose))
        assert type(n_epochs) is int, "n_epochs is not int: type(n_epochs) = {}".format(type(n_epochs))
        assert type(shuffle_buffer_size) is int, \
            "shuffle_buffer_size is not int: type(shuffle_buffer_size) = {}".format(type(shuffle_buffer_size))
        assert type(prefetch_buffer_size) is int, \
            "prefetch_buffer_size is not int: type(prefetch_buffer_size) = {}".format(type(prefetch_buffer_size))

        # assign class objects
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.len_y = len_y
        self.len_x = len_x
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.sequence_shape = sequence_shape
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.mode = mode
        self.file_path = "path"

        # log information regarding data
        log.info("Shape of input vector = {}".format(self.x_shape))
        log.info("Shape of output vector = {}".format(self.y_shape))
        log.info("Batch Size = {}".format(self.batch_size))
        log.info("Sequence Length = {}".format(self.seq_len))
        log.info("Size of input vector = {}".format(self.len_x))
        log.info("Size of label vector = {}".format(self.len_y))

        # data structures for training and inference
        self.training_labels = collections.namedtuple('training_data', ['input', 'seq_len', 'label'])
        self.inference_labels = collections.namedtuple('inference_data', ['input', 'seq_len'])

        # placeholder creation
        self.place_X = tf.placeholder(tf.float32, shape=self.x_shape, name='Input')
        self.place_Seq = tf.placeholder(tf.int32, shape=self.sequence_shape, name='Sequence_Length')
        self.place_Y = tf.placeholder(tf.int32, shape=self.y_shape, name='Label')

        # dataset creation
        self.datasetX = tf.data.Dataset.from_tensor_slices(self.place_X)
        self.datasetSeq = tf.data.Dataset.from_tensor_slices(self.place_Seq)
        self.datasetY = tf.data.Dataset.from_tensor_slices(self.place_Y)

        # optional batching, dataset and iteration creation
        self.batchX, self.batchSeq, self.batchY = self.create_batches()
        # log.info("Shape of input vector = {}".format(self.x_shape))
        self.dataset = self.create_dataset()
        self.iterator = self.create_iterator()
        if self.mode == 0 or self.mode == 1:
            self.data = self.load_data()
        self.test()

    def create_batches(self):
        """Create batch data for self.datasetX, self.datasetSeq, self.datasetY"""
        X = self.datasetX.batch(self.batch_size)
        seq_length = self.datasetSeq.batch(self.batch_size)
        y = self.datasetY.batch(self.batch_size)
        return X, seq_length, y

    def create_dataset(self):
        """Creates dataset structure"""
        # training
        if self.mode == 0:
            dataset = tf.data.Dataset.zip((self.batchX, self.batchSeq, self.batchY))
            dataset = dataset.repeat(self.n_epochs)
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        # testing
        elif self.mode == 1:
            dataset = tf.data.Dataset.zip((self.batchX, self.batchSeq, self.batchY))
        # inference
        elif self.mode == 2:
            # inference needs to be done per file
            dataset = tf.data.Dataset.from_generator(
                self.load_data_inference, (tf.float32, tf.int32), (tf.TensorShape([self.seq_len]),
                                                                   tf.TensorShape(None)))
            dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        return dataset

    def create_iterator(self):
        """Creates boilerplate iterator depending on dataset"""
        return self.dataset.make_initializable_iterator()

    def test(self):
        """Test to make sure the data was loaded correctly"""
        if self.mode == 0 or self.mode == 1:
            in_1, seq, out = self.iterator.get_next()
            with tf.Session() as sess:
                sess.run(self.iterator.initializer,
                         feed_dict={self.place_X: self.data.input,
                                    self.place_Seq: self.data.seq_len,
                                    self.place_Y: self.data.label})
                test1, test2, test3 = sess.run([in_1, seq, out])
                log.info("Dataset Creation Complete")
        elif self.mode == 1:
            in_1, seq = self.iterator.get_next()
            with tf.Session() as sess:
                sess.run(self.iterator.initializer,
                         feed_dict={self.place_X: self.data.input,
                                    self.place_Seq: self.data.seq_len})
                test1, test2 = sess.run([in_1, seq])
                log.info("Dataset Creation Complete")

        return True

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

    @staticmethod
    def padding(x, L, padding_list=None):
        """Padding the vector x to length L"""
        len_x = len(x)
        assert len_x <= L, "Length of vector x is larger than the padding length"
        zero_n = L - len_x
        if padding_list is None:
            return x + [0] * zero_n
        elif len(padding_list) < zero_n:
            return x + padding_list + [0] * (zero_n - len(padding_list))
        else:
            return x + padding_list[0:zero_n]

    @abc.abstractmethod
    def load_data(self):
        pass

    @abc.abstractmethod
    def process_output(self, graph_output):
        pass

    @abc.abstractmethod
    def load_data_inference(self):
        pass


class MotifSequence(CreateDataset):
    """Subclass of CreateDataset for dealing with data from signal and label data"""

    def __init__(self, file_list, mode=0, batch_size=10, verbose=True, seq_len=100,
                 n_epochs=5, shuffle_buffer_size=10000, prefetch_buffer_size=100, blank=True):

        """

        :param file_list: list of signal and label files within a single directory
        :param mode: int to indicate file loading protocol
        :param batch_size: integer representing number of elements in a batch
        :param verbose: bool option to print more information
        :param seq_len: estimated sequence length
        :param n_epochs: number of looping through training data
        :param shuffle_buffer_size: size of buffer for shuffle option when training
        :param prefetch_buffer_size: size of buffer for prefetch option
        """

        assert seq_len > 50, "seq_len is not greater than 50: {} !> 50".format(seq_len)
        self.file_list = file_list
        self.len_y = 3
        self.len_x = 1
        self.blank = blank

        super(MotifSequence, self).__init__(mode=mode, x_shape=[None, seq_len],
                                            y_shape=[None, None], sequence_shape=[None],
                                            batch_size=batch_size, seq_len=seq_len, len_y=self.len_y, len_x=self.len_x,
                                            n_epochs=n_epochs, verbose=verbose,
                                            shuffle_buffer_size=shuffle_buffer_size,
                                            prefetch_buffer_size=prefetch_buffer_size)

    def create_batches(self):
        """Create dataset batches for sequence and motifs data"""

        def remove_padding(y):
            """Remove padding from Y labels"""
            return tf.boolean_mask(y, tf.not_equal(y, -1))

        dataset = self.datasetY.map(remove_padding, num_parallel_calls=10)
        batch_y = dataset.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=self.batch_size,
                                                                      row_shape=[self.seq_len]))
        batch_x = self.datasetX.batch(self.batch_size)
        batch_seq_length = self.datasetSeq.batch(self.batch_size)

        return batch_x, batch_seq_length, batch_y

    def load_data(self):
        """Read in data from signal files and create specific motif comparisons"""
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
                    trim_signal = SignalLabel(name, label_name)
                    motif_generator = trim_signal.trim_to_motif(["CCAGG", "CCTGG", "CEAGG", "CETGG"],
                                                                prefix_length=0,
                                                                suffix_length=0,
                                                                methyl_index=1,
                                                                blank=self.blank)
                    for motif in motif_generator:
                        tmp_event, tmp_event_length, tmp_label, tmp_label_length = read_raw(f_signal, motif,
                                                                                            self.seq_len,
                                                                                            short=True)
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
            padded_labels.append(np.lib.pad(label[i], (0, pad_len - label_length[i]), 'constant', constant_values=-1))
        return self.training_labels(input=np.asarray(event), seq_len=np.asarray(event_length),
                                    label=padded_labels)

    def process_output(self, graph_output):
        """Process output from prediciton function"""
        # print(graph_output)
        bpreads = [SignalLabel.index2base(read) for read in graph_output]
        print(bpreads)

    def load_data_inference(self):
        """Load data in using inference functions"""
        f_signal = read_signal(self.file_path, normalize=True)
        f_signal = f_signal[self.start_index:]
        sig_len = len(f_signal)
        for indx in range(0, sig_len, self.step):
            segment_sig = f_signal[indx:indx + self.seq_len]
            segment_len = len(segment_sig)
            padded_segment_sig = self.padding(segment_sig, self.seq_len)
            yield self.inference_labels(input=np.asarray(padded_segment_sig),
                                        seq_len=np.asarray(segment_len))


class FullSignalSequence(CreateDataset):
    """Subclass of CreateDataset for dealing with data from signal and label data"""

    def __init__(self, file_list, mode=0, batch_size=10, verbose=True, seq_len=100,
                 n_epochs=5, shuffle_buffer_size=10000, prefetch_buffer_size=100, step=300, start_index=0):
        """

        :param file_list: list of signal and label files within a single directory
        :param mode: int to indicate file loading protocol
        :param batch_size: integer representing number of elements in a batch
        :param verbose: bool option to print more information
        :param seq_len: estimated sequence length
        :param n_epochs: number of looping through training data
        :param shuffle_buffer_size: size of buffer for shuffle option when training
        :param prefetch_buffer_size: size of buffer for prefetch option
        """
        self.file_list = file_list
        self.len_y = 5
        self.len_x = 1
        self.kmer = 1
        self.step = step
        self.start_index = start_index
        super(FullSignalSequence, self).__init__(mode=mode, x_shape=[None, seq_len],
                                                 y_shape=[None, None], sequence_shape=[None],
                                                 batch_size=batch_size, seq_len=seq_len, len_y=self.len_y,
                                                 len_x=self.len_x,
                                                 n_epochs=n_epochs, verbose=verbose,
                                                 shuffle_buffer_size=shuffle_buffer_size,
                                                 prefetch_buffer_size=prefetch_buffer_size)

    def create_batches(self, inference=False):
        """Create dataset batches for sequence data"""

        def remove_padding(y):
            """Remove padding from Y labels"""
            return tf.boolean_mask(y, tf.not_equal(y, -1))

        dataset = self.datasetY.map(remove_padding, num_parallel_calls=10)
        batch_y = dataset.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=self.batch_size,
                                                                      row_shape=[self.seq_len]))
        batch_x = self.datasetX.batch(self.batch_size)
        batch_seq_length = self.datasetSeq.batch(self.batch_size)
        if inference:
            self.datasetX.batch(self.batch_size)

        return batch_x, batch_seq_length, batch_y

    def load_data(self):
        """Read in data from signal files and create specific motif comparisons"""
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
                    f_label = read_label(label_name, skip_start=10, window_n=(self.kmer - 1) / 2,
                                         alphabet=self.len_y)
                    tmp_event, tmp_event_length, tmp_label, tmp_label_length = read_raw(f_signal, f_label,
                                                                                        self.seq_len)
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
            padded_labels.append(np.lib.pad(label[i], (0, pad_len - label_length[i]), 'constant', constant_values=-1))

        return self.training_labels(input=np.asarray(event), seq_len=np.asarray(event_length),
                                    label=padded_labels)

    def process_output(self, graph_output):
        """Process output from prediciton function"""
        # print(graph_output)
        bpreads = [SignalLabel.index2base(read) for read in graph_output]
        print(bpreads)

    def load_data_inference(self):
        """Load data in using inference functions"""
        f_signal = read_signal(self.file_path, normalize=True)
        f_signal = f_signal[self.start_index:]
        sig_len = len(f_signal)
        for indx in range(0, sig_len, self.step):
            segment_sig = f_signal[indx:indx + self.seq_len]
            segment_len = len(segment_sig)
            padded_segment_sig = self.padding(segment_sig, self.seq_len)
            yield self.inference_labels(input=np.asarray(padded_segment_sig),
                                   seq_len=np.asarray(segment_len))


class NumpyEventData(CreateDataset):
    """Subclass of CreateDataset for dealing with data from signal and label data"""

    def __init__(self, file_list, mode=0, batch_size=10, verbose=True, seq_len=100,
                 n_epochs=5, shuffle_buffer_size=10000, prefetch_buffer_size=100):
        """

        :param file_list: list of signal and label files within a single directory
        :param mode: int to indicate file loading protocol
        :param batch_size: integer representing number of elements in a batch
        :param verbose: bool option to print more information
        :param seq_len: estimated sequence length
        :param n_epochs: number of looping through training data
        :param shuffle_buffer_size: size of buffer for shuffle option when training
        :param prefetch_buffer_size: size of buffer for prefetch option
        """
        log = debug(verbose)
        self.file_list, self.bad_files = self.test_numpy_files(file_list)
        self.num_files = len(self.file_list)
        assert self.num_files >= 1, "There are no passing npy files to read into queue"
        if self.bad_files == 0:
            log.info("All numpy files passed")
        else:
            log.warning("{} files were unable to be loaded using np.load()".format(len(self.bad_files)))
            log.info("{}".format(self.bad_files))

        # get size of inputs and classes
        data = np.load(self.file_list[0])
        self.len_x = len(data[0][0])
        self.len_y = len(data[0][1])


        super(NumpyEventData, self).__init__(mode=mode, x_shape=[None, seq_len, self.len_x],
                                             y_shape=[None, seq_len, self.len_y], sequence_shape=[None],
                                             batch_size=batch_size, seq_len=seq_len, len_y=self.len_y,
                                             len_x=self.len_x,
                                             n_epochs=n_epochs, verbose=verbose,
                                             shuffle_buffer_size=shuffle_buffer_size,
                                             prefetch_buffer_size=prefetch_buffer_size)

    def load_data(self):
        """Load data from numpy files"""
        x = []
        y = []
        sequence_length = []
        for np_file in self.file_list:
            data = np.load(np_file)
            features = []
            labels = []
            num_batches = (len(data) // self.seq_len)
            # pad = self.seq_len - (len(data) % self.seq_len)
            batch_number = 0
            index_1 = 0
            index_2 = self.seq_len
            while batch_number < num_batches:
                next_in = data[index_1:index_2]
                features.append(np.vstack(next_in[:, 0]))
                labels.append(np.vstack(next_in[:, 1]))
                sequence_length.append(self.seq_len)
                batch_number += 1
                index_1 += self.seq_len
                index_2 += self.seq_len
            x.extend(features)
            y.extend(labels)
        features = np.asarray(x)
        labels = np.asanyarray(y)
        seq_len = np.asarray(sequence_length)
        return self.training_labels(input=features, seq_len=seq_len, label=labels)

    def process_output(self, graph_output):
        for x in graph_output:
            print(x)


if __name__ == "__main__":

    file_list = list_dir("/Users/andrewbailey/CLionProjects/nanopore-RNN/chiron/data/raw")

    motif = MotifSequence(file_list, mode=0, batch_size=10, verbose=True, seq_len=100,
                          n_epochs=5)
    full = FullSignalSequence(file_list, mode=0, batch_size=10, verbose=True, seq_len=100,
                              n_epochs=5)
    file_list = list_dir(
        "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/create_training_files/07Jul-20-11h-28m")

    full = NumpyEventData(file_list, mode=0, batch_size=10, verbose=True, seq_len=100,
                          n_epochs=5)
    print("This file is just a library", file=sys.stderr)
    raise SystemExit

