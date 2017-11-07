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

ctc_labels = collections.namedtuple('ctc_labels', ['input', 'seq_len', 'label', 'label_index'])
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
            self.n_classes = 5
            self.n_input = 1
            self.file_list = file_list
            x_shape = [None, self.seq_len]
        else:
            self.file_list, self.bad_files = self.read_numpy_files(file_list)
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
                        motif_generator = trim_signal.trim_to_motif(["CCAGG", "CCTGG", "CEAGG", "CETGG"], prefix_length=10,
                                                                    suffix_length=10)
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
                    continue
        padded_labels = []
        pad_len = max(label_length)
        for i in range(len(label)):
            padded_labels.append(np.lib.pad(label[i], (0, pad_len-label_length[i]), 'constant', constant_values=-1))
        return training_labels(input=np.asarray(event), seq_len=np.asarray(event_length),
                               label=padded_labels)

    @staticmethod
    def read_numpy_files(path_list):
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


class DataQueue:
    """Parses data and feeds inputs to the tf graph"""

    def __init__(self, file_list, batch_size=10, queue_size=100, verbose=False, pad=0, trim=True, n_steps=1,
                 min_in_queue=1, training=False, sparse=True):

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
        assert type(min_in_queue) is int, "min_in_queue is not int: type(min_in_queue) = {}".format(type(min_in_queue))

        # assign class objects
        self.batch_size = batch_size
        self.verbose = verbose
        self.trim = trim
        self.seq_len = n_steps
        # TODO implement padding
        self.pad = 0
        # test if files can be read
        self.file_list, self.bad_files = self.read_numpy_files(file_list)
        self.num_files = len(self.file_list)
        print("Creating Queue of with {} npy files".format(self.num_files), file=sys.stderr)
        print("Not using {} files".format(len(self.bad_files)), file=sys.stderr)
        assert self.num_files >= 1, "There are no passing npy files to read into queue"
        if self.verbose:
            print(self.bad_files, file=sys.stderr)

        # get size of inputs and classes
        data = np.load(self.file_list[0])
        self.n_input = len(data[0][0])
        self.n_classes = len(data[0][1])
        if self.verbose:
            print("Size of input vector = {}".format(self.n_input), file=sys.stderr)
            print("Size of label vector = {}".format(self.n_classes), file=sys.stderr)

        # The actual queue of data.
        if sparse:
            self.dataX = tf.placeholder(tf.float32, [None, self.seq_len, self.n_input], name='Input')
            self.seq_length = tf.placeholder(tf.int32, shape=[batch_size])
            self.dataY = tf.placeholder(tf.int32, [None, self.seq_len, self.n_classes], name='Label')
            # self.dataX = tf.sparse_placeholder(tf.float32, shape=None, name='Input')
            Xdataset = tf.data.Dataset.from_tensor_slices(self.dataX)
            Ydataset = tf.data.Dataset.from_tensor_slices(self.dataY)
            Xdataset = Xdataset.batch(10)
            print(Xdataset.output_shapes)
            print(Ydataset.output_shapes)
            Ydataset = Ydataset.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=10, row_shape=[self.seq_len, self.n_classes]))
            print(Xdataset.output_shapes)
            print(Ydataset.output_shapes)
            dataset = tf.data.Dataset.zip((Xdataset, Ydataset))

        # dataset = dataset.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=10, row_shape=[self.seq_len]))
        #     if training:
        #         dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat(10)

            self.iterator = dataset.make_initializable_iterator()
            data = np.load(self.file_list[0])
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
                batch_number += 1
                index_1 += self.seq_len
                index_2 += self.seq_len

            self.features = np.asarray(features)
            self.labels = np.asarray(labels)
            # print(features.ndim)
            # print(features.shape)
            x, y = self.iterator.get_next()
            with tf.Session() as sess:
                sess.run(self.iterator.initializer, feed_dict={self.dataX: self.features, self.dataY: self.labels})
                x, y = sess.run([x, y])
                print(x.shape)
                print("Y", y)
        else:
            self.dataX = tf.placeholder(tf.float32, [None, self.seq_len, self.n_input], name='Input')
            self.seq_length = tf.placeholder(tf.int32, shape=[batch_size])
            self.dataY = tf.placeholder(tf.int32, [None, self.seq_len, self.n_classes], name='Label')
            dataset = tf.data.Dataset.from_tensor_slices((self.dataX, self.dataY))#, self.seq_length))
            dataset = dataset.batch(10)
            if training:
                dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat(10)

            # dataset.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=10, row_shape=[self.seq_len]))
            self.iterator = dataset.make_initializable_iterator()
            data = np.load(self.file_list[0])
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
                batch_number += 1
                index_1 += self.seq_len
                index_2 += self.seq_len

            self.features = np.asarray(features)
            self.labels = np.asarray(labels)
            # print(features.ndim)
            # print(features.shape)
        #
        #     self.y_indexes = tf.placeholder(tf.int64, name='y_indexes')
        #     self.y_values = tf.placeholder(tf.int32, name="y_values")
        #     self.y_shape = tf.placeholder(tf.int64, name="y_shape")
        #     # self.dataY = tf.SparseTensor(y_indexes, y_values, y_shape)
        #     # tf.sparse_placeholder(dtype, shape=None, name=None)
        #
        #
        #     # self.dataY = tf.placeholder(tf.int32, [None], name='Label')
        #     # shapes = [[n_steps, self.n_input], [None], [None], [], [batch_size]]
        #     # dtypes = [tf.float32, tf.int64, tf.int32, tf.int64, tf.int32]
        #
        #     self.dataX = tf.placeholder(tf.float32, [n_steps, self.n_input], name='Input')
        #     self.dataY = tf.placeholder(tf.float32, [n_steps, self.n_classes], name='Label')
        #     self.seq_length = tf.placeholder(tf.int32, shape=[batch_size])
        #     shapes = [[n_steps, self.n_input], [n_steps, self.n_classes], [batch_size]]
        #     dtypes = [tf.float32, tf.float32, tf.int32]
        # #
        # # True if there are files that have not been read into the queue
        # self.files_left = True
        # self.files_read = 0
        # self.stop_event = threading.Event()
        # # create threading queue for file list so there can be asynchronous data batching
        # self.file_list_queue = queue.Queue()
        # for file1 in self.file_list:
        #     self.file_list_queue.put(file1)

    def join(self):
        """Kill daemon threads if needed"""
        self.stop_event.set()

    def add_to_queue(self, batch, sess):
        """Add a batch to the queue"""
        features = batch[:, 0]
        labels = batch[:, 1]
        features = np.asarray([np.asarray(features[n]) for n in range(len(features))])
        labels = np.asarray([np.asarray(labels[n]) for n in range(len(labels))])
        # labels = batch2sparse(labels)
        # print("sparse", labels)
        seq_length = np.asarray([np.asarray(len(labels[n]) for n in range(len(labels)))])
        sess.run(self.enqueue_op, feed_dict={self.dataX: features, self.y_indexes: labels[0],
                                             self.y_values: labels[0], self.y_shape: labels[1],
                                             self.seq_length: seq_length})

    def create_batches(self, data, sess):
        """Create batches from input data array"""
        num_batches = (len(data) // self.seq_len)
        # pad = self.seq_len - (len(data) % self.seq_len)
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
        #
        # if not self.trim:
        #     # moved this down because we don't care about connecting between reads right now
        #     self.add_to_queue(np.array([[str(pad), str(pad)]]), sess)
        #     next_in = data[index_1:index_2]
        #     # print(np.array([pad]))
        #     self.add_to_queue(next_in, sess, pad=pad)

        return True

    def read_in_file(self, sess):
        """Read in file from file list"""
        file_path = self.file_list_queue.get()
        data = np.load(file_path)
        features = data[:, 0]
        labels = data[:, 1]
        self.create_batches(data, sess)
        self.file_list_queue.put(file_path)
        return True

    def load_data(self, sess, stop_event):
        """Create infinite loop of adding to queue and shuffling files"""
        while not stop_event.is_set():
            self.read_in_file(sess)
            self.files_read += 1
            if self.files_read == self.num_files:
                self.files_left = False

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        reads_batch, y_indexes, y_values, y_shape, seq_length = self.queue.dequeue_many(self.batch_size, name="dequeue")
        tf.add_to_collection("read_batch", reads_batch)
        # tf.add_to_collection("y_indexes", y_indexes)
        # tf.add_to_collection("y_values", y_values)
        # tf.add_to_collection("y_shape", y_shape)
        tf.add_to_collection("seq_length", seq_length)
        return

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.load_data, args=(sess, self.stop_event))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

    @staticmethod
    def read_numpy_files(path_list):
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

        # @staticmethod
        # def pad_with_zeros(matrix, pad=0):
        #     """Pad an array with zeros so it has the correct shape for the batch"""
        #     column1 = len(matrix[0][0])
        #     column2 = len(matrix[0][1])
        #     one_row = np.array([[np.zeros([column1]), np.zeros([column2])]])
        #     new_rows = np.repeat(one_row, pad, axis=0)
        #     # print(new_rows.shape)
        #     return np.append(matrix, new_rows, axis=0)

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


# def main():
#     """Main docstring"""
#     start = timer()
#
#     # tf.set_random_seed(1234)
#     #
#     # training_files = list_dir(training_dir, ext="npy")
#     # training_dir = "/Users/andrewbailey/nanopore-RNN/test_files/create_training_files/07Jul-20-11h-28m"
#     #
#     # # Doing anything with data on the CPU is generally a good idea.
#     # data = DataQueue(training_files, batch_size=2, queue_size=10, verbose=False, pad=0, trim=True, n_steps=10)
#     # images_batch, labels_batch = data.get_inputs()
#     # images_batch1 = tf.reshape(images_batch, [-1, data.n_input])
#     # labels_batch1 = tf.reshape(labels_batch, [-1, data.n_classes])
#     #
#     # # simple model
#     # input_dim = int(images_batch1.get_shape()[1])
#     # weight = tf.Variable(tf.random_normal([input_dim, data.n_classes]))
#     # bias = tf.Variable(tf.random_normal([data.n_classes]))
#     # prediction = tf.matmul(images_batch1, weight) + bias
#     #
#     # print(tf.shape(prediction))
#     # print(tf.shape(labels_batch))
#     # print(tf.shape(labels_batch1))
#     # loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_batch1)
#     #
#     # train_op = tf.train.AdamOptimizer().minimize(loss)
#     #
#     # sess = tf.Session()
#     # init = tf.global_variables_initializer()
#     # sess.run(init)
#     #
#     # # start the tensorflow QueueRunner's
#     # tf.train.start_queue_runners(sess=sess)
#     # # start our custom queue runner's threads
#     # data.start_threads(sess)
#     #
#     # _, loss_val = sess.run([train_op, loss])
#     # print(loss_val)
#     #
#     # sess.close()
#
#     stop = timer()
#     print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    # main()
    print("This file is just a library", file=sys.stderr)
    raise SystemExit
