#!/usr/bin/env python
"""data processing for passing to network"""
########################################################################
# File: data.py
#  executable: data.py

# Author: Andrew Bailey
# History: 06/05/17 Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
import numpy as np
from multiprocessing import Process, Queue

class Data:
    """Object to manage data for shuffling data inputs"""
    def __init__(self, file_list, batch_size, seq_len, queue_size, verbose=False, pad=0, trim=True):
        self.file_list = file_list
        self.num_files = len(self.file_list)
        self.queue = Queue(maxsize=queue_size)
        self.file_index = 0
        self.batch_size = batch_size
        self.verbose = verbose
        self.process1 = Process(target=self.load_data, args=())
        self.pad = pad
        self.trim = trim
        self.seq_len = seq_len

    def shuffle(self):
        """Shuffle the input file order"""
        if self.verbose:
            print("Shuffle data files", file=sys.stderr)
        # pylint: disable=no-member
        np.random.shuffle(self.file_list)
        return True

    def add_to_queue(self, batch, wait=True, pad=0):
        """Add a batch to the queue"""
        if pad > 0:
            # print(batch[-1])
            batch = self.pad_with_zeros(batch, pad=pad)
            # print(batch[-1])
        self.queue.put(batch, wait)

    @staticmethod
    def pad_with_zeros(matrix, pad=0):
        """Pad an array with zeros so it has the correct shape for the batch"""
        column1 = len(matrix[0][0])
        column2 = len(matrix[0][1])
        one_row = np.array([[np.zeros([column1]), np.zeros([column2])]])
        new_rows = np.repeat(one_row, pad, axis=0)
        # print(new_rows.shape)
        return np.append(matrix, new_rows, axis=0)


    def get_batch(self, wait=True):
        """Get a batch from the queue"""
        batch = self.queue.get(wait)
        features = batch[:, 0]
        labels = batch[:, 1]
        features = np.asarray([np.asarray(features[n]) for n in range(len(features))])
        labels = np.asarray([np.asarray(labels[n]) for n in range(len(labels))])
        return features, labels

    def create_batches(self, data):
        """Create batches from input data array"""
        num_batches = (len(data) // self.seq_len)
        pad = self.seq_len - (len(data) % self.seq_len)
        if self.verbose:
            print("{} batches in this file".format(num_batches), file=sys.stderr)
        batch_number = 0
        more_data = True
        index_1 = 0
        index_2 = self.seq_len
        while more_data:
            next_in = data[index_1:index_2]
            self.add_to_queue(next_in)
            batch_number += 1
            index_1 += self.seq_len
            index_2 += self.seq_len
            if batch_number == num_batches:
                if not self.trim:
                    # moved this down because we dont care about connecting between reads right now
                    self.add_to_queue(np.array([[str(pad), str(pad)]]))
                    next_in = data[index_1:index_2]
                    # print(np.array([pad]))
                    self.add_to_queue(next_in, pad=pad)
                more_data = False
        return True

    def read_in_file(self):
        """Read in file from file list"""
        data = np.load(self.file_list[self.file_index])
        self.create_batches(data)
        return True

    def load_data(self):
        """Create neverending loop of adding to queue and shuffling files"""
        counter = 0
        while counter <= 10:
            self.read_in_file()
            self.file_index += 1
            if self.verbose:
                print("File Index = {}".format(self.file_index), file=sys.stderr)
            if self.file_index == self.num_files:
                self.shuffle()
                self.file_index = 0
        return True

    def start(self):
        """Start background process to keep queue filled"""
        self.process1.start()
        return True

    def end(self):
        """End bacground process"""
        self.process1.terminate()
        return True

    def interpolate(self):
        """Guess a distribution of data"""
        return "from scipy.interpolate import interp1d"


def create_proto(path):
    """Create a protobuff file from numpy array"""



def main():
    """Main docstring"""
    start = timer()

    from skdata.mnist.views import OfficialVectorClassification
    from tqdm import tqdm
    import numpy as np
    import tensorflow as tf

    # filename = "mnist.tfrecords"
    # for serialized_example in tf.python_io.tf_record_iterator(filename):
    #     example = tf.train.Example()
    #     example.ParseFromString(serialized_example)
    #
    #     # traverse the Example format to get data
    #     image = example.features.feature['image'].int64_list.value
    #     label = example.features.feature['label'].int64_list.value[0]
    #     # do something
    #     print (image, label)
    # data = OfficialVectorClassification()
    # trIdx = data.sel_idxs[:]
    # print(data.all_labels[0])
    # one MUST randomly shuffle data before putting it into one of these
    # formats. Without this, one cannot make use of tensorflow's great
    # out of core shuffling.
    # np.random.shuffle(trIdx)
    #
    data = np.load("/Users/andrewbailey/nanopore-RNN/testing.npy")
    writer = tf.python_io.TFRecordWriter("nanopore.tfrecords")
    # # iterate over each example
    # # wrap with tqdm for a progress bar
    # for example_idx in tqdm(len(data)):
    features = data[:, 0]
    labels = data[:, 1]

    def make_example(sequence, labels):
        # The object we return
        # A non-sequential feature of our example
        ex = tf.train.SequenceExample()
        sequence_length = len(sequence)
        ex.context.feature["length"].int64_list.value.append(sequence_length)
        # Feature lists for the two sequential features of our example
        fl_events = ex.feature_lists.feature_list["events"]
        fl_labels = ex.feature_lists.feature_list["labels"]
        for event, label in zip(sequence, labels):
            fl_events.feature.add().float_list.value.append(_floatlist_feature(event))
            fl_labels.feature.add().float_list.value.append(label)
        return ex

    def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    ex = tf.train.SequenceExample(features=tf.train.Features(feature={
        # 'height': _int64_feature(rows),
        # 'width': _int64_feature(cols),
        # 'depth': _int64_feature(depth),
        # 'label': _int64_feature(int(labels[index])),
        'image_raw': _float_feature(labels)}))



    # def make_example2(sequence, labels):
    #     # The object we return
    #     # A non-sequential feature of our example
    #     sequence_length = len(sequence)
    #
    #     ex = tf.train.SequenceExample()
    #     ex.context.feature["length"].int64_list.value.append(sequence_length)
    #     # Feature lists for the two sequential features of our example
    #     fl_events = ex.feature_lists.feature_list["events"]
    #     fl_labels = ex.feature_lists.feature_list["labels"]
    #     for event, label in zip(sequence, labels):
    #         fl_events.feature.add().float_list.value {event}
    #         fl_labels.feature.add().float_list.value.append(label)
    #     return ex


    sequence_length = len(features)

    # ex = make_example(features, labels)
    writer.write(ex.SerializeToString())
    writer.close()


    # construct the Example proto boject
    # example = tf.train.Example(
    #     # Example contains a Features proto object
    #     features=tf.train.Features(
    #       # Features contains a map of string to Feature proto objects
    #       feature={
    #         # A Feature contains one of either a int64_list,
    #         # float_list, or bytes_list
    #         'label': tf.train.Feature(
    #             int64_list=tf.train.Int64List(value=[label])),
    #         'image': tf.train.Feature(
    #             int64_list=tf.train.Int64List(value=features.astype("int64"))),
    #     }))
    #     # use the proto object to serialize the example to a string
    # serialized = example.SerializeToString()
    # # write the serialized object to disk
    # writer.write(serialized)

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
