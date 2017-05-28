#!/usr/bin/env python
"""nanotensor is able to label nanopore data, create training files and
then use's tensorflow to train a mulit layer BLSTM-RNN"""
########################################################################
# File: nanotensor.py
#  executable: nanotensor.py
#
# Author: Andrew Bailey
# History: TODO Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
import time
import Queue as Queue2
import numpy as np
from nanonet.fast5 import Fast5
from nanonet.eventdetection.filters import minknow_event_detect, compute_sum_sumsq, compute_tstat, short_long_peak_detector
from utils import testfast5, list_dir, project_folder
from data_preparation import save_training_file
from multiprocessing import Pool, Process, Queue
import threading

#
# q = Queue.LifoQueue()
#
# for i in range(5):
#     q.put(i)
#
# while not q.empty():
#     print q.get()

def create_training_data_from_log(log_file, file_prefix, kwargs, output_dir=project_folder()+"/training"):
    """Create npy training files"""
    counter = 0
    with open(log_file, 'r') as log:
        for line in log:
            line = line.rstrip().split('\t')
            fast5 = line[0]
            tsv = line[1]
            output_name = file_prefix+str(counter)
            save_training_file(fast5, tsv, output_name, output_dir=output_dir, kwargs=kwargs)
            counter += 1
            print("Saved {}.npy".format(output_name), file=sys.stderr)
    return True

class Data:
    """Object to manage data for shuffling data inputs"""
    def __init__(self, queue, file_list, batch_size, verbose=False):
        self.file_list = file_list
        self.num_files = len(self.file_list)
        self.queue = queue
        self.file_index = 0
        self.batch_size = batch_size
        self.verbose = verbose

    def shuffle(self):
        """Shuffle the input file order"""
        if self.verbose:
            print("Shuffle data files", file=sys.stderr)
        np.random.shuffle(self.file_list)
        return True

    def add_to_queue(self, batch, wait=True):
        """Add a batch to the queue"""
        self.queue.put(batch, wait)

    def get_batch(self, wait=True):
        """Get a batch from the queue"""
        return self.queue.get(wait)

    def create_batches(self, data):
        """Create batches from input data array"""
        num_batches = (len(data) // self.batch_size) + 1
        if self.verbose:
            print("{} batches in this file".format(num_batches), file=sys.stderr)
        batch_number = 0
        more_data = True
        index_1 = 0
        index_2 = self.batch_size
        while more_data:
            next_in = data[index_1:index_2]
            self.add_to_queue(next_in)
            batch_number += 1
            index_1 += self.batch_size
            index_2 += self.batch_size
            if batch_number == num_batches:
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


def main():
    """Main docstring"""
    start = timer()
    # output_name = "file"
    # log_file = "/Users/andrewbailey/data/log-file.1"
    # kwargs = {"eventalign": False, "prob":True, "length": 5, "alphabet": "ATGC", "nanonet": False, "deepnano": True}
    #
    # create_training_data_from_log(log_file, output_name, kwargs, output_dir=project_folder()+"/training")

    training_dir = project_folder()+"/training"

    training_files = list_dir(training_dir, ext="npy")
    # print(training_files)
    queue1 = Queue(maxsize=10)
    queue2 = Queue(maxsize=10)

    training = Data(queue1, training_files, 120, verbose=True)
    testing = Data(queue2, training_files, 120)

    # data.run()
    process1 = Process(target=training.load_data, args=())
    process2 = Process(target=testing.load_data, args=())

    process1.start()
    process2.start()

    # res.get()
    print("Doing something else")
    for i in range(1000):
        time.sleep(0.1)
        print("Testing")
        print(training.get_batch()[0][0])
        if i % 10 == 0:
            print("Training")
            print(testing.get_batch()[0][0])




    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    # f = testfast5()
    # # print(f)
    # ed_params = {'window_lengths':[3, 6]}
    # # for f in list_dir(project_folder()+"/test-files/r9/canonical", ext="fast5"):
    # window_lengths = [3, 6]
    # thresholds=[8.0, 4.0]
    # peak_height = 1.0
    # with Fast5(f) as fh:
    #     # print(fh.sample_rate)
    #     raw_data1 = fh.get_read(raw=True, scale=False)
    #     # print(max(raw_data1)-min(raw_data1))
    #     print(len(raw_data1))
    #     raw_data = fh.get_read(raw=True, scale=True)
    #     # print(min(raw_data), max(raw_data))
    #     sums, sumsqs = compute_sum_sumsq(raw_data)
    #     # print(sums, sumsqs)
    #     tstats = []
    #     for i, w_len in enumerate(window_lengths):
    #         # print(i, w_len)
    #         tstat = compute_tstat(sums, sumsqs, w_len, False)
    #         tstats.append(tstat)
    #     peaks = short_long_peak_detector(tstats, thresholds, window_lengths, peak_height)
    #     print(len(peaks))
    #     # print(tstats[0][10:100])
    #     events = minknow_event_detect(
    #         fh.get_read(raw=True), fh.sample_rate, **ed_params)
    #
    # # print()
    # print(events[0])
    # print(np.mean(raw_data[0:10]))
    # # print(len(raw_data1)/len(events))
    # print(max(np.ediff1d(peaks)))
    # print(min(np.ediff1d(peaks)))
    # print(np.mean(np.ediff1d(peaks)))
    raise SystemExit
