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
import numpy as np
from nanonet.fast5 import Fast5
from nanonet.eventdetection.filters import minknow_event_detect, compute_sum_sumsq, compute_tstat, short_long_peak_detector
from utils import testfast5, list_dir, project_folder, Data
from data_preparation import TrainingData
from multiprocessing import Process


def create_training_data(fast5_file, signalalign_file, kwargs,\
        output_name="file", output_dir=project_folder()+"/training2"):
    """Create npy training files"""
    data = TrainingData(fast5_file, signalalign_file, **kwargs)
    data.save_training_file(output_name, output_dir=output_dir)
    return True


def main():
    """Main docstring"""
    start = timer()
    output_name = "file"
    log_file = "/Users/andrewbailey/data/log-file.1"
    kwargs = {"prob":False, "kmer_len": 5, "alphabet": "ATGC", "nanonet": True}

    # create_training_data(fast5, tsv, kwargs, output_dir=project_folder()+"/training")
    jobs = []

    counter = 0
    with open(log_file, 'r') as log:
        for line in log:
            line = line.rstrip().split('\t')
            fast5 = line[0]
            tsv = line[1]
            output_name = output_name+str(counter)
            process = Process(target=create_training_data, args=(fast5, \
             tsv, kwargs, output_name, project_folder() + "/training2"))
            jobs.append(process)
            counter += 1

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()



    # get training files
    # training_dir = project_folder()
    # training_files = list_dir(training_dir, ext="npy")
    # print(training_files)
    # # create data instances
    # training = Data(training_files, 100, queue_size=10, verbose=True)
    # testing = Data(training_files, 100, queue_size=10)
    # training.start()
    # testing.start()
    # training.end()
    # testing.end()


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
