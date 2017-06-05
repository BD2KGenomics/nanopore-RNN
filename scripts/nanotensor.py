#!/usr/bin/env python
"""nanotensor is able to label nanopore data, create training files and
then use's tensorflow to train a mulit layer BLSTM-RNN"""
########################################################################
# File: nanotensor.py
#  executable: nanotensor.py
#
# Author: Andrew Bailey
# History: 05/28/17 Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
import json
from multiprocessing import Process, current_process, Manager
import numpy as np
from nanonet.fast5 import Fast5
from nanonet.eventdetection.filters import minknow_event_detect, compute_sum_sumsq, compute_tstat, short_long_peak_detector
from utils import testfast5, list_dir, project_folder
from data_preparation import TrainingData
from data import Data


def create_training_data(fast5_file, signalalign_file, kwargs,\
        output_name="file", output_dir=project_folder()):
    """Create npy training files from aligment and a fast5 file"""
    data = TrainingData(fast5_file, signalalign_file, **kwargs)
    data.save_training_file(output_name, output_dir=output_dir)
    print("FILE SAVED: {}".format(output_name+".npy"), file=sys.stderr)
    return True

def worker(work_queue, done_queue):
    """Worker function to generate training data from a queue"""
    try:
        # create training data until there are no more files
        for args in iter(work_queue.get, 'STOP'):
            create_training_data(**args)
        # catch errors
    except Exception as error:
        # pylint: disable=no-member,E1102
        done_queue.put("%s failed with %s" % (current_process().name, error.message))
        print("%s failed with %s" % (current_process().name, error.message), file=sys.stderr)



def main():
    """Main docstring"""
    start = timer()
    options = {"training-data":[{"filename": "file", "log_file": "/Users/andrewbailey/data/log-file.1", "output_folder": project_folder() + "/training2" }], "labeling-options":[{"prob":False, "kmer_len": 5, "alphabet": "ATGC", "nanonet": True}]}
    output_name = "file"
    log_file = "/Users/andrewbailey/data/log-file.1"
    kwargs = {"prob":False, "kmer_len": 5, "alphabet": "ATGC", "nanonet": True}
    with open('options.json', 'w') as outfile:
        json.dump(options, outfile, indent=4)
    #
    # with open('data.txt') as json_file:
    #     data = json.load(json_file)

    workers = 4
    work_queue = Manager().Queue()
    done_queue = Manager().Queue()
    jobs = []

    counter = 0
    with open(log_file, 'r') as log:
        for line in log:
            line = line.rstrip().split('\t')
            fast5 = line[0]
            tsv = line[1]
            name = output_name+str(counter)
            output_dir = project_folder() + "/training2"
            arguments = {"fast5_file": fast5, "signalalign_file": tsv, "kwargs": kwargs, "output_name": name, "output_dir":output_dir}
            # process = Process(target=create_training_data, args=(fast5, \
            #  tsv, kwargs, name, project_folder() + "/training2"))
            # jobs.append(process)
            work_queue.put(arguments)

            counter += 1

    # start creating files using however many workers specified
    for _ in xrange(workers):
        p = Process(target=worker, args=(work_queue, done_queue))
        p.start()
        jobs.append(p)
        work_queue.put('STOP')

    for p in jobs:
        p.join()

    done_queue.put('STOP')
    print("\n#  nanotensor - finished creating data set\n", file=sys.stderr)
    print("\n#  nanotensor - finished creating data set\n", file=sys.stdout)
    # check how long the whole program took
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
