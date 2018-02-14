#!/usr/bin/env python
"""Re-segment raw read and re-label the data"""
########################################################################
# File: event_detection.py
#  executable: event_detection.py
#
# Author: Andrew Bailey
# History: 10/6/17 Created
########################################################################

import sys
import os
import collections
import re
import numpy as np
from collections import defaultdict
from timeit import default_timer as timer
from PyPore.parsers import SpeedyStatSplit
from nanotensor.fast5 import Fast5
from nanonet.eventdetection.filters import minknow_event_detect
from nanonet.segment import segment
from nanonet.features import make_basecall_input_multi
from py3helpers.utils import test_numpy_table, list_dir, create_fastq, merge_two_dicts, TimeStamp, change_np_field_type


def create_speedy_event_table(signal, sampling_freq, start_time, min_width=5, max_width=80, min_gain_per_sample=0.008,
                              window_width=800):
    """Create new event table using SpeedyStatSplit Event detection

    :param signal: list or array of signal in pA for finding events
    :param sampling_freq: sampling frequency of ADC in Hz
    :param start_time: start time from fast5 file (time in seconds * sampling frequency
    :param min_width: param for SpeedyStatSplit
    :param max_width: param for SpeedyStatSplit
    :param min_gain_per_sample: param for SpeedyStatSplit
    :param window_width: param for SpeedyStatSplit
    :return: Table of events without model state or move information
    """
    assert np.sign(start_time) == 1, "Start time has to be positive: {}".format(start_time)
    # define speedy stat split
    parser = SpeedyStatSplit(min_width=min_width, max_width=max_width,
                             min_gain_per_sample=min_gain_per_sample,
                             window_width=window_width, sampling_freq=sampling_freq)
    # parse events
    events = parser.parse(np.asarray(signal, dtype=np.float64))
    num_events = len(events)
    # create empty event table
    event_table = np.empty(num_events, dtype=[('start', float), ('length', float),
                                              ('mean', float), ('stdv', float),
                                              ('model_state', 'S5'), ('move', '<i4'),
                                              ('raw_start', int), ('raw_length', int),
                                              ('p_model_state', float)])
    # set events into event table
    for i, event in enumerate(events):
        event_table['start'][i] = event.start / sampling_freq + (start_time / sampling_freq)
        event_table['raw_start'][i] = event.start
        event_table['length'][i] = event.duration / sampling_freq
        event_table['raw_length'][i] = event.duration
        event_table['mean'][i] = event.mean
        event_table['stdv'][i] = event.std

    return event_table


def create_minknow_event_table(signal, sampling_freq, start_time,
                               window_lengths=(16, 40), thresholds=(8.0, 4.0), peak_height=1.0):
    """Create new event table using minknow_event_detect event detection

    :param signal: list or array of signal in pA for finding events
    :param sampling_freq: sampling frequency of ADC in Hz
    :param start_time: start time from fast5 file (time in seconds * sampling frequency
    :param window_lengths: t-test windows for minknow_event_detect
    :param thresholds: t-test thresholds for minknow_event_detect
    :param peak_height: peak height param for minknow_event_detect
    :return: Table of events without model state or move information
    """
    assert np.sign(start_time) == 1, "Start time has to be positive: {}".format(start_time)
    events = minknow_event_detect(np.asarray(signal, dtype=float), sample_rate=sampling_freq,
                                  get_peaks=False, window_lengths=window_lengths,
                                  thresholds=thresholds, peak_height=peak_height)
    num_events = len(events)
    event_table = np.empty(num_events, dtype=[('start', float), ('length', float),
                                              ('mean', float), ('stdv', float),
                                              ('model_state', 'S5'), ('move', '<i4'),
                                              ('raw_start', int), ('raw_length', int),
                                              ('p_model_state', float)])
    for i, event in enumerate(events):
        event_table['start'][i] = event["start"] + (start_time / sampling_freq)
        event_table['length'][i] = event["length"]
        event_table['mean'][i] = event["mean"]
        event_table['stdv'][i] = event["stdv"]
        event_table['raw_start'][i] = np.round(event["start"] * sampling_freq)
        event_table['raw_length'][i] = np.round(event["length"] * sampling_freq)

    return event_table


def create_anchor_kmers(new_events, old_events):
    """
    Create anchor kmers for new event table.

    Basically, grab kmer and move information from previous event table and
    pull events covering the same time span into new event table.
    :param new_events: new event table
    :param old_events: event table from Fast5 file
    :return New event table
    """
    # remove all stays
    # print(old_events[:6])

    moves = np.asarray(old_events['move'], dtype=bool)
    old_events = old_events[moves]
    test_numpy_table(new_events, req_fields=('start', 'length', 'mean', 'stdv', 'model_state', 'move', 'p_model_state'))
    test_numpy_table(old_events, req_fields=('start', 'length', 'mean', 'stdv', 'model_state', 'move', 'p_model_state'))
    # current kmer
    print(old_events[:10])

    print(new_events[0:10])
    old_indx = 0
    move = 0
    # indexes to trim events based on information from basecalled events table
    start_index = 0
    end_index = len(new_events)
    most_events = 0
    for i, event in enumerate(new_events):
        # skip events that occur before labels
        try:
            if old_events[0]["start"] + old_events[0]["length"] < event["start"]:
                kmers = defaultdict(int)
                probs = defaultdict(int)
                moves = defaultdict(int)
                print(i, "old start", old_events[old_indx]["start"],  "new_end", event["start"] + event["length"])
                # if first event or event start is after current old_event start.
                while round(old_events[old_indx]["start"], 7) < round(event["start"] + event["length"], 7):
                    print(old_events[old_indx]["start"] < event["start"] + event["length"])
                    if round(old_events[old_indx+1]["start"], 7) > round(event["start"] + event["length"], 7):
                        kmers[bytes.decode(old_events[old_indx]["model_state"])] += event["start"] + event["length"] - old_events[old_indx]["start"]
                    else:
                        kmers[bytes.decode(old_events[old_indx]["model_state"])] += old_events[old_indx+1]["start"] - old_events[old_indx]["start"]

                    probs[bytes.decode(old_events[old_indx]["model_state"])] += old_events[old_indx]["p_model_state"]
                    moves[bytes.decode(old_events[old_indx]["model_state"])] += old_events[old_indx]["move"]
                    old_indx += 1
                    print(old_indx, len(kmers), kmers, probs)

                num_kmers = len(kmers.keys())

                if most_events < num_kmers:
                    most_events = num_kmers
                    if num_kmers > 50:
                        pass
                        # raise SystemExit
                time_in_event1 = event["start"] + event["length"] - old_events[old_indx]["start"]
                time_in_event2 = old_events[old_indx + 1]["start"] - (event["start"]+event["length"])

                if num_kmers == 1:
                    kmer = list(kmers.keys())[0]
                    prob = probs[kmer]
                    move = moves[kmer]
                elif num_kmers > 1:
                    kmer = max(kmers.keys(), key=lambda k: kmers[k])
                    prob = probs[kmer]
                    move = moves[kmer]
                elif num_kmers == 0:
                    prob = probs[0]
                    move = 0
                    kmer = prev_kmer

                event["p_model_state"] = prob
                event["move"] = move
                event["model_state"] = kmer

                prev_kmer = kmer
            else:
                start_index = i + 1

        except IndexError:
            # end of events
            print("most kmers", most_events)
            end_index = i
            break

    return new_events[start_index:end_index]


def resegment_reads(fast5_path, params, speedy=False, overwrite=False, name="ReSegmentBasecall_00{}"):
    """Re-segment and create anchor alignment from previously base-called fast5 file
    :param fast5_path: path to fast5 file
    :param params: event detection parameters
    :param speedy: boolean option for speedyStatSplit or minknow
    :param overwrite: overwrite a previous event re-segmented event table
    :param name: name of key where events table will be placed (Analyses/'name'/Events)
    :return True when completed
    """
    assert os.path.isfile(fast5_path), "File does not exist: {}".format(fast5_path)
    # create Fast5 object
    f5fh = Fast5(fast5_path, read='r+')
    read_id = bytes.decode(f5fh.raw_attributes['read_id'])
    sampling_freq = f5fh.sample_rate
    start_time = f5fh.raw_attributes['start_time']
    # pick event detection algorithm
    signal = f5fh.get_read(raw=True, scale=True)

    if speedy:
        event_table = create_speedy_event_table(signal, sampling_freq, start_time, **params)
        params = merge_two_dicts(params, {"event_detection": "speedy_stat_split"})

    else:
        event_table = create_minknow_event_table(signal, sampling_freq, start_time, **params)
        params = merge_two_dicts(params, {"event_detection": "minknow_event_detect"})

    keys = ["nanotensor version", "time_stamp"]
    values = ["0.2.0", TimeStamp().posix_date()]
    version_info = merge_two_dicts(params, dict(zip(keys, values)))
    attributes = merge_two_dicts(f5fh.raw_attributes, version_info)
    # gather previous event detection
    old_event_table = f5fh.get_basecall_data()
    if f5fh.is_read_rna():
        old_event_table = change_np_field_type(old_event_table, 'start', float)
        old_event_table = change_np_field_type(old_event_table, 'length', float)
        old_event_table["start"] = old_event_table["start"] / sampling_freq + (start_time / sampling_freq)
        old_event_table["length"] = old_event_table["length"] / float(sampling_freq)

    # set event table
    new_event_table = create_anchor_kmers(new_events=event_table, old_events=old_event_table)
    f5fh.set_new_event_table(name, new_event_table, attributes, overwrite=overwrite)
    # gather new sequence
    sequence = sequence_from_events(new_event_table)
    quality_scores = '!'*len(sequence)
    fastq = create_fastq(read_id+" :", sequence, quality_scores)
    # set fastq
    f5fh.set_fastq(name, fastq)
    return f5fh


def sequence_from_events(events):
    """Get new read from event table with 'model_state' and 'move' fields

    :param events: event table with 'model_state' and 'move' fields

    """
    test_numpy_table(events, req_fields=("model_state", "move"))
    bases = []
    for i, event in enumerate(events):
        if i == 0:
            bases.extend([chr(x) for x in event['model_state']])

        else:
            if event['move'] > 0:
                bases.append(bytes.decode
                             (event['model_state'][-event['move']:]))
    sequence = ''.join(bases)
    return sequence


def main():
    """Main docstring"""
    start = timer()

    dna_reads = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical/"
    rna_reads = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/rna_reads"

    dna_minknow_params = dict(window_lengths=(5, 10), thresholds=(2.0, 1.1), peak_height=1.2)
    dna_speedy_params = dict(min_width=5, max_width=80, min_gain_per_sample=0.008, window_width=800)
    rna_minknow_params = dict(window_lengths=(5, 10), thresholds=(2.0, 1.1), peak_height=1.2)
    rna_speedy_params = dict(min_width=5, max_width=10, min_gain_per_sample=0.008, window_width=800)

    rna_files = list_dir(rna_reads, ext='fast5')
    dna_files = list_dir(dna_reads, ext='fast5')
    for fast5_path in rna_files:
        resegment_reads(fast5_path, rna_speedy_params, speedy=True, overwrite=True)
        print(fast5_path)
        break
    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
