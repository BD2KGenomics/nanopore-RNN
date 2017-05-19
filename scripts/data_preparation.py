#!/usr/bin/env python
"""Create features and labels from alignment kmers"""
########################################################################
# File: data_preparation.py
#  executable: data_preparation.py
# Purpose: Give multiple options to create features from event information
#           and create multiple ways to create labels
# TODO
# Author: Andrew Bailey
# History: 05/17/17 Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
import csv
from utils import get_project_file, project_folder
import h5py
import numpy as np
import numpy.lib.recfunctions

def scrape_fast5_events(fast5_file, fields=None):
    """Scrape a fast5 file for event information"""
    # TODO Access other places than just the events within Basecall_1D_000
    # TODO May want to have error checking when grabbing fields
    assert fast5_file.endswith("fast5")
    if fields is None:
        fields = ["mean", "start", "stdv", "length", "model_state", "move"]
    with h5py.File(fast5_file, 'r+') as fast5:
        template = fast5.get("Analyses/Basecall_1D_000/BaseCalled_template/Events").value
    template = np.array(template)
    events = template[fields]
    return events

def scrape_signalalign(tsv1):
    """Grab all the event kmers from the signal align output and record probability"""
    data = dict()
    with open(tsv1) as tsv:
        reader = csv.reader(tsv, delimiter="\t")
        for line in reader:
            chromosome = np.string_(line[0])
            seq_pos = int(line[1])
            kmer = np.string_(line[15])
            name = line[3]
            template = line[4]
            event_index = int(line[5])
            prob = float(line[12])
    return False

# TODO make same method?
def scrape_eventalign(tsv1):
    """Grab all the event kmers from the signal align output and record probability"""
    data = dict()
    with open(tsv1) as tsv:
        reader = csv.reader(tsv, delimiter="\t")
        for line in reader:
            chromosome = np.string_(line[0])
            seq_pos = int(line[1])
            kmer = np.string_(line[15])
            name = line[3]
            template = line[4]
            event_index = int(line[5])
            prob = float(line[12])
    return False

def prepare_training_file(fast5, tsv, eventalign=False):
    """Gather event information from fast5 file and kmer labels from eventalign or signalalign"""
    if eventalign:
        kmers = scrape_eventalign(tsv)
    else:
        kmers = scrape_signalalign(tsv)
    events = scrape_fast5_events(fast5)
    features = create_features(events)
    labels = create_labels(kmers)
    final = match_label_with_feature(features, labels)
    return final

def match_label_with_feature(features, labels):
    """Match indexed label with correct event"""
    # TODO Match indexed label with correct event
    return False

def create_labels(kmers, prob=True, small=False):
    """Create labels from kmers"""
    # TODO create lables from kmer numpy array
    if prob:
        labels = create_prob_labels(kmers)
    if small:
        labels = create_small_labels(kmers)
    return labels

def create_small_labels(kmers):
    """Create lable vector from kmers"""
    # TODO create lables from kmer numpy array
    return False

def create_prob_labels(kmers):
    """Create lable vector from kmers"""
    # TODO create lables from kmer numpy array
    return False


def create_features(events, basic=True, other_option=False):
    """Create features from events"""
    # TODO create features from event numpy array
    if basic:
        features = basic_method(events)
    elif other_option:
        features = other_method(events)
    return features

def basic_method(events):
    """Create features """
    # TODO create features from event numpy array
    return False

def other_method(events):
    """Create features """
    # TODO create features from event numpy array
    return False


def add_field(np_struct_array, descr):
    """Return a new array that is like the structured numpy array, but has additional fields.

    descr looks like descr=[('test', '<i8')]
    """
    if np_struct_array.dtype.fields is None:
        raise ValueError("'A' must be a structured numpy array")
    new = numpy.zeros(np_struct_array.shape, dtype=np_struct_array.dtype.descr + descr)
    for name in np_struct_array.dtype.names:
        new[name] = np_struct_array[name]
    return new


def main():
    """Main docstring"""
    start = timer()

    fast5_file = \
    get_project_file("test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch146_read1209_strand1.fast5")

    signalalign_file = \
    get_project_file("/temp/tempFiles_alignment/132de6a8-df1e-468f-848b-abc960e1fc76_Basecall_2D_template.sm.backward.tsv")

    print(fast5_file)
    events = scrape_fast5_events(fast5_file)
    print(events[0])
    # np.savez(project_folder()+"/events", events)
    # np.load(project_folder()+"/events.npz")

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
