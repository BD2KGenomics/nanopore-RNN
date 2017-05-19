#!/usr/bin/env python
"""Create features and labels from alignment kmers"""
########################################################################
# File: data_preparation.py
#  executable: data_preparation.py
# Purpose: Give multiple options to create features from event information
#           and create multiple ways to create labels
# TODO Implement scraping into signalAlign C code so we can just ouptut what #      we need directly from signalAlign
# Author: Andrew Bailey
# History: 05/17/17 Created
########################################################################

from __future__ import print_function
import sys
from collections import defaultdict
from timeit import default_timer as timer
import csv
from utils import get_project_file, project_folder
import h5py
import numpy as np
import numpy.lib.recfunctions
import itertools


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
    # NOTE Memory constraints and concerns regarding reading in a very long
    #       sequence. Write to file?
    # TODO Needs more testing/ error checking with different signalalign outputs
    labels = defaultdict(list)
    with open(tsv1) as tsv:
        reader = csv.reader(tsv, delimiter="\t")
        # NOTE Hardcoded column information from signalalign
        # Must be Full tsv output
        for line in reader:
            # chromosome = (line[0])
            # seq_pos = int(line[1])
            kmer = line[15]
            # name = line[3]
            strand = line[4]
            event_index = int(line[5])
            prob = float(line[12])
            if prob != 0:
                labels[event_index].append((kmer, prob))
            # only grab template strand
            if strand == "c":
                break
    return labels

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
    # TODO create these methods
    labels = create_labels(kmers)
    features = create_features(events)
    final = match_label_with_feature(features, labels)
    return final

def match_label_with_feature(features, labels):
    """Match indexed label with correct event"""
    # TODO Match indexed label with correct event
    return False

def create_labels(kmers, prob=True, small=False, length=5, alphabet="ATGC"):
    """Create labels from kmers"""
    # create probability vector
    if prob:
        labels = create_labels(kmers, alphabet=alphabet, length=length)
    elif small:
        labels = create_labels(kmers, alphabet=alphabet, length=length, prob=False)
    return labels

def create_prob_labels(kmers, alphabet="ATGC", length=5, prob=False):
    """Create probability label vector from dictionary of kmers"""
    # create a dictionary for kmers and the location within a vector
    kmer_dict = getkmer_dict(alphabet=alphabet, length=length)
    # loop through the dictionary and create a categorical vector with probabilities
    labels = defaultdict()
    for index, kmer_list in kmers.items():
        labels[index] = create_vector(kmer_list, kmer_dict, length=length, prob=prob)
    return labels

def getkmer_dict(alphabet="ATGC", length=5, flip=False):
    """Create a dictionary for kmers and the location within a vector"""
    # http://stackoverflow.com/questions/25942528/generating-all-dna-kmers-with-python
    # make sure always alphabetical
    alphabet = ''.join(sorted(alphabet))
    # create list of kmers
    fwd_map = [''.join(p) for p in itertools.product(alphabet, repeat=length)]
    # create dictionary depending on kmer to index or index to kmer
    if flip:
        # index are keys, kmers are values
        dictionary = dict(zip(range(len(fwd_map)), fwd_map))
    else:
        # Kmers are keys, index are values
        dictionary = dict(zip(fwd_map, range(len(fwd_map))))
    return dictionary



def create_prob_vector(kmer_list, kmer_dict, length=5):
    """Create a vector with given alphabet"""
    vector = numpy.zeros(len(kmer_dict))
    for kmer in kmer_list:
        trimmed_kmer = kmer[0][-length:]
        vector[kmer_dict[trimmed_kmer]] = kmer[1]
    # make sure vector sums to one if less than one
    # NOTE we could use sklearn module normalize which makes a unit vector
    vector = sum_to_one(vector)
    return vector

def create_categorical_vector(kmer_list, kmer_dict, length=5):
    """Create a vector with given alphabet"""
    vector = numpy.zeros(len(kmer_dict))
    highest = 0
    final_kmer = str()
    # check all kmers for most probable
    for kmer in kmer_list:
        trimmed_kmer = kmer[0][-length:]
        if kmer[1] > highest:
            highest = kmer[1]
            final_kmer = trimmed_kmer
    # put most probable into vector
    vector[kmer_dict[final_kmer]] = 1
    return vector


def create_vector(kmer_list, kmer_dict, length=5, prob=False):
    """Create a vector with given alphabet"""
    if prob:
        vector = create_prob_vector(kmer_list, kmer_dict, length=length)
    else:
        vector = create_categorical_vector(kmer_list, kmer_dict, length=length)
    return vector

def sum_to_one(vector):
    """Make sure a vector sums to one, if not, create diffuse vector"""
    total = sum(vector)
    if total != 1:
        if total > 1:
            # NOTE Do we want to deal with vectors with probability over 1?
            pass
        else:
            # NOTE this is pretty slow so maybe remove it?
            leftover = 1 - total
            amount_to_add = leftover/ (len(vector) - np.count_nonzero(vector))
            for index, prob in enumerate(vector):
                if prob == 0:
                    vector[index] = amount_to_add
    return vector


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
        raise ValueError("Must be a structured numpy array")
    new = numpy.zeros(np_struct_array.shape, dtype=np_struct_array.dtype.descr + descr)
    for name in np_struct_array.dtype.names:
        new[name] = np_struct_array[name]
    return new


def main():
    """Mainly used for testing the methods within data_preparation.py"""
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
