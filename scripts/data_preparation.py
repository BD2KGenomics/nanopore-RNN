#!/usr/bin/env python
"""Create features and labels from alignment kmers"""
########################################################################
# File: data_preparation.py
#  executable: data_preparation.py
# Purpose: Give multiple options to create features from event information
#           and create multiple ways to create labels
# TODO Implement scraping into signalAlign C code so we can just ouptut what
#      we need directly from signalAlign
# Author: Andrew Bailey
# History: 05/17/17 Created
########################################################################

from __future__ import print_function
import sys
from collections import defaultdict
from timeit import default_timer as timer
import csv
import itertools
from utils import get_project_file, project_folder, sum_to_one
import h5py
import numpy as np
import numpy.lib.recfunctions
from nanonet.features import events_to_features as nanonet_features

def scrape_fast5_events(fast5_file, fields=None):
    """Scrape a fast5 file for event information"""
    # TODO Access other places than just the events within Basecall_1D_000
    # TODO May want to have error checking when grabbing fields
    assert fast5_file.endswith("fast5")
    if fields is None:
        fields = ["mean", "start", "stdv", "length"]
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
    # NOTE This takes way too long to scrape the data
    # NOTE if deepnano labels work best we may want to change data structure
    kmers = defaultdict(list)
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
            kmers[event_index].append((kmer, prob))
            # only grab template strand
            if strand == "c":
                break
    return kmers

def scrape_eventalign(tsv1):
    """Grab all the event kmers from the signal align output and record probability"""
    # TODO make same method?
    data = list()
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
    labels = create_labels(kmers)
    features = create_features(events, deepnano=True)
    final = match_label_with_feature(features, labels)
    return final

def match_label_with_feature(features, labels):
    """Match indexed label with correct event"""
    final_matrix = []
    prev_counter = -1
    for index, label in sorted(labels.items()):
        counter = index
        if prev_counter != -1:
            if counter != prev_counter+1:
                # TODO Create error message and break program
                print("This should be an error, Raise error message?", file=sys.stderr)
        final_matrix.append([features[index], label])
        prev_counter = index
    return final_matrix

def create_labels(kmers, prob=False, deepnano=False, length=5, alphabet="ATGC"):
    """Create labels from kmers"""
    # create probability vector
    if deepnano:
        # this method is not built yet
        create_deepnano_labels(kmers, alphabet=alphabet)
    else:
        # create categorical vector with probability or simple binary classification
        labels = create_kmer_labels(kmers, alphabet=alphabet, length=length, prob=prob)
    return labels

def create_deepnano_labels(kmers, alphabet="ATGC"):
    """Create labels like deepnano (XX, NX, NN) where X in {alphabet} and N
    is an unknown character"""
    # TODO Still need to know the exact shape of the label vector
    for index, kmer_list in kmers.items():
        pass
    return False

def create_kmer_labels(kmers, alphabet="ATGC", length=5, prob=False):
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
        if kmer[1] >= highest:
            highest = kmer[1]
            final_kmer = trimmed_kmer
    # put most probable into vector
    vector[kmer_dict[final_kmer]] = 1
    return vector

def create_vector(kmer_list, kmer_dict, length=5, prob=False):
    """Decide which method to use to create a vector with given alphabet from a list of kmers"""
    if prob:
        vector = create_prob_vector(kmer_list, kmer_dict, length=length)
    else:
        vector = create_categorical_vector(kmer_list, kmer_dict, length=length)
    return vector

def create_features(events, basic=False, nanonet=False, deepnano=False):
    """Create features from events"""
    if basic:
        features = basic_method(events)
    elif nanonet:
        features = nanonet_features(events).tolist()
    elif deepnano:
        features = deepnano_events(events)
    return features

def deepnano_events(events, shift=1, scale=1, scale_sd=1):
    """Replicating deepnano's feature definition"""
    new_events = []
    for event in events:
        # TODO deal with shift, scale and scale_sd
        # mean = (event["mean"] - shift) / scale
        # stdv = event["stdv"] / scale_sd
        mean = event["mean"]
        stdv = event["stdv"]
        length = event["length"]
        new_events.append(preproc_event(mean, stdv, length))
    return new_events

def preproc_event(mean, std, length):
    "Normalizing event information for deepnano feature generation"
    mean = mean / 100.0 - 0.66
    std = std - 1
    return [mean, mean*mean, std, length]

def basic_method(events):
    """Create features """
    return events





def main():
    """Mainly used for testing the methods within data_preparation.py"""
    start = timer()

    fast5_file = \
    get_project_file("test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch146_read1209_strand1.fast5")

    signalalign_file = \
    get_project_file("/temp/tempFiles_alignment/132de6a8-df1e-468f-848b-abc960e1fc76_Basecall_2D_template.sm.backward.tsv")

    # print(fast5_file)
    # events = scrape_fast5_events(fast5_file)
    # kmers = scrape_signalalign(signalalign_file)
    # print(events[0])
    # print(kmers[100])

    training_file = prepare_training_file(fast5_file, signalalign_file)
    # TODO Make this part work!
    # TODO Make function to create a file from an array
    # TODO make a function to bundule this stuff
    np.savez(project_folder()+"/events", training_file)
    np.load(project_folder()+"/events.npz")

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
