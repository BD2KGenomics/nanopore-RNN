#!/usr/bin/env python2.7
"""Create features and labels from alignment kmers"""
########################################################################
# File: data_preparation.py
#  executable: data_preparation.py
# Purpose: Give multiple options to create features from event information
#           and create multiple ways to create labels
# TODO Implement scraping into signalAlign C code so we can just ouptut what
#      we need directly from signalAlign
# TODO Create updates on when things are being completed
# Author: Andrew Bailey
# History: 05/17/17 Created
########################################################################

from __future__ import print_function
import sys
from collections import defaultdict
from timeit import default_timer as timer
import csv
import itertools
import os
from utils import get_project_file, project_folder, sum_to_one
import h5py
import numpy as np
import numpy.lib.recfunctions
from nanonet.features import events_to_features as nanonet_features


class TrainingData(object):
    """docstring for TrainingData."""
    def __init__(self, fast5_file, alignment_file, strand_name="template", prob=False, kmer_len=5, alphabet="ATGC", nanonet=True, deepnano=False):
        super(TrainingData, self).__init__()
        self.fast5_file = fast5_file
        self.alignment_file = alignment_file
        assert self.fast5_file.endswith("fast5")
        assert self.alignment_file.endswith("tsv")
        self.alphabet = ''.join(sorted(alphabet))
        self.length = kmer_len
        self.prob = prob
        self.nanonet = nanonet
        self.deepnano = deepnano
        self.strand_name = strand_name
        if self.deepnano:
            print("Deepnano events not completed", file=sys.stderr)
            #TODO create error for incomplete work

        self.events = str()
        self.kmers = str()
        self.labels = str()
        self.features = str()
        self.training_file = str()


    def scrape_fast5_events(self, fields=None):
        """Scrape a fast5 file for event information"""
        # TODO *Access other places than just the events within Basecall_1D_000
        # TODO May want to have error checking when grabbing fields
        if fields is None:
            fields = ["mean", "start", "stdv", "length"]
        with h5py.File(self.fast5_file, 'r+') as fast5:
            if self.strand_name == "template":
                template = fast5.get("Analyses/Basecall_1D_000/BaseCalled_template/Events").value
                Strand = np.array(template)
            elif self.strand_name == "complement":
                complement = fast5.get("Analyses/Basecall_1D_000/BaseCalled_complement/Events").value
                Strand = np.array(complement)
        events = Strand[fields]
        return events

    def scrape_signalalign(self):
        """Grab all the event kmers from the signal align output and record probability"""
        # NOTE Memory constraints and concerns regarding reading in a very long
        #       sequence. Write to file?
        # TODO Needs more testing/ error checking with different signalalign outputs
        # NOTE This takes way too long to scrape the data if threshold = 0
        # NOTE if deepnano labels work best we may want to change data structure
        kmers = defaultdict(list)
        with open(self.alignment_file) as tsv:
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
                # only grab template strand
                if self.strand_name == "template":
                    if strand =="t":
                        kmers[event_index].append((kmer, prob))
                elif self.strand_name == "complement":
                    if strand =="c":
                        kmers[event_index].append((kmer, prob))
        return kmers

    def scrape_eventalign(self):
        """Grab all the event kmers from the eventalign output and record probability"""
        # TODO make same method?
        # data = list()
        # with open(tsv1) as tsv:
        #     reader = csv.reader(tsv, delimiter="\t")
        #     for line in reader:
        return False

    def match_label_with_feature(self):
        """Match indexed label with correct event"""
        # TODO fix the data type so that everything is a nparray
        final_matrix = []
        prev_counter = -1
        for index, label in sorted(self.labels.items()):
            counter = index
            if prev_counter != -1:
                if counter != prev_counter+1:
                    null = self.create_null_label()
                    final_matrix.append([self.features[prev_counter+1], null])
            final_matrix.append([self.features[index], label])
            prev_counter = index
        final_matrix = np.asanyarray(final_matrix)
        return final_matrix

    def create_null_label(self):
        """For unlabelled events from signalalign create a vector with last item in vector as 1"""
        vector_len = (len(self.alphabet)**self.length)
        vector = numpy.zeros(vector_len+1)
        vector[vector_len] = 1
        return vector

    def create_labels(self):
        """Create labels from kmers"""
        # create probability vector
        if self.deepnano:
            # this method is not built yet
            labels = self.create_deepnano_labels()
        else:
            # create categorical vector with probability or simple binary classification
            labels = self.create_kmer_labels()
        return labels

    def create_deepnano_labels(self):
        """Create labels like deepnano (XX, NX, NN) where X in {alphabet} and N
        is an unknown character"""
        # TODO Still need to know the exact shape of the label vector
        # for index, kmer_list in kmers.items():
        #     pass
        return False

    def create_kmer_labels(self):
        """Create probability label vector from dictionary of kmers"""
        # create a dictionary for kmers and the location within a vector
        kmer_dict = self.getkmer_dict(self.alphabet, self.length)
        # loop through the dictionary and create a categorical vector with probabilities
        labels = defaultdict()
        for index, kmer_list in self.kmers.items():
            labels[index] = self.create_vector(kmer_list, kmer_dict)
        return labels

    @staticmethod
    def getkmer_dict(alphabet, length, flip=False):
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

    def create_vector(self, kmer_list, kmer_dict):
        """Decide which method to use to create a vector with given alphabet from a list of kmers"""
        if self.prob:
            vector = self.create_prob_vector(kmer_list, kmer_dict)
        else:
            vector = self.create_categorical_vector(kmer_list, kmer_dict)
        return vector

    def create_prob_vector(self, kmer_list, kmer_dict):
        """Create a probability vector with given alphabet dictionary and length"""
        vector = numpy.zeros(len(kmer_dict)+1)
        for kmer in kmer_list:
            trimmed_kmer = kmer[0][-self.length:]
            vector[kmer_dict[trimmed_kmer]] = kmer[1]
        # make sure vector sums to one if less than one
        # vector = sum_to_one(vector)
        return vector

    def create_categorical_vector(self, kmer_list, kmer_dict):
        """Create a vector with given alphabet"""
        vector = numpy.zeros(len(kmer_dict)+1)
        highest = 0
        final_kmer = str()
        # check all kmers for most probable
        for kmer in kmer_list:
            trimmed_kmer = kmer[0][-self.length:]
            if kmer[1] >= highest:
                highest = kmer[1]
                final_kmer = trimmed_kmer
        # put most probable into vector
        vector[kmer_dict[final_kmer]] = 1
        return vector

    def create_features(self):
        """Create features from events"""
        if self.nanonet:
            features = nanonet_features(self.events)
        else:
            features = self.deepnano_events(self.events)
        return features

    def deepnano_events(self, shift=1, scale=1, scale_sd=1):
        """Replicating deepnano's feature definition"""
        new_events = []
        for event in self.events:
            # TODO deal with shift, scale and scale_sd
            # mean = (event["mean"] - shift) / scale
            # stdv = event["stdv"] / scale_sd
            mean = event["mean"]
            stdv = event["stdv"]
            length = event["length"]
            new_events.append(self.preproc_event(mean, stdv, length))
        new_events = np.asarray(new_events)
        return new_events

    @staticmethod
    def preproc_event(mean, std, length):
        "Normalizing event information for deepnano feature generation"
        mean = mean / 100.0 - 0.66
        std = std - 1
        return [mean, mean*mean, std, length]

    def save_training_file(self, output_name, output_dir=project_folder()):
        """Create training file and save it as an .npy file"""
        assert os.path.isdir(output_dir)
        self.events = self.scrape_fast5_events()
        self.kmers = self.scrape_signalalign()
        self.labels = self.create_labels()
        self.features = self.create_features()
        self.training_file = self.match_label_with_feature()
        output_file = os.path.join(output_dir, output_name)
        np.save(output_file, self.training_file)
        return output_file

    def interpolate(self):
        """Guess a distribution of data"""
        return "from scipy.interpolate import interp1d"

def main():
    """Mainly used for testing the methods within data_preparation.py"""
    start = timer()
    #
    fast5_file = \
    get_project_file("test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch146_read1209_strand1.fast5")
    #
    signalalign_file = \
    get_project_file("/temp/tempFiles_alignment/132de6a8-df1e-468f-848b-abc960e1fc76_Basecall_2D_template.sm.backward.tsv")
    #
    kwargs = {"eventalign": False, "prob":True, "length": 5,"strand_name": "template", "alphabet": "ATGCE", "nanonet": True, "deepnano": False}

    data = TrainingData(fast5_file, signalalign_file, strand_name="template", prob=False, kmer_len=5, alphabet="ATGC", nanonet=True, deepnano=False)
    data.save_training_file("testing")
    # print(getkmer_dict()["TTTTT"])
    # create_kmer_labels({1:[["TTTTT", 1]]})
    # training_file = prepare_training_file(fast5_file, signalalign_file)
    # np.save(project_folder()+"/events", training_file)
    # np.load(project_folder()+"/events.npy")

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit


# some extra code needed for later stuff
# import numpy as np
# from nanonet.fast5 import Fast5
# from nanonet.eventdetection.filters import minknow_event_detect, compute_sum_sumsq, compute_tstat, short_long_peak_detector
# from utils import testfast5, list_dir, project_folder, merge_two_dicts

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
