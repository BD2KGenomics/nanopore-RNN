#!/usr/bin/env python2.7
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
import os
import subprocess
from nanotensor.utils import get_project_file, project_folder, sum_to_one, check_duplicate_characters
from nanotensor.error import Usage, DataPrepBug
import h5py
import numpy as np
import numpy.lib.recfunctions
from nanonet.features import events_to_features as nanonet_features
from signalalign.scripts.nanoporeParamRunner import estimate_params


class TrainingData(object):
    """docstring for TrainingData."""
    def __init__(self, fast5_file, alignment_file, strand_name="template", prob=False, kmer_len=5, alphabet="ATGC",
                 nanonet=True, deepnano=False, forward=True, cutoff=0.4,
                 template_model="../signalAlign/models/testModelR9p4_acegt_template.model",
                 complement_model="../signalAlign/models/testModelR9_complement_pop2.model"):

        self.fast5_file = fast5_file
        self.alignment_file = alignment_file
        assert self.fast5_file.endswith("fast5"), "Expecting ONT fast5 file: {}".format(fast5_file)
        assert self.alignment_file.endswith("tsv"), "Expecting signalAlign tsv file: {}".format(alignment_file)
        assert nanonet != deepnano, "Must select Deepnano or Nanonet"
        if deepnano:
            assert not prob, "Proabability vector is not an option when using deepnano data preparation"
        # define arguments
        self.alphabet = ''.join(sorted(alphabet))
        self.length = kmer_len
        self.prob = prob
        self.nanonet = nanonet
        self.deepnano = deepnano
        if self.deepnano:
            output = subprocess.check_output("estimateNanoporeParams;  exit 0", shell=True, stderr=subprocess.STDOUT)
            assert "Could not" == str(output)[:9], "estimateNanoporeParams is not in path"
        self.strand_name = strand_name
        self.forward = forward
        self.cutoff = cutoff
        self.debug = False
        # when data gets created it is stored in the class
        self.events = []
        self.kmers = []
        self.labels = []
        self.features = []
        self.training_file = []
        ## QC metrics for Deepnano labels
        self.missed = []
        self.template_lookup_table = template_model
        self.complement_lookup_table = complement_model

    def run_complete_analysis(self):
        """Run complete flow of analysis"""
        self.scrape_fast5_events()
        self.scrape_signalalign()
        self.create_labels()
        self.create_features()
        self.match_label_with_feature()
        return True

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
        self.events = events
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
                seq_pos = int(line[1])
                kmer = line[15]
                # name = line[3]
                strand = line[4]
                event_index = int(line[5])
                prob = float(line[12])
                # only grab template strand
                kmer_list = (kmer, prob, seq_pos)
                if self.strand_name == "template":
                    if strand =="t":
                        kmers[event_index].append(kmer_list)
                elif self.strand_name == "complement":
                    if strand =="c":
                        kmers[event_index].append(kmer_list)
        self.kmers = kmers
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
        final_matrix = []
        prev_counter = -1
        # go through labels based on event index
        for index, label in sorted(self.labels.items()):
            counter = index
            if prev_counter != -1:
                # if there are skipped events, label them with null labels
                while counter != prev_counter+1:
                    null = self.create_null_label()
                    final_matrix.append([self.features[prev_counter+1], null])
                    prev_counter += 1
            # label this specific event
            final_matrix.append([self.features[index], label])
            prev_counter = index
        # make np array
        final_matrix = np.asanyarray(final_matrix)
        self.training_file = final_matrix
        return final_matrix

    def create_null_label(self):
        """For unlabelled events from signalalign create a vector with last item in vector as 1"""
        if self.deepnano:
            kmer_dict = self.deepnano_dict(self.alphabet, self.length)
            vector = self.null_vector_deepnano(kmer_dict)
        elif self.prob:
            vector_len = (len(self.alphabet)**self.length)
            vector = numpy.array([1.0/vector_len]*vector_len)
        else:
            vector_len = (len(self.alphabet)**self.length)
            vector = numpy.zeros(vector_len+1)
            vector[vector_len] = 1
        return vector

    def create_labels(self):
        """Create labels from kmers"""
        # create probability vector
        if self.deepnano:
            # create deepnano labels from kmers
            labels = self.create_deepnano_labels(cutoff=self.cutoff, forward=self.forward)
        elif self.nanonet:
            # create categorical vector with probability or simple binary classification
            labels = self.create_kmer_labels()
        else:
            raise Usage("Need to select either deepnano or nanonet")
        self.labels = labels
        return labels

    def create_deepnano_labels(self, cutoff=0.2, forward=True):
        """Create labels like deepnano (XX, NX, NN) where X in {alphabet} and N
        is an unknown character"""
        # deepnano labels specificaly use 2 characters
        assert self.length == 2, "Deepnano labels have a length of 2 but {} was selected".format(self.length)

        # determine starting position depending on direction of read
        if forward:
            old_position = 0
        else: # backward
            old_position = float('inf')
        # get translation dictionary
        kmer_dict = self.deepnano_dict(self.alphabet, self.length)
        labels = defaultdict()
        first = True
        # keep track of previous probability and reference index
        old_prob = 0
        old_index = 0
        # keep track of mistakes and good events
        if self.debug:
            first_position = 0
            self.good_counter = 0
            self.counter_list = []
            probs = []
        # loop through all kmers based on event index
        for index, kmer_list in sorted(self.kmers.items()):
            # get most probable kmer for a given event
            best_kmer, prob, position = self.get_most_probable_kmer(kmer_list)
            if self.debug:
                probs.append(prob)
                if first:
                    first_position = position
            # check if reference position is save as previously assigned kmer
            if old_position == position:
                # if same but new event has a higher probability
                if old_prob < prob:
                    # label current index with correct label
                    labels[index] = labels[old_index]
                    # replace previously assigned kmer with a null vector
                    labels[old_index] = self.null_vector_deepnano(kmer_dict)
                else:
                    # current event has lower proability than previous so assign null vector
                    labels[index] = self.null_vector_deepnano(kmer_dict)
            elif prob < cutoff: # assign low probability events with null labels
                labels[index] = self.null_vector_deepnano(kmer_dict)
            else:
                # NOTE looping around the start and end of circular chromosome?
                if forward:
                    diff = position - old_position
                else: # backward
                    diff = old_position - position
                if diff > 0 or first: # if we moved in correct direction assign kmer
                    labels[index] = self.create_deepnano_vector(kmer_dict, diff, best_kmer, index, position, prob)
                    old_position, old_prob, old_index = position, prob, index
                    # not first event
                    first = False
                else: # moved in wrong direction
                    # assign null label
                    labels[index] = self.null_vector_deepnano(kmer_dict)
                    # print("###### CREATED null LABEL ###########", index, position, diff, prob)
                    # if difference is greater than -10 the alignment was reset and
                    if diff < -10:# or prob > cutoff:
                        # print("REASSIGN", index, position, diff, prob)
                        old_position = position
                        old_prob = prob
                        old_index = index
                        if self.debug:
                            self.counter_list.append(self.good_counter)
                            # print("NEW COUNTER = ", self.good_counter, diff, prob)
                            self.good_counter = 0
                    else:
                        if self.debug:
                            print("JUST A SKIP BACK = ", diff, index)

        if self.debug:
            print(sum(self.counter_list)/len(self.counter_list))
            print(max(self.counter_list), min(self.counter_list))
            print("Length of sequence =", abs(first_position - position), sum(self.counter_list))
            print("Number of misses =", len(self.missed))
            print("Skipped Bases =", sum([x[0] for x in self.missed][1:]))
            print("Percent Missed =", float(sum([x[0] for x in self.missed][1:]))/float(abs(first_position - position)) * 100)
            print((self.counter_list))
            print(sum(self.counter_list)/len(self.counter_list))
            print("Average Proabability = ", sum(probs)/(len(probs)))
        return labels

    def null_vector_deepnano(self, kmer_dict):
        "Creat null_vector_deepnano"
        vector = numpy.zeros(len(kmer_dict))
        zero_label = "N"*self.length
        vector[kmer_dict[zero_label]] = 1
        return vector

    def create_deepnano_vector(self, kmer_dict, diff, best_kmer, index=0, position=0, prob=0):
        """Label indices with correct deepnano label"""
        vector = numpy.zeros(len(kmer_dict))
        if diff <= self.length:
            kmer = best_kmer[-diff:]+("N" * (self.length-diff))
            if self.debug:
                self.good_counter += diff
        else:
            kmer = best_kmer[-self.length:]
            if self.debug:
                print("NEW COUNTER = ", self.good_counter, "Missed {}".format(diff-self.length), index, position, prob)
                self.counter_list.append(self.good_counter)
                self.good_counter = 2
                # print("Missed a base!!!", index, position, diff-self.length, prob)
                self.missed.append([diff-self.length, index, position])
        assert len(kmer) == self.length, "Length of kmer is not equal to defined length: len({}) != {}".format(kmer, self.length)
        try:
            # make sure if one kmer has two probabilities we assign the highest probability
            vector[kmer_dict[kmer]] = 1
            vector = sum_to_one(vector, prob=False)
        except KeyError as error:
            raise DataPrepBug("Kmer: {} not in reference kmer dictionary, check alphabet or length".format(error))
        return vector

    @staticmethod
    def get_most_probable_kmer(kmer_list):
        """Get most probable kmer"""
        prob = 0
        best_kmer = str()
        position = int()
        for kmer in kmer_list:
            new_kmer = kmer[0]
            # gets farthest to the right kmer if probabilities are equal
            if kmer[1] >= prob:
                prob = kmer[1]
                best_kmer = new_kmer
                position = kmer[2]
        return best_kmer, prob, position

    def deepnano_dict(self, alphabet, length, flip=False):
        """Create translation dictionary for deepnano labels"""
        assert "N" not in alphabet
        # remove =
        kmer_dict = self.getkmer_dict(alphabet+"N", length, flip=flip, deepnano=True)
        return kmer_dict

    def create_kmer_labels(self):
        """Create probability label vector from dictionary of kmers"""
        # create a dictionary for kmers and the location within a vector
        kmer_dict = self.getkmer_dict(self.alphabet, self.length, prob=self.prob)
        # loop through the dictionary and create a categorical vector with probabilities
        labels = defaultdict()
        for index, kmer_list in self.kmers.items():
            labels[index] = self.create_kmer_vector(kmer_list, kmer_dict)
        return labels

    @staticmethod
    def getkmer_dict(alphabet, length, flip=False, deepnano=False, prob=False):
        """Create a dictionary for kmers and the location within a vector"""
        # make sure there are no duplicates in the alphabet
        alphabet = check_duplicate_characters(alphabet)
        # make sure always alphabetical
        alphabet = ''.join(sorted(alphabet))
        # create list of kmers
        kmers = [''.join(p) for p in itertools.product(alphabet, repeat=length)]
        if deepnano:
            # remove kmers with incorrect syntax
            kmers = [kmer for kmer in kmers if kmer.find("N") == -1 or kmer ==\
                    kmer[:kmer.find("N")]+("N"*(length-kmer.find("N")))]
        else:
            # add null label for categorical classification
            if not prob:
                kmers.append("N"*length)

        # create dictionary depending on kmer to index or index to kmer
        if flip:
            # index are keys, kmers are values
            dictionary = dict(zip(range(len(kmers)), kmers))
        else:
            # Kmers are keys, index are values
            dictionary = dict(zip(kmers, range(len(kmers))))
        return dictionary

    def create_kmer_vector(self, kmer_list, kmer_dict):
        """Decide which method to use to create a vector with given alphabet from a list of kmers"""
        assert not self.deepnano, "create_kmer_vector should not be called when using deepnano data preparation"
        if self.prob:
            vector = self.create_prob_vector(kmer_list, kmer_dict)
        else:
            vector = self.create_categorical_vector(kmer_list, kmer_dict)
        return vector

    def create_prob_vector(self, kmer_list, kmer_dict):
        """Create a probability vector with given alphabet dictionary and length"""
        vector = numpy.zeros(len(kmer_dict))
        for kmer in kmer_list:
            trimmed_kmer = kmer[0][-self.length:]
            assert len(trimmed_kmer) == self.length, "Length of kmer is not equal to defined length: len({}) != {}".format(trimmed_kmer, self.length)
            try:
                # make sure if one kmer has two probabilities we assign the highest probability
                if vector[kmer_dict[trimmed_kmer]] < kmer[1]:
                    vector[kmer_dict[trimmed_kmer]] = kmer[1]
                # make sure vector sums to one if less than one
                vector = sum_to_one(vector, prob=True)
            except KeyError as error:
                raise DataPrepBug("Kmer: {} not in reference kmer dictionary, check alphabet or length".format(error))
        return vector

    def create_categorical_vector(self, kmer_list, kmer_dict):
        """Create a vector with given alphabet"""
        # check all kmers for most probable
        best_kmer, _, _ = self.get_most_probable_kmer(kmer_list)
        final_kmer = best_kmer[-self.length:]
        assert len(final_kmer) == self.length, "Length of kmer is not equal to defined length: len({}) != {}".format(final_kmer, self.length)
        try: # to put most probable kmer into vector
            vector = numpy.zeros(len(kmer_dict))
            vector[kmer_dict[final_kmer]] = 1
        except KeyError as error:
            raise DataPrepBug("Kmer: {} not in reference kmer dictionary, check alphabet".format(error))
        # check if sums to one
        vector = sum_to_one(vector, prob=False)
        return vector

    def create_features(self):
        """Create features from events"""
        if self.nanonet:
            features = nanonet_features(self.events)
        else:
            features = self.deepnano_features(self.events)
        self.features = features
        return features

    def deepnano_features(self, events):
        """Replicating deepnano's feature definition"""
        params = estimate_params(self.fast5_file, binary_path="estimateNanoporeParams",
                                 template_lookup_table=self.template_lookup_table,
                                 complement_lookup_table=self.complement_lookup_table,
                                 twoD=True, verbose=False)
        # print(params)
        new_events = []
        try:
            for event in events:
                mean = (event["mean"] - params["shift"]) / params["scale"]
                stdv = event["stdv"] / params["scale_sd"]
                # mean = event["mean"]
                # stdv = event["stdv"]
                length = event["length"]
                new_events.append(self.preproc_event(mean, stdv, length))
            new_events = np.asarray(new_events)
        except ValueError as error:
            raise DataPrepBug("Event data has {}. Check fields selection from scrape_fast5_events.".format(error))
        return new_events

    @staticmethod
    def preproc_event(mean, std, length):
        "Normalizing event information for deepnano feature generation"
        mean = mean / 100.0 - 0.66
        std = std - 1
        return [mean, mean*mean, std, length]

    def save_training_file(self, output_name, output_dir):
        """Create training file and save it as an .npy file"""
        assert os.path.isdir(output_dir)
        output_file = os.path.join(output_dir, output_name)
        self.run_complete_analysis()
        np.save(output_file, self.training_file)
        return output_file+".npy"

    def interpolate(self):
        """Guess a distribution of data"""
        return "from scipy.interpolate import interp1d"

def main():
    """Mainly used for testing the methods within data_preparation.py"""
    start = timer()
    #
    canonical_fast5 = \
    get_project_file("test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_sequencing_run_AMS_158_R9_WGA_Ecoli_08_20_16_43623_ch100_read104_strand.fast5")


    canonical_tsv = \
    get_project_file("test_files/signalalignment_files/canonical/18a21abc-7827-4ed7-8919-c27c9bd06677_Basecall_2D_template.sm.forward.tsv")
    #
    DEEPNANO = TrainingData(canonical_fast5, canonical_tsv, strand_name="template", prob=False, kmer_len=2, alphabet="ATGC", nanonet=False, deepnano=True)

    # CATEGORICAL = TrainingData(canonical_fast5, canonical_tsv, strand_name="template", prob=False, kmer_len=5, alphabet="ATGCE", nanonet=True, deepnano=False)
    #
    # T = TrainingData(canonical_fast5, canonical_tsv, strand_name="template", prob=True, kmer_len=5, alphabet="ATGCE", nanonet=True, deepnano=False)
    #
    # C = TrainingData(canonical_fast5, canonical_tsv, strand_name="complement", prob=True, kmer_len=5, alphabet="ATGCE", nanonet=True, deepnano=False)

    # data = TrainingData(canonical_fast5, canonical_tsv, strand_name="template", prob=False, kmer_len=2, alphabet="ATGC", nanonet=False, deepnano=True)
    # data.scrape_signalalign()
    # labels = data.create_deepnano_labels()
    a = DEEPNANO.run_complete_analysis()
    print(a)
    # b = CATEGORICAL.run_complete_analysis()
    # print(b)
    # c = T.run_complete_analysis()
    # print(c)
    # d = C.run_complete_analysis()
    # print(d)

    # kmer_dict = {"NN":0}
    # kmer_list = [["4MER", 1, 100], ["4MER", 1, 101], ["4MER", .1, 102]]
    #
    # data.create_categorical_vector(kmer_list, kmer_dict)
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
