#!/usr/bin/env python
"""Plot speeds of maximum expected accuracy methods"""
########################################################################
# File: plot_mea_speeds.py
#  executable: plot_mea_speeds.py
#
# Author: Andrew Bailey
# History: Created 02/24/18
########################################################################

from __future__ import print_function
import sys
import os
from timeit import default_timer as timer
import pysam
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import numpy as np
from collections import defaultdict
import scipy.stats as stats
from py3helpers.utils import list_dir, time_it, test_numpy_table
from nanotensor.fast5 import Fast5
import tempfile
from scipy import sparse
from py3helpers.seq_tools import ReferenceHandler, initialize_pysam_wrapper, ReverseComplement
from nanotensor.mea_algorithm import maximum_expected_accuracy_alignment, mea_slow, \
    mea_slower, create_random_prob_matrix, get_mea_params_from_events, match_events_with_signalalign
import unittest
from signalalign.signalAlignment import SignalAlignment


class AlignedSignal(object):
    """Labeled nanopore signal data"""

    def __init__(self, scaled_signal):
        """Initialize the scaled signal and label

        :param scaled_signal: scaled signal to pA
        """
        self.scaled_signal = None
        self.raw_signal = None
        self._add_scaled_signal(scaled_signal)
        self.signal_length = len(self.scaled_signal)
        self.minus_strand = None
        # label can be used for neural network training with all signal continuously labelled
        self.label = defaultdict()
        # predictions can have multiple labels for different sections of current
        self.prediction = defaultdict()
        # guides are sections that we are confident in (guide alignments)
        self.guide = defaultdict()

    def add_raw_signal(self, signal):
        """Add raw signal to class

        :param signal: raw current signal in ADC counts
        """
        assert int(signal[0]) == signal[0], "Raw signal are always integers"
        assert len(signal) == len(self.scaled_signal) and len(signal) == self.signal_length, \
            "Raw signal must be same size as scaled signal input:{} != scale:{}".format(signal, self.scaled_signal)
        self.raw_signal = signal

    def _add_scaled_signal(self, signal):
        """Add scaled signal to class

        :param signal: normalized current signal to pA
        """
        if type(signal) is np.ndarray:
            signal = signal.tolist()
        assert type(signal[0]) == float, "scaled signal must be a float"
        self.scaled_signal = signal

    def add_label(self, label, name, location):
        """Add labels to class

        :param label: label numpy array with required fields ['raw_start', 'raw_length', 'reference_index',
                                                              'kmer', 'posterior_probability']
        :param name: name of the label for signal
        :param location: data structure to add signal labels :['label', 'prediction', 'guide']
        """
        assert location in ['label', 'prediction', 'guide'], \
            "{} not in ['label', 'prediction', 'guide']: Must select an acceptable location".format(location)
        test_numpy_table(label, req_fields=('raw_start', 'raw_length', 'reference_index',
                                            'kmer', 'posterior_probability'))

        label.sort(order='raw_start', kind='mergesort')
        # check the labels are in the correct format
        assert min(label["raw_start"]) >= 0, "Raw start cannot be less than 0"
        assert 0 <= max(label["posterior_probability"]) <= 1, \
            "posterior_probability must be between zero and one {}".format(row["posterior_probability"])

        # make sure last label can actually index the signal correctly
        try:
            self.scaled_signal[label[-1]["raw_start"]:label[-1]["raw_start"] + label[-1]["raw_length"]]
        except IndexError:
            raise IndexError("labels are longer than signal")

        # infer strand alignment of read
        if label[0]["reference_index"] >= label[-1]["reference_index"]:
            minus_strand = True
        else:
            minus_strand = False
        if self.minus_strand is not None:
            if label[0]["raw_start"] != label[-1]["raw_start"]:
                assert self.minus_strand == minus_strand, "New label has different strand direction, check label"
        else:
            self.minus_strand = minus_strand

        # set label with the specified name
        if location == 'label':
            self.label[name] = label
        elif location == 'prediction':
            self.prediction[name] = label
        elif location == 'guide':
            self.guide[name] = label

    def add_prediction(self, prediction, name):
        """Add prediction to class. Predictions are similar to labels but can have gaps and duplicates have many

        :param prediction: prediction numpy array with required fields ['raw_start', 'raw_length', 'reference_index',
                                                              'kmer', 'posterior_probability']
        :param name: name of the label for signal
        """
        test_numpy_table(prediction, req_fields=('raw_start', 'raw_length', 'reference_index',
                                                 'kmer', 'posterior_probability'))
        prediction.sort(order='raw_start', kind='mergesort')
        # check the labels are in the correct format
        assert min(prediction["raw_start"]) >= 0, "Raw start cannot be less than 0"
        assert 0 <= max(prediction["posterior_probability"]) <= 1, \
            "posterior_probability must be between zero and one {}".format(max(prediction["posterior_probability"]))

        # make sure last label can actually index the signal correctly
        try:
            self.scaled_signal[prediction[-1]["raw_start"]:prediction[-1]["raw_start"] + prediction[-1]["raw_length"]]
        except IndexError:
            raise IndexError("labels are longer than signal")

        # infer strand alignment of read
        if prediction[0]["reference_index"] > prediction[-1]["reference_index"]:
            minus_strand = True
        else:
            minus_strand = False
        if self.minus_strand is not None:
            assert self.minus_strand == minus_strand, "New prediction has different strand direction, check label"
        else:
            self.minus_strand = minus_strand

        # set label with the specified name
        self.prediction[name] = prediction

    def generate_label_mapping(self, name, scaled=True):
        """Create a generator of the mapping between the signal and the label

        :param name: name of mapping to create label mapping
        :param scaled: boolean option for returning scaled or unscaled signal
        """
        label = self.label[name]
        len_label = len(label)
        if scaled:
            signal = self.scaled_signal
        else:
            assert self.raw_signal is not None, "Must set raw signal in order to generate raw signal alignments"
            signal = self.raw_signal
        for i, segment in enumerate(label):
            start = segment["raw_start"]
            if i < len_label-1:
                end = label[i+1]["raw_start"] - segment["raw_start"]
            else:
                end = segment["raw_start"] + segment["raw_length"]
            yield signal[start:end], segment['kmer'], segment['posterior_probability'], segment['reference_index']


class AlignedSignalTest(unittest.TestCase):
    """Test the class AlignedSignal"""

    @classmethod
    def setUpClass(cls):
        super(AlignedSignalTest, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-2])
        print(cls.HOME)
        cls.dna_file = os.path.join(cls.HOME,
                                    "tests/test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_sequencing_run_AMS_158_R9_WGA_Ecoli_08_20_16_43623_ch100_read280_strand.fast5")
        cls.modified_file = os.path.join(cls.HOME,
                                         "tests/test_files/minion-reads/methylated/DEAMERNANOPORE_20160805_FNFAD19383_MN16450_sequencing_run_MA_821_R9_gEcoli_MG1655_08_05_16_89825_ch100_read5189_strand.fast5")
        cls.rna_file = os.path.join(cls.HOME,
                                    "tests/test_files/minion-reads/rna_reads/DEAMERNANOPORE_20170922_FAH26525_MN16450_sequencing_run_MA_821_R94_NA12878_mRNA_09_22_17_67136_read_61_ch_151_strand.fast5")
        cls.handle = AlignedSignal(scaled_signal=[1.1, 2.2, 1.1, 2.2, 1.1, 2.2])

    def test__add_label(self):
        """Test _add_label method"""
        label = np.zeros(4, dtype=[('raw_start', int), ('raw_length', int), ('reference_index', int),
                                   ('posterior_probability', float), ('kmer', 'S5')])
        label["raw_start"] = [0, 1, 2, 3]
        label["raw_length"] = [1, 1, 1, 1]
        label["reference_index"] = [0, 1, 2, 3]
        label["posterior_probability"] = [1, 1, 1, 1]
        label["kmer"] = ["AAT", "A", "B", "C"]

        self.handle.add_label(label, name="test")
        self.handle.add_label(label, name="test2")
        self.handle.add_label(label, name="test3")
        with self.assertRaises(KeyError):
            label = np.zeros(0, dtype=[('fake', int), ('raw_length', int), ('reference_index', int),
                                       ('posterior_probability', float), ('kmer', 'S5')])
            self.handle.add_label(label, name="test")

    def test_add_raw_signal(self):
        """Test add_raw_signal method"""
        self.handle.add_raw_signal(np.asanyarray([1, 2, 3, 4, 5, 6]))
        self.handle.add_raw_signal([1, 2, 3, 4, 5, 6])

        with self.assertRaises(AssertionError):
            self.handle.add_raw_signal([1.1, 2.2, 1.1, 4])
            self.handle.add_raw_signal([1.1, 2, 1, 2, 3, 6])

    def test__add_scaled_signal(self):
        """Test _add_scaled_signal method"""
        self.handle._add_scaled_signal(np.asanyarray([1.1, 2.2, 1.1, 2.2, 1.1, 2.2]))
        self.handle._add_scaled_signal([1.1, 2.2, 1.1, 2.2, 1.1, 2.2])

        with self.assertRaises(AssertionError):
            self.handle._add_scaled_signal([1, 2.2, 1.1, 4])
            self.handle._add_scaled_signal([1, 2, 1, 2, 3, 6])

    def test_generate_label_mapping(self):
        """Test generate_label_mapping method"""
        label = np.zeros(4, dtype=[('raw_start', int), ('raw_length', int), ('reference_index', int),
                                   ('posterior_probability', float), ('kmer', 'S5')])
        label["raw_start"] = [0, 1, 2, 3]
        label["raw_length"] = [1, 1, 1, 1]
        label["reference_index"] = [0, 1, 2, 3]
        label["posterior_probability"] = [1, 1, 1, 1]
        label["kmer"] = ["AAT", "A", "B", "C"]
        handle = AlignedSignal(scaled_signal=[1.1, 2.2, 1.1, 2.2, 1.1, 2.2])

        handle.add_label(label, name="test")
        handle.add_label(label, name="test2")
        handle.add_label(label, name="test3")

        test = handle.generate_label_mapping(name='test')
        for i, return_tuple in enumerate(test):
            self.assertEqual(return_tuple[0], handle.scaled_signal[i:i + 1])
            self.assertEqual(return_tuple[1], label["kmer"][i])
            self.assertEqual(return_tuple[2], label["posterior_probability"][i])
            self.assertEqual(return_tuple[3], label["reference_index"][i])

        test = handle.generate_label_mapping(name='test2')
        for i, return_tuple in enumerate(test):
            self.assertEqual(return_tuple[0], handle.scaled_signal[i:i + 1])
            self.assertEqual(return_tuple[1], label["kmer"][i])
            self.assertEqual(return_tuple[2], label["posterior_probability"][i])
            self.assertEqual(return_tuple[3], label["reference_index"][i])

        with self.assertRaises(KeyError):
            handle.generate_label_mapping(name="fake").__next__()
        with self.assertRaises(AssertionError):
            handle.generate_label_mapping(name="test2", scaled=False).__next__()

    def test_create_signal_align_labels(self):
        """Test create_signal_align_labels function"""
        for file_path in [self.dna_file]:
            handle = create_signal_align_labels(file_path)
            self.assertIsInstance(handle, AlignedSignal)


class PlotSignal(object):
    """Handles alignments between nanopore signals, events and reference sequences"""

    def __init__(self, aligned_signal):
        """Plot alignment

        :param fname: path to fast5 file
        """
        assert isinstance(aligned_signal, AlignedSignal), "aligned_signal must be an instance of AlignedSignal"
        self.signal_h = aligned_signal
        self.names = []
        self.alignments = []
        self.predictions = []
        self.guide_alignments = []

    def get_alignments(self):
        """Format alignments from AlignedSignal to make it possible to plot easily"""
        min_ref_position = np.inf
        max_ref_position = 0
        for name, label in self.signal_h.label.items():
            self.names.append(name)
            self.alignments.append([label['raw_start'], label['reference_index']])
            # gather tail ends of alignments
            temp_min_position = min(label['reference_index'])
            if min_ref_position > temp_min_position:
                min_ref_position = temp_min_position
            temp_max_position = max(label['reference_index'])
            if max_ref_position < temp_max_position:
                max_ref_position = temp_max_position

        # predictions can have multiple alignments for each event and are not a path
        for name, prediction in self.signal_h.prediction.items():
            self.names.append(name)
            self.predictions.append([prediction['raw_start'], prediction['reference_index'],
                                     prediction['posterior_probability']])
            # gather tail ends of alignments
            temp_min_position = min(prediction['reference_index'])
            if min_ref_position > temp_min_position:
                min_ref_position = temp_min_position
            temp_max_position = max(prediction['reference_index'])
            if max_ref_position < temp_max_position:
                max_ref_position = temp_max_position

        for name, guide in self.signal_h.guide.items():
            self.guide_alignments.append([guide['raw_start'], guide['reference_index']])
            # gather tail ends of alignments
            temp_min_position = min(guide['reference_index'])
            if min_ref_position > temp_min_position:
                min_ref_position = temp_min_position
            temp_max_position = max(guide['reference_index'])
            if max_ref_position < temp_max_position:
                max_ref_position = temp_max_position
        if self.guide_alignments:
            self.names.append(name)

        # scale reference position accordingly
        for events, ref_pos, _ in self.predictions:
            if self.signal_h.minus_strand:
                ref_pos -= max_ref_position
                ref_pos *= -1
            else:
                ref_pos -= min_ref_position

        # scale reference position accordingly
        for events, ref_pos in self.alignments:
            if self.signal_h.minus_strand:
                ref_pos -= max_ref_position
                ref_pos *= -1
            else:
                ref_pos -= min_ref_position

        # scale reference position accordingly
        for events, ref_pos in self.guide_alignments:
            if self.signal_h.minus_strand:
                ref_pos -= max_ref_position
                ref_pos *= -1
            else:
                ref_pos -= min_ref_position

    def plot_alignment(self):
        """Plot the alignment between events and reference with the guide alignment and mea alignment

        signal: normalized signal
        posterior_matrix: matrix with col = reference , rows=events
        guide_cigar: guide alignment before signal align
        events: doing stuff
        mea_alignment: final alignment between events and reference
        """

        self.get_alignments()

        plt.figure(figsize=(6, 8))
        panel1 = plt.axes([0.1, 0.22, .8, .7])
        panel2 = plt.axes([0.1, 0.09, .8, .1], sharex=panel1)
        panel1.set_xlabel('Events')
        panel1.set_ylabel('Reference')
        panel1.xaxis.set_label_position('top')
        panel1.invert_yaxis()
        panel1.xaxis.tick_top()
        panel1.grid(color='black', linestyle='-', linewidth=1)

        handles = list()
        colors = ['blue', 'green', 'red']
        # # # plot signal alignments

        for i, alignment in enumerate(self.alignments):
            handle, = panel1.plot(alignment[0], alignment[1], color=colors[i])
            handles.append(handle)

        for i, prediction in enumerate(self.predictions):
            rgba_colors = np.zeros((len(prediction[0]), 4))
            rgba_colors[:, 3] = prediction[2].tolist()
            handle = panel1.scatter(prediction[0].tolist(), prediction[1].tolist(), marker='.', c=rgba_colors)
            handles.append(handle)

        if self.guide_alignments:
            for i, guide in enumerate(self.guide_alignments):
                handle, = panel1.plot(guide[0], guide[1], color='magenta')
            handles.append(handle)


        panel2.set_xlabel('Time')
        panel2.set_ylabel('Current (pA)')
        if len(self.predictions) > 0:
            panel2.set_xticks(self.predictions[0][0], minor=True)
        else:
            panel2.set_xticks(self.alignments[0][0], minor=True)
        panel2.axvline(linewidth=0.1, color="k")

        handle, = panel2.plot(self.signal_h.scaled_signal, color="black", lw=0.2)
        handles.append(handle)
        self.names.append("scaled_signal")

        panel1.legend(handles, self.names, loc='upper right')

        plt.show()


def create_signal_align_labels(fast5_path):
    """Gather signal-aligned labelled information from fast5 file.

    Will create an AlignedSignal class with required fields filled out

    :param fast5_path: path to fast5 file
    """
    fast5 = Fast5(fast5_path)
    scaled_signal = fast5.get_read(raw=True, scale=True)
    raw_signal = fast5.get_read(raw=True, scale=False)
    # add raw signal information to AlignedSignal
    aligned_signal = AlignedSignal(scaled_signal)
    aligned_signal.add_raw_signal(raw_signal)
    # TODO call signalalign if not called
    mea_alignment = fast5.get_signalalign_events(mea=True)
    aligned_signal.add_label(mea_alignment, name="mea_signalalign", location='label')

    return aligned_signal


def create_signal_align_prediction(fast5_path):
    """Create prediction using probabilities from full output format from signalAlign

    :param fast5_path: path to fast5 file
    """
    f5fh = Fast5(fast5_path)
    # TODO call signalalign if not called
    signal_alignment = f5fh.get_signalalign_events()
    sa_events = np.unique(signal_alignment)
    events = f5fh.get_resegment_basecall()
    labels = match_events_with_signalalign(sa_events=sa_events, event_detections=events)
    return labels


def test_unique_signalalign_output(signal_align_output):
    """Make sure np.unique does not remove any rows"""
    n_rows = len(signal_align_output)
    sa_unique = np.unique(signal_align_output)
    n_unique_rows = len(sa_unique)
    n_duplicates = 0
    test = []
    for y in signal_alignment:
        if y in test:
            n_duplicates += 1
        else:
            test.append(y)
    assert n_rows - n_duplicates == n_unique_rows, "np.unique is not working correctly"
    posterior_matrix1, _, _ = get_mea_params_from_events(signal_align_output)
    posterior_matrix, _, _ = get_mea_params_from_events(sa_unique)
    assert posterior_matrix.tolist() == posterior_matrix1.tolist()

    return True


def create_labels_from_guide_alignment(events, sam_string, rna=False, reference_path=None):
    """Create labeled signal from a guide alignment with only matches being reported

    :param events: path to fast5 file
    :param sam_string: sam alignment string
    :param rna: if read is rna, reverse again
    :param reference_path: if sam_string has MDZ field the reference sequence can be inferred, otherwise, it is needed
    """
    # test if the required fields are in structured numpy array
    test_numpy_table(events, req_fields=('raw_start', 'model_state', 'p_model_state', 'raw_length', 'move'))
    psam_h = initialize_pysam_wrapper(sam_string, reference_path=reference_path)
    # print(psam_h.alignment_segment.is_reverse)
    # create an indexed map of the events and their corresponding bases
    # TODO should be it's own method

    bases, base_raw_starts, base_raw_lengths, probs = index_bases_from_events(events)

    # check if string mapped to reverse strand
    if psam_h.alignment_segment.is_reverse:
        probs = probs[::-1]
        base_raw_starts = base_raw_starts[::-1]
        # rna reads go 3' to 5' so we dont need to reverse if it mapped to reverse strand
        if not rna:
            bases = ReverseComplement().reverse(''.join(bases))
    # rna reads go 3' to 5' so we do need to reverse if it mapped to forward strand
    elif rna:
        bases = ReverseComplement().reverse(''.join(bases))

    # print("Real sequence", ''.join(bases))
    # all 'matches' and 'mismatches'
    matches_map = psam_h.seq_alignment.matches_map
    # matches_map = psam_h.seq_alignment.matches_map
    # set labels
    raw_start = []
    raw_length = []
    reference_index = []
    kmer = []
    posterior_probability = []
    cigar_labels = []
    prev = matches_map[0].reference_index
    for i, alignment in enumerate(matches_map):
        if i == 0 or alignment.reference_index == prev+1:
            raw_start.append(base_raw_starts[alignment.query_index])
            raw_length.append(base_raw_lengths[alignment.query_index])
            reference_index.append(alignment.reference_index + psam_h.alignment_segment.reference_start)
            kmer.append(bases[alignment.query_index])
            posterior_probability.append(probs[alignment.query_index])
            prev = alignment.reference_index
        else:
            # initialize labels
            cigar_label = np.zeros(len(raw_start), dtype=[('raw_start', int), ('raw_length', int), ('reference_index', int),
                                                          ('posterior_probability', float), ('kmer', 'S5')])
            # assign labels
            cigar_label['raw_start'] = raw_start
            cigar_label['raw_length'] = raw_length
            cigar_label['reference_index'] = reference_index
            cigar_label['kmer'] = kmer
            cigar_label['posterior_probability'] = posterior_probability
            # add to other blocks
            cigar_labels.append(cigar_label)
            # reset trackers
            raw_start = [base_raw_starts[alignment.query_index]]
            raw_length = [base_raw_lengths[alignment.query_index]]
            reference_index = [alignment.reference_index + psam_h.alignment_segment.reference_start]
            kmer = [bases[alignment.query_index]]
            posterior_probability = [probs[alignment.query_index]]
            prev = alignment.reference_index

        # # initialize labels
        # cigar_labels = np.zeros(len(matches_map), dtype=[('raw_start', int), ('raw_length', int), ('reference_index', int),
        #                                                  ('posterior_probability', float), ('kmer', 'S5')])
        #
        #
        # cigar_labels['raw_start'] = [base_raw_starts[i] for i in matches_query_indexes]
        # cigar_labels['raw_length'] = [base_raw_lengths[i] for i in matches_query_indexes]
        # cigar_labels['reference_index'] = [alignment.reference_index+psam_h.alignment_segment.reference_start for alignment in matches_map]
        # cigar_labels['kmer'] = [bases[i] for i in matches_query_indexes]
        # cigar_labels['posterior_probability'] = [probs[i] for i in matches_query_indexes]

    return cigar_labels


def index_bases_from_events(events):
    """Map basecalled sequence to events from a table with required fields

    :param events: original base-called events with required fields
    """

    test_numpy_table(events, req_fields=('raw_start', 'model_state', 'p_model_state', 'raw_length', 'move'))
    probs = []
    base_raw_starts = []
    bases = []
    base_raw_lengths = []
    for i, event in enumerate(events):
        if i == 0:
            # initialize with first kmer
            base_raw_starts.extend([event['raw_start'] for _ in event['model_state']])
            probs.extend([event['p_model_state'] for _ in event['model_state']])
            bases.extend([chr(x) for x in event['model_state']])
            base_raw_lengths.extend([event['raw_length'] for _ in event['model_state']])
        else:
            # if there was a move, gather the information for each base by index
            if event['move'] > 0:
                char_moves = bytes.decode(event['model_state'][-event['move']:])
                for x in range(event['move']):
                    base_raw_starts.append(event['raw_start'])
                    probs.append(event['p_model_state'])
                    bases.append(char_moves[x])
                    base_raw_lengths.extend([event['raw_length'] for _ in event['model_state']])
    # the index of each corresponds to the index of the final sequence
    return bases, base_raw_starts, base_raw_lengths, probs


def main():
    """Main docstring"""
    start = timer()
    # sam = "/Users/andrewbailey/CLionProjects/nanopore-RNN/signalAlign/bin/test_output/tempFiles_alignment/tempFiles_miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch138_read23_strand/temp_sam_file_5048dffc-a463-4d84-bd3b-90ca183f488a.sam"\

    # rna_read = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/rna_reads/DEAMERNANOPORE_20170922_FAH26525_MN16450_sequencing_run_MA_821_R94_NA12878_mRNA_09_22_17_67136_read_36_ch_218_strand.fast5"
    # dna_read = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_sequencing_run_AMS_158_R9_WGA_Ecoli_08_20_16_43623_ch100_read280_strand.fast5"
    dna_read = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/tests/test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_sequencing_run_AMS_158_R9_WGA_Ecoli_08_20_16_43623_ch100_read280_strand.fast5"
    dna_read2 = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch138_read23_strand.fast5"
    # dna_read3 = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical/over_run/miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch138_read23_strand.fast5"

    reference = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/reference-sequences/ecoli_k12_mg1655.fa"

    # rh = ReferenceHandler(reference)
    # seq = rh.get_sequence(chromosome_name="Chromosome", start=623566, stop=623572)
    # print(seq)


    f5fh = Fast5(dna_read2)
    test_sam = f5fh.get_signalalign_events(sam=True)
    events = f5fh.get_resegment_basecall()
    cigar_labels = create_labels_from_guide_alignment(events=events, sam_string=test_sam)
    handle = create_signal_align_labels(dna_read2)
    signal_align_tsv_prediction = create_signal_align_prediction(dna_read2)
    handle.add_label(signal_align_tsv_prediction, name="full_signalalign", location='prediction')
    for i, block in enumerate(cigar_labels):
        # print(block)
        handle.add_label(block, name="guide_alignment{}".format(i), location='guide')

    ps = PlotSignal(handle)
    ps.plot_alignment()


    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    unittest.main()
    raise SystemExit
    #
    # prev_e = 0
    # prev_r = 0
    # max_e = 0
    # max_r = 0
    # for i, x in enumerate(mea_alignment):
    #     if i == 0:
    #         prev_e = x["event_index"]
    #         prev_r = x["reference_index"]
    #     else:
    #         if x["event_index"] > prev_e + 1:
    #             e_diff = np.abs(x["event_index"] - prev_e)
    #             if e_diff >  max_e:
    #                 max_e = e_diff
    #                 print("Max Event skip", max_e)
    #                 print(event_detect[x["event_index"]])
    #
    #         if x["reference_index"] < prev_r - 1:
    #             r_diff = np.abs(x["reference_index"] - prev_r)
    #             if r_diff > max_r:
    #                 max_r = r_diff
    #                 print("Max Reference skip", max_r)
    #                 print(event_detect[x["event_index"]])
    #         prev_e = x["event_index"]
    #         prev_r = x["reference_index"]
    #
