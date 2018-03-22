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
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import numpy as np
from nanotensor.alignedsignal import AlignedSignal, CreateLabels

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
        for name, label in self.signal_h.label.items():
            self.names.append(name)
            self.alignments.append([label['raw_start'], label['reference_index']])

        # predictions can have multiple alignments for each event and are not a path
        for name, prediction in self.signal_h.prediction.items():
            self.names.append(name)
            self.predictions.append([prediction['raw_start'], prediction['reference_index'],
                                     prediction['posterior_probability']])

        for name, guide in self.signal_h.guide.items():
            self.guide_alignments.append([guide['raw_start'], guide['reference_index']])
            # gather tail ends of alignments
        if self.guide_alignments:
            self.names.append(name)

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
            handle, = panel1.plot(alignment[0], alignment[1], color=colors[i], alpha=0.8)
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

        handle, = panel2.plot(self.signal_h.scaled_signal, color="black", lw=0.4)
        handles.append(handle)
        self.names.append("scaled_signal")

        panel1.legend(handles, self.names, loc='upper right')

        plt.show()


def main():
    """Main docstring"""
    start = timer()
    # sam = "/Users/andrewbailey/CLionProjects/nanopore-RNN/signalAlign/bin/`output/tempFiles_alignment/tempFiles_miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch138_read23_strand/temp_sam_file_5048dffc-a463-4d84-bd3b-90ca183f488a.sam"\

    # rna_read = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/rna_reads/DEAMERNANOPORE_20170922_FAH26525_MN16450_sequencing_run_MA_821_R94_NA12878_mRNA_09_22_17_67136_read_36_ch_218_strand.fast5"
    # dna_read = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_sequencing_run_AMS_158_R9_WGA_Ecoli_08_20_16_43623_ch100_read280_strand.fast5"
    dna_read = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/tests/test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_sequencing_run_AMS_158_R9_WGA_Ecoli_08_20_16_43623_ch100_read280_strand.fast5"
    dna_read2 = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch138_read23_strand.fast5"
    # dna_read3 = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical/over_run/miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch138_read23_strand.fast5"

    reference = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/reference-sequences/ecoli_k12_mg1655.fa"

    # rh = ReferenceHandler(reference)
    # seq = rh.get_sequence(chromosome_name="Chromosome", start=623566, stop=623572)
    # print(seq)
    # MINKNOW = dict(window_lengths=(5, 10), thresholds=(2.0, 1.1), peak_height=1.2)
    # resegment_reads(dna_read2, MINKNOW, speedy=False, overwrite=True)

    test = CreateLabels(dna_read2)
    test.add_guide_alignment()
    test.add_mea_labels()
    test.add_signal_align_predictions()
    test.add_nanoraw_labels(reference)
    test.add_eventalign_labels()
    ps = PlotSignal(test.aligned_signal)
    print("Plotting")
    ps.plot_alignment()


    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
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
