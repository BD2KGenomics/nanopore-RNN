#!/usr/bin/env python
"""Plot information needed file"""
########################################################################
# File: plot_raw_read_alignment.py
#  executable: plot_raw_read_alignment.py
#
# Author: Andrew Bailey
# History: Created 12/01/17
########################################################################

from __future__ import print_function
import sys
import os
from timeit import default_timer as timer
import pysam
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import numpy as np
import scipy.stats as stats
import seaborn as sns
from nanotensor.trim_signal import read_label, SignalLabel
from nanotensor.utils import list_dir

from nanonet.eventdetection.filters import minknow_event_detect


def raw_scatter_plot(signal_data, label_data, outpath, interval):
    """plot accuracy distribution of reads"""
    # define figure size
    size = (interval[1] - interval[0])/100
    plt.figure(figsize=(size, 4))
    panel1 = plt.axes([0.01, 0.1, .95, .9])
    # longest = max(data[0]) + data[1])
    # panel1.set_xlim(0, 1000)
    mean = np.mean(signal_data)
    stdv = np.std(signal_data)
    panel1.set_ylim(mean - (3*stdv), mean + (3*stdv))
    panel1.set_xlim(interval[0], interval[1])

    # panel1.set_xscale("log")
    plt.scatter(x=range(len(signal_data)), y=signal_data, s=1, c="k")

    plt.title('Nanopore Read')
    for i in range(len(label_data.start)):
        if interval[0] < label_data.start[i] < interval[1]:
            panel1.text(label_data.start[i]+(label_data.length[i]/2), 2, "{}".format(label_data.base[i]), fontsize=10, va="bottom", ha="center")
            panel1.axvline(label_data.start[i])
            panel1.axvline(label_data.start[i]+label_data.length[i])

    plt.savefig(outpath)


def raw_scatter_plot_with_events(signal_data, label_data, outpath, interval, events):
    """plot accuracy distribution of reads"""
    # define figure size
    size = (interval[1] - interval[0])/75
    plt.figure(figsize=(size, 4))
    panel1 = plt.axes([0.01, 0.1, .95, .9])
    # longest = max(data[0]) + data[1])
    # panel1.set_xlim(0, 1000)
    mean = np.mean(signal_data)
    stdv = np.std(signal_data)
    panel1.set_ylim(mean - (3*stdv), mean + (3*stdv))
    panel1.set_xlim(interval[0], interval[1])

    # panel1.set_xscale("log")
    plt.scatter(x=range(len(signal_data)), y=signal_data, s=1, c="k")

    plt.title('Nanopore Read')
    for i in range(len(label_data.start)):
        if interval[0] < label_data.start[i] < interval[1]:
            panel1.text(label_data.start[i]+(label_data.length[i]/2), 2, "{}".format(label_data.base[i]), fontsize=10, va="bottom", ha="center")
            panel1.axvline(label_data.start[i])
            panel1.axvline(label_data.start[i]+label_data.length[i])

    for event_peak in events:
        if interval[0] < event_peak < interval[1]:
            panel1.axvline(event_peak, linestyle='--', color='r')



    plt.savefig(outpath)


def main():
    """Main docstring"""
    start = timer()
    rna_event_lengths = []
    DNA_reads = list_dir("/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/dna_training/training", ext="label")
    RNA_reads = list_dir("/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/rna_training/training", ext="label")

    signal_file = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/rna_training/training/DEAMERNANOPORE_20170922_FAH26525_MN16450_mux_scan_MA_821_R94_NA12878_mRNA_09_22_17_34495_read_12_ch_3_strand.signal"
    label_file = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/rna_training/training/DEAMERNANOPORE_20170922_FAH26525_MN16450_mux_scan_MA_821_R94_NA12878_mRNA_09_22_17_34495_read_12_ch_3_strand.label"
    fileh = SignalLabel(signal_file=signal_file, label_file=label_file)
    signal = fileh.read_signal(normalize=True)
    label = fileh.read_label(skip_start=10, window_n=0, bases=True)
    rna_sample_rate = 3012.0
    # defaults from nanoraw
    params = dict(window_lengths=[3, 6], thresholds=[1.4, 1.1], peak_height=0.2)
    # testing params
    params = dict(window_lengths=[3, 6], thresholds=[1., 1.], peak_height=0.2)

    count = 0
    size = 10000
    for i, label_file in enumerate(DNA_reads):
        if count < 10:
            start1 = timer()
            readpath, ext = os.path.splitext(label_file)
            signal_file = readpath+".signal"
            if os.path.isfile(readpath+".signal"):
                fileh = SignalLabel(signal_file=signal_file, label_file=label_file)
                signal = fileh.read_signal(normalize=True)
                label = fileh.read_label(skip_start=10, window_n=0, bases=True)
                events = minknow_event_detect(np.asarray(signal, dtype=float), sample_rate=rna_sample_rate, get_peaks=True, **params)
                for indx, i in enumerate(range(0, len(signal), size)):
                    begin = i
                    end = min(size*(indx+1), len(signal))
                    interval = [begin, end]
                    outpath = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/rna_training/event_detect_rna_plots"
                    outpath = os.path.join(outpath, os.path.basename(readpath)+str(i)+".png")
                    print("Plotting {} to {} of read {}".format(begin, end, outpath), file=sys.stderr)
                    raw_scatter_plot_with_events(signal_data=signal, label_data=label, interval=interval,
                                                 outpath=outpath, events=events)
                    stop1 = timer()
                    print("Finished part {} to {} in {} seconds".format(begin, end, stop1 - start1), file=sys.stderr)
                count += 1
    # print(signal)
    # print(label)
    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
