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
from PyPore.parsers import SpeedyStatSplit
from nanonet.eventdetection.filters import minknow_event_detect
from nanotensor.fast5 import Fast5


def raw_scatter_plot(signal_data, label_data, outpath, interval):
    """plot accuracy distribution of reads"""
    # define figure size
    size = (interval[1] - interval[0]) / 100
    plt.figure(figsize=(size, 4))
    panel1 = plt.axes([0.01, 0.1, .95, .9])
    # longest = max(data[0]) + data[1])
    # panel1.set_xlim(0, 1000)
    mean = np.mean(signal_data)
    stdv = np.std(signal_data)
    panel1.set_ylim(mean - (3 * stdv), mean + (3 * stdv))
    panel1.set_xlim(interval[0], interval[1])

    # panel1.set_xscale("log")
    plt.scatter(x=range(len(signal_data)), y=signal_data, s=1, c="k")

    plt.title('Nanopore Read')
    for i in range(len(label_data.start)):
        if interval[0] < label_data.start[i] < interval[1]:
            panel1.text(label_data.start[i] + (label_data.length[i] / 2), 2, "{}".format(label_data.base[i]),
                        fontsize=10, va="bottom", ha="center")
            panel1.axvline(label_data.start[i])
            panel1.axvline(label_data.start[i] + label_data.length[i])
    plt.show()

    # plt.savefig(outpath)


def raw_scatter_plot_with_events(signal_data, label_data, outpath, interval, events):
    """plot accuracy distribution of reads"""
    # define figure size
    size = (interval[1] - interval[0]) / 75
    plt.figure(figsize=(size, 4))
    panel1 = plt.axes([0.01, 0.1, .95, .9])
    # longest = max(data[0]) + data[1])
    # panel1.set_xlim(0, 1000)
    mean = np.mean(signal_data)
    stdv = np.std(signal_data)
    panel1.set_ylim(mean - (3 * stdv), mean + (3 * stdv))
    panel1.set_xlim(interval[0], interval[1])

    # panel1.set_xscale("log")
    plt.scatter(x=range(len(signal_data)), y=signal_data, s=1, c="k")

    plt.title('Nanopore Read')
    for i in range(len(label_data.start)):
        if interval[0] < label_data.start[i] < interval[1]:
            panel1.text(label_data.start[i] + (label_data.length[i] / 2), 2, "{}".format(label_data.base[i]),
                        fontsize=10, va="bottom", ha="center")
            panel1.axvline(label_data.start[i])
            panel1.axvline(label_data.start[i] + label_data.length[i])

    for event_peak in events:
        if interval[0] < event_peak < interval[1]:
            panel1.axvline(event_peak, linestyle='--', color='r')

    plt.show()
    # plt.savefig(outpath)


def plot_raw_reads(current, events, sampling_freq, minknow_peaks=None, rna_table=True):
    """Plot raw reads using ideas from Ryan Lorig-Roach's script"""
    fig1 = plt.figure(figsize=(24, 3))
    panel = fig1.add_subplot(111)
    prevMean = 0
    handles = list()
    handle, = panel.plot(current, color="black", lw=0.2)
    handles.append(handle)
    prev_kmer = ""
    for j, segment in enumerate(events):
        if rna_table:
            kmer = segment["model_state"]
            x0 = segment["start"] - 1
            x1 = x0 + segment["length"]
            mean = segment['mean']
        else:
            x0 = segment.start - 1
            x1 = x0 + segment.duration
            mean = segment.mean
        y0 = current[int(round(x0))]
        y1 = current[int(round(x1))]
        diff = abs(y1 - y0)

        # sdList.append(sd)
        # meanList.append(mean)

        # print j,x0,x1,mean,sd,diff
        # print(y0)

        textColor = "red"

        # color coding by the range of each segment to see if it indicates a bad segment
        # if diff > 3*sd:
        #     textColor = "red"
        # else:
        #     textColor = "green"

        color = [.082, 0.282, 0.776]

        # color coding segments by stdev to see whether stdev is an indication of bad segments
        # if sd > 3:
        # color = "blue"
        # panel.text(x1, mean, "%d" % j, ha="right", va="top", color=textColor, fontsize="5")
        # else:
        # color = "purple"

        # Uncomment for in-plot labelling of each segment mean
        # panel.text(x1, mean, "%d"%mean, ha="right", va="top", color=textColor, fontsize="5")

        # Uncomment for ugly demarcation of segment boundary:
        # panel.plot([x0,x1],[y0,y1],marker='o',mfc="green",mew=0,markersize=3,linewidth=0)

        # segment plotting:
        # if prev_kmer != kmer:
        handle, = panel.plot([x0, x1], [mean, mean], color=color, lw=0.8)
        panel.plot([x0, x0], [prevMean, mean], color=color, lw=0.5)  # <-- uncomment for pretty square wave

        if j == len(events) - 1:
            handles.append(handle)
            box = panel.get_position()
            panel.set_position([box.x0, box.y0, box.width * 0.95, box.height])
            plt.legend(handles, ["Raw", "Segmented"], loc='upper left', bbox_to_anchor=(1, 1))

        prevMean = mean

        panel.set_title("Signal")

        panel.set_xlabel("Time (ms)")
        panel.set_ylabel("Current (pA)")

    if minknow_peaks is not None:
        color = [1, 0.282, 0.176]

        for indx, peak in enumerate(minknow_peaks):
            if indx == 0:
                x0 = peak
                prevMean = 0
            else:
                x1 = peak
                y0 = current[int(round(x0))]
                y1 = current[int(round(x1))]
                mean = np.mean(current[x0:x1])
                handle, = panel.plot([x0, x1], [mean, mean], color=color, lw=0.8)
                panel.plot([x0, x0], [prevMean, mean], color=color, lw=0.5)  # <-- uncomment for pretty square wave
                x0 = peak
                prevMean = mean
    plt.show()


def main():
    """Main docstring"""
    start = timer()
    rna_event_lengths = []
    DNA_reads = list_dir(
        "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/dna_training/training", ext="label")
    RNA_reads = list_dir(
        "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/rna_training/training", ext="label")

    signal_file = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/rna_training/training/DEAMERNANOPORE_20170922_FAH26525_MN16450_mux_scan_MA_821_R94_NA12878_mRNA_09_22_17_34495_read_12_ch_3_strand.signal"
    label_file = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/rna_training/training/DEAMERNANOPORE_20170922_FAH26525_MN16450_mux_scan_MA_821_R94_NA12878_mRNA_09_22_17_34495_read_12_ch_3_strand.label"
    fileh = SignalLabel(signal_file=signal_file, label_file=label_file)
    signal = fileh.read_signal(normalize=True)
    label = fileh.read_label(skip_start=10, window_n=0, bases=True)
    rna_sample_rate = 3012.0
    # defaults from nanoraw
    params = dict(window_lengths=[3, 6], thresholds=[1.4, 1.1], peak_height=0.2)
    # testing params
    params = dict(window_lengths=[3, 6], thresholds=[2.5, 2], peak_height=1)

    fast5_file = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch467_read35_strand.fast5"
    fast5_file = "//Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/rna_reads/DEAMERNANOPORE_20170922_FAH26525_MN16450_sequencing_run_MA_821_R94_NA12878_mRNA_09_22_17_67136_read_36_ch_218_strand.fast5"
    f5fh = Fast5(fast5_file, read='r+')
    events = f5fh.get_basecall_data()
    print(events)


    min_width = 5
    max_width = 80
    min_gain_per_sample = 0.008
    window_width = 800
    # max_width = 100
    # min_gain_per_sample = 0.08
    # window_width = 1000

    sampling_freq = f5fh.sample_rate

    parser = SpeedyStatSplit(min_width=min_width, max_width=max_width,
                             min_gain_per_sample=min_gain_per_sample,
                             window_width=window_width, sampling_freq=sampling_freq)
    signal = f5fh.get_read(raw=True, scale=True)
    # speedy_events = parser.parse(np.asarray(signal, dtype=np.float64))
    params = dict(window_lengths=[16, 40], thresholds=[8.0, 4.0], peak_height=1)
    minknow_peaks = minknow_event_detect(np.asarray(signal, dtype=np.float64), sample_rate=sampling_freq,
                                          get_peaks=True, **params)
    plot_raw_reads(signal, events, sampling_freq, minknow_peaks=minknow_peaks, rna_table=True)




    # userInput = sys.stdin.readline()

    # count = 0
    # size = 10000
    # for i, label_file in enumerate(DNA_reads):
    #     if count < 10:
    #         start1 = timer()
    #         readpath, ext = os.path.splitext(label_file)
    #         signal_file = readpath+".signal"
    #         if os.path.isfile(readpath+".signal"):
    #             fileh = SignalLabel(signal_file=signal_file, label_file=label_file)
    #             signal = fileh.read_signal(normalize=True)
    #             label = fileh.read_label(skip_start=10, window_n=0, bases=True)
    #             events = minknow_event_detect(np.asarray(signal, dtype=float), sample_rate=rna_sample_rate, get_peaks=True, **params)
    #             for indx, i in enumerate(range(0, len(signal), size)):
    #                 begin = i
    #                 end = min(size*(indx+1), len(signal))
    #                 interval = [begin, end]
    #                 outpath = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/rna_training/event_detect_rna_plots"
    #                 outpath = os.path.join(outpath, os.path.basename(readpath)+str(i)+".png")
    #                 print("Plotting {} to {} of read {}".format(begin, end, outpath), file=sys.stderr)
    #                 # raw_scatter_plot_with_events(signal_data=signal, label_data=label, interval=interval,
    #                 #                              outpath=outpath, events=events)
    #                 raw_scatter_plot(signal_data=signal, label_data=label, interval=interval, outpath=outpath)
    #                 stop1 = timer()
    #                 print("Finished part {} to {} in {} seconds".format(begin, end, stop1 - start1), file=sys.stderr)
    #             count += 1
    # print(signal)
    # print(label)
    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
