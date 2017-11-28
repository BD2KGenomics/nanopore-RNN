#!/usr/bin/env python
"""Plot information needed file"""
########################################################################
# File: plot_accuracy.py
#  executable: plot_accuracy.py
#
# Author: Andrew Bailey
# History: Created 11/22/17
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
from nanotensor.trim_signal import read_label
from nanotensor.utils import list_dir


def densityplot_events(data, outpath, manutest=False):
    """plot accuracy distribution of reads"""
    # define figure size
    plt.figure(figsize=(10, 4))
    panel1 = plt.axes([0.1, 0.1, .7, .7])
    # longest = max(data[0]) + data[1])
    panel1.set_xlim(0, 1000)
    # panel1.set_xscale("log")

    plt.title('Event Lengths')

    #plot distributions
    # sns.distplot(data[0], bins=1000, label="rna")
    # sns.distplot(data[1], bins=1000, label="dna")
    sns.distplot(data[0], label="rna")
    sns.distplot(data[1], label="dna")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # fix labels
    panel1.set_xlabel('Event Lengths')
    # plot means
    avg_0 = np.average(data[0])
    avg_1 = np.average(data[1])
    # panel1.axvline(avg_0)
    # panel1.axvline(avg_1)
    # panel1.text(avg_0-5, 0.03, "average = {0:.3f}".format(avg_0), fontsize=10, va="bottom", ha="center")
    # panel1.text(avg_1+5, 0.03, "average = {0:.3f}".format(avg_1), fontsize=10, va="bottom", ha="center")

    # plot mann-whitney u test
    if manutest:
        mwu_1_2 = round(stats.mannwhitneyu(data[0], data[1], alternative='two-sided')[1], 3)
        middle = np.average([np.average(data[0]),  np.average(data[1])])
        panel1.plot([np.average(data[0]), np.average(data[1])], [12, 12], linewidth=1, color="black")
        panel1.text(middle, 12, "p={0:.3f}".format(mwu_1_2), fontsize=10, va="bottom", ha="center")

    plt.savefig(outpath)

def plot_histogram(rna_data, dna_data, outpath):
    """plot accuracy distribution of reads"""
    # define figure size
    plt.figure(figsize=(10, 4))
    panel1 = plt.axes([0.12, 0.12, .7, .7])
    longest = max(rna_data)
    panel1.set_xlim(0, 200)


    #plot histogram
    plt.hist(rna_data, bins=12000, normed=True, label="RNA", histtype="barstacked", alpha=0.5)
    plt.hist(dna_data, bins=3000, normed=True, label="DNA", histtype="barstacked", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    panel1.text(75, .02, "rna_data median = {}".format(np.median(rna_data)), fontsize=10, va="bottom", ha="center")
    panel1.text(75, .03, "rna_data max = {}".format(longest), fontsize=10, va="bottom", ha="center")
    panel1.text(75, .04, "rna_data min = {}".format(min(rna_data)), fontsize=10, va="bottom", ha="center")
    panel1.text(150, .02, "dna_data median = {}".format(np.median(dna_data)), fontsize=10, va="bottom", ha="center")
    panel1.text(150, .03, "dna_data max = {}".format(max(dna_data)), fontsize=10, va="bottom", ha="center")
    panel1.text(150, .04, "dna_data min = {}".format(min(dna_data)), fontsize=10, va="bottom", ha="center")

    # fix labels
    panel1.set_xlabel('Event Length')
    panel1.set_ylabel('Number of events')

    plt.savefig(outpath)


def main():
    """Main docstring"""
    start = timer()
    # label_dir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/chiron/data/raw"
    label_dir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/rna_training/training"
    label_files = list_dir(label_dir, ext='label')
    rna_event_lengths = []
    for label_f in label_files:
        events = read_label(label_f)
        rna_event_lengths.extend(events.length)
    label_dir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/dna_training/training"
    label_files = list_dir(label_dir, ext='label')
    dna_event_lengths = []
    for label_f in label_files:
        events = read_label(label_f)
        dna_event_lengths.extend(events.length)

    outpath = "event_hist.png"
    plot_histogram(rna_event_lengths, dna_event_lengths, outpath)
    # densityplot_events([rna_event_lengths, dna_event_lengths], "event_density.png")
    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
