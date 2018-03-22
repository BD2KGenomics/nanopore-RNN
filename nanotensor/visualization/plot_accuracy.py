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


class SamBam(object):
    """Object to handle sam bam files"""

    def __init__(self, bamsam):
        assert os.path.isfile(bamsam), "{} file does not exist".format(bamsam)
        ext = os.path.splitext(os.path.basename(bamsam))[1]
        assert ext == '.bam' or ext == '.sam', "{} is not sam or bam file".format(bamsam)
        if ext == '.bam':
            self.samfile = pysam.AlignmentFile(bamsam, "rb")
        elif ext == '.sam':
            self.samfile = pysam.AlignmentFile(bamsam, "r")

    def get_accuracies(self):
        """Get list of accuracies"""
        accuracies = []
        # go through each line in sam file
        try:
            while True:
                read = self.samfile.next()
                cigar_stats = read.get_cigar_stats()
                # sum matches, mismatches, inserts and deletions
                alignment_len = cigar_stats[0][0] + cigar_stats[0][1] + cigar_stats[0][2] + cigar_stats[0][7] + \
                                cigar_stats[0][8]
                matches = cigar_stats[0][0]
                if alignment_len == 0:
                    pass
                else:
                    accuracies.append(matches / float(alignment_len))
        except StopIteration:
            return accuracies


def violin_plot(data, outpath, manutest=False):
    """Plot very basic violin plot"""
    plt.figure(figsize=(3, 8))
    panel1 = plt.axes([0.15, 0.15, .8, .8])
    panel1.set_ylim(0.5, 1)
    plt.title('Basecalling Accuracy')

    # plot distributions
    sns.violinplot(data=data)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # fix labels
    panel1.set_ylabel('Accuracy')

    # plot means
    avg_0 = np.average(data[0])
    avg_1 = np.average(data[1])
    panel1.axhline(avg_0)
    panel1.axhline(avg_1)
    panel1.text(1, avg_0, "average = {0:.3f}".format(avg_0), fontsize=10, va="bottom", ha="center")
    panel1.text(0, avg_1,  "average = {0:.3f}".format(avg_1), fontsize=10, va="bottom", ha="center")

    # plot mann-whitney u test
    if manutest:
        mwu_1_2 = round(stats.mannwhitneyu(data[0], data[1], alternative='two-sided')[1], 3)
        middle = np.average([np.average(data[0]),  np.average(data[1])])
        panel1.plot([0, 1], [middle, middle], linewidth=1, color="black")
        panel1.text(0.5, middle, "p={0:.3f}".format(mwu_1_2), fontsize=10, va="bottom", ha="center")

    plt.savefig(outpath)


def densityplot_accuracy(data, outpath, manutest=False):
    """plot accuracy distribution of reads"""
    # define figure size
    plt.figure(figsize=(10, 4))
    panel1 = plt.axes([0.1, 0.2, .7, .7])
    panel1.set_xlim(0.5, 1)
    plt.title('Basecalling Accuracy')

    #plot distributions
    sns.distplot(data[0], bins=100, label="chiron")
    sns.distplot(data[1], bins=100, label="albacore")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # fix labels
    panel1.set_xlabel('Accuracy')
    # plot means
    avg_0 = np.average(data[0])
    avg_1 = np.average(data[1])
    panel1.axvline(avg_0)
    panel1.axvline(avg_1)
    panel1.text(avg_0-0.01, 25, "average = {0:.3f}".format(avg_0), fontsize=10, va="bottom", ha="center")
    panel1.text(avg_1-0.01, 25, "average = {0:.3f}".format(avg_1), fontsize=10, va="bottom", ha="center")

    # plot mann-whitney u test
    if manutest:
        mwu_1_2 = round(stats.mannwhitneyu(data[0], data[1], alternative='two-sided')[1], 3)
        middle = np.average([np.average(data[0]),  np.average(data[1])])
        panel1.plot([np.average(data[0]), np.average(data[1])], [12, 12], linewidth=1, color="black")
        panel1.text(middle, 12, "p={0:.3f}".format(mwu_1_2), fontsize=10, va="bottom", ha="center")

    plt.savefig(outpath)


def main():
    """Main docstring"""
    start = timer()
    all_rna_albacore = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/primary.mapped.bam"
    chiron_recent_model = "/Users/andrewbailey/CLionProjects/nanopore-RNN/nanotensor/visualization/all.sorted.bam"
    sam = SamBam(all_rna_albacore)
    all_rna_accuracies = sam.get_accuracies()

    sam = SamBam(chiron_recent_model)
    chiron_accuracies = sam.get_accuracies()
    # print(accuracies)
    outpath = "accuracy_density.png"
    data = [chiron_accuracies, all_rna_accuracies]
    densityplot_accuracy(data, outpath)
    outpath = "accuracy_violin.png"
    violin_plot(data, outpath)
    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
