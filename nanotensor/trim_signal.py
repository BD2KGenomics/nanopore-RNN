#!/usr/bin/env python
"""Trim signal file to only have signal aligned from label file"""
########################################################################
# File: trim_signal.py
#  executable: trim_signal.py
#
# Author: Andrew Bailey
# History: 10/6/17 Created
########################################################################

from __future__ import print_function
import sys
import os
import collections
import re
from timeit import default_timer as timer
from chiron.chiron_input import read_signal
from nanotensor.utils import list_dir
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from collections import defaultdict

raw_labels = collections.namedtuple('raw_labels', ['start', 'length', 'base'])

ALPHABET = ['A', 'C', 'G', 'T', 'E']


def index2base(read):
    bpread = [ALPHABET[x] for x in read]
    bpread = ''.join(x for x in bpread)
    return bpread


def base2ind(base):
    """base to 1-hot vector,
    Input Args:
        base: current base,can be AGCT, or AGCTE for methylation.
        alphabet_n: can be 4 or 5, related to normal DNA or methylation call.
        """
    return ALPHABET.index(base.upper())


class SignalLabel:
    """Class to  handle signal and label files """
    def __init__(self, signal_file, label_file):
        """Initiate signal and label files"""
        self.signal_file = signal_file
        self.label_file = label_file
        assert os.path.isfile(self.signal_file), "{} does not exist".format(self.signal_file)

    def trim_complement_signal(self, outdir):
        """Trim signal file to only have signal aligned from label file"""
        assert os.path.isdir(outdir), "{} does not exist".format(outdir)
        assert os.path.isfile(self.label_file), "{} does not exist".format(self.label_file)
        out_path = os.path.join(outdir, self.signal_file.split('/')[-1].split('.')[0] + ".trim.signal")
        signal = read_signal(self.signal_file, normalize=False)
        label = read_label(self.label_file)
        start = label.start[0]
        end = label.start[-1]
        final_signal = signal[start:end]
        out_path = self.write_signal(final_signal, out_path)
        return out_path

    @staticmethod
    def write_signal(signal, out_path):
        """Write out a signal file from list of values"""
        with open(out_path, 'w+') as f_signal:
            f_signal.write(" ".join(str(val) for val in signal))
        return out_path

    def get_sequence(self):
        """Return sequence from label file"""
        label = read_label(self.label_file, skip_start=0, bases=True)
        seq = label.base
        seq = "".join(str(val) for val in seq)
        return seq

    def motif_search(self, motif):
        """Return motif indexes from label file"""
        label = read_label(self.label_file, skip_start=0, bases=True)
        seq = label.base
        seq = "".join(str(val) for val in seq)
        indexes = [x.start() for x in re.finditer(motif, seq)]
        return [[x, x+len(motif)] for x in indexes]

    def trim_to_motif(self, motifs, prefix_length=0, suffix_length=0, methyl_index=-1, blank=False):
        """Trim labels around a motif"""
        assert type(methyl_index) is int
        label = read_label(self.label_file, skip_start=0, bases=False)
        indices = []
        for motif in motifs:
            indices.extend(self.motif_search(motif))
        for index_pair in indices:
            prefix = index_pair[0]-prefix_length
            suffix = index_pair[1]+suffix_length
            base = label.base[prefix:suffix]
            if blank:
                b = [0]*len(base)
                b[methyl_index] = base[methyl_index]
                base = b
            # print(base, label.start[prefix:suffix], sum(label.length[prefix:suffix]))
            # if methyl_index > -1:
            #     base[methyl_index] = base2ind('E')
            yield raw_labels(start=label.start[prefix:suffix],
                             length=label.length[prefix:suffix],
                             base=base)


def read_label(file_path, skip_start=10, window_n=0, bases=False):
    """Method taken from chiron_input.py https://github.com/haotianteng/chiron"""
    f_h = open(file_path, 'r')
    f_h.seek(0, 0)
    start = list()
    length = list()
    base = list()
    all_base = list()
    count = 0
    if skip_start < window_n:
        skip_start = window_n
    for count, line in enumerate(f_h):
        if count < skip_start:
            continue
        record = line.split()
        if bases:
            all_base.append(record[2])
        else:
            all_base.append(base2ind(record[2]))
        start.append(int(record[0]))
        length.append(int(record[1]) - int(record[0]))
    # print("len of allbases", file_len)
    if window_n > 0:
        f_h.seek(0, 0)  # Back to the start
        file_len = len(all_base)
        for count, line in enumerate(f_h):
            record = line.split()
            if count < skip_start or count > (file_len - skip_start - 1):
                continue
            start.append(int(record[0]))
            length.append(int(record[1]) - int(record[0]))
            k_mer = 0
            for i in range(window_n * 2 + 1):
                k_mer = k_mer * 4 + all_base[count + i - window_n]
            base.append(k_mer)
        all_base = base
    return raw_labels(start=start, length=length, base=all_base)


def trim_signal(signal_file, label_file, outdir):
    """Trim signal file to only have signal aligned from label file"""
    outpath = os.path.join(outdir, signal_file.split('/')[-1].split('.')[0] + ".trim.signal")
    signal = read_signal(signal_file, normalize=False)
    label = read_label(label_file)
    start = label.start[0]
    end = label.start[-1]
    final_signal = signal[start:end]
    with open(outpath, 'w+') as f_signal:
        f_signal.write(" ".join(str(val) for val in final_signal))
    return outpath


def trim_signal_wrapper(dir, outdir):
    """Wrapper for trim signal function used for whole directory of signal and label files"""
    signal_files = list_dir(dir, ext='signal')
    labels_files = list_dir(dir, ext='label')
    out_files = []
    for signal_f in signal_files:
        try:
            file_pre = os.path.splitext(signal_f)[0]
            f_label = file_pre + '.label'
            assert os.path.isfile(f_label)
            outpath = trim_signal(signal_f, f_label, outdir)
            out_files.append(outpath)
        except (AssertionError, ValueError) as e:
            print("cannot find {}".format(f_label), file=sys.stderr)
            continue
    return out_files


def create_alignment(fasta, label):
    """Get aligment score from fasta file and label file"""
    ref = ''
    with open(label, 'r') as label_f:
        for line in label_f:
            ref += str(line.split()[2])
    with open(fasta, 'r') as fasta_f:
        fasta_f.readline()
        fasta_seq = str(fasta_f.readline())
    # print(fasta_seq)
    alignments = pairwise2.align.globalms(ref.upper(), fasta_seq.upper(), 2, -0.5, -1, -0.3,
                                          one_alignment_only=True)
    # print(format_alignment(*alignments[0]))
    return {'reference': alignments[0][0], 'query': alignments[0][1]}


def alignment_stats(alignment):
    """Return alignment accuracies"""
    # create dictionary to keep alignment info
    alphabet = set(alignment['query'] + alignment['reference'])
    base_counts = {key: {'matches': 0, 'deletions': 0, 'insertions': 0, 'ref_mismatches': 0, 'query_mismatches': 0}
                   for key in alphabet}
    total_counts = {'matches': 0, 'deletions': 0, 'insertions': 0, 'mismatches': 0, "reference": 0, 'read': 0}
    for ref, query in zip(alignment['reference'], alignment["query"]):
        if ref == query:
            base_counts[ref]['matches'] += 1
            total_counts['matches'] += 1
            total_counts['reference'] += 1
            total_counts['read'] += 1
        elif ref == '-':
            base_counts[query]['insertions'] += 1
            total_counts['insertions'] += 1
            total_counts['read'] += 1

        elif query == '-':
            base_counts[ref]['deletions'] += 1
            total_counts['deletions'] += 1
            total_counts['reference'] += 1
        else:
            total_counts['reference'] += 1
            total_counts['read'] += 1
            total_counts['mismatches'] += 1
            base_counts[ref]['ref_mismatches'] += 1
            base_counts[query]['query_mismatches'] += 1

    return total_counts, base_counts


#
def create_summary_stats(total_counts):
    """Report summary alignment stats from total counts created by alignment_stats"""
    print("Reference sequence length: {}".format(total_counts["reference"]))
    print("Read sequence length: {}".format(total_counts["read"]))
    print("Identity Rate: {}".format(float(total_counts['matches']) / total_counts["reference"]))
    print("Mismatch Rate: {}".format(float(total_counts["mismatches"]) / total_counts['reference']))
    print("Insertion Rate: {}".format(float(total_counts["insertions"]) / total_counts['reference']))
    print("Deletion Rate: {}".format(float(total_counts["deletions"]) / total_counts['reference']))


def find_accuracy(fasta_dir, label_dir):
    fastas = list_dir(fasta_dir)
    for fasta in fastas:
        name = fasta.split('/')[-1].split('.')[0]
        label_file = os.path.join(label_dir, name + '.label')
        alignment = create_alignment(fasta, label_file)
        total_counts, base_counts = alignment_stats(alignment)
        create_summary_stats(total_counts)

    return True


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def match_label_fasta(fasta_dir, label_dir):
    """match up label files with fasta files from chiron output"""
    pairs = []
    for fasta in list_dir(fasta_dir, ext='fasta'):
        pref = os.path.splitext(fasta)[0].split('/')[-1]
        label = os.path.join(label_dir, pref + '.label')
        if os.path.exists(label):
            pairs.append([fasta, label])
        else:
            print("file not found: {}".format(label))
    return pairs


def print_summary_stats_for_base(base_counts_list, char='E'):
    """Print out summary statistics for alignment data for a base"""
    total = 0
    matches = 0
    insertions = 0
    ref_mismatches = 0
    query_mismatches = 0
    deletions = 0
    for base_counts in base_counts_list:
        try:
            total += base_counts[char]["matches"] + base_counts[char]["insertions"] + base_counts[char]["ref_mismatches"] \
                    + base_counts[char]["deletions"]
            matches += base_counts[char]["matches"]
            insertions += base_counts[char]["insertions"]
            ref_mismatches += base_counts[char]["ref_mismatches"]
            query_mismatches += base_counts[char]["query_mismatches"]
            deletions += base_counts[char]["deletions"]

        except KeyError:
            print("Base '{}' not found:".format(char))
    print("True Positives = {}".format(float(matches)/total))
    print("False Positives (insertions) = {}".format(float(insertions)/total))
    print("False Positives (query mismatch) = {}".format(float(query_mismatches)/total))
    print("False Negatives = {}".format(float(deletions)/total))
    print("False Negatives (ref mismatch) = {}".format(float(ref_mismatches)/total))

def main():
    """Main docstring"""
    start = timer()
    label = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/ch467_read35.label"
    signal = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/ch467_read35.signal"
    outdir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/"
    fasta = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/test_minion.fa"
    labeled_data = "/Users/andrewbailey/CLionProjects/nanopore-RNN/chiron/data/raw"
    fasta_dir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/chiron/data/result"

    handler = SignalLabel(signal, label)
    # handler.trim_signal("/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/")
    seq = handler.get_sequence()
    # indexs = handler.motif_search("CCAGG")
    indexs = handler.trim_to_motif(["CCAGG", "CCTGG"], prefix_length=0, suffix_length=0, methyl=1)
    # print(indexs)
    for index in indexs:
        print(index)
        break





    # pairs = match_label_fasta(fasta_dir, labeled_data)
    # base_counts_list = []
    # for pair in pairs:
    #     print(pair)
    #     alignment = create_alignment(pair[0], pair[1])
    #     total_counts, base_counts = alignment_stats(alignment)
    #     base_counts_list.append(base_counts)
    #     create_summary_stats(total_counts)
    # print_summary_stats_for_base(base_counts_list, char='C')


    # True Positives = 0.0328947368421
    # False Positives (insertions) = 0.388157894737
    # False Positives (query mismatch) = 0.302631578947
    # False Negatives = 0.276315789474
    # False Negatives (ref mismatch) = 0.302631578947
    # print(base_counts['C'])
    # outpath = trim_signal(signal, label, outdir)
    # trim_signal_wrapper(outdir, outdir)
    # alignment = pairwise2.align.globalms("ATGCEE".upper(), "ATGCATGCE".upper(), 2, -0.5, -1, -0.3,
    #                                      one_alignment_only=True)
    # print(format_alignment(*alignment[0]))
    # alignment2 = {'reference': alignment[0][0], 'query': alignment[0][1]}
    # alignment = create_alignment(fasta, label)
    # total_counts, base_counts = alignment_stats(alignment)
    # create_summary_stats(total_counts)


    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
# om Bio import pairwise2
# >>> alignments = pairwise2.align.globalxx("ACCGT", "ACG")
# >>> from Bio.pairwise2 import format_alignment
# >>> print(format_alignment(*alignments[0]))
