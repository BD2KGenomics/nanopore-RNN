#!/usr/bin/env python
"""Create features and labels from alignment kmers"""
########################################################################
# File: data_preparation.py
#  executable: data_preparation.py
# Purpose: Give multiple options to create features from event information
#           and create multiple ways to create labels
#
# Author: Andrew Bailey
# History: 05/17/17 Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
from utils import get_project_file, project_folder
import h5py
import numpy as np
import csv

def scrape_fast5(fast5_file):
    """Scrape a fast5 file for event information"""
    assert fast5_file.endswith("fast5")
    with h5py.File(fast5_file, 'r+') as fast5:
        template = fast5.get("Analyses/Basecall_1D_000/BaseCalled_template/Events").value
    print(template.dtype.names)
    events = template[["mean", "start", "stdv", "length", "model_state", "move"]]
    return events

def scrape_signalalign(tsv1, events):
    """Grab the event kmers from the signal align output"""
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


def add_column(nparray, name):
    """Add a column of zeros to a numpy array"""

    b = np.zeros((N,N+1))
    b[:,:-1] = a


def main():
    """Main docstring"""
    start = timer()

    fast5_file = \
    get_project_file("test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch146_read1209_strand1.fast5")

    signalalign_file = \
    get_project_file("/temp/tempFiles_alignment/132de6a8-df1e-468f-848b-abc960e1fc76_Basecall_2D_template.sm.backward.tsv")

    print(fast5_file)
    events = scrape_fast5(fast5_file)
    print(events[0])
    # np.savez(project_folder()+"/events", events)
    # np.load(project_folder()+"/events.npz")

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
