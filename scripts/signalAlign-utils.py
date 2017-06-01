#!/usr/bin/env python
"""
This is a place for small scripts and utility functions
"""
########################################################################
# File: utils.py
#  executable: utils.py
# Purpose: maintain some simple functions as needed
#   make sure all events are represented from output from signalalign

#   stderr: errors and status
#   stdout:
#
# Author: Rojin Safavi
# History: 06/01/2017 Created
import sys
import os
import numpy as np
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import glob
import random
import re


def get_motif_complement(motif):
    '''get the complement of a motif, cuurrently works with A,T,C,G,E,X,O
    ex: the complement of ATCO is TAGO'''
    dna = Seq(motif)
    motif_complement = str(dna.complement())
    return motif_complement

def store_seq_and_name(reference_modified_Path):
    sequence_list = ""
    seq_name = ""
    with open(reference_modified_Path, 'r') as infile:
        for line in infile:
            if ">" in line:
                seq_name = seq_name + line.rsplit()[0].split(">")[1]
            else:
                sequence_list = sequence_list + line
    return seq_name,sequence_list

def replace_nucleotide(motif, modified):
    modified_nuc = motif[[x for x in range(len(motif)) if motif[x] != modified[x]][0]]
    return modified_nuc

def nuc_position(seq_str):
    motif_position = [m.start() for m in re.finditer('F', seq_str)]
    return motif_position

def make_bed_file(ref_modified_path, bed_path, char, *args):
    seq_str_fwd = store_seq_and_name(ref_modified_path)[1]
    seq_name = store_seq_and_name(ref_modified_path)[0]
    seq_str_bwd = store_seq_and_name(ref_modified_path)[1]
    for pair in args:
        motif = pair[0]
        modified = pair[1]
        motif_comp = get_motif_complement(motif)
        modified_comp = get_motif_complement(modified)
        '''outputs the nucleotide that is been modified, it can be A,T,C, or G'''
        modified_nuc = replace_nucleotide(motif, modified)
        seq_str_fwd = seq_str_fwd.replace(motif, modified)
        seq_str_bwd = seq_str_bwd.replace(motif_comp, modified_comp)
    with open(bed_path, "a") as outfile:
        nuc_positions = nuc_position(seq_str_fwd)
        for pos in nuc_positions:
            outfile.write(seq_name + "\t" + np.str(pos) + "\t" + "+" + "\t" + modified_nuc +"\t" + char + "\n")
        nuc_positions = nuc_position(seq_str_bwd)
        for pos in nuc_positions:
            outfile.write(seq_name + "\t" + np.str(pos) + "\t" + "-" + "\t" + modified_nuc +"\t" + char + "\n")

## Concatenate control and experimental assignments
def concat_assignments (assignments_path1, assignments_path2, output):
    '''concatenates control and experimental assignments'''
    read_files = glob.glob(assignments_path1 + "/*.assignments") + glob.glob(assignments_path2 + "/*.assignments")
    with open(output, "w") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())

## for each kmer in assignmnets get 50 assignment or less
def get_sample_assignments(concat_assign_path, output_sampled):
    kmerDict = dict()
    with open(concat_assign_path, "r") as infile:
        for i in infile:
            key = i.split("\t")[0]
            value = "\t".join(i.split("\t")[1:])
            if kmerDict.has_key(key):
                kmerDict[key].append(value)
            else:
                kmerDict[key] = [value]
    with open(output_sampled, "w") as outfile:
        for key, value in kmerDict.iteritems():
            mylist = kmerDict[key]
            if len(mylist) >= 50:
                rand_smpl = [mylist[i] for i in random.sample(range(len(mylist)), 50)]
                for g in rand_smpl:
                    string = ''.join(g)
                    outfile.write(key + "\t" + string)
            elif len(mylist) < 50:
                rand_smpl = [mylist[i] for i in random.sample(range(len(mylist)), len(mylist))]
                for g in rand_smpl:
                    string = ''.join(g)
                    outfile.write(key + "\t" + string)
