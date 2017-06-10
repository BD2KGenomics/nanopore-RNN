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
from __future__ import print_function
import sys
from timeit import default_timer as timer
import glob
import random
import re
import numpy as np
from Bio.Seq import Seq
from utils import get_project_file, project_folder
# from Bio.Alphabet import generic_dna

def get_refrence_and_edit(referencePath, reference_Modified_Path):
    """Get fast5 file and remove \n from the ends"""
    with open(reference_Modified_Path, 'w') as outfile, open(referencePath, 'r') as infile:
        for line in infile:
            if ">" in line:
                outfile.write(line)
            else:
                T = line.rstrip()
                outfile.write(T)

def get_motif_complement(motif):
    """get the complement of a motif, cuurrently works with A,T,C,G,E,X,O
    ex: the complement of ATCO is TAGO"""
    dna = Seq(motif)
    motif_complement = str(dna.complement())
    return motif_complement

def get_motif_REVcomplement(motif):
    """get the complement of a motif, cuurrently works with A,T,C,G,E,X,O
    ex: the REVcomplement of ATCO is OGAT"""
    dna = Seq(motif)
    rev_complement = str(dna.reverse_complement())
    return rev_complement

def store_seq_and_name(reference_modified_Path):
    sequence_list = ""
    seq_name = ""
    with open(reference_modified_Path, 'r') as infile:
        for line in infile:
            if ">" in line:
                seq_name = seq_name + line.rsplit()[0].split(">")[1]
            else:
                sequence_list = sequence_list + line
    return seq_name, sequence_list


def replace_nucleotide(motif, replacement):
    """compares motifs and modifed motif and
        tells you what nucleotide is modified
        ex: ("CCAGG","CFAGG") => C"""
    pos = [i for i in range(len(motif)) if motif[i] != replacement[i]][0]
    old_char = motif[pos]
    new_char = replacement[pos]
    rev_comp_pos = len(motif)-pos-1

    return pos, old_char, new_char, rev_comp_pos

def nuc_position(seq_str, char):
    """Finds all positions of specific character
        withing sequence"""
    motif_position = [m.start() for m in re.finditer(char, seq_str)]
    return motif_position


def make_bed_file(ref_modified_path, bed_path, *args):
    """ex: args = [("CCAGG","CEAGG"), ("CCTGG","CETGG")]"""
    seq_name, seq_str_fwd  = store_seq_and_name(ref_modified_path)
    seq_name, seq_str_bwd  = store_seq_and_name(ref_modified_path)
    with open(bed_path, "w") as outfile:
        for pair in args:
            motif = pair[0]
            modified = pair[1]
            # get pos, old character and the replacement character
            pos, old_char, new_char, rev_comp_pos = replace_nucleotide(motif, modified)
            # get get rev_complement of motif and modified
            motif_comp = get_motif_REVcomplement(motif)
            # changed from rev complement to expand the alphabet and not contain
            # replacements to a single character, it can be different across motifs
            modified_comp = motif_comp[:rev_comp_pos] + new_char + \
                                   motif_comp[rev_comp_pos+1:]

            seq_str_fwd = seq_str_fwd.replace(motif, modified)
            seq_str_bwd = seq_str_bwd.replace(motif_comp, modified_comp)
        nuc_positions = nuc_position(seq_str_fwd, new_char)
        for pos in nuc_positions:
            outfile.write(seq_name + "\t" + np.str(pos) + "\t" + "+" + "\t"
                          + old_char +"\t" + new_char + "\n")
        nuc_positions = nuc_position(seq_str_bwd, new_char)
        for pos in nuc_positions:
            outfile.write(seq_name + "\t" + np.str(pos) + "\t" + "-" + "\t"
                          + old_char +"\t" + new_char + "\n")


## Concatenate control and experimental assignments
## ex : concatenation of non methylated and methylated assignments
def concat_assignments (assignments_path1, assignments_path2, output, op_prefix):
    """concatenates control and experimental assignments"""
    read_files = glob.glob(assignments_path1 + "/*." + op_prefix) + glob.glob(assignments_path2 + "/*." + op_prefix)
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



def main():
    """Test the methods"""
    start = timer()

    ref_seq = get_project_file("/testing/reference-sequences/ecoli_k12_mg1655.fa")
    reference_modified_path = project_folder()+"/testing/reference-sequences/ecoli_k12_mg1655_modified.fa"
    get_refrence_and_edit(ref_seq, reference_modified_path)
    bed_file_path = project_folder()+"/testing/reference-sequences/CCAGG_modified2.bed"
    char = "E"
    make_bed_file(reference_modified_path, bed_file_path, ["CCTGG","CETGG"], ["CCAGG","CEAGG"])

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
