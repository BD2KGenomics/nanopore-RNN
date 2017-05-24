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
# Author: Andrew Bailey, Rojin Safavi
# History: 5/16/2017 Created
from __future__ import print_function
from timeit import default_timer as timer
import sys
import os
import boto
from error import PathError
import numpy as np


import numpy as np
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import glob
import random

def no_skipped_events(filepath):
    """Find if there are any skipped events in a signalalign file"""
    # this is quite slow but it works
    set1 = set()
    with open(filepath, 'r') as file_handle:
        for line in file_handle:
            set1.add(int(line.rstrip().split()[5]))
    return check_sequential(set1)

def check_sequential(list_of_integers):
    """Make sure there are no gaps in a list of integers"""
    # returns true if there are no gaps
    return bool(sorted(list_of_integers) == list(range(min(list_of_integers),\
     max(list_of_integers)+1)))

def grab_s3_files(bucket_path, ext=""):
    """Grab the paths to files with an extention in a s3 bucket or in a local directory"""
    # connect to s3
    bucket_path = bucket_path.split("/")
    conn = boto.connect_s3()
    test = conn.lookup(bucket_path[0])
    if test is None:
        print("There is no bucket with this name!", file=sys.stderr)
        return 1
    else:
        bucket = conn.get_bucket(bucket_path[0])
    file_paths = []
    # check file in each bucket
    for key in bucket.list("/".join(bucket_path[1:])):
        if ext == "":
            file_paths.append(os.path.join("s3://", bucket_path[0], key.name))
        else:
            if key.name.split(".")[-1] == ext:
                file_paths.append(os.path.join("s3://", bucket_path[0], key.name))
    return file_paths

def list_dir(path, ext=""):
    """get all fast5 file paths from local directory"""
    if ext == "":
        onlyfiles = [os.path.join(os.path.abspath(path), f) for f in \
        os.listdir(path) if \
        os.path.isfile(os.path.join(os.path.abspath(path), f))]
    else:
        onlyfiles = [os.path.join(os.path.abspath(path), f) for f in \
        os.listdir(path) if \
        os.path.isfile(os.path.join(os.path.abspath(path), f)) \
        if f.split(".")[-1] == ext]
    return onlyfiles

def check_events(directory):
    """Check if all the tsv files from signal align match each event"""
    counter = 0
    good_files = []
    # make sure each file has all events
    for file1 in list_dir(directory, ext="tsv"):
        if no_skipped_events(file1):
            good_files.append(file1)
        else:
            counter += 1
    # print how many failed and return files that passed
    print("{} files had missing events".format(counter))
    return good_files

def project_folder():
    """Find the project folder path from any script"""
    current = os.path.abspath(__file__).split("/")
    path = '/'.join(current[:current.index("nanopore-RNN")+1])
    if os.path.exists(path):
        return path
    else:
        PathError("Path to directory does not exist!")

def get_project_file(localpath):
    """Get the path to an internal project file"""
    if localpath != "":
        if not localpath.startswith('/'):
            localpath = '/'+localpath
    path = project_folder()+localpath
    if os.path.isfile(path):
        return path
    else:
        raise PathError("Path to file does not exist!")

#signalAlign
def get_refrence_andEdit(referencePath, reference_Modified_Path):
    '''Get fast5 file and remove \n from the ends'''
    with open(reference_Modified_Path, 'w') as outfile, open(referencePath, 'r') as infile:
        for line in infile:
            if ">" in line:
                outfile.write(line)
            else:
                T = line.rstrip()
                outfile.write(T)

def get_motif_complement(motif):
    '''get the complement of a motif'''
    dna = Seq(motif)
    motif_complement = str(dna.complement())
    return motif_complement

def make_Bed_file (reference_modified_Path, BED_file_path, motif1,modified_motif1,modified_motif1_comp, alphabet, motif2 = False, modified_motif2 = False, modified_motif2_comp = False):
    sequence_list = ""
    seq_name = ""
    string1 = motif1[[i for i in range(len(motif1)) if motif1[i] != modified_motif1[i]][0]]
    motif1_comp = get_motif_complement(motif1)
    with open(reference_modified_Path, 'r') as infile:
        for line in infile:
            if ">" in line:
                 seq_name = seq_name + line.rsplit()[0].split(">")[1]
            else:
                sequence_list = sequence_list + line
    with open(BED_file_path, "w") as output:
        motif1_replaced = sequence_list.replace(motif1, modified_motif1)
        motif1_position = [m.start() for m in re.finditer('M', motif1_replaced)]
        motif1_comp_replaced = sequence_list.replace(motif1_comp, modified_motif1_comp)
        motif1_comp_position = [m.start() for m in re.finditer('M', motif1_comp_replaced)]
        if motif2 == False:
            for i in motif1_position:
                output.write(seq_name + "\t" + np.str(i) + "\t" + "+" + "\t" + string1 +"\t" + alphabet + "\n")
            for i in motif1_comp_position:
                output.write(seq_name + "\t" + np.str(i) + "\t" + "-" + "\t" + string1 +"\t" + alphabet + "\n")
        elif motif2 != False:
            motif2_comp = get_motif_complement(motif2)
            motif_1and2_replaced = motif1_replaced.replace(motif2, modified_motif2)
            motif_1and2_positions = [m.start() for m in re.finditer('M', motif_1and2_replaced)]
            motif_1and2_comp_replaced = motif1_comp_replaced.replace(motif2_comp, modified_motif2_comp)
            motif_1and2_comp_positions = [m.start() for m in re.finditer('M', motif_1and2_comp_replaced)]
            for i in motif_1and2_positions:
                output.write(seq_name + "\t" + np.str(i) + "\t" + "+" + "\t" + string1 +"\t" + alphabet + "\n")
            for i in motif_1and2_comp_positions:
                output.write(seq_name + "\t" + np.str(i) + "\t" + "-" + "\t" + string1 +"\t" + "E" + "\n")

## Concatenate control and experimental assignments
def concatenate_assignments (assignments_path1, assignments_path2, output):
    '''concatenates control and experimental assignments'''
    read_files = glob.glob(assignments_path1 + "/*.assignments") + glob.glob(assignments_path2 + "/*.assignments")
    with open(output, "w") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())

## for each kmer in assignmnets get 50 assignment or less
def get_sample_assignments(concatenated_assignmnets_path, sampled_assignments):
    kmerDict = dict()
    with open(concatenated_assignmnets_path, "r") as infile:
        for i in infile:
            key = i.split("\t")[0]
            value = "\t".join(i.split("\t")[1:])
            if kmerDict.has_key(key):
                kmerDict[key].append(value)
            else:
                kmerDict[key] = [value]
    with open(sampled_assignments, "w") as outfile:
        for key, value in kmerDict.iteritems():
            mylist = kmerDict[key]
            if len(mylist) >= 50:
                rand_smpl = [mylist[i] for i in random.sample(range(len(mylist)),50)]
                for g in rand_smpl:
                    string = ''.join(g)
                    outfile.write(key + "\t" + string)
            elif len(mylist) < 50:
                rand_smpl = [mylist[i] for i in random.sample(range(len(mylist)),len(mylist))]
                for g in rand_smpl:
                    string = ''.join(g)
                    outfile.write(key + "\t" + string)

                
def sum_to_one(vector):
    """Make sure a vector sums to one, if not, create diffuse vector"""
    total = sum(vector)
    if total != 1:
        if total > 1:
            # NOTE Do we want to deal with vectors with probability over 1?
            pass
        else:
            # NOTE this is pretty slow so maybe remove it?
            leftover = 1 - total
            amount_to_add = leftover/ (len(vector) - np.count_nonzero(vector))
            for index, prob in enumerate(vector):
                if prob == 0:
                    vector[index] = amount_to_add
    return vector


def add_field(np_struct_array, descr):
    """Return a new array that is like the structured numpy array, but has additional fields.
    descr looks like descr=[('test', '<i8')]
    """
    if np_struct_array.dtype.fields is None:
        raise ValueError("Must be a structured numpy array")
    new = np.zeros(np_struct_array.shape, dtype=np_struct_array.dtype.descr + descr)
    for name in np_struct_array.dtype.names:
        new[name] = np_struct_array[name]
    return new

def main():
    """Test the methods"""
    start = timer()
    # file1 = """/Users/andrewbailey/nanopore-RNN/temp/tempFiles_alignment/132de6a8-df1e-468f-848b-abc960e1fc76_Basecall_2D_template.sm.backward.tsv"""
    # dir1 = "/Users/andrewbailey/nanopore-RNN/temp/tempFiles_alignment/"
    # print(len(grab_s3_files("bailey-nanonet/fast5files2", ext="a")))
    # check_events(dir1)
    # print(len(list_dir(dir1, ext="a")))
    # print(find_skipped_events(file1))
    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
