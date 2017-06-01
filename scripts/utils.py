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
import re
import glob
import random
from multiprocessing import Process, Queue
import boto
from error import PathError
import numpy as np
from Bio.Seq import Seq

# from Bio.Alphabet import generic_dna
#TODO create debug function and verbose options


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
    """get all file paths from local directory with extention"""
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
    path = os.path.join(project_folder()+localpath)
    if os.path.isfile(path):
        return path
    else:
        raise PathError("Path to file does not exist!")

def testfast5():
    """Get the path to one of our test fast5 files"""
    return get_project_file("test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch146_read1209_strand1.fast5")
#signalAlign
def remove_fasta_newlines(reference_path, reference_modified_path):
    """Get fasta file and remove \n from the ends"""
    with open(reference_modified_path, 'w') as outfile, open(reference_path, 'r') as infile:
        for line in infile:
            if ">" in line:
                outfile.write(line)
            else:
                newline = line.rstrip()
                outfile.write(newline)

def get_motif_complement(motif):
    """get the complement of a motif"""
    dna = Seq(motif)
    motif_complement = str(dna.complement())
    return motif_complement

def make_bed_file(reference_modified_path, bed_file_path, motif1, modified_motif1, modified_motif1_comp, replace):
    """This method does something"""
    sequence_list = str()
    seq_name = str()
    string1 = motif1[[i for i in range(len(motif1)) if motif1[i] != modified_motif1[i]][0]]
    print(string1)
    motif1_comp = get_motif_complement(motif1)
    with open(reference_modified_path, 'r') as infile:
        for line in infile:
            if ">" in line:
                seq_name = seq_name + line.rsplit()[0].split(">")[1]
            else:
                sequence_list = sequence_list + line
    with open(bed_file_path, "w") as output:
        motif1_replaced = sequence_list.replace(motif1, modified_motif1)
        motif1_position = [m.start() for m in re.finditer(replace, motif1_replaced)]
        motif1_comp_replaced = sequence_list.replace(replace, modified_motif1_comp)
        motif1_comp_position = [m.start() for m in re.finditer(replace, motif1_comp_replaced)]
        for i in motif1_position:
            output.write(seq_name + "\t" + np.str(i) + "\t" + "+" + "\t" + string1 + "\t" + replace + "\n")
        for i in motif1_comp_position:
            output.write(seq_name + "\t" + np.str(i) + "\t" + "-" + "\t" + string1 +"\t" + replace + "\n")

## Concatenate control and experimental assignments
def concatenate_assignments(assignments_path1, assignments_path2, output):
    """concatenates control and experimental assignments"""
    read_files = glob.glob(assignments_path1 + "/*.assignments") +\
    glob.glob(assignments_path2 + "/*.assignments")
    with open(output, "w") as outfile:
        for file1 in read_files:
            with open(file1, "rb") as infile:
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


class Data:
    """Object to manage data for shuffling data inputs"""
    def __init__(self, file_list, batch_size, queue_size, verbose=False, pad=0, trim=True):
        self.file_list = file_list
        self.num_files = len(self.file_list)
        self.queue = Queue(maxsize=queue_size)
        self.file_index = 0
        self.batch_size = batch_size
        self.verbose = verbose
        self.process1 = Process(target=self.load_data, args=())
        self.pad = pad
        self.trim = trim

    def shuffle(self):
        """Shuffle the input file order"""
        if self.verbose:
            print("Shuffle data files", file=sys.stderr)
        np.random.shuffle(self.file_list)
        return True

    def add_to_queue(self, batch, wait=True, pad=0):
        """Add a batch to the queue"""
        if pad > 0:
            # print(batch[-1])
            batch = self.pad_with_zeros(batch, pad=pad)
            # print(batch[-1])
        self.queue.put(batch, wait)

    @staticmethod
    def pad_with_zeros(matrix, pad=0):
        """Pad an array with zeros so it has the correct shape for the batch"""
        column1 = len(matrix[0][0])
        column2 = len(matrix[0][1])
        one_row = np.array([[np.zeros([column1]), np.zeros([column2])]])
        new_rows = np.repeat(one_row, pad, axis=0)
        # print(new_rows.shape)
        return np.append(matrix, new_rows, axis=0)


    def get_batch(self, wait=True):
        """Get a batch from the queue"""
        batch = self.queue.get(wait)
        features = batch[:, 0]
        labels = batch[:, 1]
        features = np.asarray([np.asarray(features[n]) for n in range(len(features))])
        labels = np.asarray([np.asarray(labels[n]) for n in range(len(labels))])
        return features, labels

    def create_batches(self, data):
        """Create batches from input data array"""
        num_batches = (len(data) // self.batch_size)
        pad = self.batch_size - (len(data) % self.batch_size)
        if self.verbose:
            print("{} batches in this file".format(num_batches), file=sys.stderr)
        batch_number = 0
        more_data = True
        index_1 = 0
        index_2 = self.batch_size
        while more_data:
            next_in = data[index_1:index_2]
            self.add_to_queue(next_in)
            batch_number += 1
            index_1 += self.batch_size
            index_2 += self.batch_size
            if batch_number == num_batches:
                self.add_to_queue(np.array([[str(pad), str(pad)]]))
                if not self.trim:
                    next_in = data[index_1:index_2]
                    # print(np.array([pad]))
                    self.add_to_queue(next_in, pad=pad)
                more_data = False
        return True

    def read_in_file(self):
        """Read in file from file list"""
        data = np.load(self.file_list[self.file_index])
        self.create_batches(data)
        return True

    def load_data(self):
        """Create neverending loop of adding to queue and shuffling files"""
        counter = 0
        while counter <= 10:
            self.read_in_file()
            self.file_index += 1
            if self.verbose:
                print("File Index = {}".format(self.file_index), file=sys.stderr)
            if self.file_index == self.num_files:
                self.shuffle()
                self.file_index = 0
        return True

    def start(self):
        """Start background process to keep queue filled"""
        self.process1.start()
        return True

    def end(self):
        """End bacground process"""
        self.process1.terminate()
        return True

def merge_two_dicts(dict1, dict2):
    """Given two dicts, merge them into a new dict as a shallow copy.
    source:https://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression"""
    final = dict1.copy()
    final.update(dict2)
    return final

def main():
    """Test the methods"""
    start = timer()
    # file1 = """/Users/andrewbailey/nanopore-RNN/temp/tempFiles_alignment/132de6a8-df1e-468f-848b-abc960e1fc76_Basecall_2D_template.sm.backward.tsv"""
    # dir1 = "/Users/andrewbailey/nanopore-RNN/temp/tempFiles_alignment/"
    # print(len(grab_s3_files("bailey-nanonet/fast5files2", ext="a")))
    # check_events(dir1)
    # print(len(list_dir(dir1, ext="a")))
    # print(find_skipped_events(file1))
    ref_seq = get_project_file("/reference-sequences/ecoli_k12_mg1655.fa")
    reference_modified_path = get_project_file("/reference-sequences/ecoli_k12_mg1655_modified.fa")
    # remove_fasta_newlines(ref_seq, ref_seq+"1")
    bed_file_path = project_folder()+"/reference-sequences/CCAGG_modified.bed"
    motif1 = "CCAGG"
    # motif2 = "CCTGG"
    modified_motif1 = "CEAGG"
    modified_motif1_comp = "GGTCC"
    replace = "E"
    make_bed_file(reference_modified_path, bed_file_path, motif1, modified_motif1, modified_motif1_comp, replace)
    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
