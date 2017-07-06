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
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from nanotensor.error import PathError
import numpy as np

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

def merge_two_dicts(dict1, dict2):
    """Given two dicts, merge them into a new dict as a shallow copy.
    source: https://stackoverflow.com/questions/38987/
    how-to-merge-two-python-dictionaries-in-a-single-expression"""
    final = dict1.copy()
    final.update(dict2)
    return final

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def upload_file_to_s3(bucket_path, filepath, key):
    """Upload a file or directory to an aws bucket"""
    # s3_conn = S3Connection(host='s3-us-west-1.amazonaws.com')


    conn = S3Connection(host='s3-us-west-2.amazonaws.com')
    test = conn.lookup(bucket_path)
    if test is None:
        print("There is no bucket with this name!", file=sys.stderr)
        return 1
    else:
        bucket = conn.get_bucket(bucket_path)

    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()

    k = Key(bucket)
    k.key = key
    k.set_contents_from_filename(filepath,
            cb=percent_cb, num_cb=10)
    sys.stdout.write('\n')

def upload_model(bucket, files, dir_name):
    """Upload all files to bucket"""
    for file1 in files:
        name = file1.split("/")[-1]
        key = os.path.join(dir_name, name)
        upload_file_to_s3(bucket, file1, key)

def main():
    """Test the methods"""
    start = timer()
    file_path = "/Users/andrewbailey/nanopore-RNN/logs/06Jun-29-11h-11m/checkpoint"
    file_list = ["/Users/andrewbailey/nanopore-RNN/kmers.txt", "/Users/andrewbailey/nanopore-RNN/logs/06Jun-29-11h-11m/checkpoint", "/Users/andrewbailey/nanopore-RNN/logs/06Jun-29-11h-11m/my_test_model-3215.data-00000-of-00001"]
    bucket = "neuralnet-accuracy"
    dir_name = "06Jun-29-11h-30m-11.0%"
    # upload_file_to_s3(bucket, file_path, "12.0%/checkpoint")
    upload_model(bucket, file_list, dir_name)
    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
