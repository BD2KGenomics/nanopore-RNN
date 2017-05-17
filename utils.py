#!/usr/bin/env python
########################################################################
# File: utils.py
#  executable: utils.py
# Purpose: maintain some simple functions as needed
    # TODO make sure all events are represented from output from signalalign

#   stderr: errors and status
#   stdout:
#
# Author: Andrew Bailey
# History: 5/16/2017 Created
from __future__ import print_function
from timeit import default_timer as timer
import sys
import boto

def find_skipped_events(filepath):
    """Find if there are any skipped events in a signalalign file or an event align file"""
    set1 = set()
    with open(filepath, 'r') as fh:
        for line in fh:
            # print(line.rstrip().split()[5])
            set1.add(int(line.rstrip().split()[5]))
    return check_sequential(set1)


def check_sequential(list_of_integers):
    """Make sure there are no gaps in a list of integers"""
    l = list_of_integers
    if sorted(l) == list(range(min(l), max(l)+1)):
        return True
    else:
        return False

def grab_s3_files(bucket_path, ):
    """Grab the paths to all fast5 files in a s3 bucket or in a local directory"""
    bucket = bucket_path.split("/")
    c = boto.connect_s3()
    test = c.lookup(bucket[0])
    if test is None:
        print("There is no bucket with this name!", file=sys.stderr)
        return 1
    else:
        b = c.get_bucket(bucket[0])
    file_paths = []
    for key in b.list("/".join(bucket[1:])):
        if key.name[-5:] == "fast5":
            file_paths.append(os.path.join("s3://", bucket[0], key.name))
    return file_paths

def list_dir(path, fast5=True):
    """get all fast5 file paths from local directory"""
    if fast5:
        onlyfiles = [os.path.join(os.path.abspath(path), f) for f in \
        os.listdir(bucket_path) if \
        os.path.isfile(os.path.join(os.path.abspath(path), f)) \
        if f[-5:] == "fast5"]
    else:
        onlyfiles = [os.path.join(os.path.abspath(path), f) for f in os.listdir(bucket_path)]
    # print(onlyfiles)
    return onlyfiles



def main():
    start = timer()
    file1 = "/Users/andrewbailey/nanopore-RNN/temp/tempFiles_alignment/132de6a8-df1e-468f-848b-abc960e1fc76_Basecall_2D_template.sm.backward.tsv"
    dir1 = "/Users/andrewbailey/nanopore-RNN/temp/tempFiles_alignment/"


    print(find_skipped_events(file1))
    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__=="__main__":
    main()
    raise SystemExit
