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
import collections
import boto
import json
from datetime import datetime
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from nanotensor.error import PathError
import numpy as np
from multiprocessing import Process, current_process, Manager
import tarfile
import logging as log

def no_skipped_events(file_path):
    """Find if there are any skipped events in a signalalign file"""
    # this is quite slow but it works
    set1 = set()
    with open(file_path, 'r') as file_handle:
        for line in file_handle:
            set1.add(int(line.rstrip().split()[5]))
    return check_sequential(set1)


def check_sequential(list_of_integers):
    """Make sure there are no gaps in a list of integers"""
    # returns true if there are no gaps
    return bool(sorted(list_of_integers) == list(range(min(list_of_integers), max(list_of_integers) + 1)))


def grab_s3_files(bucket_path, ext=""):
    """Grab the paths to files with an extension in a s3 bucket or in a local directory"""
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
    """get all file paths from local directory with extension"""
    if ext == "":
        only_files = [os.path.join(os.path.abspath(path), f) for f in \
                      os.listdir(path) if \
                      os.path.isfile(os.path.join(os.path.abspath(path), f))]
    else:
        only_files = [os.path.join(os.path.abspath(path), f) for f in \
                      os.listdir(path) if \
                      os.path.isfile(os.path.join(os.path.abspath(path), f)) \
                      if f.split(".")[-1] == ext]
    return only_files


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
    path = '/'.join(current[:current.index("nanopore-RNN") + 1])
    if os.path.exists(path):
        return path
    else:
        PathError("Path to directory does not exist!")


def get_project_file(local_path):
    """Get the path to an internal project file"""
    if local_path != "":
        if not local_path.startswith('/'):
            local_path = '/' + local_path
    path = os.path.join(project_folder() + local_path)
    if os.path.isfile(path):
        return path
    else:
        raise PathError("Path to file does not exist!")


def sum_to_one(vector, prob=False):
    """Make sure a vector sums to one, if not, create diffuse vector"""
    sum1 = sum(vector)
    assert sum != 0.0, "Vector of probabilities sum's to zero"
    if prob:
        vector = [n / sum1 for n in vector]
        sum1 = sum(vector)
    assert round(sum1, 10) == np.float(1.0), "Vector does not sum to one: {} != 1.0".format((round(sum1, 10) == 1.0))
    return vector


def add_field(np_struct_array, description):
    """Return a new array that is like the structured numpy array, but has additional fields.
    description looks like description=[('test', '<i8')]
    """
    if np_struct_array.dtype.fields is None:
        raise ValueError("Must be a structured numpy array")
    new = np.zeros(np_struct_array.shape, dtype=np_struct_array.dtype.descr + description)
    for name in np_struct_array.dtype.names:
        new[name] = np_struct_array[name]
    return new


def merge_two_dicts(dict1, dict2):
    """Given two dicts, merge them into a new dict as a shallow copy.
    source: https://stackoverflow.com/questions/38987/
    how-to-merge-two-python-dictionaries-in-a-single-expression"""
    assert type(dict1) is dict or type(dict1) is DotDict, "Both arguments must be dictionaries: type(arg1) = {}".format(
        type(dict1))
    assert type(dict2) is dict or type(dict2) is DotDict, "Both arguments must be dictionaries: type(arg2) = {}".format(
        type(dict2))
    final = dict1.copy()
    final.update(dict2)
    return final


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def upload_file_to_s3(bucket_path, file_path, name):
    """Upload a file or directory to an aws bucket"""
    # s3_conn = S3Connection(host='s3-us-west-1.amazonaws.com')

    conn = S3Connection(host='s3-us-west-2.amazonaws.com')
    test = conn.lookup(bucket_path)
    if test is None:
        print("There is no bucket with this name!", file=sys.stderr)
        return 0
    else:
        bucket = conn.get_bucket(bucket_path)

    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()

    k = Key(bucket)
    k.key = name
    k.set_contents_from_filename(file_path,
                                 cb=percent_cb, num_cb=10)
    sys.stdout.write('\n')


def upload_model(bucket, files, dir_name):
    """Upload all files to bucket"""
    for file1 in files:
        name = file1.split("/")[-1]
        key = os.path.join(dir_name, name)
        upload_file_to_s3(bucket, file1, key)


def test_aws_connection(bucket_path):
    """Test to see if there is a configured aws connection"""
    try:
        conn = S3Connection(host='s3-us-west-2.amazonaws.com')
        test = conn.lookup(bucket_path)
        if test is None:
            print("There is no bucket with this name!", file=sys.stderr)
            return False
    except:
        return False
    return True

def check_duplicate_characters(string):
    """make sure there are no duplicates in the alphabet"""
    results = collections.Counter(string)
    len_string = len(string)
    num_characters = len(results.items())
    assert len_string == num_characters, "String '{}' has repeat characters".format(string)
    return string


def load_json(path):
    """Load a json file and make sure that path exists"""
    path = os.path.abspath(path)
    assert os.path.isfile(path), "Json file does not exist: {}".format(path)
    with open(path) as json_file:
        args = json.load(json_file)
    return args


def save_json(dict1, path):
    """Save a python object as a json file"""
    path = os.path.abspath(path)
    with open(path, 'w') as outfile:
        json.dump(dict1, outfile)
    assert os.path.isfile(path)
    return path


def multiprocess_data(num_workers, target, arg_generator):
    """Create processes with number of workers a target function and argument generator"""
    work_queue = Manager().Queue()
    done_queue = Manager().Queue()
    jobs = []

    # start executing worker function using however many workers specified
    for args in arg_generator:
        # print(type(args))
        # print(dict1._id)

        work_queue.put(args)

    for _ in range(num_workers):
        process = Process(target=worker, args=(work_queue, done_queue, target))
        process.start()
        jobs.append(process)
        work_queue.put('STOP')

    for process in jobs:
        process.join()

    done_queue.put('STOP')
    return True


def worker(work_queue, done_queue, target_function):
    """Worker function to generate training data from a queue"""
    try:
        # create training data until there are no more files
        for args in iter(work_queue.get, 'STOP'):
            target_function(args)
            # catch errors
    except Exception as error:
        done_queue.put("%s failed with %s" % (current_process().name, error.message))
        print("%s failed with %s" % (current_process().name, error.message), file=sys.stderr)


def create_time_directory(output_dir):
    # create new output dir
    assert os.path.exists(output_dir)

    log_folder_path = os.path.join(output_dir,
                                   datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
    os.makedirs(log_folder_path)
    return log_folder_path


def save_config_file(config_data, log_folder_path, name="create_training_data.config.json"):
    """Save configuration dictionary as json specified log folder"""
    assert os.path.exists(log_folder_path), "Log folder path does not exist: {}".format(log_folder_path)
    assert type(config_data) is dict or type(config_data) is list or type(config_data) is DotDict
    config_path = os.path.join(log_folder_path, name)
    # save config file for training data
    save_json(config_data, config_path)
    return config_path


def create_log_file(home, old_log, new_path):
    """Create a log file which works on anyone's computer"""
    with open(old_log, 'r') as file1:
        with open(new_path, 'w+') as file2:
            for line in file1:
                line = line.rstrip().split('\t')
                # get file paths
                fast5 = os.path.join(home, line[0])
                tsv = os.path.join(home, line[1])
                assert os.path.exists(fast5), "Fast5 file does not exist: {}".format(fast5)
                assert os.path.exists(tsv), "Alignment file does not exist: {}".format(tsv)
                file2.write(fast5 + '\t' + tsv + '\n')
    return new_path


def tarball_files(tar_name, file_paths, output_dir='.', prefix=''):
    """
    Creates a tarball from a group of files
    :param str tar_name: Name of tarball
    :param list[str] file_paths: Absolute file paths to include in the tarball
    :param str output_dir: Output destination for tarball
    :param str prefix: Optional prefix for files in tarball
    """
    if tar_name.endswith(".tar.gz"):
        tar_name = tar_name
    else:
        tar_name = tar_name + ".tar.gz"
    tar_path = os.path.join(output_dir, tar_name)
    with tarfile.open(tar_path, 'w:gz') as f_out:
        for file_path in file_paths:
            if not file_path.startswith('/'):
                raise ValueError('Path provided is relative not absolute.')
            file_name = prefix + os.path.basename(file_path)
            f_out.add(file_path, arcname=file_name)
    return tar_path


def debug(verbose=False):
    """Method for setting log statements with verbose or not verbose"""
    assert type(verbose) is bool, "Verbose needs to be a boolean"
    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")
        log.info("This should not print.")
    return log

# def main():
#     """Test the methods"""
#     start = timer()
#     if test_aws_connection("neuralnet-accuracy"):
#         print("True")
#     # file_path = "/Users/andrewbailey/nanopore-RNN/test_files/create_training_files/07Jul-19-16h-48m"
#     # files = list_dir(file_path)
#     # tarball_files("test_tar", files)
#     # file_list = ["/Users/andrewbailey/nanopore-RNN/kmers.txt",
#     #              "/Users/andrewbailey/nanopore-RNN/logs/06Jun-29-11h-11m/checkpoint",
#     #              "/Users/andrewbailey/nanopore-RNN/logs/06Jun-29-11h-11m/my_test_model-3215.data-00000-of-00001"]
#     # bucket = "neuralnet-accuracy"
#     # dir_name = "06Jun-29-11h-30m-11.0%"
#     # upload_file_to_s3("nanotensor-data", "/Users/andrewbailey/nanopore-RNN/test_files/create_training_files/07Jul-19-16h-48m/create_training_data.config.json", "create_training_data.config.json")
#     # upload_model(bucket, file_list, dir_name)
#
#     stop = timer()
#     print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
