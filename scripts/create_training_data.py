#!/usr/bin/env python
"""create_training_data is able to label nanopore data and create training files """
########################################################################
# File: create_training_data.py
#  executable: create_training_data.py
#
# Author: Andrew Bailey
# History: 05/28/17 Created
########################################################################

from __future__ import print_function
import sys
import os
from timeit import default_timer as timer
from datetime import datetime
import json
from multiprocessing import Process, current_process, Manager
import argparse
from utils import project_folder, merge_two_dicts
from data_preparation import TrainingData
from error import Usage



class CommandLine(object):
    '''
    Handle the command line, usage and help requests.

    attributes:
    myCommandLine.args is a dictionary which includes each of the available
    command line arguments as myCommandLine.args['option']

    methods:
    do_usage_and_die()
    prints usage and help and terminates with an error.

    '''

    def __init__(self, inOpts=None):
        '''CommandLine constructor.

        Implements a parser to interpret the command line argv string using
        argparse.
        '''
        # define program description, usage and epilog
        self.parser = argparse.ArgumentParser(description='This program \
        parses a log file with fast5 and tsv file paths. It then parses both and\
        creates npy files meant for training. Using a configuration file is \
        recommended', epilog="Dont forget to tar the files",
                                              usage='%(prog)s use "-h" for help')

        # create mutually exclusive argument for log file and json file
        self.exclusive = self.parser.add_mutually_exclusive_group(required=True)

        # optional arguments
        self.exclusive.add_argument('-l', '--log-file',
                                    help='path to log file with fast5 paths as \
                                    the first column and tsv \
                                    alinment files as the second column')


        self.exclusive.add_argument('-c', '--config',
                                    help='path to a json config file with all \
                                    required arguments defined')

        self.parser.add_argument('-f', '--file-prefix',
                                 help='prefix for the name of each numpy file',
                                 default="file")

        self.parser.add_argument('-o', '--output-dir',
                                 help='path to the output directory where \
                                 output folder will be placed', default=project_folder())

        self.parser.add_argument('-p', '--prob',
                                 help='boolean option to use probability labels', default=False)

        self.parser.add_argument('-k', '--kmer-len',
                                 help='label kmer length', default=5)

        self.parser.add_argument('-a', '--alphabet',
                                 help='alphabet used to create kmers. default="ATGC"',\
                                 default="ATGC")

        self.parser.add_argument('-n', '--nanonet-features',
                                 help='use nanonet feature definitions', default=True,
                                 dest='nanonet')

        self.parser.add_argument('-t', '--num-cpu', help='number of cpus available for\
                                 compute', default=1)

        # allow optional arguments not passed by the command line
        if inOpts is None:
            self.args = vars(self.parser.parse_args())
        else:
            self.args = vars(self.parser.parse_args(inOpts))


    def do_usage_and_die(self, message):
        ''' Print string and usage then return 2

        If a critical error is encountered, where it is suspected that the
        program is not being called with consistent parameters or data, this
        method will write out an error string (str), then terminate execution
        of the program.
        '''
        print(message, file=sys.stderr)
        self.parser.print_help(file=sys.stderr)
        return 2

def create_training_data(fast5_file, signalalign_file, prob=False, kmer_len=5, \
        alphabet="ATGC", nanonet=True, output_name="file", output_dir=project_folder()):
    """Create npy training files from aligment and a fast5 file"""
    data = TrainingData(fast5_file, signalalign_file, prob=prob, kmer_len=kmer_len,
                        alphabet=alphabet, nanonet=nanonet)
    data.save_training_file(output_name, output_dir=output_dir)
    print("FILE SAVED: {}".format(output_name+".npy"), file=sys.stderr)
    return True


def check_args(args):
    """Check arguments, save config file in new folder if correct"""
    # make sure output dir exists
    assert os.path.isdir(args["output_dir"])
    # create new output dir
    logfolder_path = os.path.join(args["output_dir"],
                                  datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
    os.makedirs(logfolder_path)
    config_path = os.path.join(logfolder_path, 'config.json')
    # save config file for training data
    with open(config_path, 'w') as outfile:
        json.dump(args, outfile, indent=4)
    args["output_dir"] = logfolder_path
    return args


def multiprocess_data(workers, target, arg_generator):
    """Create processes with number of workers a target function and argument generator"""
    work_queue = Manager().Queue()
    done_queue = Manager().Queue()
    jobs = []
    # start executing worker function using however many workers specified
    for args in arg_generator:
        work_queue.put(args)
    # pylint: disable=e0602
    for _ in xrange(workers):
        process = Process(target=worker, args=(work_queue, done_queue, target))
        process.start()
        jobs.append(process)
        work_queue.put('STOP')

    for proccess in jobs:
        proccess.join()

    done_queue.put('STOP')
    return True


def worker(work_queue, done_queue, target_function):
    """Worker function to generate training data from a queue"""
    try:
        # create training data until there are no more files
        for args in iter(work_queue.get, 'STOP'):
            target_function(**args)
        # catch errors
    # pylint: disable=W0703
    except Exception as error:
        # pylint: disable=no-member,E1102
        done_queue.put("%s failed with %s" % (current_process().name, error.message))
        print("%s failed with %s" % (current_process().name, error.message), file=sys.stderr)


def create_args(log_file, args):
    """Create generator of specific arguments for create_training_data"""
    prefix = args.pop("file_prefix")
    counter = 0
    with open(log_file, 'r') as log:
        for line in log:
            line = line.rstrip().split('\t')
            # get file paths
            fast5 = line[0]
            tsv = line[1]
            # define new file name
            name = str(prefix)+str(counter)
            paths = {"fast5_file": fast5, "signalalign_file": tsv, "output_name": name}
            # create final arguments and add to queue
            arguments = merge_two_dicts(paths, args)
            counter += 1
            yield arguments


def main(command_line=None):
    """Main docstring"""
    start = timer()

    # allow for a command line to be input into main
    if command_line is None:
        command_line = CommandLine()

    try:
        # get config and log files
        config = command_line.args["config"]

        # define arguments with config or arguments
        if config:
            config = os.path.abspath(config)
            assert os.path.isfile(config)
            print("Using config file {}".format(config), file=sys.stderr)
            with open(config) as json_file:
                args = json.load(json_file)
                args = check_args(args)
                log_file = args.pop("log_file")

        else:
            log_file = os.path.abspath(command_line.args.pop("log_file"))
            assert os.path.isfile(log_file)
            print("Using log file {}".format(log_file), file=sys.stderr)
            args = command_line.args
            args.pop("config")
            args = check_args(args)

        # define number of workers and create queues
        num_workers = args.pop("num_cpu")
        arg_generator = create_args(log_file, args)
        target = create_training_data
        multiprocess_data(num_workers, target, arg_generator)

        print("\n#  nanotensor - finished creating data set\n", file=sys.stderr)
        print("\n#  nanotensor - finished creating data set\n", file=sys.stdout)
        # check how long the whole program took

        stop = timer()
        print("Running Time = {} seconds".format(stop-start), file=sys.stderr)


    except Usage as err:
        command_line.do_usage_and_die(err.msg)

if __name__ == "__main__":
    main()
    raise SystemExit
