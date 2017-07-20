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

import argparse
import os
import sys
from timeit import default_timer as timer

from nanotensor.data_preparation import TrainingData
from nanotensor.error import Usage
from nanotensor.utils import merge_two_dicts, load_json, DotDict, multiprocess_data, create_time_directory, \
    save_config_file


class CommandLine(object):
    """
    Handle the command line, usage and help requests.

    attributes:
    myCommandLine.args is a dictionary which includes each of the available
    command line arguments as myCommandLine.args['option']

    methods:
    do_usage_and_die()
    prints usage and help and terminates with an error.

    """

    def __init__(self, in_opts=None):
        """CommandLine constructor.

        Implements a parser to interpret the command line argv string using
        argparse.
        """
        # define program description, usage and epilog
        self.parser = argparse.ArgumentParser(description='This program \
        parses a log file with fast5 and tsv file paths. It then parses both files and\
        creates npy files meant for training. Using a configuration file is \
        recommended', epilog="Don't forget to tar the files",
                                              usage='%(prog)s use "-h" for help')

        # create mutually exclusive argument for log file and json file
        self.parser.add_argument('-v', '--verbose',
                                 help='verbose output', action='store_true')

        self.parser.add_argument('--debug',
                                 help="Output error messages and use process instead of multiprocessing",
                                 action='store_true')

        self.parser.add_argument('-t', '--num-cpu', help='number of CPUs available for compute',
                                 default=1)

        self.parser.add_argument('--cutoff',
                                 help='cutoff probability for assignment when using deepnano labeling',
                                 default=0.2)

        self.parser.add_argument('-p', '--prob',
                                 help='Boolean option to use probability labels. Only used for nanonet-features',
                                 action='store_true')

        config_group = self.parser.add_argument_group('Recommended Usage: Config File')

        config_group.add_argument('-c', '--config',
                                  help='path to a json config file with all \
                                    required arguments defined')

        # required if config is not selected
        def config_not_selected():
            """If log file is selected return true for required. Else return false"""
            if in_opts:
                return not ('-c' in in_opts or '--config' in in_opts)
            else:
                return not ('-c' in sys.argv or '--config' in sys.argv)

        no_config_group = self.parser.add_argument_group('Command Line Usage', 'Required arguments if configuration\
         file is not selected')

        no_config_group.add_argument('-l', '--log-file',
                                     help='path to log file with fast5 paths as \
                                    the first column and tsv \
                                    alignment files as the second column', required=config_not_selected())

        no_config_group.add_argument('-f', '--file-prefix',
                                     help='prefix for the name of each numpy file',
                                     default="file", required=config_not_selected())

        no_config_group.add_argument('-o', '--output-dir',
                                     help='path to the output directory where \
                                     output folder will be placed', required=config_not_selected())

        no_config_group.add_argument('-k', '--kmer-len',
                                     help='label kmer length', required=config_not_selected())

        no_config_group.add_argument('-a', '--alphabet',
                                     help='alphabet used to create kmers. default="ATGC"',
                                     required=config_not_selected())

        no_config_group.add_argument('-s', '--strand-name',
                                     help='"template" or "complement". default="template"', default="template",
                                     required=config_not_selected())

        exclusive_labeling = self.parser.add_mutually_exclusive_group(required=config_not_selected())

        exclusive_labeling.add_argument('-n', '--nanonet',
                                        help='use nanonet feature and and label definitions', action="store_true",
                                        dest='nanonet')

        exclusive_labeling.add_argument('-d', '--deepnano',
                                        help='use deepnano feature and and label definitions', action="store_true",
                                        dest='deepnano')

        # allow optional arguments not passed by the command line
        if in_opts is None:
            self.args = vars(self.parser.parse_args())
        else:
            self.args = vars(self.parser.parse_args(in_opts))

    def do_usage_and_die(self, message):
        """ Print string and usage then return 2

        If a critical error is encountered, where it is suspected that the
        program is not being called with consistent parameters or data, this
        method will write out an error string (str), then terminate execution
        of the program.
        """
        print(message, file=sys.stderr)
        self.parser.print_help(file=sys.stderr)
        return 2

    @staticmethod
    def check_args(args):
        """Check arguments, save config file in new folder if correct"""
        # make sure output dir exists
        args = DotDict(args)

        assert os.path.isdir(args.output_dir), "Output directory does not exist: {}".format(args.output_dir)
        assert os.path.isfile(args.log_file), "Log file does not exist: {}".format(args.log_file)
        assert type(args.prob) is bool, "Prob must be boolean: {}".format(args.prob)
        assert type(args.kmer_len) is int, "kmer-len must be integer: {}".format(args.kmer_len)
        assert type(args.alphabet) is unicode or str, "alphabet must be string: {}".format(args.alphabet)
        assert type(args.nanonet) is bool, "nanonet-features option must be boolean: {}".format(args.nanonet)
        assert type(args.num_cpu) is int, "num-cpu must be integer: {}".format(args.num_cpu)
        assert type(args.deepnano) is bool, "deepnano must be integer: {}".format(args.deepnano)
        assert type(args.file) is unicode or str, "file-prefix must be string: {}".format(args.file)
        assert type(args.verbose) is bool, "verbose option must be a boolean: {}".format(args.verbose)
        assert type(args.debug) is bool, "debug option must be a boolean: {}".format(args.debug)

        return args


def create_training_data(args):
    """Create npy training files from alignment and a fast5 file"""
    args = DotDict(args)
    data = TrainingData(args.fast5_file, args.signalalign_file, args.strand_name, prob=args.prob,
                        kmer_len=args.kmer_len, alphabet=args.alphabet, nanonet=args.nanonet, deepnano=args.deepnano,
                        forward=args.forward, cutoff=args.cutoff)
    output_file_path = data.save_training_file(args.output_name, output_dir=args.output_dir)
    if args.verbose:
        print("FILE SAVED: {}".format(args.output_name + ".npy"), file=sys.stderr)
    return output_file_path


def create_training_data_args(log_file, prefix, args, exception=AssertionError):
    """Create generator of specific arguments for create_training_data"""
    assert os.path.exists(log_file), "Log file does not exist: {}".format(log_file)
    assert type(prefix) is str or type(prefix) is unicode
    counter = 0
    with open(log_file, 'r') as log:
        for line in log:
            try:
                line = line.rstrip().split('\t')
                # get file paths
                fast5 = os.path.abspath(line[0])
                tsv = os.path.abspath(line[1])
                assert os.path.exists(fast5), "Fast5 file does not exist: {}".format(fast5)
                assert os.path.exists(tsv), "alignment file does not exist: {}".format(tsv)
                # define new file name
                name = str(prefix) + str(counter)
                if "forward" in tsv:
                    paths = {"fast5_file": fast5, "signalalign_file": tsv, "output_name": name, "forward": True}
                elif "backward" in tsv:
                    paths = {"fast5_file": fast5, "signalalign_file": tsv, "output_name": name, "forward": False}
                else:
                    raise Usage("TSV does not have forward or backward in it's name")
                # create final arguments and add to queue
                arguments = merge_two_dicts(paths, args)
                counter += 1
                yield arguments
            except exception:
                continue


def get_arguments(command_line):
    """Get arguments from config file or from the command line"""
    config = command_line.args["config"]
    if config:
        config = os.path.abspath(config)
        assert os.path.isfile(config), "There is no configuration file: {}".format(config)
        print("Using config file {}".format(config), file=sys.stderr)
        args = load_json(config)
    else:
        args = command_line.args

    return args


def main(command_line=None):
    """Main docstring"""
    start = timer()

    # allow for a command line to be input into main
    if command_line is None:
        command_line = CommandLine()

    try:
        args = get_arguments(command_line)
        args = command_line.check_args(args)
        log_dir_path = create_time_directory(args.output_dir)
        save_config_file(args, log_dir_path)
        args.output_dir = log_dir_path

        log_file = args.log_file
        print("Using log file {}".format(log_file), file=sys.stderr)
        # define number of workers and create queues
        print(type(args.prefix))
        arg_generator = create_training_data_args(log_file, args.file_prefix, args)
        if args.debug:
            for arg in arg_generator:
                create_training_data(arg)
        else:
            num_workers = args.num_cpu
            target = create_training_data
            multiprocess_data(num_workers, target, arg_generator)

        print("\n#  nanotensor - finished creating data set\n", file=sys.stderr)
        print("\n#  nanotensor - finished creating data set\n", file=sys.stderr)
        # check how long the whole program took

        stop = timer()
        print("Running Time = {} seconds".format(stop - start), file=sys.stderr)

    except Usage as err:
        command_line.do_usage_and_die(err.msg)


if __name__ == "__main__":
    main()
    raise SystemExit
