#!/usr/bin/env python
"""Test script for create_training_data.py """
########################################################################
# File  create_training_data_test.py
#  executable  create_training_data_test.py
#
# Author  Andrew Bailey
# History  Created 07/18/17
########################################################################

from __future__ import print_function

import os
import types
import unittest

from nanotensor.create_training_data import CommandLine, get_arguments, create_training_data_args, create_training_data, main
from nanotensor.utils import create_log_file


# noinspection PyTypeChecker
class CreateTrainingDataTest(unittest.TestCase):
    """Test the functions in network.py"""

    @classmethod
    def setUpClass(cls):
        super(CreateTrainingDataTest, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-3])
        cls.TEST_DIR = os.path.join(cls.HOME, "test_files/")
        cls.old_log_file = os.path.join(cls.HOME, "test_files/test_log_files/canonical.log.txt")
        new_log_path = os.path.join(cls.HOME, "test_files/test_log_files/real.canonical.log.txt")
        cls.log_file = create_log_file(cls.HOME, cls.old_log_file, new_log_path)
        cls.args = dict(nanonet=True, alphabet="ATGC", file="canonical", num_cpu=5, kmer_len=5, output_dir=cls.HOME,
                        strand_name="template", prob=False, deepnano=False, log_file=cls.log_file, verbose=False,
                        cutoff=0.4, debug=False, save2s3=True, tar=True)
        with open(cls.log_file, 'r') as log:
            line = log.readline()
            line = line.rstrip().split('\t')
            # get file paths
            cls.fast5 = os.path.abspath(line[0])
            cls.tsv = os.path.abspath(line[1])

    def test_check_args(self):
        """Test_check_args in CommandLine class"""
        commandline = CommandLine(in_opts=["--config", "path"])
        bad_args = dict(output_dir="/nanopore-RNN/test_files/")
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file="test")
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file=self.log_file, prob="test")
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file=self.log_file, prob=bool(), kmer_len="test")
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file=self.log_file, prob=bool(), kmer_len=int(), alphabet=0)
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file=self.log_file, prob=bool(), kmer_len=int(), alphabet=str(),
                        nanonet="test")
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file=self.log_file, prob=bool(), kmer_len=int(), alphabet=0,
                        nanonet=bool(),
                        num_cpu="test")
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file=self.log_file, prob=bool(), kmer_len=int(), alphabet=0,
                        nanonet=bool(),
                        num_cpu=int(), deepnano="test")
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file=self.log_file, prob=bool(), kmer_len=int(), alphabet=0,
                        nanonet=bool(),
                        num_cpu=int(), deepnano=bool(), file=0)
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file=self.log_file, prob=bool(), kmer_len=int(), alphabet=0,
                        nanonet=bool(),
                        num_cpu=int(), deepnano=bool(), file=str(), verbose="test")
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        bad_args = dict(output_dir=self.HOME, log_file=self.log_file, prob=bool(), kmer_len=int(), alphabet=0,
                        nanonet=bool(),
                        num_cpu=int(), deepnano=bool(), file=str(), verbose=bool(), debug="test")
        self.assertRaises(AssertionError, commandline.check_args, bad_args)
        new_args = commandline.check_args(self.args)
        self.assertEqual(self.args["output_dir"], new_args.output_dir)
        self.assertEqual(self.args["log_file"], new_args.log_file)
        self.assertEqual(self.args["prob"], new_args.prob)
        self.assertEqual(self.args["kmer_len"], new_args.kmer_len)
        self.assertEqual(self.args["alphabet"], new_args.alphabet)
        self.assertEqual(self.args["nanonet"], new_args.nanonet)
        self.assertEqual(self.args["num_cpu"], new_args.num_cpu)
        self.assertEqual(self.args["deepnano"], new_args.deepnano)
        self.assertEqual(self.args["file"], new_args.file)
        self.assertEqual(self.args["verbose"], new_args.verbose)
        self.assertEqual(self.args["debug"], new_args.debug)

    def test_get_arguments(self):
        """Test get_arguments function"""
        args = ["--config", "path/to/config"]
        commandline = CommandLine(in_opts=args)
        self.assertRaises(AssertionError, get_arguments, commandline)
        args = ["--log-file", "/Users/andrewbailey/data/log-file.1",
                "--kmer-len", "5",
                "--output-dir", self.TEST_DIR,
                "--strand-name", "template",
                "--deepnano",
                "--file-prefix", "canonical",
                "--alphabet", "ATGC"]
        commandline = CommandLine(in_opts=args)
        self.assertEqual(commandline.args, get_arguments(commandline))

    def test_command_line_argparse(self):
        """Test_command_line_argparse"""
        args = ["--log-file", "/Users/andrewbailey/data/log-file.1"]
        self.assertRaises(SystemExit, CommandLine, in_opts=args)
        args.extend(["--kmer-len", "5",
                     "--output-dir", self.TEST_DIR,
                     "--strand-name", "template",
                     "--deepnano",
                     "--file-prefix", "canonical",
                     "--alphabet", "ATGC"])
        self.assertIsInstance(CommandLine(in_opts=args), CommandLine)
        args.extend(["--nanonet"])
        self.assertRaises(SystemExit, CommandLine, in_opts=args)
        args = ["--config", "path/to/config"]
        self.assertIsInstance(CommandLine(in_opts=args), CommandLine)

    def test_create_training_data_args(self):
        """Test create_training_data_args"""
        prefix = 1
        log_file = self.args["log_file"]
        arg_generator = create_training_data_args(log_file, prefix, self.args)
        self.assertRaises(AssertionError, next, arg_generator)
        prefix = "file"
        log_file = "test"
        arg_generator = create_training_data_args(log_file, prefix, self.args)
        self.assertRaises(AssertionError, next, arg_generator)
        prefix = "file"
        log_file = self.old_log_file
        arg_generator = create_training_data_args(log_file, prefix, self.args)
        self.assertRaises(StopIteration, next, arg_generator)
        prefix = "file"
        log_file = self.old_log_file
        arg_generator = create_training_data_args(log_file, prefix, self.args, exception=None)
        self.assertRaises(AssertionError, next, arg_generator)
        # passes
        prefix = "file"
        log_file = self.args["log_file"]
        arg_generator = create_training_data_args(log_file, prefix, self.args)
        self.assertIsInstance(arg_generator, types.GeneratorType)
        args1 = next(arg_generator)
        self.assertIsInstance(args1, dict)

    def test_create_training_data(self):
        """test_create_training_data"""
        # fails
        args = dict(cutoff=0.4, nanonet=False, verbose=True, strand_name='template', deepnano=False, debug=False,
                    file='canonical', num_cpu=5, alphabet='ATGC', kmer_len=5, signalalign_file=self.tsv,
                    output_dir=self.TEST_DIR, forward=True, log_file=self.log_file,
                    output_name='file1', prob=False, fast5_file=self.fast5)
        self.assertRaises(AssertionError, create_training_data, args)
        args = dict(cutoff=0.4, nanonet=False, verbose=True, strand_name='template', deepnano=True, debug=False,
                    file='canonical', num_cpu=5, alphabet='ATGC', kmer_len=5, signalalign_file=self.tsv,
                    output_dir=self.TEST_DIR, forward=True, log_file=self.log_file,
                    output_name='file1', prob=False, fast5_file=self.fast5)
        self.assertRaises(AssertionError, create_training_data, args)
        args = dict(cutoff=0.4, nanonet=False, verbose=True, strand_name='template', deepnano=True, debug=False,
                    file='canonical', num_cpu=5, alphabet='ATGC', kmer_len=4, signalalign_file=self.tsv,
                    output_dir=self.TEST_DIR, forward=True, log_file=self.log_file,
                    output_name='file1', prob=True, fast5_file=self.fast5)
        self.assertRaises(AssertionError, create_training_data, args)

        # passes
        args = dict(cutoff=0.4, nanonet=False, verbose=False, strand_name='template', deepnano=True, debug=False,
                    file='canonical', num_cpu=5, alphabet='ATGC', kmer_len=2, signalalign_file=self.tsv,
                    output_dir=self.TEST_DIR, forward=True, log_file=self.log_file,
                    output_name='deepnano1', prob=False, fast5_file=self.fast5)
        output_file_path = create_training_data(args)
        self.assertTrue(os.path.exists(output_file_path))
        os.remove(output_file_path)

        args = dict(cutoff=0.4, nanonet=True, verbose=False, strand_name='template', deepnano=False, debug=False,
                    file='canonical', num_cpu=5, alphabet='ATGC', kmer_len=2, signalalign_file=self.tsv,
                    output_dir=self.TEST_DIR, forward=True, log_file=self.log_file,
                    output_name='prob1', prob=True, fast5_file=self.fast5)
        output_file_path = create_training_data(args)
        self.assertTrue(os.path.exists(output_file_path))
        os.remove(output_file_path)

        args = dict(cutoff=0.4, nanonet=True, verbose=False, strand_name='template', deepnano=False, debug=False,
                    file='canonical', num_cpu=5, alphabet='ATGC', kmer_len=5, signalalign_file=self.tsv,
                    output_dir=self.TEST_DIR, forward=True, log_file=self.log_file,
                    output_name='nanonet1', prob=False, fast5_file=self.fast5)
        output_file_path = create_training_data(args)
        self.assertTrue(os.path.exists(output_file_path))
        os.remove(output_file_path)

    # def test_main(self):
    #     """Test main function of create_training_data"""
    #     main()

if __name__ == '__main__':
    unittest.main()
