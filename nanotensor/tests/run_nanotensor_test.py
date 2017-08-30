#!/usr/bin/env python
"""
    Place unit tests for run_nanotensor.py
"""
########################################################################
# File: run_nanotensor_test.py
#  executable: run_nanotensor_test.py
# Purpose: run_nanotensor test functions
#
# Author: Andrew Bailey
# History: 08/08/2017 Created
########################################################################
from __future__ import print_function
import unittest
import os
import numpy as np
import threading
import shutil
import time
import tensorflow as tf
from nanotensor.run_nanotensor import main
import unittest


class RunNanotensorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(RunNanotensorTest, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-3])
        cls.data_dir = os.path.join(cls.HOME, "test_files/create_training_files/07Jul-20-11h-28m")
        cls.output_dir = os.path.join(cls.HOME, "test_files/test_models")
        cls.trained_model = os.path.join(cls.HOME, "test_files/test_models/08Aug-25-16h-33m/")
        cls.trained_model_path = os.path.join(cls.HOME, "test_files/test_models/08Aug-25-16h-33m/test_model-30")
        cls.inference_output = os.path.join(cls.HOME, "kmers.txt")

        cls.args_train = dict(
            testing_dir=cls.data_dir,
            training_dir=cls.data_dir,
            output_dir=cls.output_dir, model_name="test_model", num_gpu=0,
            num_threads=2, queue_size=10, training_iters=30, save_model=60, record_step=10, n_steps=30, batch_size=1,
            learning_rate=0.001, binary_cost=True, network=
            [dict(bias=5.0, type="blstm", name="blstm_layer_1", size=5)],
            train=True, inference=False, testing_accuracy=False, load_trained_model=False, alphabet="ATGC",
            kmer_len=5, inference_output=cls.inference_output, trained_model=cls.trained_model,
            trained_model_path=cls.trained_model_path,
            use_checkpoint=False, save_s3=False, trace_name="timeline.json", save_trace=False, profile=False,
            s3bucket="neuralnet-accuracy")

        cls.args_test = dict(
            testing_dir=cls.data_dir,
            training_dir=cls.data_dir,
            output_dir=cls.output_dir, model_name="test_model", num_gpu=0,
            num_threads=2, queue_size=10, training_iters=30, save_model=60, record_step=10, n_steps=30, batch_size=1,
            learning_rate=0.001, binary_cost=True, network=
            [dict(bias=5.0, type="blstm", name="blstm_layer1", size=5)],
            train=False, inference=False, testing_accuracy=True, load_trained_model=False, alphabet="ATGC",
            kmer_len=5, inference_output=cls.inference_output, trained_model=cls.trained_model,
            trained_model_path=cls.trained_model_path,
            use_checkpoint=False, save_s3=True, trace_name="timeline.json", save_trace=False, profile=False,
            s3bucket="neuralnet-accuracy")

    def test_main(self):
        with tf.variable_scope("tests"):
            output_dir = main(in_opts=self.args_train)
        shutil.rmtree(output_dir)
        # main(in_opts=self.args_test)


if __name__ == '__main__':
    unittest.main()
