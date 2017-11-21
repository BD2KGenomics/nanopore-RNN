#!/usr/bin/env python
"""nanotensor is able to label nanopore data, create training files and
then use's tensorflow to train a mulit layer BLSTM-RNN"""
########################################################################
# File: nanotensor.py
#  executable: nanotensor.py
#
# Author: Andrew Bailey
# History: 05/28/17 Created
########################################################################

from __future__ import print_function
import logging as log
import sys
import os
import re
import subprocess
from timeit import default_timer as timer
import json
import argparse
from datetime import datetime
import numpy as np
import time
from itertools import izip
from nanotensor.utils import project_folder, list_dir, DotDict, upload_model, load_json, save_config_file, \
    test_aws_connection, merge_two_dicts, time_it, debug
from nanotensor.error import Usage
from nanotensor.queue import FullSignalSequence, MotifSequence, NumpyEventData
from nanotensor.network import CtcLoss, CrossEntropy
import tensorflow as tf
from tensorflow.python.client import timeline


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
        self.parser = argparse.ArgumentParser(description='This program takes two config files one with the network '
                                                          'configurations and one with the paths to data directories.',
                                              epilog="Dont forget to tar the files",
                                              usage='%(prog)s use "-h" for help')

        # create mutually exclusive argument for log file and json file
        self.exclusive = self.parser.add_mutually_exclusive_group(required=True)

        # optional arguments
        self.exclusive.add_argument('-c', '--config',
                                    help='path to a json config file with all \
                                    required arguments defined')

        # self.parser.add_argument('-v', '--verbose', action='store_true',
        #                          help='print out more information')
        #
        # allow optional arguments not passed by the command line
        if in_opts is None:
            self.args = vars(self.parser.parse_args())
        elif type(in_opts) is list:
            self.args = vars(self.parser.parse_args(in_opts))
        else:
            self.args = in_opts

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
    def check_args(args, config):
        """Check arguments, save config file in new folder if correct"""
        # make sure output dir exists
        assert os.path.isdir(args["CreateDataset"]["training_dir"]), \
            "{} does not exist".format(args["CreateDataset"]["training_dir"])
        assert os.path.isdir(args["CreateDataset"]["validation_dir"]), \
            "{} does not exist".format(args["CreateDataset"]["validation_dir"])
        assert int(args["train"]) + int(args["inference"]) + int(args["test"]) == 1, "Must select either train, " \
                                                                                     "inference or test."
        # create new output dir
        if args["train"] and not args["load_trained_model"]:
            logfolder_path = os.path.join(args["output_dir"],
                                          datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
            try:
                os.makedirs(logfolder_path)
            except OSError:
                pass
            config_path = save_config_file(args, logfolder_path, name=os.path.basename(config))
            # define log file dir
            args["output_dir"] = logfolder_path
            args["config_path"] = config_path
        args = DotDict(args)
        args.CreateDataset = DotDict(args.CreateDataset)
        args.CreateDataset.dataset_args = DotDict(args.CreateDataset.dataset_args)
        args.BuildGraph = DotDict(args.BuildGraph)
        args.BuildGraph.graph_args = DotDict(args.BuildGraph.graph_args)
        return args


class RunTensorflow(object):
    """Class for running a tensorflow model."""

    def __init__(self, args):
        self.args = args
        # get correct data input pipeline
        dataset_options = {"FullSignalSequence": FullSignalSequence, "MotifSequence": MotifSequence,
                           "NumpyEventData": NumpyEventData}
        self.Dataset = dataset_options[self.args.CreateDataset.dataset]
        # pick graph
        graph_options = {"CtcLoss": CtcLoss, "CrossEntropy": CrossEntropy}
        self.Graph = graph_options[self.args.BuildGraph.graph]

        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        if self.args.train:
            self.training_files = list_dir(self.args.CreateDataset.training_dir)
            self.validation_files = list_dir(self.args.CreateDataset.validation_dir)
            self.training = "CreateDataset"
            self.validation = "CreateDataset"
            self.train_op = "train_op"
            self.training_model = "BuildGraph"
            self.validation_model = "BuildGraph"
            self.cost_diff_summary = None
            self.tower_grads = []
            learning_rate = tf.train.exponential_decay(self.args.learning_rate, self.global_step,
                                                       100000, 0.96, staircase=True)
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        elif self.args.test:
            self.test_files = list_dir(self.args.CreateDataset.test_dir)
            self.testing = "CreateDataset"
            self.testing_model = "BuildGraph"
            self.testing_opts = []

        elif self.args.inference:
            self.inference_files = list_dir(self.args.CreateDataset.inference_dir)
            self.inference = "CreateDataset"
            self.inference_model = "BuildGraph"
            self.inference_opts = []
        # load data
        log.info("Data Loading Started")
        speed = time_it(self.load_data)
        log.info("Data Loading took {} seconds to complete".format(speed))

        # look for GPU's
        self.gpu_indexes = test_for_nvidia_gpu(self.args.num_gpu)

        # initialize model
        log.info("Initializing Model")
        speed = time_it(self.initialize_model)
        log.info("Model Initialization took {} seconds to complete".format(speed))

        self.summaries = tf.summary.merge_all()
        self.start = datetime.now()
        if self.args.use_checkpoint:
            self.model_path = tf.train.latest_checkpoint(self.args.trained_model)
        else:
            self.model_path = self.args.trained_model_path

    def initialize_model(self):
        """Initialize the model with multi-gpu options"""
        with tf.variable_scope(self.args.model_name, reuse=tf.AUTO_REUSE):
            # if gpus are availiable
            if self.gpu_indexes:
                log.info("Using GPU's {}".format(self.gpu_indexes))
                for count, i in enumerate(self.gpu_indexes):
                    with tf.device('/gpu:%d' % i):
                        # create graph
                        graph_type = self.create_model(validation=(count == 0))
                        log.info("Created {} graph on gpu:{}".format(graph_type, i))
                with tf.device('/cpu:0'):
                    # Create validation graph on cpu
                    self.validation_model = self.Graph(network=self.args.network, dataset=self.validation,
                                                       summary_name="Validation")
            else:
                log.info("No GPU's available, using CPU for computation")
                graph_type = self.create_model(validation=True)
                log.info("Created {} graph on CPU".format(graph_type))

            if self.args.training:
                # update graph with new weights after backprop
                if self.gpu_indexes:
                    grads = average_gradients(self.tower_grads)
                else:
                    grads = self.tower_grads[0]
                with tf.variable_scope("apply_gradients", reuse=tf.AUTO_REUSE):
                    self.train_op = self.opt.apply_gradients(grads, global_step=self.global_step)

                self.cost_diff_summary = tf.summary.scalar('cost_diff',
                                                           (self.training_model.cost - self.validation_model.cost))

                # variable_averages = tf.train.ExponentialMovingAverage(
                # cifar10.MOVING_AVERAGE_DECAY, global_step)
                # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    def create_model(self, validation=False):
        """Create inference, test or training model"""
        if self.args.inference:
            self.inference_model = self.Graph(network=self.args.network, dataset=self.inference,
                                              summary_name="Inference")
            self.inference_opts.append(self.inference_model.prediction)
            graph_type = "Inference"
        elif self.args.test:
            self.testing_model = self.Graph(network=self.args.network, dataset=self.testing,
                                            summary_name="Testing")
            self.testing_opts.append(self.testing_model.accuracy)
            self.testing_opts.append(self.testing_model.batch_size)
            graph_type = "Testing"
        elif self.args.train:
            self.training_model = self.Graph(network=self.args.network, dataset=self.training, summary_name="Training")
            # compute gradients
            gradients = self.opt.compute_gradients(self.training_model.cost)
            self.tower_grads.append(gradients)
            graph_type = "Training"
            if validation:
                with tf.device('/cpu:0'):
                    self.validation_model = self.Graph(network=self.args.network, dataset=self.validation,
                                                       summary_name="Validation")
                    graph_type = "Training and Validation"

        return graph_type

    def load_data(self):
        """Create training and testing queues from training and testing files"""
        if self.args.train:
            train_dataset_args = merge_two_dicts(self.args.CreateDataset.dataset_args,
                                                 {"mode": 0, "verbose": self.args.verbose})
            self.training = self.Dataset(self.training_files, **train_dataset_args)
            self.validation = self.Dataset(self.validation_files, **train_dataset_args)
        if self.args.test:
            test_dataset_args = merge_two_dicts(self.args.CreateDataset.dataset_args,
                                                {"mode": 1, "verbose": self.args.verbose})
            self.testing = self.Dataset(self.test_files, **test_dataset_args)
        if self.args.inference:
            inference_dataset_args = merge_two_dicts(self.args.CreateDataset.dataset_args,
                                                     {"mode": 2, "verbose": self.args.verbose})
            self.inference = self.Dataset(self.inference_files, **inference_dataset_args)
        return 0

    # @profile
    def train_model(self, intra_op_parallelism_threads=8, log_device_placement=True, allow_soft_placement=True):
        """Run training steps
        use `dmesg` to get error message if training is killed
        """
        config = tf.ConfigProto(log_device_placement=log_device_placement,
                                intra_op_parallelism_threads=intra_op_parallelism_threads,
                                allow_soft_placement=allow_soft_placement)
        # shows gpu memory usage
        config.gpu_options.allow_growth = True
        run_metadata = tf.RunMetadata()

        # with tf.train.MonitoredTrainingSession(config=config) as sess:
        with tf.Session(config=config) as sess:
            # create logs
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            step = 0
            if self.args.load_trained_model:
                writer = tf.summary.FileWriter(self.args.trained_model, sess.graph)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.restore(sess, self.model_path)
            else:
                # initialize
                writer = tf.summary.FileWriter(self.args.output_dir, sess.graph)
                sess.run(tf.global_variables_initializer())
                self.model_path = os.path.join(self.args.output_dir, self.args.model_name)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.save(sess, self.model_path,
                           global_step=self.global_step)
            log.info("Model Path: {}".format(self.model_path))

            # load data into placeholders
            sess.run(self.training.iterator.initializer,
                     feed_dict={self.training.place_X: self.training.data.input,
                                self.training.place_Y: self.training.data.label,
                                self.training.place_Seq: self.training.data.seq_len})
            sess.run(self.validation.iterator.initializer,
                     feed_dict={self.validation.place_X: self.validation.data.input,
                                self.validation.place_Y: self.validation.data.label,
                                self.validation.place_Seq: self.validation.data.seq_len})

            # Keep training until reach max iterations
            print("Training Has Started!", file=sys.stderr)
            try:
                # TODO have a better termination method
                while step < self.args.training_iters:
                    for _ in range(self.args.record_step):
                        # Run optimization training step
                        _ = sess.run([self.train_op])
                        step += 1
                    # get training accuracy stats
                    self.report_summary_stats(sess, writer)
                    # if it has been enough time save model and print training stats
                    if self.test_time() and self.args.profile:
                        # very expensive profiling steps to add time statistics
                        self.profile_training(sess, writer, run_metadata, run_options)

                    saver.save(sess, self.model_path,
                               global_step=self.global_step, write_meta_graph=False)
            except tf.errors.OutOfRangeError:
                print("End of dataset")  # ==> "End of dataset"
        sess.close()
        writer.close()

        print("Training Finished!")

    # TODO refactor to work correctly
    def run_model(self):
        """Run a model from a saved model path"""
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            saver.restore(sess, self.model_path)
            for file_path in self.inference_files:
                if file_path.endswith(".signal"):
                    self.inference.file_path = file_path
                    sess.run(self.inference.iterator.initializer)
                    prediction_list = []
                    try:
                        while True:
                            evaluate_pred = sess.run([self.inference_opts])
                            prediction_list.extend(np.asarray(np.vstack(evaluate_pred[0])))
                    except tf.errors.OutOfRangeError:
                        print("New File")
                        self.inference.process_output(np.asarray(prediction_list), file_path)
                        continue
            print("Finished Inference")

    def test_model(self, intra_op_parallelism_threads=8, log_device_placement=False):
        """Get testing accuracy and save model along with configuration file on s3"""
        if self.args.save_s3:
            aws_test = test_aws_connection(self.args.s3bucket)
            assert aws_test is True, "Selected save to s3 option but cannot connect to specified bucket"
        # testing_accuracy = tf.group(self.testing_opts)
        with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement,
                                              intra_op_parallelism_threads=intra_op_parallelism_threads)) as sess:
            # restore model
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)
            acc_sum = 0
            # Keep training until reach max iterations
            step = 0
            sess.run(self.testing.iterator.initializer,
                     feed_dict={self.testing.place_X: self.testing.data.input,
                                self.testing.place_Seq: self.testing.data.seq_len,
                                self.testing.place_Y: self.testing.data.label})

            try:
                while True:
                    acc = sess.run(self.testing_opts)
                    a = iter(acc)
                    zips = izip(a, a)
                    for acc, batch_size in zips:
                        acc_sum += acc * batch_size
                        step += batch_size
                        log.info("Iter " + str(step) + ", Testing Accuracy= " + "{:.5f}".format(acc))
            except tf.errors.OutOfRangeError:
                print("End of dataset", file=sys.stderr)

            sess.close()
            # Calculate average accuracy
            print("Average Accuracy = {:.3f}".format(acc_sum / step * 100), file=sys.stderr)

        if self.args.save:
            file_list = self.get_model_files(args.config_path)
            final_acc = str(acc_sum / step * 100)[:5] + "%" + datetime.now().strftime("%m%b-%d-%Hh-%Mm")
            # print(file_list)
            print("Uploading Model to neuralnet-accuracy s3 Bucket", file=sys.stderr)
            upload_model("neuralnet-accuracy", file_list, final_acc)

    def profile_training(self, sess, writer, run_metadata, run_options):
        """Expensive profile step so we can track speed of operations of the model"""
        _, summary, global_step = sess.run(
            [self.train_op, self.summaries, self.global_step],
            run_metadata=run_metadata, options=run_options)
        # add summary statistics
        writer.add_summary(summary, global_step)
        writer.add_run_metadata(run_metadata, "step{}_train".format(global_step))
        if self.args.save_trace:
            self.chrome_trace(run_metadata, self.args.trace_name)

    @staticmethod
    def chrome_trace(metadata_proto, f_name):
        """Save a chrome trace json file.
        To view json vile go to - chrome://tracing/
        """
        time_line = timeline.Timeline(metadata_proto.step_stats)
        ctf = time_line.generate_chrome_trace_format()
        with open(f_name, 'w') as file1:
            file1.write(ctf)

    def report_summary_stats(self, sess, writer):
        # Calculate batch loss and accuracy for training
        sum_stats, global_step, val_acc, val_cost, train_acc, train_cost = sess.run([self.summaries, self.global_step,
                                                                                     self.validation_model.accuracy,
                                                                                     self.validation_model.cost,
                                                                                     self.training_model.accuracy,
                                                                                     self.training_model.cost])

        # add summary stats to tensorboard
        writer.add_summary(sum_stats, global_step)
        # print summary stats
        print("Iter " + str(global_step) + ", Training Cost= " +
              "{:.6f}".format(train_cost) + ", Validation Cost= " +
              "{:.5f}".format(val_cost), file=sys.stderr)
        print("Iter " + str(global_step) + ", Training Accuracy= " +
              "{:.6f}".format(train_acc) + ",  Validation Accuracy= " +
              "{:.5f}".format(val_acc), file=sys.stderr)

    def test_time(self):
        """Return true if it is time to save the model"""
        delta = (datetime.now() - self.start).total_seconds()
        if delta > self.args.save_model:
            self.start = datetime.now()
            return True
        return False

    def get_model_files(self, *files):
        """Collect neccessary model files for upload"""
        file_list = [self.model_path + ".data-00000-of-00001", self.model_path + ".index"]
        for file1 in files:
            file_list.append(file1)
        return file_list


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    # # print(tower_grads)
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # print(grad_and_vars)
        grads = []
        for g, v in grad_and_vars:
            # print(g)
            # print(v)
            # print("Another gradient and variable")
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        #
        # # Keep in mind that the Variables are redundant because they are shared
        # # across towers. So .. we will just return the first tower's pointer to
        # # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def test_for_nvidia_gpu(num_gpu):
    assert type(num_gpu) is int, "num_gpu option must be integer"
    if num_gpu == 0:
        return False
    else:
        try:
            utilization = re.findall(r"Utilization.*?Gpu.*?(\d+).*?Memory.*?(\d+)",
                                     subprocess.check_output(["nvidia-smi", "-q"]),
                                     flags=re.MULTILINE | re.DOTALL)
            indices = [i for i, x in enumerate(utilization) if x == ('0', '0')]
            assert len(indices) >= num_gpu, "Only {0} GPU's are available, change num_gpu parameter to {0}".format(
                len(indices))
            return indices[:num_gpu]
        except OSError:
            log.info("No GPU's found. Using CPU.")
            return False


def main():
    """Main docstring"""
    start = timer()

    # get arguments from command line or config file
    command_line = CommandLine()
    # get config and log files
    config_path = command_line.args["config"]
    print("Using config file {}".format(config_path), file=sys.stderr)
    # define arguments with config or arguments
    args = load_json(config_path)
    try:
        # check arguments and define
        log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
        args = command_line.check_args(args, config_path)
        config = args.config_path
        debug(verbose=args.verbose)
        # Parameters
        if args.train:
            train = RunTensorflow(args)
            train.train_model(intra_op_parallelism_threads=8, log_device_placement=False, allow_soft_placement=True)
            print("\n#  nanotensor - finished training \n", file=sys.stderr)
            print("\n#  nanotensor - finished training \n", file=sys.stderr)

        elif args.inference:
            infer = RunTensorflow(args)
            infer.run_model()
            print("\n#  nanotensor - finished inference \n", file=sys.stderr)
            print("\n#  nanotensor - finished inference \n", file=sys.stderr)

        elif args.test:
            test = RunTensorflow(args)
            test.test_model(config)
            print("\n#  nanotensor - finished testing accuracy \n", file=sys.stderr)
            print("\n#  nanotensor - finished testing accuracy \n", file=sys.stderr)

        else:
            raise Usage("\nTrain, inference or testing accuracy must be set to true in the configuration file\n")

        # check how long the whole program took
        stop = timer()
        print("Running Time = {} seconds".format(stop - start), file=sys.stderr)

        return args.output_dir

    except Usage as err:
        command_line.do_usage_and_die(err.msg)


if __name__ == "__main__":
    main()
    raise SystemExit
