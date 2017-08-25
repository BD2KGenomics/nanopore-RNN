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
from nanotensor.utils import project_folder, list_dir, DotDict, upload_model, load_json
from nanotensor.error import Usage
from nanotensor.queue import DataQueue
from nanotensor.network import BuildGraph
from nanotensor.data_preparation import TrainingData
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
        self.parser = argparse.ArgumentParser(description='This program \ takes a config file with network '
                                                          'configurations and paths to data directories.',
                                              epilog="Dont forget to tar the files",
                                              usage='%(prog)s use "-h" for help')

        # create mutually exclusive argument for log file and json file
        self.exclusive = self.parser.add_mutually_exclusive_group(required=True)

        # optional arguments
        self.exclusive.add_argument('-c', '--config',
                                    help='path to a json config file with all \
                                    required arguments defined')

        self.parser.add_argument('-v', '--verbose', action='store_true',
                                 help='print out more information')

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

    def debug(self, string):
        """
        Controls debugging messages.
        """
        if not self.args['verbose']:
            pass
        elif self.args['verbose']:
            sys.stdout.write('{}\n'.format(string))

    @staticmethod
    def check_args(args):
        """Check arguments, save config file in new folder if correct"""
        # make sure output dir exists
        assert os.path.isdir(args["training_dir"])
        assert os.path.isdir(args["training_dir"])
        # assert isinstance(args["blstm_layer_sizes"], list)
        # create new output dir
        if args["train"] and not args["load_trained_model"]:
            logfolder_path = os.path.join(args["output_dir"],
                                          datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
            os.makedirs(logfolder_path)
            config_path = os.path.join(logfolder_path, 'config.json')
            # save config file for training data
            with open(config_path, 'w') as outfile:
                json.dump(args, outfile, indent=4)
            # define log file dir
            args["output_dir"] = logfolder_path
            args["config_path"] = config_path
        args = DotDict(args)
        return args


class TrainModel(object):
    """Class for running a tensorflow model."""

    def __init__(self, args):
        super(TrainModel, self).__init__()
        self.args = args
        self.training_files = list_dir(self.args.training_dir, ext="npy")
        self.testing_files = list_dir(self.args.testing_dir, ext="npy")
        self.testing_bool = tf.placeholder(dtype=bool, shape=[], name='testing_bool')
        self.n_input = int()
        self.n_classes = int()
        self.training = "DataQueue"
        self.testing = "DataQueue"
        self.start = datetime.now()
        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        # self.save_model_path = os.path.join(self.args.output_dir, self.args.model_name)
        self.train_op = "train_op"
        self.model = self.initialize_model()

        if self.args.use_checkpoint:
            self.trained_model_path = tf.train.latest_checkpoint(self.args.trained_model)
        else:
            self.trained_model_path = self.args.trained_model_path

    def initialize_model(self):
        """Initialize the model with multi-gpu options"""
        tower_grads = []
        self.load_data()
        gpu_indexes = test_for_nvidia_gpu(self.args.num_gpu)
        reuse = None
        opt = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        if gpu_indexes and self.args.train:
            print("Using GPU's {}".format(gpu_indexes), file=sys.stderr)
            with tf.variable_scope(tf.get_variable_scope()):
                for i in list(gpu_indexes):
                    events, labels = self.training.get_inputs()
                    with tf.device('/gpu:%d' % i):
                        model = BuildGraph(self.n_input, self.n_classes, self.args.learning_rate,
                                           n_steps=self.args.n_steps,
                                           network=self.args.network, x=events, y=labels,
                                           binary_cost=self.args.binary_cost,
                                           reuse=reuse)
                        tf.get_variable_scope().reuse_variables()
                        reuse = True
                        gradients = opt.compute_gradients(model.cost)
                        tower_grads.append(gradients)
                        # print(gradients)
                        # print(len(gradients))
            grads = average_gradients(tower_grads)
        else:
            print("No GPU's available, using CPU for computation", file=sys.stderr)
            if self.args.train:
                events, labels = self.training.get_inputs()
            else:
                events, labels = self.testing.get_inputs()

            model = BuildGraph(self.n_input, self.n_classes, self.args.learning_rate, n_steps=self.args.n_steps,
                               network=self.args.network, x=events, y=labels, binary_cost=self.args.binary_cost,
                               reuse=reuse)
            # self.train_op = model.optimizer
            grads = opt.compute_gradients(model.cost)
        with tf.variable_scope("apply_gradients"):
            self.train_op = opt.apply_gradients(grads, global_step=self.global_step)
        # variable_averages = tf.train.ExponentialMovingAverage(
        # cifar10.MOVING_AVERAGE_DECAY, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())

        return model

    def load_data(self):
        """Create training and testing queues from training and testing files"""
        self.training = DataQueue(self.training_files, self.args.batch_size,
                                  queue_size=self.args.queue_size, verbose=False, pad=0, trim=True,
                                  n_steps=self.args.n_steps)
        self.testing = DataQueue(self.testing_files, self.args.batch_size,
                                 queue_size=self.args.queue_size, verbose=False, pad=0, trim=True,
                                 n_steps=self.args.n_steps)
        assert self.training.n_input == self.testing.n_input
        assert self.training.n_classes == self.testing.n_classes
        self.n_input = self.training.n_input
        self.n_classes = self.training.n_classes
        return 0

    # @profile
    def run_training(self, intra_op_parallelism_threads=8, log_device_placement=True, allow_soft_placement=True):
        """Run training steps
        use `dmesg` to get error message if training is killed
        """
        config = tf.ConfigProto(log_device_placement=log_device_placement,
                                intra_op_parallelism_threads=intra_op_parallelism_threads,
                                allow_soft_placement=allow_soft_placement)
        # shows gpu memory usage
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # create logs
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            step = 0
            if self.args.load_trained_model:
                writer = tf.summary.FileWriter(self.args.trained_model, sess.graph)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.restore(sess, self.trained_model_path)
                # TODO could have a bug here if using wrong config file with wrong model name
                save_model_path = os.path.join(self.args.trained_model, self.args.model_name)
            else:
                # initialize
                writer = tf.summary.FileWriter(self.args.output_dir, sess.graph)
                sess.run(tf.global_variables_initializer())
                save_model_path = os.path.join(self.args.output_dir, self.args.model_name)
                saver = tf.train.Saver()
                saver.save(sess, save_model_path,
                           global_step=self.global_step)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            # start queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            self.training.start_threads(sess, n_threads=self.args.num_threads)
            run_metadata = tf.RunMetadata()
            # Keep training until reach max iterations
            # print("Training Has Started!")
            while step < self.args.training_iters:
                for _ in range(self.args.record_step):
                    # Run optimization training step
                    _ = sess.run([self.train_op])
                    step += 1
                # get training accuracy stats
                self.report_summary_stats(sess, writer, run_metadata, run_options)
                # if it has been enough time save model and print training stats
                if self.test_time() and self.args.profile:
                    # very expensive profiling steps to add time statistics
                    self.profile_training(sess, writer, run_metadata, run_options)

            saver.save(sess, save_model_path,
                       global_step=self.global_step, write_meta_graph=False)

            coord.request_stop()
            coord.join(threads)
            sess.close()
            writer.close()

            print("Training Finished!")

    def profile_training(self, sess, writer, run_metadata, run_options):
        """Expensive profile step so we can track speed of operations of the model"""
        _, acc, summary, cost, global_step = sess.run(
            [self.train_op, self.model.accuracy, self.model.train_summary,
             self.model.cost, self.global_step],
            run_metadata=run_metadata, options=run_options)
        # add summary statistics
        writer.add_summary(summary, global_step)
        writer.add_run_metadata(run_metadata, "step{}_train".format(global_step))
        if self.args.save_trace:
            self.chrome_trace(run_metadata, self.args.trace_name)

    def report_summary_stats(self, sess, writer, run_metadata, run_options):
        # Calculate batch loss and accuracy for training
        summary, global_step, train_acc, train_cost = sess.run([self.model.train_summary,
                                                                self.global_step,
                                                                self.model.accuracy, self.model.cost],
                                                               run_metadata=run_metadata,
                                                               options=run_options)
        # add summary statistics
        writer.add_summary(summary, global_step)
        # print summary stats
        print("Iter " + str(global_step) + ", Training Cost= " +
              "{:.6f}".format(train_cost) + ", Training Accuracy= " +
              "{:.5f}".format(train_acc), file=sys.stderr)

    def chrome_trace(self, metadata_proto, f_name):
        """Save a chrome trace json file.
        To view json vile go to - chrome://tracing/
        """
        time_line = timeline.Timeline(metadata_proto.step_stats)
        ctf = time_line.generate_chrome_trace_format()
        with open(f_name, 'w') as file1:
            file1.write(ctf)

    def test_time(self):
        """Return true if it is time to save the model"""
        delta = (datetime.now() - self.start).total_seconds()
        if delta > self.args.save_model:
            self.start = datetime.now()
            return True
        return False

    # TODO refactor to work correctly
    def call(self):
        """Run a model from a saved model path"""
        new_saver = tf.train.Saver()
        translate = TrainingData.getkmer_dict(alphabet=self.args.alphabet, length=self.args.kmer_len, flip=True)
        translate[1024] = "NNNNN"
        with tf.Session() as sess:
            # new_saver = tf.train.import_meta_graph(
            # '/Users/andrewbailey/nanopore-RNN/logs/06Jun-26-18h-04m/my_test_model-0.meta')
            new_saver.restore(sess, self.trained_model_path)
            # graph = tf.get_default_graph()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            self.testing.start_threads(sess)
            evaluate_pred = sess.run([self.model.evaluate_pred])
            kmers = [translate[index] for index in evaluate_pred[0]]
            with open(self.args.inference_output, "w+") as f1:
                f1.write("{}".format(kmers))

            # print([translate[index] for index in evaluate_pred[0]])
            # add summary statistics
            coord.request_stop()
            coord.join(threads)
            sess.close()

    def testing_accuracy(self, config_path, save=True, intra_op_parallelism_threads=8, log_device_placement=False):
        """Get testing accuracy and save model along with configuration file on s3"""
        with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement,
                                              intra_op_parallelism_threads=intra_op_parallelism_threads)) as sess:
            # restore model
            saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            saver.restore(sess, self.trained_model_path)

            # save_model_path = os.path.join(self.args.trained_model, self.args.model_name)
            # start queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            self.testing.start_threads(sess)

            acc_sum = 0
            # Keep training until reach max iterations
            step = 0

            while self.testing.files_left:
                # Calculate batch loss and accuracy
                acc = sess.run([self.model.accuracy])
                # print(acc)
                acc_sum += acc[0]
                print("Iter " + str(step * self.args.batch_size) + ", Testing Accuracy= " + "{:.5f}".format(acc[0]))
                step += 1

            coord.request_stop()
            coord.join(threads)
            sess.close()
            final_acc = str(acc_sum / step * 100)[:5] + "%" + datetime.now().strftime("%m%b-%d-%Hh-%Mm")
            print("Average Accuracy = {:.3f}".format(acc_sum / step * 100))

        if save:
            file_list = self.get_model_files(config_path)
            print(file_list)
            print("Uploading Model to neuralnet-accuracy s3 Bucket")
            upload_model("neuralnet-accuracy", file_list, final_acc)

    def get_model_files(self, *files):
        """Collect neccessary model files for upload"""
        file_list = [self.trained_model_path + ".data-00000-of-00001", self.trained_model_path + ".index"]
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
            print(v)
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
    try:
        utilization = re.findall(r"Utilization.*?Gpu.*?(\d+).*?Memory.*?(\d+)",
                                 subprocess.check_output(["nvidia-smi", "-q"]),
                                 flags=re.MULTILINE | re.DOTALL)
        assert len(utilization) >= num_gpu, "Not enough GPUs are available"
        indices = [i for i, x in enumerate(utilization) if x == ('0', '0')]
        assert len(indices) >= num_gpu, "Only {0} GPU's are not being used,  change num_gpu parameter to {0}".format(
            len(indices))
        return indices
    except OSError:
        return False


def main():
    """Main docstring"""
    start = timer()

    command_line = CommandLine()
    debug = command_line.debug

    try:
        # get config and log files
        config = command_line.args["config"]
        print("Using config file {}".format(config), file=sys.stderr)
        # define arguments with config or arguments
        args = load_json(config)
        # check arguments and define
        args = command_line.check_args(args)
        # Parameters
        if args.train:

            train = TrainModel(args)
            train.run_training(intra_op_parallelism_threads=8, log_device_placement=False, allow_soft_placement=True)
            print("\n#  nanotensor - finished training \n", file=sys.stderr)
            print("\n#  nanotensor - finished training \n", file=sys.stderr)

        elif args.inference:
            infer = TrainModel(args)
            infer.call()
            print("\n#  nanotensor - finished inference \n", file=sys.stderr)
            print("\n#  nanotensor - finished inference \n", file=sys.stderr)

        elif args.testing_accuracy:
            test = TrainModel(args)
            test.testing_accuracy(config, save=args.save_s3)
            print("\n#  nanotensor - finished testing accuracy \n", file=sys.stderr)
            print("\n#  nanotensor - finished testing accuracy \n", file=sys.stderr)

        else:
            raise Usage("\nTrain, inference or testing accuracy must be set to true in the configuration file\n")

        # check how long the whole program took
        stop = timer()
        print("Running Time = {} seconds".format(stop - start), file=sys.stderr)

    except Usage as err:
        command_line.do_usage_and_die(err.msg)


if __name__ == "__main__":
    main()
    raise SystemExit
