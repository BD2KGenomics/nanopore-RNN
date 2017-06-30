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
from timeit import default_timer as timer
import json
import argparse
import pickle
import numpy as np
import time
from datetime import datetime
from utils import project_folder, list_dir, DotDict, upload_model
from error import Usage
from data import DataQueue
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import timeline
from network import BuildGraph
from data_preparation import TrainingData

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
        takes a config file with network configurations and paths to data directories.'
        , epilog="Dont forget to tar the files",
                                              usage='%(prog)s use "-h" for help')

        # create mutually exclusive argument for log file and json file
        self.exclusive = self.parser.add_mutually_exclusive_group(required=True)

        # optional arguments
        self.exclusive.add_argument('-c', '--config',
                                    help='path to a json config file with all \
                                    required arguments defined')

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

    @staticmethod
    def check_args(args):
        """Check arguments, save config file in new folder if correct"""
        # make sure output dir exists
        assert os.path.isdir(args["training_dir"])
        assert os.path.isdir(args["training_dir"])
        assert isinstance(args["blstm_layer_sizes"], list)
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
        # self.save_model_path = os.path.join(self.args.output_dir, self.args.model_name)
        self.model = self.models()
        if self.args.use_checkpoint:
            self.trained_model_path = tf.train.latest_checkpoint(self.args.trained_model)
        else:
            self.trained_model_path = self.args.trained_model_path

    def models(self):
        events, labels = self.load_data()
        model = BuildGraph(self.n_input, self.n_classes, self.args.learning_rate, n_steps=self.args.n_steps, \
                        layer_sizes=self.args.blstm_layer_sizes, batch_size=self.args.batch_size, x=events, y=labels)
        # with open('/Users/andrewbailey/nanopore-RNN/logs/' + 'graph1' + '.pkl', 'wb+') as output:
        #     pickle.dump(model, output)
        return model

    def load_data(self):
        """Create training and testing queues from training and testing files"""
        if self.args.train:
            self.training = DataQueue(self.training_files, self.args.batch_size, \
                queue_size=self.args.queue_size, verbose=False, pad=0, trim=True, \
                n_steps=self.args.n_steps)
            self.testing = DataQueue(self.testing_files, self.args.batch_size, \
                queue_size=self.args.queue_size, verbose=False, pad=0, trim=True, \
                n_steps=self.args.n_steps)
        # events, labels = self.training.get_inputs()
            events, labels = tf.cond(self.testing_bool, \
                        lambda: self.testing.get_inputs(),\
                        lambda: self.training.get_inputs(), name="events")
            assert self.training.n_input == self.testing.n_input
            assert self.training.n_classes == self.testing.n_classes
            self.n_input = self.training.n_input
            self.n_classes = self.training.n_classes
        else:
            self.testing = DataQueue(self.testing_files, self.args.batch_size, \
                queue_size=self.args.queue_size, verbose=False, pad=0, trim=True, \
                n_steps=self.args.n_steps)
            events, labels = self.testing.get_inputs()
            # TODO -  should get this info from data preparation configuration
            self.n_input = self.testing.n_input
            self.n_classes = self.testing.n_classes

        return events, labels

    def run_training(self, intra_op_parallelism_threads=8, log_device_placement=False):
        """Run training steps"""
        with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement,\
                    intra_op_parallelism_threads=intra_op_parallelism_threads)) as sess:
            # create logs
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

            step = 1
            if self.args.load_trained_model:
                writer = tf.summary.FileWriter((self.args.trained_model), sess.graph)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
                saver.restore(sess, self.trained_model_path)
                save_model_path = os.path.join(self.args.trained_model, self.args.model_name)
            else:
            # initialize
                writer = tf.summary.FileWriter((self.args.output_dir), sess.graph)
                sess.run(tf.global_variables_initializer())
                save_model_path = os.path.join(self.args.output_dir, self.args.model_name)
                saver = tf.train.Saver()
                saver.save(sess, save_model_path, \
                                global_step=self.model.global_step)
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            # start queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            self.training.start_threads(sess)
            self.testing.start_threads(sess)
            run_metadata = tf.RunMetadata()

            # Keep training until reach max iterations
            while step * self.args.batch_size < self.args.training_iters:
                # Run optimization and update layers
                _ = sess.run([self.model.optimizer, self.model.zero_state], \
                                feed_dict={self.testing_bool:False})
                if step % self.args.display_step == 0:
                    # Calculate batch loss and accuracy
                    acc, summary, loss, global_step = sess.run([self.model.accuracy, self.model.merged_summaries\
                                        , self.model.cost, self.model.global_step],\
                                        run_metadata=run_metadata, options=run_options, feed_dict={self.testing_bool:True})
                    # add summary statistics
                    writer.add_summary(summary, global_step)
                    writer.add_run_metadata(run_metadata, "step{}".format(global_step))
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)
                        
                    print("Iter " + str(step*self.args.batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Testing Accuracy= " + \
                          "{:.5f}".format(acc))
                    saver.save(sess, save_model_path, \
                                    global_step=self.model.global_step, write_meta_graph=False)
                step += 1
            saver.save(sess, save_model_path, \
                            global_step=self.model.global_step, write_meta_graph=False)

            coord.request_stop()
            coord.join(threads)
            sess.close()

            print("Optimization Finished!")
            writer.close()

    def call(self):
        """Run a model from a saved model path"""
        new_saver = tf.train.Saver()
        translate = TrainingData.getkmer_dict(alphabet=self.args.alphabet, length=self.args.kmer_len, flip=True)
        translate[1024] = "NNNNN"
        with tf.Session() as sess:
            # new_saver = tf.train.import_meta_graph('/Users/andrewbailey/nanopore-RNN/logs/06Jun-26-18h-04m/my_test_model-0.meta')
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
        with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement,\
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
                print("Iter " + str(step*self.args.batch_size) + ", Testing Accuracy= " + "{:.5f}".format(acc[0]))
                step += 1
            coord.request_stop()
            coord.join(threads)
            sess.close()
            final_acc = str(acc_sum/step *100)[:5]+"%"+datetime.now().strftime("%m%b-%d-%Hh-%Mm")
            print("Average Accuracy = {:.3f}".format(acc_sum/step *100))

        if save:
            file_list = self.get_model_files(config_path)
            # print(file_list)
            print("Uploading Model to neuralnet-accuracy s3 Bucket")
            upload_model("neuralnet-accuracy", file_list, final_acc)

    def get_model_files(self, *files):
        """Collect neccessary model files for upload"""
        file_list = []
        file_list.append(self.trained_model_path+".data-00000-of-00001")
        file_list.append(self.trained_model_path+".index")
        for file1 in files:
            file_list.append(file1)
        return file_list


def main(command_line=None):
    """Main docstring"""
    start = timer()

    if command_line is None:
        command_line = CommandLine()

    try:
        # get config and log files
        config = command_line.args["config"]
        # define arguments with config or arguments
        config = os.path.abspath(config)
        assert os.path.isfile(config)

        print("Using config file {}".format(config), file=sys.stderr)
        with open(config) as json_file:
            args = json.load(json_file)
        # check arguments and define
        args = command_line.check_args(args)
        # Parameters
        if args.train:
            train = TrainModel(args)
            train.run_training(log_device_placement=False)
        elif args.inference:
            infer = TrainModel(args)
            infer.call()
        else:
            test = TrainModel(args)
            test.testing_accuracy(config, save=args.save_s3)

        print("\n#  nanotensor - finished training \n", file=sys.stderr)
        print("\n#  nanotensor - finished training \n", file=sys.stdout)
        # check how long the whole program took
        stop = timer()
        print("Running Time = {} seconds".format(stop-start), file=sys.stderr)


    except Usage as err:
        command_line.do_usage_and_die(err.msg)


if __name__ == "__main__":
    main()
    raise SystemExit
