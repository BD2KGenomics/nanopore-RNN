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
from utils import project_folder, list_dir
import argparse
from error import Usage
from datetime import datetime
from data import DataQueue
import tensorflow as tf
from tensorflow.contrib import rnn
from network import BuildGraph

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
        # create new output dir
        # logfolder_path = os.path.join(args["output_dir"],
        #                               datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
        # os.makedirs(logfolder_path)
        # config_path = os.path.join(logfolder_path, 'config.json')
        # # save config file for training data
        # with open(config_path, 'w') as outfile:
        #     json.dump(args, outfile, indent=4)
        # args["output_dir"] = logfolder_path
        return args

#
# if "CUDA_HOME" in os.environ:
#     utilization = re.findall(r"Utilization.*?Gpu.*?(\d+).*?Memory.*?(\d+)",
#                              subprocess.check_output(["nvidia-smi", "-q"]),
#                              flags=re.MULTILINE | re.DOTALL)
#     print("GPU Utilization", utilization)
#
#     if ('0', '0') in utilization:
#         print("Using GPU Device:", utilization.index(('0', '0')))
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(utilization.index(('0', '0')))
#         os.environ["CUDA_DEVICE_ORDER"]  = "PCI_BUS_ID"  # To ensure the index matches
#     else:
#         print("All GPUs in Use")
#         exit
# else:
#     print("Running using CPU, NOT GPU")

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
        command_line.check_args(args)
        # Parameters
        learning_rate = args["learning_rate"]
        training_iters = args["training_iters"]
        batch_size = args["batch_size"]
        queue_size = args["queue_size"]
        display_step = args["display_step"]
        n_steps = args["n_steps"] # one vector per timestep
        layer_sizes = args["blstm_layer_sizes"] # hidden layer num of features
        training_dir = args["training_dir"]
        training_files = list_dir(training_dir, ext="npy")
        # continually load data on the CPU
        with tf.device("/cpu:0"):
            data = DataQueue(training_files, batch_size, queue_size=queue_size, verbose=False, \
                    pad=0, trim=True, n_steps=n_steps)
            images_batch, labels_batch = data.get_inputs()

        # build model
        model = BuildGraph(data.n_input, data.n_classes, learning_rate, n_steps=n_steps, \
                layer_sizes=layer_sizes, batch_size=batch_size, x=images_batch, y=labels_batch)
        cost = model.cost
        accuracy = model.accuracy
        merged_summaries = model.merged_summaries
        optimizer = model.optimizer
        # define what we want from the optimizer run
        init = model.init
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        # Launch the graph
        with tf.Session(config=tf.ConfigProto(log_device_placement=True, intra_op_parallelism_threads=8)) as sess:
            # create logs
            logfolder_path = os.path.join(project_folder(), 'logs/', \
                            datetime.now().strftime("%m%b-%d-%Hh-%Mm"))
            writer = tf.summary.FileWriter((logfolder_path), sess.graph)
            # initialize
            sess.run(init)
            step = 1
            # start queue
            tf.train.start_queue_runners(sess=sess)
            data.start_threads(sess)
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                # Run optimization and update layers
                output_states = sess.run([optimizer, model.update_op])
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    run_metadata = tf.RunMetadata()
                    acc, summary, loss = sess.run([accuracy, merged_summaries, cost], run_metadata=run_metadata)
                    # add summary statistics
                    writer.add_summary(summary, step)
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                step += 1

            print("Optimization Finished!")
            # Calculate accuracy for a bunch of test data

            saver.save(sess, project_folder()+'/testing/my_test_model.ckpt', global_step=model.global_step)

            print("Testing Accuracy: {}".format(sess.run(accuracy)))
            writer.close()



        #
        # with tf.Session() as sess:
        #     # To initialize values with saved data
        #     sess.run(init)
        #     tf.train.start_queue_runners(sess=sess)
        #     data.start_threads(sess)
        #
        #     saver.restore(sess, '/Users/andrewbailey/nanopore-RNN/testing/my_test_model.ckpt-49')
        #     # new_saver.restore(sess, tf.train.latest_checkpoint('/Users/andrewbailey/nanopore-RNN/testing/'))
        #     print(sess.run(accuracy)) # returns 1000

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
