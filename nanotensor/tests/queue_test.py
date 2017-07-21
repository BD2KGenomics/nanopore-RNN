#!/usr/bin/env python
"""
Place unit tests for data_preparation.py
"""
########################################################################
# File: data_preparation_test.py
#  executable: data_preparation_test.py
# Purpose: data_preparation test functions
#
# Author: Rojin Safavi
# History: 6/25/2017 Created
########################################################################

from __future__ import print_function
import unittest
import os
import numpy as np
from nanotensor.queue import DataQueue
from nanotensor.utils import list_dir
import tensorflow as tf

class QueueTest(unittest.TestCase):
    """Test the functions in data_preparation.py"""

    @classmethod
    def setUpClass(cls):
        super(QueueTest, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-3])
        cls.training_files = list_dir(os.path.join(cls.HOME, "test_files/create_training_files/07Jul-20-11h-28m"))
        cls.no_npy = list_dir(os.path.join(cls.HOME, "test_files/create_training_files/"))
        cls.data = DataQueue(cls.training_files, batch_size=2, queue_size=10, verbose=False, pad=0, trim=True, n_steps=10)

    def test_init(self):
        """Test init of DataQueue class"""
        self.assertRaises(AssertionError, DataQueue, self.no_npy, batch_size=2, queue_size=10, verbose=False, pad=0, trim=True, n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=10, queue_size=10, verbose=False, pad=0, trim=True, n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=2, queue_size=10, verbose="False", pad=0, trim=True, n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=2, queue_size=10, verbose=False, pad="0", trim=True, n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=2, queue_size=10, verbose=False, pad=0, trim="True", n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=2, queue_size=10, verbose=False, pad=0, trim=True, n_steps="10")
        self.assertIsInstance(self.data, DataQueue)
        self.assertIsInstance(self.data, DataQueue)
        self.assertEqual(len(self.data.bad_files), 2)
        self.assertEqual(len(self.data.file_list), 5)
        self.assertEqual((10, 1025), self.data.dataY.shape)
        self.assertEqual((10, 12), self.data.dataX.shape)
        self.assertTrue(self.data.files_left)

    def test_read_numpy_files(self):
        """Test if read numpy files function works correctly"""
        passing, fails = self.data.read_numpy_files(self.training_files)
        self.assertEqual(len(passing), 5)
        self.assertEqual(len(fails), 2)

    def test_shuffle(self):
        """Test if shuffle of read files works"""
        np.random.seed(1)
        testlist = self.data.file_list[:]
        self.data.shuffle()
        self.assertTrue(testlist != self.data.file_list)
        new_indexes = [2, 1, 4, 0, 3]
        for index in range(len(testlist)):
            self.assertTrue(self.data.file_list[index] == testlist[new_indexes[index]])
        self.assertFalse(self.data.files_left)

    def test_add_to_queue(self):
        """Test add to queue function"""

        dequeue_x_op, dequeue_y_op = self.data.queue.dequeue_many(1)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        data = np.asanyarray([[np.random.rand(12), np.random.rand(1025)] for _ in range(10)])
        data_x = np.asarray([np.asarray(data[:, 0][n]) for n in range(len(data[:, 0]))])
        data_y = np.asarray([np.asarray(data[:, 1][n]) for n in range(len(data[:, 1]))])

        for _ in range(10):
            self.data.add_to_queue(data, sess)

        same_data_x, same_data_y = sess.run([dequeue_x_op, dequeue_y_op])

        self.assertTrue(np.isclose(same_data_x[0], data_x).all())
        self.assertTrue(np.isclose(same_data_y[0], data_y).all())


if __name__ == '__main__':
    unittest.main()
