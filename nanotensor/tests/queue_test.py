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
import threading
import time
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
        cls.data = DataQueue(cls.training_files, batch_size=2, queue_size=1000, verbose=True, pad=0, trim=True,
                             n_steps=10, min_in_queue=1)

    def test_init(self):
        """Test init of DataQueue class"""
        self.assertRaises(AssertionError, DataQueue, self.no_npy, batch_size=2, queue_size=10, verbose=False, pad=0,
                          trim=True, n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=10, queue_size=10, verbose=False,
                          pad=0, trim=True, n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=2, queue_size=10, verbose="False",
                          pad=0, trim=True, n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=2, queue_size=10, verbose=False,
                          pad="0", trim=True, n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=2, queue_size=10, verbose=False,
                          pad=0, trim="True", n_steps=10)
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=2, queue_size=10, verbose=False,
                          pad=0, trim=True, n_steps="10")
        self.assertRaises(AssertionError, DataQueue, self.training_files, batch_size=2, queue_size=10, verbose=False,
                          pad=0, trim=True, n_steps=10, min_in_queue="10")
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

    def test_add_to_queue(self):
        """Test add to queue function"""
        # define dequeue operations
        dequeue_x_op, dequeue_y_op = self.data.queue.dequeue_many(1)
        # initialize and start tf session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        # create data
        data = np.asanyarray([[np.random.rand(12), np.random.rand(1025)] for _ in range(10)])
        data_x = np.asarray([np.asarray(data[:, 0][n]) for n in range(len(data[:, 0]))])
        data_y = np.asarray([np.asarray(data[:, 1][n]) for n in range(len(data[:, 1]))])
        # add data to queue
        for _ in range(10):
            self.data.add_to_queue(data, sess)
        # make sure data coming out is same as it was when it went in
        same_data_x, same_data_y = sess.run([dequeue_x_op, dequeue_y_op])
        self.assertTrue(np.isclose(same_data_x[0], data_x).all())
        self.assertTrue(np.isclose(same_data_y[0], data_y).all())

        sess.close()

    def test_get_inputs(self):
        """Test get inputs function"""
        # define dequeue operations
        dequeue_x_op, dequeue_y_op = self.data.get_inputs()
        # initialize and start tf session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        # create data
        data = np.asanyarray([[np.random.rand(12), np.random.rand(1025)] for _ in range(10)])
        data_x = np.asarray([np.asarray(data[:, 0][n]) for n in range(len(data[:, 0]))])
        data_y = np.asarray([np.asarray(data[:, 1][n]) for n in range(len(data[:, 1]))])
        # add data to queue
        enqueue_op = self.data.queue.enqueue([self.data.dataX, self.data.dataY])

        for _ in range(10):
            sess.run(enqueue_op, feed_dict={self.data.dataX: data_x, self.data.dataY: data_y})
        # make sure data coming out is same as it was when it went in
        same_data_x, same_data_y = sess.run([dequeue_x_op, dequeue_y_op])
        self.assertTrue(np.isclose(same_data_x[0], data_x).all())
        self.assertTrue(np.isclose(same_data_y[0], data_y).all())

        sess.close()

    def test_create_batches(self):
        """Test create batches method"""
        tf.set_random_seed(1)
        # define dequeue operations
        dequeue_x_op, dequeue_y_op = self.data.queue.dequeue_many(1)
        # initialize and start tf session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        # create data
        data = np.asanyarray([[np.random.rand(12), np.random.rand(1025)] for _ in range(100)])
        data_x = np.asarray([np.asarray(data[:, 0][n]) for n in range(len(data[:, 0]))])
        data_y = np.asarray([np.asarray(data[:, 1][n]) for n in range(len(data[:, 1]))])
        # add data to queue
        self.data.create_batches(data, sess)
        # grab batches
        same_data_x, same_data_y = sess.run([dequeue_x_op, dequeue_y_op])
        same_data_x2, same_data_y2 = sess.run([dequeue_x_op, dequeue_y_op])
        same_data_x3, same_data_y3 = sess.run([dequeue_x_op, dequeue_y_op])
        same_data_x4, same_data_y4 = sess.run([dequeue_x_op, dequeue_y_op])

        # testing shuffle queue seed is same and we should get same data out from queue
        self.assertTrue(np.isclose(same_data_x[0], data_x[70:80]).all())
        self.assertTrue(np.isclose(same_data_y[0], data_y[70:80]).all())
        self.assertTrue(np.isclose(same_data_x2[0], data_x[80:90]).all())
        self.assertTrue(np.isclose(same_data_y2[0], data_y[80:90]).all())
        self.assertTrue(np.isclose(same_data_x3[0], data_x[10:20]).all())
        self.assertTrue(np.isclose(same_data_y3[0], data_y[10:20]).all())
        self.assertTrue(np.isclose(same_data_x4[0], data_x[30:40]).all())
        self.assertTrue(np.isclose(same_data_y4[0], data_y[30:40]).all())

        sess.close()

    def test_read_in_file(self):
        """Test read_in_file method"""
        tf.set_random_seed(1)
        # define dequeue operations
        dequeue_x_op, dequeue_y_op = self.data.queue.dequeue_many(1)
        # initialize and start tf session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        # get test data
        data = np.load(self.data.file_list[0])
        data_x = np.asarray([np.asarray(data[:, 0][n]) for n in range(len(data[:, 0]))])
        data_y = np.asarray([np.asarray(data[:, 1][n]) for n in range(len(data[:, 1]))])

        # read in first file
        self.data.read_in_file(sess)
        # make sure file index was increased

        # get data
        same_data_x, same_data_y = sess.run([dequeue_x_op, dequeue_y_op])
        same_data_x2, same_data_y2 = sess.run([dequeue_x_op, dequeue_y_op])
        same_data_x3, same_data_y3 = sess.run([dequeue_x_op, dequeue_y_op])
        same_data_x4, same_data_y4 = sess.run([dequeue_x_op, dequeue_y_op])

        # make sure it is same data we expected
        self.assertTrue(np.isclose(same_data_x[0], data_x[270:280]).all())
        self.assertTrue(np.isclose(same_data_y[0], data_y[270:280]).all())
        self.assertTrue(np.isclose(same_data_x2[0], data_x[170:180]).all())
        self.assertTrue(np.isclose(same_data_y2[0], data_y[170:180]).all())
        self.assertTrue(np.isclose(same_data_x3[0], data_x[330:340]).all())
        self.assertTrue(np.isclose(same_data_y3[0], data_y[330:340]).all())
        self.assertTrue(np.isclose(same_data_x4[0], data_x[100:110]).all())
        self.assertTrue(np.isclose(same_data_y4[0], data_y[100:110]).all())
        sess.close()

    def test_load_data(self):
        """Test read_in_file method"""
        tf.set_random_seed(2)
        data_queue = DataQueue(self.training_files, batch_size=2, queue_size=100, verbose=True, pad=0, trim=True,
                               n_steps=10)
        # define dequeue operations
        dequeue_x_op, dequeue_y_op = data_queue.queue.dequeue_many(1)

        # initialize and start tf session
        init = tf.global_variables_initializer()
        coord = tf.train.Coordinator()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess, coord=coord)
        # send process to background so we can continue
        stop_event = threading.Event()

        t = threading.Thread(target=data_queue.load_data, args=(sess, stop_event))
        t.daemon = True  # thread will close when parent quits
        t.start()

        # making sure the queue is full before dequeue in order to maintain consistancy with a random queue
        time.sleep(1)
        same_data_x, same_data_y = sess.run([dequeue_x_op, dequeue_y_op])
        time.sleep(1)
        same_data_x2, same_data_y2 = sess.run([dequeue_x_op, dequeue_y_op])
        time.sleep(1)
        same_data_x3, same_data_y3 = sess.run([dequeue_x_op, dequeue_y_op])
        # load data from first two files to make sure we are getting correct data
        data = np.load(data_queue.file_list[0])
        data_x_0 = np.asarray([np.asarray(data[:, 0][n]) for n in range(len(data[:, 0]))])
        data_y_0 = np.asarray([np.asarray(data[:, 1][n]) for n in range(len(data[:, 1]))])
        data = np.load(data_queue.file_list[1])
        data_x_1 = np.asarray([np.asarray(data[:, 0][n]) for n in range(len(data[:, 0]))])
        data_y_1 = np.asarray([np.asarray(data[:, 1][n]) for n in range(len(data[:, 1]))])

        # make sure it is same data we expected
        self.assertTrue(np.isclose(same_data_x[0], data_x_1[240:250]).all())
        self.assertTrue(np.isclose(same_data_y[0], data_y_1[240:250]).all())
        self.assertTrue(np.isclose(same_data_x2[0], data_x_0[270:280]).all())
        self.assertTrue(np.isclose(same_data_y2[0], data_y_0[270:280]).all())
        self.assertTrue(np.isclose(same_data_x3[0], data_x_0[0:10]).all())
        self.assertTrue(np.isclose(same_data_y3[0], data_y_0[0:10]).all())

        dequeue_x_op, dequeue_y_op = data_queue.queue.dequeue_many(50)
        _, _ = sess.run([dequeue_x_op, dequeue_y_op])
        _, _ = sess.run([dequeue_x_op, dequeue_y_op])
        _, _ = sess.run([dequeue_x_op, dequeue_y_op])
        time.sleep(1)
        stop_event.set()
        self.assertFalse(data_queue.files_left)
        sess.close()

    def test_start_threads(self):
        """Test start threads method"""
        tf.set_random_seed(2)
        data_queue = DataQueue(self.training_files, batch_size=1, queue_size=3, verbose=True, pad=0, trim=True,
                               n_steps=10)
        # define dequeue operations
        dequeue_x_op, dequeue_y_op = data_queue.queue.dequeue_many(1)

        # initialize and start tf session
        init = tf.global_variables_initializer()
        coord = tf.train.Coordinator()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess, coord=coord)
        # send process to background so we can continue
        threads = data_queue.start_threads(sess, n_threads=3)
        # three threads on three different files have started and stopped
        self.assertEqual(len(threads), 3)
        self.assertTrue(data_queue.file_list_queue.get() in data_queue.file_list)
        self.assertTrue(data_queue.file_list_queue.get() in data_queue.file_list)
        # self.assertTrue(data_queue.file_list_queue.get() in data_queue.file_list)
        # self.assertFalse(data_queue.file_list_queue.empty())
        coord.request_stop()
        data_queue.join()
        sess.close()

    # def test_main(self):
    #     """Test that main does not error"""
    #     self.assertIsNone(main())

if __name__ == '__main__':
    unittest.main()

