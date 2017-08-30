#!/usr/bin/env python
"""
    Place unit tests for network.py
    """
########################################################################
# File: data_preparation_test.py
#  executable: data_preparation_test.py
# Purpose: data_preparation test functions
#
# Author: Rojin Safavi
# History: 08/01/2017 Created
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
from nanotensor.network import BuildGraph
import unittest


class NetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(NetworkTest, cls).setUpClass()

        cls.images_batch = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                                     [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]).astype(np.float32)

        cls.labels_batch = np.array(
            [[[1, 2, 3, 4, 1, 2, 3, 4], [5, 6, 7, 8, 5, 6, 7, 8], [9, 10, 11, 12, 9, 10, 11, 12]],
             [[13, 14, 15, 16, 13, 14, 15, 16], [17, 18, 19, 20, 17, 18, 19, 20],
              [21, 22, 23, 24, 21, 22, 23, 24]]]).astype(np.float32)

        cls.images_batch_reshaped = tf.reshape(cls.images_batch, [-1, 4])
        cls.labels_batch_flat = tf.reshape(cls.labels_batch, [-1, cls.labels_batch.shape[2]])
        cls.n_classes = cls.labels_batch.shape[2]
        cls.n_input = cls.images_batch.shape[2]
        cls.model = BuildGraph(n_input=cls.n_input, n_classes=cls.n_classes, learning_rate=0.001, n_steps=3,
                               forget_bias=5.0, x=cls.images_batch, y=cls.labels_batch,
                               network=[{"bias": 5.0, 'type': 'blstm', 'name': 'blstm_layer1', 'size': 128},
                                        {"bias": 1.0, 'type': 'tanh', 'name': 'subsample_level_1', 'size': 64},
                                        {"bias": 1.0, 'type': 'blstm', 'name': 'blstm_layer2', 'size': 128},
                                        {"bias": 1.0, 'type': 'tanh', 'name': 'subsample_level_2', 'size': 64}],
                               binary_cost=True)
        # cls.model2 = BuildGraph(n_input=cls.n_input, n_classes=cls.n_classes, learning_rate=0.001, n_steps=3,
        #                         forget_bias=5.0, x=cls.images_batch, y=cls.labels_batch,
        #                         network=[{"bias": 5.0, 'type': 'blstm', 'name': 'blstm_layer1', 'size': 128},
        #                                  {"bias": 1.0, 'type': 'tanh', 'name': 'subsample_level_1', 'size': 64},
        #                                  {"bias": 1.0, 'type': 'blstm', 'name': 'blstm_layer2', 'size': 128},
        #                                  {"bias": 1.0, 'type': 'tanh', 'name': 'subsample_level_2', 'size': 64}],
        #                         binary_cost=False)

    def test_init(self):
        """Test init of BuildGraph class"""
        self.assertIsInstance(self.model, BuildGraph)

    def test_prediction_function(self):
        self.model.pred = tf.constant([[1, 4, 3], [15, 5, 5]])  # 1,0
        self.model.y_flat = tf.constant([[1, 4, 1], [7, 1, 1]])  # 1,0
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        h = sess.run(self.model.prediction_function())
        self.assertTrue(h.all)
        sess.close()

    def test_blstm(self):
        blstm_model = self.model.blstm(input_vector=self.images_batch, layer_name="TEST", n_hidden=11)
        self.assertEqual(str((2, 3, 22)), str(blstm_model.shape))
        self.assertNotEqual(str((2, 3, 11)), str(blstm_model.shape))

    def test_combine_arguments(self):
        self.model.T1 = tf.constant([[1, 4, 3], [15, 5, 5]])
        self.model.T2 = tf.constant([[1, 4, 1], [1, 7, 1]])
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        h = sess.run(self.model.combine_arguments([self.model.T1, self.model.T2], name="combined_tensors"))
        h_asanarray = np.asanyarray(h)
        h_asanarray_shape = h_asanarray.shape
        np.testing.assert_array_equal(h_asanarray[0], np.asanyarray([[1, 4, 3], [15, 5, 5]]))
        np.testing.assert_array_equal(h_asanarray[1], np.asanyarray([[1, 4, 1], [1, 7, 1]]))
        self.assertEqual(str((2, 2, 3)), str(h_asanarray_shape))
        sess.close()

    def test_accuracy_function(self):
        self.model.correct_pred = np.array([True, False, False, False])
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        j = sess.run(self.model.accuracy_function())
        self.assertEqual(j, 0.25)
        sess.close()

    def test_fulconn_layer(self):
        output_dim = self.n_classes
        with tf.variable_scope("test"):
            output = self.model.fulconn_layer(self.images_batch_reshaped, output_dim)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        nn = sess.run(output)
        sess.close()
        np.testing.assert_array_almost_equal(nn[0], np.matmul((self.images_batch.reshape([-1, 4])), nn[1]) + nn[2],
                                             decimal=3)


if __name__ == '__main__':
    unittest.main()
