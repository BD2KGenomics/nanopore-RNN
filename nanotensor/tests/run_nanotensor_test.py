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
import time
from nanotensor.queue import DataQueue, main
from nanotensor.utils import list_dir
import tensorflow as tf
# from nanotensor.run_nanotensor import
import unittest


class NetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(NetworkTest, cls).setUpClass()







if __name__ == '__main__':
    unittest.main()
