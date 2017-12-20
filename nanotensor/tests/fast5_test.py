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
from nanotensor.fast5 import Fast5
import unittest


class Fast5Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(Fast5Test, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-3])
        fast5_file = os.path.join(cls.HOME, "test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_sequencing_run_AMS_158_R9_WGA_Ecoli_08_20_16_43623_ch100_read104_strand.fast5")
        cls.fast5handle = Fast5(fast5_file)

    def test_get_fastq(self):
        """Test get_fastq method of Fast5 class"""
        self.fast5handle.get_fastq()

    def test_get_basecall_data(self):
        """Test get_basecall_data method of Fast5 class"""
        self.fast5handle.get_basecall_data()

    def test_get_read_stats(self):
        """Test get_read_stats from Fast5 class"""
        self.fast5handle.get_read_stats()

    def test_get_read(self):
        """Test get_read from Fast5 class"""
        self.fast5handle.get_read(raw=True, scale=False)

    def test_get_corrected_events(self):
        """Test get_corrected_events from Fast5 class"""
        self.fast5handle.get_corrected_events()

if __name__ == '__main__':
    unittest.main()
