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
import unittest
import pickle
import os
import numpy as np
from nanotensor.data_preparation import TrainingData


def load_pickle(path):
    """load a python object from pickle"""
    path = os.path.abspath(path)
    with open(path, "rb") as file1:
        loading = pickle.load(file1)
    return loading

def save_pickle(obj, path):
    """Save a python object as file"""
    path = os.path.abspath(path)
    loading = pickle.dump(obj, open(path, "wb"))
    return loading



class Test_data_preparation(unittest.TestCase):
    """Test the functions in data_preparation.py"""

    def setUp(self):
        self.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-3])
        canonical_fast5 = os.path.join(self.HOME,\
        "test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch467_read35_strand.fast5")
        canonical_tsv = os.path.join(self.HOME, \
         "test_files/signalalignment_files/canonical/3c6e13d8-2ecb-475a-9499-d0d5c48dd1c8_Basecall_2D_template.sm.forward.tsv")
        self.G = TrainingData(canonical_fast5, canonical_tsv, strand_name="template", prob=True, kmer_len=5, alphabet="ATGCE", nanonet=True, deepnano=False)
        self.G.run_complete_analysis()

    def test_scrape_signalalign(self):
        '''test_scrape_signalalign'''
        events = self.G.scrape_signalalign()
        #these values manually extracted from SA output
        dict_real = {1010:12, 12684:12, 991:6}
        #extracted through scrape_signalalign method
        dict_test = {1010:len(events[1010]), 12684:len(events[12684]), 991:len(events[991])}
        # save_pickle(events[1010], "check_vals.p")
        for (k, v), (k2, v2) in zip(dict_real.items(), dict_test.items()):
            self.assertEqual((k, v), (k2, v2)) #check if they length matches
        # '''for position 23, check kmers match their probabilities'''
        pos1010 = events[1010] #extracted through SA method
        probabilities = os.path.join(self.HOME, \
         "test_files/pos1010-kmer-prob.txt")

        loading = load_pickle(probabilities) # pos23-actual-prob.txt extracted manually
        self.assertListEqual(pos1010, loading)

    def test_null(self):
        """test_null"""
        null = self.G.create_null_label()
        null_len = len(null)
        self.assertEqual(null[null_len - 1], 1)

    def test_length(self):
        """test_length"""
        fast5_length = len(self.G.scrape_fast5_events())
        features_length = len(self.G.create_features())
        self.assertEqual(fast5_length, features_length)

    def test_create_kmer_labels(self):
        """test_create_kmer_labels"""
        pos1010 = self.G.create_kmer_labels()[1010]
        labels = os.path.join(self.HOME, \
         "test_files/pos1010-label-prob.txt")
        test_1010 = load_pickle(labels)
        self.assertEqual(sorted(pos1010), sorted(test_1010))



if __name__ == '__main__':
    # suite = unittest.TestLoader().loadTestsFromTestCase(Test_data_preparation)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
