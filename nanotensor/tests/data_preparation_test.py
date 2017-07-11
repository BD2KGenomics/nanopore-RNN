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

def conver_np_void_to_list(np_void):
    D = []
    for i in np_void:
        D.append(i)
    return D

def non_zero_element(array_prob):
    for i in array_prob:
        if i != 0:
            return i



class Test_data_preparation(unittest.TestCase):
    """Test the functions in data_preparation.py"""

    def setUp(self):
        self.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-3])
        canonical_fast5 = os.path.join(self.HOME,\
        "test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch467_read35_strand.fast5")
        canonical_tsv = os.path.join(self.HOME, \

        "test_files/signalalignment_files/test_canonical/3c6e13d8-2ecb-475a-9499-d0d5c48dd1c8_Basecall_2D_2d.sm.forward.tsv")

        self.T = TrainingData(canonical_fast5, canonical_tsv, strand_name="template", prob=True, kmer_len=5, alphabet="ATGCE", nanonet=True, deepnano=False)
        self.T.run_complete_analysis()

        self.C = TrainingData(canonical_fast5, canonical_tsv, strand_name="complement", prob=True, kmer_len=5, alphabet="ATGCE", nanonet=True, deepnano=False)
        self.C.run_complete_analysis()

    def test_scrape_signalalign_temp(self):
        '''test_scrape_signalalign'''
        events = self.T.scrape_signalalign()
        dict_real = {6324:9, 6321:4, 6356:6}
        dict_test = {6324:len(events[6324]), 6321:len(events[6321]), 6356:len(events[6356])}
        for (k, v), (k2, v2) in zip(dict_real.items(), dict_test.items()):
            self.assertEqual((k, v), (k2, v2))
        pos6324 = events[6324]
        probabilities = os.path.join(self.HOME, \
         "test_files/pos6324-kmer-prob.txt")

        loading = load_pickle(probabilities) # pos23-actual-prob.txt extracted manually
        self.assertListEqual(pos6324, loading)

    def test_scrape_signalalign_comp(self):
        '''test_scrape_signalalign'''
        events = self.C.scrape_signalalign()
        dict_real = {2851:8, 936:3, 1262:5}
        dict_test = {2851:len(events[2851]), 936:len(events[936]), 1262:len(events[1262])}
        for (k, v), (k2, v2) in zip(dict_real.items(), dict_test.items()):
            self.assertEqual((k, v), (k2, v2))
        pos2851 = events[2851] #extracted through SA method
        probabilities = os.path.join(self.HOME, \
         "test_files/pos2851-kmer-prob.txt")

        loading = load_pickle(probabilities)
        self.assertListEqual(pos2851, loading)

    def test_null(self):
        """test_null"""
        null = self.T.create_null_label()

        null_len = len(null)
        self.assertEqual(null[null_len - 1], 1)

    def test_length(self):
        """test_length"""
        fast5_length = len(self.T.scrape_fast5_events())
        features_length = len(self.T.create_features())
        self.assertEqual(fast5_length, features_length)

    def test_create_kmer_labels_temp(self):
        """test_create_kmer_labels"""
        pos6324 = self.T.create_kmer_labels()[6324]
        labels = os.path.join(self.HOME, \
         "test_files/pos6324-label-prob.txt")
        test_6324 = load_pickle(labels)
        self.assertEqual(sorted(pos6324), sorted(test_6324))

    def test_create_kmer_labels_comp(self):
        """test_create_kmer_labels"""
        pos2851 = self.C.create_kmer_labels()[2851]
        labels = os.path.join(self.HOME, \
         "test_files/pos2851-label-prob.txt")
        test_2851 = load_pickle(labels)
        self.assertEqual(sorted(pos2851), sorted(test_2851))

    def test_scrape_fast5_events(self):
        row10_real_temp=  [83.52333150227787, 221.1755, 1.3181877787337306, 0.0015]
        row10_real_comp=  [68.825253092462106,256.58775000000003,2.0204697774551619,0.0052500000000000003]
        row10_temp= conver_np_void_to_list(self.T.scrape_fast5_events()[10])
        row10_comp= conver_np_void_to_list(self.C.scrape_fast5_events()[10])
        self.assertEqual(row10_temp, row10_real_temp)
        self.assertEqual(row10_comp, row10_real_comp)

    def test_getkmer_dict(self):
        first = "AAAAA"
        last = "TTTTT"
        sorted_dict = sorted(self.T.getkmer_dict("ATCGE", 5))
        self.assertEqual(first, sorted_dict[0])
        self.assertEqual(last, sorted_dict[-1])

    def test_create_labels(self):
        comp_7 = self.C.scrape_signalalign()[7][0][1]
        temp_2942 = self.T.scrape_signalalign()[2942][0][1]
        comp_7_CL = non_zero_element(self.C.create_labels()[7])
        temp_2942_CL = non_zero_element(self.T.create_labels()[2942])
        self.assertEqual(comp_7, comp_7_CL)
        self.assertEqual(temp_2942, temp_2942_CL)

    def test_run_complete_analysis(self):
        EQ_t = self.T.run_complete_analysis()
        EQ_c = self.C.run_complete_analysis()
        self.assertEqual(EQ_t, True)
        self.assertEqual(EQ_c, True)

    def test_scrape_eventalign(self):
        self.assertEqual(self.T.scrape_eventalign(), False)
        self.assertEqual(self.C.scrape_eventalign(), False)

    def test_create_deepnano_labels(self):
        self.assertEqual(self.T.create_deepnano_labels(), False)
        self.assertEqual(self.C.create_deepnano_labels(), False)



if __name__ == '__main__':
    # suite = unittest.TestLoader().loadTestsFromTestCase(Test_data_preparation)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
