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
import pickle
import os
import sys
from timeit import default_timer as timer
import numpy as np
from nanotensor.data_preparation import TrainingData
from nanotensor.error import DataPrepBug


def load_pickle(path):
    """load a python object from pickle"""
    path = os.path.abspath(path)
    with open(path, "rb") as file1:
        loading = pickle.load(file1)
    return loading


# def save_pickle(obj, path):
#     """Save a python object as file"""
#     path = os.path.abspath(path)
#     loading = pickle.dump(obj, open(path, "wb"))
#     return loading



class DataPreparationTest(unittest.TestCase):
    """Test the functions in data_preparation.py"""

    @classmethod
    def setUpClass(cls):
        super(DataPreparationTest, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-3])
        canonical_fast5 = os.path.join(cls.HOME,
                                       "test_files/minion-reads/canonical"
                                       "/miten_PC_20160820_FNFAD20259_MN17223_mux_scan_AMS_158_R9_WGA_Ecoli_08_20_16_83098_ch467_read35_strand.fast5")
        # canonical_fast5 = os.path.join(cls.HOME,\
        # "test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_sequencing_run_AMS_158_R9_WGA_Ecoli_08_20_16_43623_ch100_read104_strand.fast5")
        canonical_tsv = os.path.join(cls.HOME,
                                     'test_files/signalalignment_files/test_canonical/3c6e13d8-2ecb-475a-9499'
                                     '-d0d5c48dd1c8_Basecall_2D_2d.sm.forward.short.tsv')
        # canonical_tsv = os.path.join(cls.HOME, \
        # "test_files/signalalignment_files/canonical/18a21abc-7827-4ed7-8919-c27c9bd06677_Basecall_2D_template.sm.forward.tsv")
        template_model = os.path.join(cls.HOME, "signalAlign/models/testModelR9p4_acegt_template.model")
        complement_model = os.path.join(cls.HOME, "signalAlign/models/testModelR9_complement_pop2.model")

        cls.DEEPNANO = TrainingData(canonical_fast5, canonical_tsv, strand_name="template", prob=False, kmer_len=2,
                                    alphabet="ATGC", nanonet=False, deepnano=True, template_model=template_model,
                                    complement_model=complement_model)

        cls.CATEGORICAL = TrainingData(canonical_fast5, canonical_tsv, strand_name="template", prob=False, kmer_len=5,
                                       alphabet="ATGCE", nanonet=True, deepnano=False)

        cls.T = TrainingData(canonical_fast5, canonical_tsv, strand_name="template", prob=True, kmer_len=5,
                             alphabet="ATGCE", nanonet=True, deepnano=False)
        #
        cls.C = TrainingData(canonical_fast5, canonical_tsv, strand_name="complement", prob=True, kmer_len=5,
                             alphabet="ATGCE", nanonet=True, deepnano=False)

        cls.eq_deepnano = cls.DEEPNANO.run_complete_analysis()
        cls.eq_categorical = cls.CATEGORICAL.run_complete_analysis()
        cls.eq_t = cls.T.run_complete_analysis()
        cls.eq_c = cls.C.run_complete_analysis()

    def test_run_complete_analysis(self):
        """test_run_complete_analysis"""
        self.assertEqual(self.eq_t, True)
        self.assertEqual(self.eq_c, True)
        self.assertEqual(self.eq_categorical, True)
        self.assertEqual(self.eq_deepnano, True)

    def test_scrape_signalalign(self):
        '''test_scrape_signalalign'''
        kmers_t = self.T.scrape_signalalign()
        dict_real = {6998: 2, 6999: 3, 6987: 1}
        dict_test = {6998: len(kmers_t[6998]), 6999: len(kmers_t[6999]), 6987: len(kmers_t[6987])}
        for (k, v), (k2, v2) in zip(dict_real.items(), dict_test.items()):
            self.assertEqual((k, v), (k2, v2))
        pos6998 = kmers_t[6998]
        self.assertListEqual(pos6998, [('CAACA', 0.468349, 3777116), ('CAACC', 0.013211, 3777119)])
        kmers_c = self.C.scrape_signalalign()
        dict_real = {54: 2, 52: 2, 60: 1}
        dict_test = {54: len(kmers_c[54]), 52: len(kmers_c[52]), 60: len(kmers_c[60])}
        for (k, v), (k2, v2) in zip(dict_real.items(), dict_test.items()):
            self.assertEqual((k, v), (k2, v2))
        pos54 = kmers_c[54]
        self.assertListEqual(pos54, [('TGGCG', 0.390422, 3780657), ('GGCGG', 0.096091, 3780656)])
        kmers_categorical = self.CATEGORICAL.scrape_signalalign()
        kmers_deepnano = self.DEEPNANO.scrape_signalalign()
        self.assertEqual(kmers_t, kmers_categorical)
        self.assertEqual(kmers_t, kmers_deepnano)
        self.assertTrue((kmers_deepnano == self.DEEPNANO.kmers))
        self.assertTrue((kmers_t == self.T.kmers))
        self.assertTrue((kmers_c == self.C.kmers))
        self.assertTrue((kmers_categorical == self.CATEGORICAL.kmers))

    def test_create_features(self):
        """test_create_features"""
        features_deepnano = self.DEEPNANO.create_features()
        features_t = self.T.create_features()
        features_c = self.C.create_features()
        features_categorical = self.CATEGORICAL.create_features()
        self.assertNotEqual(features_deepnano, features_t)
        self.assertNotEqual(features_deepnano, features_c)
        self.assertNotEqual(features_deepnano, features_categorical)
        self.assertTrue(np.isclose(features_t, features_categorical).all())
        deepnano_test = [0.30412248, 0.09249048, 0.86009045, 0.0025]
        t_test = [0.0, 0.0, 0.0, 0.0, 0.40229428, 0.18797024, 0.10281436, 0.0, 0.49232142, -0.25025325, -0.11723997,
                  0.0971415]
        c_test = [0.0, 0.0, 0.0, 0.0, 2.26279692, 1.12409658, 2.77720919, 0.0, 2.130275, 1.17222898, 1.01273878,
                  -0.13769097]
        categorical_test = [0.0, 0.0, 0.0, 0.0, 0.40229428, 0.18797024, 0.10281436, 0.0, 0.49232142, -0.25025325,
                            -0.11723997, 0.0971415]
        for i in range(4):
            self.assertAlmostEqual(deepnano_test[i], features_deepnano[0][i])
        for i in range(12):
            self.assertAlmostEqual(t_test[i], features_t[0][i])
            self.assertAlmostEqual(c_test[i], features_c[0][i])
            self.assertAlmostEqual(categorical_test[i], features_categorical[0][i])
        self.assertTrue(np.isclose(features_deepnano, self.DEEPNANO.features).all())
        self.assertTrue(np.isclose(features_t, self.T.features).all())
        self.assertTrue(np.isclose(features_c, self.C.features).all())
        self.assertTrue(np.isclose(features_categorical, self.CATEGORICAL.features).all())
    #
    def test_match_label_with_feature(self):
        """test_match_label_with_feature"""
        final_deepnano = self.DEEPNANO.match_label_with_feature()
        final_t = self.T.match_label_with_feature()
        final_c = self.C.match_label_with_feature()
        final_categorical = self.CATEGORICAL.match_label_with_feature()

        for x in range(len(final_deepnano)):
            self.assertListEqual(list(final_deepnano[x]), list(self.DEEPNANO.training_file[x]))
            self.assertListEqual(list(final_c[x]), list(self.C.training_file[x]))
            self.assertListEqual(list(final_t[x]), list(self.T.training_file[x]))
            self.assertListEqual(list(final_categorical[x]), list(self.CATEGORICAL.training_file[x]))

        self.assertEqual(len(final_deepnano), len(final_t))
        self.assertEqual(len(final_deepnano), len(final_categorical))
        self.assertEqual((15, 2), final_deepnano.shape)

    def test_create_labels(self):
        """test_create_labels"""
        labels_c = self.C.create_labels()
        labels_t = self.T.create_labels()
        labels_deepnano = self.DEEPNANO.create_labels()
        labels_categorical = self.CATEGORICAL.create_labels()
        self.assertTrue((labels_deepnano == self.DEEPNANO.labels))
        self.assertTrue((labels_t == self.T.labels))
        self.assertTrue((labels_c == self.C.labels))
        self.assertTrue((labels_categorical == self.CATEGORICAL.labels))
        self.assertEqual(3125, len(labels_c[7]))
        self.assertEqual(3125, len(labels_t[6998]))
        self.assertEqual(3126, len(labels_categorical[6998]))
        self.assertEqual(21, len(labels_deepnano[6998]))

    def test_create_deepnano_labels(self):
        self.assertRaises(AssertionError, self.T.create_deepnano_labels)
        self.assertRaises(AssertionError, self.C.create_deepnano_labels)
        self.assertRaises(AssertionError, self.CATEGORICAL.create_deepnano_labels)
        labels = self.DEEPNANO.create_deepnano_labels()
        test_6999 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0]
        self.assertListEqual(list(labels[6999]), test_6999)

    def test_scrape_eventalign(self):
        """test_scrape_eventalign"""
        self.assertEqual(self.T.scrape_eventalign(), False)
        self.assertEqual(self.C.scrape_eventalign(), False)

    def test_length(self):
        """test_length"""
        fast5_length = len(self.T.scrape_fast5_events())
        features_length = len(self.T.create_features())
        self.assertEqual(fast5_length, features_length)
        fast5_length = len(self.C.scrape_fast5_events())
        features_length = len(self.C.create_features())
        self.assertEqual(fast5_length, features_length)
        fast5_length = len(self.CATEGORICAL.scrape_fast5_events())
        features_length = len(self.CATEGORICAL.create_features())
        self.assertEqual(fast5_length, features_length)
        fast5_length = len(self.DEEPNANO.scrape_fast5_events())
        features_length = len(self.DEEPNANO.create_features())
        self.assertEqual(fast5_length, features_length)

    def test_create_kmer_labels(self):
        """test_create_kmer_labels"""
        # Template strand
        test_6999 = self.T.create_kmer_labels()[6999]
        labels = os.path.join(self.HOME,
                              "test_files/data_prep_test_files/pos6999-template-label-prob.pkl")
        pos6999 = load_pickle(labels)
        self.assertEqual(sorted(pos6999), sorted(test_6999))
        # Complement strand
        test_55 = self.C.create_kmer_labels()[55]
        labels = os.path.join(self.HOME,
                              "test_files/data_prep_test_files/pos55-complement-label-prob.pkl")
        pos55 = load_pickle(labels)
        self.assertEqual(sorted(pos55), sorted(test_55))
        # DEEPNANO doesnt use kmer_labels
        self.assertRaises(AssertionError, self.DEEPNANO.create_kmer_labels)
        # # CATEGORICAL
        test_6998 = self.CATEGORICAL.create_kmer_labels()[6998]
        labels = os.path.join(self.HOME,
                              "test_files/data_prep_test_files/pos6998-CATEGORICAL-label-prob.pkl")
        pos6998 = load_pickle(labels)
        self.assertEqual(sorted(pos6998), sorted(test_6998))
        # make sure categorical and probability assignments are different
        self.assertNotEqual(self.CATEGORICAL.create_kmer_labels()[6999], test_6999)

    def test_scrape_fast5_events(self):
        row10_real_temp = [83.52333150227787, 221.1755, 1.3181877787337306, 0.0015]
        row10_real_comp = [68.825253092462106, 256.58775000000003, 2.0204697774551619, 0.0052500000000000003]
        row10_temp = list(self.T.scrape_fast5_events()[10])
        row10_comp = list(self.C.scrape_fast5_events()[10])
        row10_categorical = list(self.CATEGORICAL.scrape_fast5_events()[10])
        row10_deepnano = list(self.DEEPNANO.scrape_fast5_events()[10])
        self.assertEqual(row10_temp, row10_real_temp)
        self.assertEqual(row10_comp, row10_real_comp)
        self.assertEqual(row10_categorical, row10_real_temp)
        self.assertEqual(row10_deepnano, row10_real_temp)

    def test_create_null_label(self):
        """test create_null_labels"""
        # deepnano null label
        null = self.DEEPNANO.create_null_label()
        self.assertEqual(null[15], 1)
        self.assertEqual(1, sum(null))
        # categorical vector label for nanonet
        null = self.CATEGORICAL.create_null_label()
        null_len = len(null)
        self.assertEqual(1, sum(null))
        self.assertEqual(null[null_len - 1], 1)
        # probability vector
        null = self.T.create_null_label()
        self.assertAlmostEqual(1, sum(null))
        for prob in null:
            # make sure all values are the same
            self.assertEqual(null[0], prob)

    def test_getkmer_dict(self):
        first = 'AAAAA'
        last = 'TTTTT'
        kmer_dict = self.T.getkmer_dict("ATCGE", 5, flip=False)
        self.assertEqual(0, kmer_dict[first])
        self.assertEqual(3124, kmer_dict[last])
        kmer_dict_flip = self.T.getkmer_dict("ATCGE", 5, flip=True)
        self.assertEqual(first, kmer_dict_flip[0])
        self.assertEqual(last, kmer_dict_flip[3124])

    def test_deepnano_dict(self):
        """Test if deepnano_dictionary creates correct dictionary"""
        # No N in alphabet
        length = 2
        test1 = "AN"
        self.assertRaises(AssertionError, self.DEEPNANO.deepnano_dict, test1, length)
        # No duplicate characters
        test2 = "AAT"
        self.assertRaises(AssertionError, self.DEEPNANO.deepnano_dict, test2, length)
        #
        test3 = "ATGC"
        length = 2
        test_dict = self.DEEPNANO.deepnano_dict(test3, length, flip=False)
        test_dict_flip = self.DEEPNANO.deepnano_dict(test3, length, flip=True)
        self.assertEqual(0, test_dict["AA"])
        self.assertEqual(20, test_dict["TT"])
        self.assertEqual("AA", test_dict_flip[0])
        # make sure none start with "N" except for "NN"
        for key, value in test_dict.items():
            first_n = key.find("N")
            if first_n != -1:
                self.assertTrue(key == key[:first_n] + ("N" * (length - first_n)))

            self.assertTrue(value <= 20)
        # make sure none start with "N" except for "NN"
        for key, value in test_dict_flip.items():
            first_n = value.find("N")
            if first_n != -1:
                self.assertTrue(value == value[:first_n] + ("N" * (length - first_n)))
            self.assertTrue(key <= 20)

    def test_null_vector_deepnano(self):
        """test_null_vector_deepnano"""
        # deepnano null label
        kmer_dict = {"NN": 0}
        null = self.DEEPNANO.null_vector_deepnano(kmer_dict)
        self.assertEqual(null[0], 1)
        self.assertEqual(1, sum(null))

    def test_get_most_probable_kmer(self):
        """test_get_most_probable_kmer"""
        kmer_list = [["KMER", 1, 100], ["KMER2", 1, 101], ["KMER3", .1, 102]]
        best_kmer, prob, position = self.T.get_most_probable_kmer(kmer_list)
        self.assertEqual("KMER2", best_kmer)
        self.assertEqual(1, prob)
        self.assertEqual(101, position)

    def test_create_categorical_vector(self):
        """test_create_categorical_vector"""
        # catch length error
        kmer_dict = {"NN": 0}
        kmer_list = [["4MER", 1, 100], ["4MER", 1, 101], ["4MER", .1, 102]]
        self.assertRaises(AssertionError, self.CATEGORICAL.create_categorical_vector, kmer_list, kmer_dict)
        kmer_list = [["4MER", 1, 100], ["4MER1", 1, 101], ["4MER", .1, 102]]

        self.assertRaises(DataPrepBug, self.CATEGORICAL.create_categorical_vector, kmer_list, kmer_dict)

        # make sure it works
        kmer_dict = {"NNNNN": 0}
        kmer_list = [["NNNNA", 1, 100], ["NNNNN", 1, 101], ["NNBNN", .1, 102]]
        vector = self.CATEGORICAL.create_categorical_vector(kmer_list, kmer_dict)
        self.assertEqual(vector[0], 1)

    def test_create_prob_vector(self):
        """test_create_prob_vector"""
        # test error handling
        kmer_dict = {"4MER": 0}
        kmer_list = [["4MER", 1, 100], ["4MER", 1, 101], ["4MER", .1, 102]]
        self.assertRaises(AssertionError, self.T.create_prob_vector, kmer_list, kmer_dict)
        kmer_list = [["5MER1", 1, 100], ["5MER1", 1, 101], ["5MER1", .1, 102]]
        self.assertRaises(DataPrepBug, self.T.create_prob_vector, kmer_list, kmer_dict)
        # make sure it works
        kmer_dict = {"NNNNN": 0, "NNNNA": 1, "NNBNN": 2}
        kmer_list = [["NNNNA", 0.5, 100], ["NNNNN", 0.5, 101], ["NNBNN", 0.2, 102], \
                     ["NNBNN", 0.0, 102]]

        vector = self.T.create_prob_vector(kmer_list, kmer_dict)
        self.assertAlmostEqual(vector[2], 0.16666666666666669)
        self.assertEqual(len(vector), 3)
        self.assertEqual(sum(vector), 1)

    def test_preproc_event(self):
        """test_preproc_event"""
        mean = 100
        std = 100
        length = 100
        mean, mean2, std, length = self.T.preproc_event(mean, std, length)
        self.assertAlmostEqual(.34, mean)
        self.assertAlmostEqual(.1156, mean2)
        self.assertEqual(99, std)
        self.assertEqual(100, length)

    def test_deepnano_features(self):
        """test_deepnano_features"""
        events = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0)], dtype=[('mean', 'f4'), ('length', 'f4'), ('stdv', 'f4')])
        labels = self.DEEPNANO.deepnano_features(events)
        test_answers = np.array([-0.57135018, 0.32644103, 1.58086051, 2.0])
        np.testing.assert_array_almost_equal(test_answers, labels[0])
        events = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                          dtype=[('something', 'f4'), ('length', 'f4'), ('stdv', 'f4')])
        self.assertRaises(DataPrepBug, self.DEEPNANO.deepnano_features, events)

    def test_create_kmer_vector(self):
        """test_create_kmer_vector"""
        kmer_dict = {"NNNNN": 0, "NNNNA": 1, "NNBNN": 2}
        kmer_list = [["NNNNA", 0.5, 100], ["NNNNN", 0.5, 101], ["NNBNN", 0.2, 102], \
                     ["NNBNN", 0.0, 102]]

        probability = self.T.create_kmer_vector(kmer_list, kmer_dict)
        categorical = self.CATEGORICAL.create_kmer_vector(kmer_list, kmer_dict)
        self.assertAlmostEqual(probability[1], 0.55555555555555558)
        self.assertAlmostEqual(categorical[0], 1)
        self.assertRaises(AssertionError, self.DEEPNANO.create_kmer_vector, kmer_list, kmer_dict)

    def test_create_deepnano_vector(self):
        """test_create_deepnano_vector"""
        kmer_dict = {"NN": 0, "AN": 1, "BN": 2}
        diff = 2
        best_kmer = "AA"
        self.assertRaises(DataPrepBug, self.DEEPNANO.create_deepnano_vector, kmer_dict, diff, best_kmer)
        diff = 2
        best_kmer = "A"
        self.assertRaises(AssertionError, self.DEEPNANO.create_deepnano_vector, kmer_dict, diff, best_kmer)
        diff = 1
        best_kmer = "ATGCB"
        vector = self.DEEPNANO.create_deepnano_vector(kmer_dict, diff, best_kmer)
        self.assertEqual(vector[2], 1)


if __name__ == '__main__':
    unittest.main()
