#!/usr/bin/env python
"""
Place unit tests in this file
"""
########################################################################
# File: tests.py
#  executable: tests.py
# Purpose: test functions
#
# Author: Andrew Bailey
# History: 5/17/2017 Created
########################################################################

import os
import unittest
from nanotensor.utils import list_dir, DotDict, check_duplicate_characters


class UtilsTest(unittest.TestCase):
    """Test the functions in utils.py"""

    def test_list_dir(self):
        """Test if list_dir is working"""
        canonical = '/'.join(os.path.abspath(__file__).split("/")[:-3])
        canonical = os.path.join(canonical, "test_files/reference-sequences/")
        expected = ["CCAGG_modified.bed", "ecoli_k12_mg1655.fa", "ecoli_k12_mg1655_modified.fa"]
        expected_files = sorted([os.path.join(canonical, x) for x in expected])
        self.assertEqual(sorted(list_dir(canonical)), expected_files)
        self.assertEqual(list_dir(canonical, ext="bed"), expected_files[0:1])

    def test_dot_dict(self):
        """Test DotDict class"""
        test_dict = {"asdf": 0}
        new_dict = DotDict(test_dict)
        self.assertEqual(test_dict["asdf"], new_dict.asdf)

    def test_check_duplicate_characters(self):
        """Make sure that check_duplicate_characters works as expected"""
        test1 = "ABCD"
        test2 = "AABCD"
        self.assertEqual(test1, check_duplicate_characters(test1))
        self.assertRaises(AssertionError, check_duplicate_characters, test2)



if __name__ == '__main__':
    unittest.main()
