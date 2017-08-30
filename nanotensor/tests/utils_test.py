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
from nanotensor.utils import list_dir, DotDict, check_duplicate_characters, create_time_directory, save_config_file, \
    merge_two_dicts, create_log_file


class UtilsTest(unittest.TestCase):
    """Test the functions in utils.py"""

    @classmethod
    def setUpClass(cls):
        super(UtilsTest, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-3])

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
        test_dict = {"test": 0}
        new_dict = DotDict(test_dict)
        self.assertEqual(test_dict["test"], new_dict.test)

    def test_check_duplicate_characters(self):
        """Make sure that check_duplicate_characters works as expected"""
        test1 = "TEMP"
        test2 = "TEST"
        self.assertEqual(test1, check_duplicate_characters(test1))
        self.assertRaises(AssertionError, check_duplicate_characters, test2)

    def test_create_time_directory(self):
        """Test create_time_directory method"""
        self.assertRaises(AssertionError, create_time_directory, "test")
        log_folder_path = create_time_directory(self.HOME)
        self.assertTrue(os.path.exists(log_folder_path))
        os.rmdir(log_folder_path)

    def test_save_config_file(self):
        """Test save_config_file"""
        self.assertRaises(AssertionError, save_config_file, {"test": 1}, "test")
        self.assertRaises(AssertionError, save_config_file, "test", self.HOME)
        self.assertRaises(AssertionError, save_config_file, ["test", 1], "test")

        config_path = save_config_file({"test": 1}, self.HOME, name="test.config.json")
        self.assertTrue(os.path.exists(config_path))
        os.remove(config_path)

    def test_merge_two_dicts(self):
        """Test merge_two_dicts"""
        self.assertRaises(AssertionError, merge_two_dicts, {"test": 1}, "test")
        self.assertRaises(AssertionError, merge_two_dicts, {"test": 1}, 1)
        self.assertRaises(AssertionError, merge_two_dicts, ["test", 1], {"test": 1})
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        merge_dict = merge_two_dicts(dict1, dict2)
        self.assertEqual(dict1["a"], merge_dict["a"])
        self.assertEqual(dict2["b"], merge_dict["b"])

    def test_create_log_file(self):
        """Test create_log_file"""
        bad_log_file = os.path.join(self.HOME, "test_files/test_log_files/test.paths.log.txt")
        old_log_file = os.path.join(self.HOME, "test_files/test_log_files/canonical.log.txt")
        new_log_path = os.path.join(self.HOME, "test_files/test_log_files/test.canonical.log.txt")
        self.assertRaises(AssertionError, create_log_file, self.HOME, bad_log_file, new_log_path)
        log_path = create_log_file(self.HOME, old_log_file, new_log_path)
        self.assertTrue(os.path.exists(log_path))
        print(log_path)
        os.remove(log_path)

if __name__ == '__main__':
    unittest.main()
