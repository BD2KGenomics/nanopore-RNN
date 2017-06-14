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
from utils import list_dir

class UtilsTest(unittest.TestCase):
    """Test the functions in utils.py"""

    def test_list_dir(self):
        """Test if list_dir is working"""
        canonical = os.path.abspath(__file__).split("/")[:-2]
        canonical.append("test-files/r9/canonical/")
        canonical = '/'.join(canonical)
        dir_list = \
        ['/Users/andrewbailey/nanopore-RNN/test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch146_read1209_strand1.fast5', \
        '/Users/andrewbailey/nanopore-RNN/test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch176_read15517_strand1.fast5', \
        '/Users/andrewbailey/nanopore-RNN/test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch215_read2042_strand1.fast5',\
        '/Users/andrewbailey/nanopore-RNN/test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch348_read11114_strand.fast5',\
        '/Users/andrewbailey/nanopore-RNN/test-files/r9/canonical/AlexisLucattini_20160918_FNFAD24297_MN19582_sequencing_run_E_COLI_NON_MTHYLTD_R9_77950_ch94_read2151_strand.fast5']
        self.assertEqual(list_dir(canonical), dir_list)
        self.assertEqual(list_dir(canonical, ext="txt"), [])


#TODO test signalalign in path
#TODO test signalalign output is the same as signalalign files


# runSignalAlign -d /Users/andrewbailey/nanopore-RNN/testing/minion-reads/methylated/ -o /Users/andrewbailey/nanopore-RNN/testing/signalalignment_files/methylated/ -r /Users/andrewbailey/nanopore-RNN/testing/reference-sequences/ecoli_k12_mg1655.fa -p /Users/andrewbailey/nanopore-RNN/testing/reference-sequences/CCAGG_modified.bed -t 0.0001 --log_file /Users/andrewbailey/nanopore-RNN/testing/test_log_files/methylated.log.txt --debug
#
#
# runSignalAlign -d /Users/andrewbailey/nanopore-RNN/testing/minion-reads/canonical/ -o /Users/andrewbailey/nanopore-RNN/testing/signalalignment_files/canonical/ -r /Users/andrewbailey/nanopore-RNN/testing/reference-sequences/ecoli_k12_mg1655.fa -t 0.0001 --log_file /Users/andrewbailey/nanopore-RNN/testing/test_log_files/canonical.log.txt --debug


if __name__ == '__main__':
    unittest.main()
