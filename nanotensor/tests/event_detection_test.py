#!/usr/bin/env python
"""
    Place unit tests for event_detection.py
"""
########################################################################
# File: event_detection_test.py
#  executable: event_detection_test.py
# Purpose: event_detection test functions
#
# Author: Andrew Bailey
# History: 12/21/2017 Created
########################################################################
import unittest
import os
import numpy as np
import threading
import time
from nanotensor.fast5 import Fast5
from nanotensor.event_detection import create_speedy_event_table, create_minknow_event_table, \
    create_anchor_kmers, resegment_reads
import unittest


class EventDetectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(EventDetectTests, cls).setUpClass()
        cls.HOME = '/'.join(os.path.abspath(__file__).split("/")[:-3])
        cls.fast5_file = os.path.join(cls.HOME,
                                  "test_files/minion-reads/canonical/miten_PC_20160820_FNFAD20259_MN17223_sequencing_run_AMS_158_R9_WGA_Ecoli_08_20_16_43623_ch100_read104_strand.fast5")
        fast5handle = Fast5(cls.fast5_file, 'r+')
        cls.fast5handle = fast5handle.create_copy("test.fast5")

    def test_create_speedy_event_table(self):
        """Test create_speedy_event_table"""
        sampling_freq = self.fast5handle.sample_rate
        signal = self.fast5handle.get_read(raw=True, scale=True)
        start_time = self.fast5handle.raw_attributes['start_time']

        events = create_speedy_event_table(signal=signal, sampling_freq=sampling_freq, start_time=start_time,
                                           min_width=5, max_width=80, min_gain_per_sample=0.008,
                                           window_width=800)

        events_to_check = np.random.randint(0, len(events), 10)
        for x in events_to_check:
            event = events[x]
            signal_mean = np.mean(signal[event["raw_start"]:event["raw_start"] + event["raw_length"]])
            signal_std = np.std(signal[event["raw_start"]:event["raw_start"] + event["raw_length"]])

            self.assertAlmostEqual(event["mean"], signal_mean)
            self.assertAlmostEqual(event["stdv"], signal_std)
            self.assertAlmostEqual(event['raw_start'], (event['start'] - (start_time / sampling_freq)) * sampling_freq)
            self.assertAlmostEqual(event['raw_length'], event['length'] * sampling_freq)

        with self.assertRaises(TypeError):
            create_speedy_event_table(signal=1, sampling_freq=sampling_freq, start_time=start_time,
                                      min_width=5, max_width=80, min_gain_per_sample=0.008,
                                      window_width=800)
        with self.assertRaises(AssertionError):
            create_speedy_event_table(signal=signal, sampling_freq=sampling_freq, start_time=-120,
                                      min_width=5, max_width=80, min_gain_per_sample=0.008,
                                      window_width=800)

    def test_create_minknow_event_table(self):
        """Test create_minknow_event_table"""
        sampling_freq = self.fast5handle.sample_rate
        signal = self.fast5handle.get_read(raw=True, scale=True)
        start_time = self.fast5handle.raw_attributes['start_time']
        events = create_minknow_event_table(signal=signal, sampling_freq=sampling_freq, start_time=start_time,
                                            window_lengths=(16, 40), thresholds=(8.0, 4.0), peak_height=1)

        events_to_check = np.random.randint(0, len(events), 10)
        for x in events_to_check:
            event = events[x]
            signal_mean = np.mean(signal[event["raw_start"]:event["raw_start"] + event["raw_length"]])
            signal_std = np.std(signal[event["raw_start"]:event["raw_start"] + event["raw_length"]])

            self.assertAlmostEqual(event["mean"], signal_mean)
            self.assertAlmostEqual(event["stdv"], signal_std)
            self.assertAlmostEqual(event['raw_start'], (event['start'] - (start_time / sampling_freq)) * sampling_freq)
            self.assertAlmostEqual(event['raw_length'], event['length'] * sampling_freq)

        with self.assertRaises(TypeError):
            create_minknow_event_table(signal=1, sampling_freq=sampling_freq, start_time=start_time,
                                       window_lengths=(16, 40), thresholds=(8.0, 4.0), peak_height=1)
        with self.assertRaises(AssertionError):
            create_minknow_event_table(signal=signal, sampling_freq=sampling_freq, start_time=-10,
                                       window_lengths=(16, 40), thresholds=(8.0, 4.0), peak_height=1)

    def test_create_anchor_kmers(self):
        """Test create anchor kmers method"""
        sampling_freq = self.fast5handle.sample_rate
        signal = self.fast5handle.get_read(raw=True, scale=True)
        start_time = self.fast5handle.raw_attributes['start_time']
        old_event_table = self.fast5handle.get_basecall_data()

        event_table = create_minknow_event_table(signal=signal, sampling_freq=sampling_freq, start_time=start_time,
                                                 window_lengths=(16, 40), thresholds=(8.0, 4.0), peak_height=1)

        new_event_table = create_anchor_kmers(new_events=event_table, old_events=old_event_table)

        events_to_check = np.random.randint(0, len(new_event_table), 10)
        for x in events_to_check:
            event = new_event_table[x]
            signal_mean = np.mean(signal[event["raw_start"]:event["raw_start"] + event["raw_length"]])
            signal_std = np.std(signal[event["raw_start"]:event["raw_start"] + event["raw_length"]])

            self.assertAlmostEqual(event["mean"], signal_mean)
            self.assertAlmostEqual(event["stdv"], signal_std)
            self.assertAlmostEqual(event['raw_start'], (event['start'] - (start_time / sampling_freq)) * sampling_freq)
            self.assertAlmostEqual(event['raw_length'], event['length'] * sampling_freq)

        with self.assertRaises(TypeError):
            create_anchor_kmers(new_events=1, old_events=old_event_table)
        with self.assertRaises(KeyError):
            mean_table = event_table["mean"]
            create_anchor_kmers(new_events=np.array(range(10)), old_events=old_event_table)
            create_anchor_kmers(new_events=mean_table, old_events=old_event_table)

    def test_resegment_reads(self):
        """Test resegment_reads method"""
        minknow_params = dict(window_lengths=(5, 10), thresholds=(2.0, 1.1), peak_height=1.2)
        speedy_params = dict(min_width=5, max_width=80, min_gain_per_sample=0.008, window_width=800)
        resegment_reads(self.fast5_file, minknow_params, speedy=False, overwrite=True)
        fasthandle = resegment_reads(self.fast5_file, speedy_params, speedy=True, overwrite=True)
        with self.assertRaises(AssertionError):
            resegment_reads("fakepath/path", speedy_params, speedy=True, overwrite=True)
            resegment_reads(self.fast5_file, speedy_params, speedy=True, overwrite=False)
        with self.assertRaises(TypeError):
            resegment_reads(self.fast5_file, speedy_params, speedy=False, overwrite=True)
        fasthandle.delete("Analyses/ReSegmentBasecall_000")

    @classmethod
    def tearDownClass(cls):
        """Remove test fast5 file"""
        os.remove("test.fast5")


if __name__ == '__main__':
    unittest.main()
