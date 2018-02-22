#!/usr/bin/env python
"""Test maximum expected accuracy alignment algorithm"""
########################################################################
# File: mea_algorithm_test.py
#  executable: mea_algorithm_test.py
#
# Author: Andrew Bailey
# History: 1/25/18 Created
########################################################################


import sys
import numpy as np
import unittest
from collections import defaultdict
from scipy import sparse
from nanotensor.mea_algorithm import *


class Mea(unittest.TestCase):
    """Test the functions in mea_algorithm.py"""

    # def test_maximum_expected_accuracy_alignment(self):
    #     """Test maximum_expected_accuracy_alignment function"""
    #     # ref x event (opposite of how we have been thinking about the chart)
    #     posterior_matrix = [[0.2, 0.3, 0.2, 0.2, 0.1],
    #                         [0.2, 0.5, 0.3, 0.0, 0.0],
    #                         [0.3, 0.1, 0.0, 0.3, 0.3],
    #                         [0.0, 0.0, 0.0, 0.4, 0.1],
    #                         [0.0, 0.0, 0.0, 0.2, 0.5],
    #                         ]
    #
    #     # correct input
    #     shortest_ref_per_event = [0, 0, 0, 3, 3]
    #     forward_edges = maximum_expected_accuracy_alignment(posterior_matrix, shortest_ref_per_event, return_all=True)
    #     # trim unnecessary edges
    #     self.assertEqual(3, len(forward_edges))
    #     # 0.3->0.3->0.0->0.0->0.0 = 0.6 Keep horizontal moves through gaps
    #     self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[0]), forward_edges[0][3])
    #     self.assertAlmostEqual(0.6, forward_edges[0][3])
    #     # 0.2->0.5->0.1->0.4->0.2 = 1.1 (Don't count last horizontal move and move from 0.5 to 0.1)
    #     self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[1]) - 0.2 - 0.1, forward_edges[1][3])
    #     self.assertAlmostEqual(1.1, forward_edges[1][3])
    #     # 0.2->0.5->0.1->0.4->0.5 = 1.6 (Don't count horizontal move from 0.5 to 0.1)
    #     self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[2]) - 0.1, forward_edges[2][3])
    #     self.assertAlmostEqual(1.6, forward_edges[2][3])
    #     # test if mae returns most probable edge
    #     most_probable_edge = maximum_expected_accuracy_alignment(posterior_matrix, shortest_ref_per_event)
    #     self.assertEqual(forward_edges[2], most_probable_edge)
    #
    #     # test passing through a sparse matrix
    #     forward_edges = maximum_expected_accuracy_alignment(sparse.coo_matrix(posterior_matrix), shortest_ref_per_event,
    #                                                         sparse_posterior_matrix=True,
    #                                                         return_all=True)
    #     self.assertEqual(3, len(forward_edges))
    #     # 0.3->0.3->0.0->0.0->0.0 = 0.6 Keep horizontal moves through gaps
    #     self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[0]), forward_edges[0][3])
    #     self.assertAlmostEqual(0.6, forward_edges[0][3])
    #     # 0.2->0.5->0.1->0.4->0.2 = 1.1 (Don't count last horizontal move and move from 0.5 to 0.1)
    #     self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[1]) - 0.2 - 0.1, forward_edges[1][3])
    #     self.assertAlmostEqual(1.1, forward_edges[1][3])
    #     # 0.2->0.5->0.1->0.4->0.5 = 1.6 (Don't count horizontal move from 0.5 to 0.1)
    #     self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[2]) - 0.1, forward_edges[2][3])
    #     self.assertAlmostEqual(1.6, forward_edges[2][3])
    #
    #     # incorrect min ref lengths
    #     shortest_ref_per_event = [0, 1, 1, 1, 1]
    #     forward_edges = maximum_expected_accuracy_alignment(posterior_matrix, shortest_ref_per_event, return_all=True)
    #     self.assertEqual(5, len(forward_edges))
    #
    # def test_binary_search_for_edge(self):
    #     """Test binary_search_for_edge"""
    #     forward_edges = [[0, 1, .1, .1], [1, 1, .1, .1], [2, 1, .1, .1], [3, 1, .1, .1], [4, 1, .1, .1],
    #                      [5, 1, .1, .1], [6, 1, .1, .1]]
    #
    #     binary_edge = binary_search_for_edge(forward_edges, 1.1, 1, 0.1)
    #     self.assertEqual([1.1, 1, .1, .2, [1, 1, .1, .1]], binary_edge)
    #     random_indexes = np.random.uniform(0, 7, 10)
    #     for index in random_indexes:
    #         edge = self.slow_search_for_edge(forward_edges, index, 1, 0.1)
    #         binary_edge = binary_search_for_edge(forward_edges, index, 1, 0.1)
    #         self.assertEqual(edge, binary_edge)
    #
    #     with self.assertRaises(AssertionError):
    #         binary_search_for_edge(forward_edges, -1, 1, 0.1)
    #
    # @staticmethod
    # def slow_search_for_edge(forward_edges, ref_index, event_index, posterior):
    #     """Search the forward edges list for best ref index comparison
    #     :param forward_edges: list of forward edges to search
    #     :param ref_index: index to match with forward edges list
    #     :param event_index: information to be passed into new forward edge link
    #     :param posterior: posterior probability of event for a kmer at ref_index
    #     :return: new forward edge
    #     """
    #     assert forward_edges[0][0] <= ref_index, "Ref index cannot be smaller than smallest forward edge"
    #
    #     inxs = []
    #     probs = []
    #     for j, forward_edge in enumerate(forward_edges):
    #         if forward_edge[0] < ref_index:
    #             # track which probabilities with prev edge
    #             inxs.append(j)
    #             probs.append(posterior + forward_edge[3])
    #         elif forward_edge[0] == ref_index:
    #             # stay at reference position
    #             # add probability of event if we want to promote sideways movement
    #             inxs.append(j)
    #             probs.append(forward_edge[3])
    #     # add most probable connecting edge if better than creating an new edge
    #     # deal with multiple edges with equal probabilty
    #     probs = probs[::-1]
    #     inxs = inxs[::-1]
    #     connecting_edge = forward_edges[inxs[int(np.argmax(probs))]]
    #     return [ref_index, event_index, posterior, max(probs), connecting_edge]
    #
    # @staticmethod
    # def sum_forward_edge_accuracy(forward_edge):
    #     """Add up all probabilities from a forward edge"""
    #     sum_prob = 0
    #
    #     def get_probability_info(forward_edge):
    #         """Get event information from maximum_expected_accuracy_alignment"""
    #         nonlocal sum_prob
    #         sum_prob = sum_prob + forward_edge[2]
    #         if forward_edge[4] is not None:
    #             get_probability_info(forward_edge[4])
    #         else:
    #             pass
    #
    #     get_probability_info(forward_edge)
    #
    #     return sum_prob

    def test_mae_random_matrix(self):
        """Create random alignment matrix for mae alignment"""
        for _ in range(20):
            size = 5
            posterior_matrix = np.zeros([size, size])
            shortest_ref_per_event = np.zeros(size)
            shortest_ref = size
            start = np.random.randint(0, 3)
            start = 0
            for x in range(size-1, start-1, -1):
                a = np.random.random(np.random.randint(0, size))
                a /= a.sum()
                # go through events backward to make sure the shortest ref per event is calculated at the same time
                a = np.sort(a)
                print(a)
                for i in a:
                    ref_indx = np.random.randint(0, size)
                    # using full event so we can use the same matrix to assemble the training data later
                    posterior_matrix[x][ref_indx] = i
                    if ref_indx < shortest_ref:
                        shortest_ref_per_event[x] = ref_indx
                        shortest_ref = ref_indx
            # check with naive NxM algorithm
            # posterior_matrix = [[0.2, 0.3, 0.2, 0.2, 0.1],
            #                     [0.2, 0.5, 0.3, 0.0, 0.0],
            #                     [0.3, 0.1, 0.0, 0.3, 0.3],
            #                     [0.0, 0.0, 0.0, 0.4, 0.1],
            #                     [0.0, 0.0, 0.0, 0.2, 0.5],
            #                     ]
#             posterior_matrix = [[ 0.0,          0.0,          0.0,          0.70241264,  0.29758736],
#                                 [ 0.48147804,  0.0,          0.18241497,  0.19156012,  0.0        ],
#                                 [ 0.0,          0.0,          0.0,          1.0,          0.0        ],
#                                 [ 0.0,          1.0,          0.0,          0.0,          0.0        ],
#                                 [ 0.27177382,  0.0,          0.0,          0.48519885,  0.0        ]]
#
# #
#             shortest_ref_per_event = [0, 0, 0, 0, 0]

            # # correct input
            # shortest_ref_per_event = [0, 0, 0, 0, 0]
            most_probable_edge = maximum_expected_accuracy_alignment_edits(posterior_matrix, shortest_ref_per_event)

            another_probable_edge = self.mae_slow(posterior_matrix, shortest_ref_per_event)
            print(posterior_matrix.T)
            print(most_probable_edge)
            print(another_probable_edge)
            # print("len(most_probable_edge)", len(most_probable_edge), "len(another_probable_edge)", len(another_probable_edge))
            # print("MAE EDITS:")
            # for x in most_probable_edge:
            #     print(x)
            # print("MAE SLOW:")
            # for x in another_probable_edge:
            #     print(x)
            self.assertAlmostEqual(most_probable_edge[3], another_probable_edge[3])

            # check total probability and traceback
            ref_indx = most_probable_edge[0]
            event_indx = most_probable_edge[1]
            prob = most_probable_edge[2]
            sum_prob = 0
            total_prob = most_probable_edge[3]
            prev_event = most_probable_edge[4]
            while prev_event:
                # step or stay with reference
                self.assertGreaterEqual(ref_indx, prev_event[0])
                # must step for each event
                self.assertGreater(event_indx, prev_event[1])
                # gather correct probabilities
                if ref_indx != prev_event[0]:
                    sum_prob += prob
                ref_indx = prev_event[0]
                event_indx = prev_event[1]
                prob = prev_event[2]
                prev_event = prev_event[4]
            # include first probability
            sum_prob += prob
            self.assertAlmostEqual(sum_prob, total_prob)

    @staticmethod
    def mae_slow(posterior_matrix, shortest_ref_per_event, return_all=False):
        """Computes the maximum expected accuracy alignment along a reference with given events and probabilities.

        Computes a very slow but thorough search through the matrix

        :param posterior_matrix: matrix of posterior probabilities with reference along x axis and events along y
        :param shortest_ref_per_event: shortest ref position per event
        :param return_all: return all forward edges
        """
        ref_len = len(posterior_matrix[0])
        events_len = len(posterior_matrix)
        ref_index = 0
        initialize = True
        forward_edges = list()
        new_edges = list()
        # step through all events
        for event_index in range(events_len):
            max_prob = 0
            if initialize:
                ref_index = 0
                while ref_index < ref_len:
                    # intitialize forward edges with first event alignments
                    # if type(posterior_matrix[ref_index][event_index]) is not int:
                    posterior = posterior_matrix[event_index][ref_index]
                    event_data = [ref_index, event_index, posterior, posterior, None]
                    if 0 < posterior >= max_prob:
                        # print("True", posterior, max_prob)
                        new_edges.append(event_data)
                        max_prob = posterior
                    ref_index += 1
                print("INITIALIZE", new_edges, max_prob)
                if len(new_edges) != 0:
                    forward_edges = new_edges
                    new_edges = list()
                    initialize = False
            else:
                # print(forward_edges)
                ref_index = 0
                top_edge = []
                while ref_index < ref_len:
                    posterior = posterior_matrix[event_index][ref_index]
                    if posterior >= max_prob:
                        # no possible connecting edges and is needed for other other events create a new one
                        if ref_index < shortest_ref_per_event[event_index]:
                            top_edge.append([ref_index, event_index, posterior, posterior, None])
                            max_prob = posterior
                    ref_index += 1
                # add top edge if needed
                if top_edge:
                    new_edges.append(top_edge[-1])
                print("STARTING MAX PROB", max_prob)
                ref_index = 0
                while ref_index < ref_len:
                    # print("START_NEW_EDGES", ref_index, new_edges)

                    inxs = []
                    probs = []
                    posterior = posterior_matrix[event_index][ref_index]
                    for j, forward_edge in enumerate(forward_edges):
                        if forward_edge[0] < ref_index:
                            # track which probabilities with prev edge
                            inxs.append(j)
                            probs.append(posterior + forward_edge[3])
                            # if needed, keep edges aligned to ref positions previous than the current ref position
                        elif forward_edge[0] == ref_index:
                            # stay at reference position
                            # add probability of event if we want to promote sideways movement
                            inxs.append(j)
                            probs.append(forward_edge[3])

                    # add new edge
                    inxs = inxs[::-1]
                    probs = probs[::-1]
                    if len(probs) != 0:
                        if max(probs) > max_prob and ref_index >= shortest_ref_per_event[event_index]:
                            connecting_edge = forward_edges[inxs[int(np.argmax(probs))]]
                            new_edges.append([ref_index, event_index, posterior, max(probs), connecting_edge])
                            max_prob = max(probs)
                    else:
                        if forward_edges[0][0] > ref_index and posterior > max_prob:
                            new_edges.append([ref_index, event_index, posterior, posterior, None])
                            max_prob = posterior

                    ref_index += 1
                print("END_NEW_EDGES", new_edges)
                forward_edges = new_edges
                new_edges = list()

        # grab and return the highest probability edge
        if return_all:
            return forward_edges
        else:
            highest_prob = 0
            best_forward_edge = 0
            for x in forward_edges:
                if x[3] > highest_prob:
                    highest_prob = x[3]
                    best_forward_edge = x
            return best_forward_edge
    #
    # def test_get_indexes_from_best_path(self):
    #     """test get_get_indexes_from_best_path"""
    #     fake_mea = [4, 4, 1, 0.1, [3, 3, 0.9, 0.1, [2, 2, 0.8, 0.1, [1, 1, 0.7, 0.1, [0, 0, 0.6, 0.6, None]]]]]
    #     alignment = get_indexes_from_best_path(fake_mea)
    #     self.assertEqual(alignment, [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    #     fake_mea = [0.1, [3, 3, 0.9, 0.1, [2, 2, 0.8, 0.1, [1, 1, 0.7, 0.1, [0, 0, 0.6, 0.6, None]]]]]
    #     with self.assertRaises(IndexError):
    #         get_indexes_from_best_path(fake_mea)
    #
    # def test_get_events_from_path(self):
    #     """Test get_events_from_path"""
    #     path = [[0, 0], [1, 1], [2, 2], [3, 3]]
    #     event1 = np.zeros(4, dtype=[('contig', 'S10'), ('reference_index', '<i8'), ('reference_kmer', 'S5'),
    #                                 ('strand', 'S1'),
    #                                 ('event_index', '<i8'), ('event_mean', '<f8'), ('event_noise', '<f8'),
    #                                 ('event_duration', '<f8'), ('aligned_kmer', 'S5'),
    #                                 ('scaled_mean_current', '<f8'), ('scaled_noise', '<f8'),
    #                                 ('posterior_probability', '<f8'), ('descaled_event_mean', '<f8'),
    #                                 ('ont_model_mean', '<f8'), ('path_kmer', 'S5')])
    #     event_matrix = [[event1[0], 0, 0, 0],
    #                     [0, event1[0], 0, 0],
    #                     [0, 0, event1[0], 0],
    #                     [0, 0, 0, event1[0]]]
    #     events = get_events_from_path(event_matrix, path)
    #     self.assertSequenceEqual(events.tolist(), event1.tolist())
    #
    #     with self.assertRaises(TypeError):
    #         event_matrix = [[0, 0, 0, 0],
    #                         [0, 0, 0, 0],
    #                         [0, 0, 0, 0],
    #                         [0, 0, 0, 0]]
    #
    #         get_events_from_path(event_matrix, path)
    #
    # def test_get_mea_params_from_events(self):
    #     """Test get_mea_params_from_events"""
    #     events = np.zeros(4, dtype=[('contig', 'S10'), ('reference_index', '<i8'), ('reference_kmer', 'S5'),
    #                                 ('strand', 'S1'),
    #                                 ('event_index', '<i8'), ('event_mean', '<f8'), ('event_noise', '<f8'),
    #                                 ('event_duration', '<f8'), ('aligned_kmer', 'S5'),
    #                                 ('scaled_mean_current', '<f8'), ('scaled_noise', '<f8'),
    #                                 ('posterior_probability', '<f8'), ('descaled_event_mean', '<f8'),
    #                                 ('ont_model_mean', '<f8'), ('path_kmer', 'S5')])
    #     events["posterior_probability"] = [0.1, 0.2, 0.3, 0.4]
    #     events["event_index"] = [0, 1, 2, 3]
    #     events["reference_index"] = [0, 1, 2, 3]
    #     test_event_matrix = [[events[0], 0, 0, 0],
    #                          [0, events[1], 0, 0],
    #                          [0, 0, events[2], 0],
    #                          [0, 0, 0, events[3]]]
    #
    #     posterior_matrix, shortest_ref, event_matrix = get_mea_params_from_events(events)
    #     self.assertSequenceEqual(posterior_matrix.tolist(), [[0.1, 0, 0, 0],
    #                                                          [0, 0.2, 0, 0],
    #                                                          [0, 0, 0.3, 0],
    #                                                          [0, 0, 0, 0.4]])
    #     self.assertSequenceEqual(shortest_ref, [0, 1, 2, 3])
    #     self.assertSequenceEqual(event_matrix, test_event_matrix)
    #     with self.assertRaises(KeyError):
    #         events = np.zeros(4, dtype=[('reference_index', '<i8'), ('reference_kmer', 'S5'),
    #                                     ('strand', 'S1'),
    #                                     ('event_index', '<i8'), ('event_mean', '<f8'), ('event_noise', '<f8'),
    #                                     ('event_duration', '<f8'), ('aligned_kmer', 'S5'),
    #                                     ('scaled_mean_current', '<f8'), ('scaled_noise', '<f8'),
    #                                     ('posterior_probability', '<f8'), ('descaled_event_mean', '<f8'),
    #                                     ('ont_model_mean', '<f8'), ('path_kmer', 'S5')])
    #         get_mea_params_from_events(events)

    # def test_match_events_with_mea(self):
    #     """Test match_events_with_mea"""
    #


if __name__ == '__main__':
    unittest.main()
