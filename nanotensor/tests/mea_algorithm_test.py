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
from py3helpers.utils import time_it


class Mea(unittest.TestCase):
    """Test the functions in mea_algorithm.py"""

    def test_maximum_expected_accuracy_alignment(self):
        """Test maximum_expected_accuracy_alignment function"""
        # ref x event
        posterior_matrix = [[0.2, 0.2, 0.3, 0.0, 0.0],
                            [0.3, 0.5, 0.1, 0.0, 0.0],
                            [0.2, 0.3, 0.0, 0.0, 0.0],
                            [0.2, 0.0, 0.3, 0.4, 0.2],
                            [0.1, 0.0, 0.3, 0.1, 0.5]]

        posterior_matrix = np.asanyarray(posterior_matrix).T
        # correct input
        shortest_ref_per_event = [0, 0, 0, 3, 3]
        forward_edges = maximum_expected_accuracy_alignment(posterior_matrix, shortest_ref_per_event, return_all=True)
        # trim unnecessary edges
        self.assertEqual(3, len(forward_edges))
        # 0.2->0.5->0.1 = 0.7 dont count horizontal move through 0.1
        self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[0]) - 0.1, forward_edges[0][3])
        self.assertAlmostEqual(0.7, forward_edges[0][3])
        # 0.2->0.5->0.1->0.4->0.2 = 1.1 (Don't count last horizontal move and move from 0.4 to 0.2)
        self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[1]) - 0.2 - 0.1, forward_edges[1][3])
        self.assertAlmostEqual(1.1, forward_edges[1][3])
        # 0.2->0.5->0.1->0.4->0.5 = 1.6 (Don't count horizontal move from 0.5 to 0.1)
        self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[2]) - 0.1, forward_edges[2][3])
        self.assertAlmostEqual(1.6, forward_edges[2][3])

        # test passing through a sparse matrix
        forward_edges = maximum_expected_accuracy_alignment(sparse.coo_matrix(posterior_matrix), shortest_ref_per_event,
                                                            sparse_posterior_matrix=True,
                                                            return_all=True)
        self.assertEqual(3, len(forward_edges))
        # 0.2->0.5->0.1 = 0.7 dont count horizontal move through 0.1
        self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[0]) - 0.1, forward_edges[0][3])
        self.assertAlmostEqual(0.7, forward_edges[0][3])
        # 0.2->0.5->0.1->0.4->0.2 = 1.1 (Don't count last horizontal move and move from 0.4 to 0.2)
        self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[1]) - 0.2 - 0.1, forward_edges[1][3])
        self.assertAlmostEqual(1.1, forward_edges[1][3])
        # 0.2->0.5->0.1->0.4->0.5 = 1.6 (Don't count horizontal move from 0.5 to 0.1)
        self.assertAlmostEqual(self.sum_forward_edge_accuracy(forward_edges[2]) - 0.1, forward_edges[2][3])
        self.assertAlmostEqual(1.6, forward_edges[2][3])

        # incorrect min ref lengths
        shortest_ref_per_event = [0, 1, 1, 1, 1]
        forward_edges = maximum_expected_accuracy_alignment(posterior_matrix, shortest_ref_per_event, return_all=True)
        self.assertEqual(4, len(forward_edges))

    def test_binary_search_for_edge(self):
        """Test binary_search_for_edge"""
        forward_edges = [[0, 1, .1, .1], [1, 1, .1, .1], [2, 1, .1, .1], [3, 1, .1, .1], [4, 1, .1, .1],
                         [5, 1, .1, .1], [6, 1, .1, .1]]

        binary_edge = binary_search_for_edge(forward_edges, 1.1, 1, 0.1)
        self.assertEqual([1.1, 1, .1, .2, [1, 1, .1, .1]], binary_edge)
        random_indexes = np.random.uniform(0, 7, 10)
        for index in random_indexes:
            edge = self.slow_search_for_edge(forward_edges, index, 1, 0.1)
            binary_edge = binary_search_for_edge(forward_edges, index, 1, 0.1)
            self.assertEqual(edge, binary_edge)

        with self.assertRaises(AssertionError):
            binary_search_for_edge(forward_edges, -1, 1, 0.1)

    @staticmethod
    def slow_search_for_edge(forward_edges, ref_index, event_index, posterior):
        """Search the forward edges list for best ref index comparison
        :param forward_edges: list of forward edges to search
        :param ref_index: index to match with forward edges list
        :param event_index: information to be passed into new forward edge link
        :param posterior: posterior probability of event for a kmer at ref_index
        :return: new forward edge
        """
        assert forward_edges[0][0] <= ref_index, "Ref index cannot be smaller than smallest forward edge"

        inxs = []
        probs = []
        for j, forward_edge in enumerate(forward_edges):
            if forward_edge[0] < ref_index:
                # track which probabilities with prev edge
                inxs.append(j)
                probs.append(posterior + forward_edge[3])
            elif forward_edge[0] == ref_index:
                # stay at reference position
                # add probability of event if we want to promote sideways movement
                inxs.append(j)
                probs.append(forward_edge[3])
        # add most probable connecting edge if better than creating an new edge
        # deal with multiple edges with equal probabilty
        probs = probs[::-1]
        inxs = inxs[::-1]
        connecting_edge = forward_edges[inxs[int(np.argmax(probs))]]
        return [ref_index, event_index, posterior, max(probs), connecting_edge]

    @staticmethod
    def sum_forward_edge_accuracy(forward_edge):
        """Add up all probabilities from a forward edge"""
        sum_prob = 0

        def get_probability_info(forward_edge):
            """Get event information from maximum_expected_accuracy_alignment"""
            nonlocal sum_prob
            sum_prob = sum_prob + forward_edge[2]
            if forward_edge[4] is not None:
                get_probability_info(forward_edge[4])
            else:
                pass

        get_probability_info(forward_edge)

        return sum_prob

    def test_mae_random_matrix(self):
        """Create random alignment matrix for mae alignment"""
        max_size = 40
        for j in range(20):
            row, col = np.random.randint(int(max_size/2), max_size, 2)
            # create random inputs
            posterior_matrix, shortest_ref_per_event = self.create_random_prob_matrix(row=row, col=col)
            # test 3 implementations
            most_probable_edge = maximum_expected_accuracy_alignment(posterior_matrix, shortest_ref_per_event)
            yet_another_implementation = self.maximum_expected_accuracy_alignment2(posterior_matrix, shortest_ref_per_event)
            another_probable_edge = self.mae_slow(posterior_matrix, shortest_ref_per_event)

            self.assertAlmostEqual(most_probable_edge[3], another_probable_edge[3])
            self.assertAlmostEqual(yet_another_implementation[3], most_probable_edge[3])

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

    def create_random_prob_matrix(self, row=None, col=None, gaps=True):
        """Create a matrix of random probability distributions along each row

        :param row: number of rows
        :param col: number of columns
        :param gaps: if have start gap, middle gaps or end gap
        """
        assert row is not None, "Must set row option"
        assert col is not None, "Must set col option"
        prob_matrix = np.zeros([row, col])
        shortest_future_col_per_row = np.zeros(row)
        shortest_col = col
        start = np.random.randint(0, 3)
        skip = 0
        if not gaps:
            start = 0
            skip = 1
        col_indexes = [x for x in range(col)]
        for row_i in range(row-1, start-1, -1):
            a = np.random.random(np.random.randint(skip, col))
            a /= a.sum()
            # go through events backward to make sure the shortest ref per event is calculated at the same time
            a = np.sort(a)
            np.random.shuffle(col_indexes)
            # make sure we dont have gaps at ends of columns
            if not gaps and row_i == row-1:
                col_indexes.remove(col-1)
                col_indexes.insert(0, col-1)
            if not gaps and row_i == 0:
                col_indexes.remove(0)
                col_indexes.insert(0, 0)

            for prob, col_i in zip(a, col_indexes):
                # using full event so we can use the same matrix to assemble the training data later
                prob_matrix[row_i][col_i] = prob
                if col_i <= shortest_col:
                    shortest_col = col_i
            shortest_future_col_per_row[row_i] = shortest_col
        # check with naive NxM algorithm
        prob_matrix = np.asanyarray(prob_matrix)
        # print(prob_matrix)
        # print(prob_matrix.T)
        #
        # print(shortest_future_col_per_row)
        assert self.matrix_event_length_pairs_test(prob_matrix, shortest_future_col_per_row), \
            "Did not create accurate prob matrix and shortest_future_col_per_row"
        return prob_matrix, shortest_future_col_per_row

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
                # print("INITIALIZE", new_edges, max_prob)
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
                ref_index = 0
                while ref_index < ref_len:
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
                        if max(probs) > max_prob:
                            connecting_edge = forward_edges[inxs[int(np.argmax(probs))]]
                            new_edges.append([ref_index, event_index, posterior, max(probs), connecting_edge])
                            max_prob = max(probs)
                    else:
                        if forward_edges[0][0] > ref_index and posterior > max_prob:
                            new_edges.append([ref_index, event_index, posterior, posterior, None])
                            max_prob = posterior
                    ref_index += 1
                # print("END_NEW_EDGES", new_edges)
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

    @staticmethod
    def maximum_expected_accuracy_alignment2(posterior_matrix, shortest_ref_per_event, return_all=False,
                                             sparse_posterior_matrix=None):
        """Computes the maximum expected accuracy alignment along a reference with given events and probabilities

        NOTE: Slower than other version

        :param posterior_matrix: matrix of posterior probabilities with reference along x axis (col) and
                                events along y axis (row).
        :param shortest_ref_per_event: list of the highest possible reference position for all future events at a
                                        given index
        :param return_all: option to return all the paths through the matrix
        :param sparse_posterior_matrix: bool sparse matrix option

        :return best_path: a nested list of lists-> last_event = [ref_index, event_index, prob, sum_prob, [prev_event]]
        """
        # optional convert to sparse matrix
        if sparse_posterior_matrix:
            sparse_posterior_matrix = posterior_matrix
        else:
            sparse_posterior_matrix = sparse.coo_matrix(posterior_matrix)

        forward_edges = list()
        # get the index of the largest probability for the first event
        smallest_event = min(sparse_posterior_matrix.row)
        first_events = sparse_posterior_matrix.row == smallest_event
        num_first_event = sum(first_events)

        largest_start_prob = np.argmax(sparse_posterior_matrix.data[first_events])
        # gather leading edges for all references above max
        for x in range(int(largest_start_prob) + 1):
            event_data = [sparse_posterior_matrix.col[x], sparse_posterior_matrix.row[x],
                          sparse_posterior_matrix.data[x], sparse_posterior_matrix.data[x], None]
            forward_edges.append(event_data)
        # number of values for first event
        prev_event = sparse_posterior_matrix.row[num_first_event]
        new_edges = list()
        first_pass = True
        prev_ref_pos = 0
        fill_gap = False
        # go through rest of events
        num_events = len(sparse_posterior_matrix.row)
        for i in range(num_first_event, num_events):
            event_index = sparse_posterior_matrix.row[i]
            posterior = sparse_posterior_matrix.data[i]
            ref_index = sparse_posterior_matrix.col[i]
            # update forward edges if new event
            if prev_event != event_index:
                prev_event = event_index
                # capture edges that are further along than the current event
                for forward_edge in forward_edges:
                    if forward_edge[0] > prev_ref_pos:
                        new_edges.append(forward_edge)
                forward_edges = new_edges
                new_edges = list()
                first_pass = True
            # check if there is a gap between reference
            if prev_ref_pos + 1 != ref_index and not first_pass:
                fill_gap = True
                gap_indicies = [x for x in range(prev_ref_pos + 1, ref_index)]
            # keep track of probabilities to select best connecting edge to new node
            inxs = []
            probs = []
            # event_data = [ref_index, event_index, prob, sum_prob, None]
            for j, forward_edge in enumerate(forward_edges):
                if forward_edge[0] < ref_index:
                    # track which probabilities with prev edge
                    inxs.append(j)
                    probs.append(posterior + forward_edge[3])
                    # if needed, keep edges aligned to ref positions previous than the current ref position
                    if first_pass and shortest_ref_per_event[event_index] < forward_edge[0] + 2:
                        new_edges.append(forward_edge)
                elif forward_edge[0] == ref_index:
                    # stay at reference position
                    # add probability of event if we want to promote sideways movement
                    inxs.append(j)
                    probs.append(forward_edge[3])
                if fill_gap:
                    if forward_edge[0] in gap_indicies:
                        # add edges that pass through gaps in the called events
                        new_edges.append(forward_edge)
            # add most probable connecting edge if better than creating an new edge
            if probs:
                connecting_edge = forward_edges[inxs[int(np.argmax(probs))]]
                new_edges.append([ref_index, event_index, posterior, max(probs), connecting_edge])
            else:
                # no possible connecting edges or connecting edges decrease probability, create a new one
                new_edges.append([ref_index, event_index, posterior, posterior, None])

            # reset trackers
            first_pass = False
            prev_ref_pos = ref_index
            fill_gap = False

        # add back last edges which may not have been connected
        for forward_edge in forward_edges:
            if forward_edge[0] > prev_ref_pos:
                new_edges.append(forward_edge)
        forward_edges = new_edges
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

    def test_get_indexes_from_best_path(self):
        """test get_get_indexes_from_best_path"""
        fake_mea = [4, 4, 1, 0.1, [3, 3, 0.9, 0.1, [2, 2, 0.8, 0.1, [1, 1, 0.7, 0.1, [0, 0, 0.6, 0.6, None]]]]]
        alignment = get_indexes_from_best_path(fake_mea)
        self.assertEqual(alignment, [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        fake_mea = [0.1, [3, 3, 0.9, 0.1, [2, 2, 0.8, 0.1, [1, 1, 0.7, 0.1, [0, 0, 0.6, 0.6, None]]]]]
        with self.assertRaises(IndexError):
            get_indexes_from_best_path(fake_mea)

    def test_get_events_from_path(self):
        """Test get_events_from_path"""
        path = [[0, 0], [1, 1], [2, 2], [3, 3]]
        event1 = np.zeros(4, dtype=[('contig', 'S10'), ('reference_index', '<i8'), ('reference_kmer', 'S5'),
                                    ('strand', 'S1'),
                                    ('event_index', '<i8'), ('event_mean', '<f8'), ('event_noise', '<f8'),
                                    ('event_duration', '<f8'), ('aligned_kmer', 'S5'),
                                    ('scaled_mean_current', '<f8'), ('scaled_noise', '<f8'),
                                    ('posterior_probability', '<f8'), ('descaled_event_mean', '<f8'),
                                    ('ont_model_mean', '<f8'), ('path_kmer', 'S5')])
        event_matrix = [[event1[0], 0, 0, 0],
                        [0, event1[0], 0, 0],
                        [0, 0, event1[0], 0],
                        [0, 0, 0, event1[0]]]
        events = get_events_from_path(event_matrix, path)
        self.assertSequenceEqual(events.tolist(), event1.tolist())

        with self.assertRaises(TypeError):
            event_matrix = [[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]

            get_events_from_path(event_matrix, path)

    def test_get_mea_params_from_events(self):
        """Test get_mea_params_from_events"""
        for _ in range(20):
            max_size = 20
            row, col = np.random.randint(int(max_size/2), max_size, 2)
            # create random probability matrix
            true_posterior_matrix, true_shortest_ref_per_event = self.create_random_prob_matrix(row=row, col=col,
                                                                                                gaps=False)
            # print(true_posterior_matrix.T)
            # print(true_shortest_ref_per_event)
            # generate events and event matrix to match
            events, true_event_matrix = self.generate_events_from_probability_matrix(true_posterior_matrix)
            # get events from random signal align output
            posterior_matrix, shortest_ref, event_matrix = get_mea_params_from_events(events)
            self.assertSequenceEqual(posterior_matrix.tolist(), true_posterior_matrix.tolist())
            self.assertSequenceEqual(shortest_ref, true_shortest_ref_per_event.tolist())
            self.assertSequenceEqual(event_matrix, true_event_matrix)
        with self.assertRaises(KeyError):
            events = np.zeros(4, dtype=[('reference_index', '<i8'), ('reference_kmer', 'S5'),
                                        ('strand', 'S1'),
                                        ('event_index', '<i8'), ('event_mean', '<f8'), ('event_noise', '<f8'),
                                        ('event_duration', '<f8'), ('aligned_kmer', 'S5'),
                                        ('scaled_mean_current', '<f8'), ('scaled_noise', '<f8'),
                                        ('posterior_probability', '<f8'), ('descaled_event_mean', '<f8'),
                                        ('ont_model_mean', '<f8'), ('path_kmer', 'S5')])
            get_mea_params_from_events(events)

    @staticmethod
    def matrix_event_length_pairs_test(posterior_matrix, shortest_events):
        """Test if the shortest events list matches what is in the posterior matrix"""
        # posterior matrix is events x ref
        current_min = np.inf
        shortest_events = shortest_events[::-1]
        for i, row in enumerate(posterior_matrix[::-1]):
            if sum(row) > 0:
                min_event_in_row = min(np.nonzero(row)[0])
                if min_event_in_row < current_min:
                    current_min = min_event_in_row
                if shortest_events[i] != current_min:
                    return False
        return True

    @staticmethod
    def generate_events_from_probability_matrix(matrix):
        """Create events from probability matrix for testing get_mea_params_from_events"""
        event_indexs = []
        probs = []
        ref_indexs = []

        for row_index, row in enumerate(matrix):
            for col_index, prob in enumerate(row):
                if prob != 0:
                    probs.append(prob)
                    event_indexs.append(row_index)
                    ref_indexs.append(col_index)

        ref_indexs = np.asanyarray(ref_indexs)
        event_indexs = np.asanyarray(event_indexs)
        # create events table
        n_events = len(probs)
        events = np.zeros(n_events, dtype=[('contig', 'S10'), ('reference_index', '<i8'), ('reference_kmer', 'S5'),
                                           ('strand', 'S1'),
                                           ('event_index', '<i8'), ('event_mean', '<f8'), ('event_noise', '<f8'),
                                           ('event_duration', '<f8'), ('aligned_kmer', 'S5'),
                                           ('scaled_mean_current', '<f8'), ('scaled_noise', '<f8'),
                                           ('posterior_probability', '<f8'), ('descaled_event_mean', '<f8'),
                                           ('ont_model_mean', '<f8'), ('path_kmer', 'S5')])

        #  add to ref and event starts
        ref_start = np.random.randint(0, 10000)
        event_start = np.random.randint(0, 10000)
        # decide if reverse strand or not
        # minus_strand = False
        # if ref_indexs[0] > ref_indexs[-1]:
        #     minus_strand = True
        # print(event_indexs)
        # print(ref_indexs)
        # print("minus_strand", minus_strand)
        # if minus_strand:
        #     max_start = max(ref_indexs)
        #     ref_indexs *= -1
        #     ref_indexs += max_start

        events["reference_index"] = ref_indexs + ref_start
        events["posterior_probability"] = probs
        events["event_index"] = event_indexs + event_start

        # create comparison event matrix
        event_matrix = np.zeros(matrix.shape).tolist()

        for i in range(n_events):
            event_matrix[event_indexs[i]][ref_indexs[i]] = events[i]

        return events, event_matrix

    # def test_match_events_with_mea(self):
    #     """Test match_events_with_mea"""



if __name__ == '__main__':
    unittest.main()
