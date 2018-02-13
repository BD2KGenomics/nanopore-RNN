#!/usr/bin/env python
"""Create maximum expected accuracy alignment algorithm for signal align best alignment path"""
########################################################################
# File: mea_algorithm.py
#  executable: mea_algorithm.py
#
# Author: Andrew Bailey
# History: 1/17/18 Created
########################################################################


import sys
import os
import numpy as np
from scipy import sparse
from timeit import default_timer as timer
from nanotensor.fast5 import Fast5
from py3helpers.utils import list_dir
from collections import defaultdict


def maximum_expected_accuracy_alignment(posterior_matrix, shortest_ref_per_event, return_all=False,
                                        sparse_posterior_matrix=None):
    """Computes the maximum expected accuracy alignment along a reference with given events and probabilities

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
    for x in range(int(largest_start_prob)+1):
        event_data = [sparse_posterior_matrix.col[x], sparse_posterior_matrix.row[x],
                      sparse_posterior_matrix.data[x], sparse_posterior_matrix.data[x],  None]
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
        if prev_ref_pos+1 != ref_index and not first_pass:
            fill_gap = True
            gap_indicies = [x for x in range(prev_ref_pos+1, ref_index)]
        # keep track of probabilities to select best connecting edge to new node
        inxs = []
        probs = []
        # event_data = [ref_index, event_index, prob, sum_prob, None]
        for j, forward_edge in enumerate(forward_edges):
            if forward_edge[0] < ref_index:
                # track which probabilities with prev edge
                inxs.append(j)
                probs.append(posterior+forward_edge[3])
                # if needed, keep edges aligned to ref positions previous than the current ref position
                if first_pass and shortest_ref_per_event[event_index] < forward_edge[0]+2:
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
        if probs and max(probs) > posterior:
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


def get_events_from_best_path_sparse(best_path):
    """Recursively grab the reference and event index of the best path from the
    maximum_expected_accuracy_alignment_sparse function.
    :param best_path: output from maximum_expected_accuracy_alignment
    :return: list of events [[ref_pos, event_pos]...]
    """
    events = []

    while best_path[4]:
        ref_pos = best_path[0]
        event_pos = best_path[1]
        events.append([ref_pos, event_pos])
        best_path = best_path[4]

    return events


def get_mea_alignment_path(fast5_path, events=None):
    """Get the maximum expected alignment from a nanopore read fast5 file which has signalalign data

    :param fast5_path: path to fast5 file
    :param events: directly pass events in via a numpy array
    """
    if events is None:
        assert os.path.isfile(fast5_path)
        fileh = Fast5(fast5_path)
        events = fileh.get_signalalign_events()
    # sort events before or min/max args
    ref_start = min(events["reference_index"])
    ref_end = max(events["reference_index"])

    events = np.sort(events, order=['event_index'], kind='mergesort')
    event_start = events["event_index"][0]
    event_end = events["event_index"][-1]

    negative_strand = False
    if events[0]["reference_index"] > events[-1]["reference_index"]:
        negative_strand = True
    ref_length = ref_end - ref_start + 1
    event_length = event_end - event_start + 1

    # print(event_length, ref_length)
    # initialize data structures
    event_matrix = [[0 for _ in range(ref_length)] for _ in range(event_length)]
    posterior_matrix = np.zeros([event_length, ref_length])
    shortest_ref_per_event = [np.inf for _ in range(event_length)]

    max_shortest_ref = np.inf
    # print(ref_start, ref_end)
    # go through events backward to make sure the shortest ref per event is calculated at the same time
    for i in range(1, len(events)+1):
        event = events[-i]
        event_indx = event["event_index"] - event_start
        if negative_strand:
            ref_indx = event["reference_index"] - ref_end
            ref_indx *= -1
        else:
            ref_indx = event["reference_index"] - ref_start

        # using full event so we can use the same matrix to assemble the training data later
        posterior_matrix[event_indx][ref_indx] = event['posterior_probability']
        event_matrix[event_indx][ref_indx] = event
        if shortest_ref_per_event[event_indx] > ref_indx:
            shortest_ref_per_event[event_indx] = max(max_shortest_ref, ref_indx)
            if max_shortest_ref < ref_indx:
                max_shortest_ref = ref_indx

    mea_alignemnt = maximum_expected_accuracy_alignment(posterior_matrix, shortest_ref_per_event)
    # get raw index values from alignment data structure
    best_path = get_events_from_best_path_sparse(mea_alignemnt)
    return best_path


def main():
    """Main docstring"""
    start1 = timer()
    fast5_path = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical"
    files = list_dir(fast5_path, ext='fast5')
    for file in files:
        start = timer()
        path = get_mea_alignment_path(file)
        stop = timer()
        print("Total Time = {} seconds".format(stop - start), file=sys.stderr)

    stop = timer()
    print("Running Time = {} seconds".format(stop - start1), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
