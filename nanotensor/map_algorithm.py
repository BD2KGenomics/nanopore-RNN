#!/usr/bin/env python
"""Create map algorithm for signal align best alignment path"""
########################################################################
# File: mea_algorithm.py
#  executable: map_algorithm.py
#
# Author: Andrew Bailey
# History: 1/17/18 Created
########################################################################


import sys
import numpy as np
from scipy import sparse
from collections import defaultdict
from timeit import default_timer as timer
from nanotensor.fast5 import Fast5
from py3helpers.utils import list_dir


def maximum_expected_accuracy_alignment(posterior_matrix):
    """Computes the maximum expected accuracy alignment along a reference with given events and probabilities

    :param posterior_matrix: matrix of posterior probabilities with reference along y axis and events along x.
    :return best_path: a matrix with the first column as the reference position and second column is event index
    """
    # access previous edges based on reference position
    forward_edges = defaultdict()
    ref_len = len(posterior_matrix)
    events_len = len(posterior_matrix[0])
    ref_index = 0
    most_prob = 0
    best_ref = 0
    initialize = True
    # step through all events
    for event_index in range(events_len):
        if initialize:
            while ref_index < ref_len:
                # intitialize forward edges with first event alignments
                # if type(posterior_matrix[ref_index][event_index]) is not int:
                prob = posterior_matrix[ref_index][event_index]
                event_data = [prob, ref_index, event_index, None]
                forward_edges[ref_index] = [prob, event_data]
                ref_index += 1
            initialize = False
        else:
            # connect with most probable previous node
            new_edges = defaultdict()
            ref_index = 0
            step_prob = 0
            while ref_index < ref_len:
                prob = posterior_matrix[ref_index][event_index]
                stay_prob = forward_edges[ref_index][0]
                if ref_index != 0:
                    step_prob = forward_edges[ref_index-1][0]
                # Find best connection from previous node
                if stay_prob < step_prob:
                    new_prob = prob+step_prob
                    event_data = [new_prob, ref_index, event_index, forward_edges[ref_index-1][1]]
                    new_edges[ref_index] = [new_prob, event_data]
                else:
                    new_prob = prob+stay_prob
                    event_data = [new_prob, ref_index, event_index, forward_edges[ref_index][1]]
                    new_edges[ref_index] = [new_prob, event_data]

                # keep track of best in order to easily return
                if new_prob > most_prob:
                    most_prob = new_prob
                    best_ref = ref_index
                ref_index += 1
            # create new forward edges
            forward_edges = new_edges

    return forward_edges[best_ref][1]


def maximum_expected_accuracy_alignment_sparse(sparse_posterior_matrix):
    """Computes the maximum expected accuracy alignment along a reference with given events and probabilities

    :param posterior_matrix: scipy.sparse matrix of posterior probabilities with reference along x axis (col) and
                            events along y axis (row).
    :return best_path: a matrix with the first column as the reference position and second column is event index
    """
    assert sparse.issparse(sparse_posterior_matrix), "Sparse matrix must be of the scipy.sparse format"

    forward_edges = list()
    # get the index of the largest probability for the first event
    largest_start_prob = np.argmax(sparse_posterior_matrix.data[sparse_posterior_matrix.row == 0])
    # gather leading edges for all references above max
    for x in range(int(largest_start_prob)+1):
        event_data = [sparse_posterior_matrix.col[x], sparse_posterior_matrix.row[x],
                      sparse_posterior_matrix.data[x], sparse_posterior_matrix.data[x],  None]
        forward_edges.append(event_data)
    # number of values for first event
    num_first_event = sum(sparse_posterior_matrix.row == 0)
    prev_event = 1
    new_edges = list()
    first_pass = True
    # max_prob = 0
    # max_prob_indx = 0
    # go through rest of events
    for i, event_index in enumerate(sparse_posterior_matrix.row[num_first_event:]):
        posterior = sparse_posterior_matrix.data[i]
        ref_index = sparse_posterior_matrix.col[i]
        # update forward edges if new event
        if prev_event != event_index:
            prev_event = event_index
            forward_edges = new_edges
            new_edges = list()
            first_pass = True
        # keep track of probabilities to select best connecting edge to new node
        inxs = []
        probs = []
        # event_data = [ref_index, event_index, prob, sum_prob, None]
        for j, forward_edge in enumerate(forward_edges):
            if forward_edge[0] < ref_index:
                # track which probabilities with prev edge
                inxs.append(j)
                probs.append(posterior+forward_edge[3])
                if first_pass:
                    # add continuing edges
                    new_edges.append(forward_edge)
            elif forward_edge[0] == ref_index:
                inxs.append(j)
                # this is where we would decide if a vertical move is ok.
                probs.append(forward_edge[3])
            else:
                pass
                # new, earlier reference position
                # print("This happens", forward_edge[0], ref_index)
                # new_edges.append([ref_index, event_index, posterior, posterior, None])
        if len(probs) != 0:
            connecting_edge = forward_edges[inxs[int(np.argmax(probs))]]
            new_edges.append([ref_index, event_index, posterior, max(probs), connecting_edge])
        first_pass = False

    return forward_edges


def get_events_from_best_path_sparse(best_path):
    """Recursively grab the reference and event index of the best path from the maximum_expected_accuracy_alignment
    function.
    :param best_path: output from maximum_expected_accuracy_alignment
    """
    events = []

    def get_event_info(best_path):
        """Get event information from """
        ref_pos = best_path[0]
        event_pos = best_path[1]
        events.append([ref_pos, event_pos])
        if best_path[4] is not None:
            get_event_info(best_path[4])
        else:
            pass

    get_event_info(best_path)

    return events



def get_events_from_best_path(best_path):
    """Recursively grab the reference and event index of the best path from the maximum_expected_accuracy_alignment
    function.
    :param best_path: output from maximum_expected_accuracy_alignment
    """
    events = []

    def get_event_info(best_path):
        """Get event information from """
        ref_pos = best_path[1]
        event_pos = best_path[2]
        events.append([ref_pos, event_pos])
        if best_path[3] is not None:
            get_event_info(best_path[3])
        else:
            pass

    get_event_info(best_path)

    return events


def main():
    """Main docstring"""
    start = timer()
    fast5_path = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/canonical"
    files = list_dir(fast5_path, ext='fast5')

    fileh = Fast5(files[0])
    events = fileh.get_signalalign_events()
    # sort before or min/max args
    # events = np.sort(events, order=['event_index', 'reference_index'])
    # # print(events[:10])
    # ref_start = events["reference_index"][0]
    # ref_end = events["reference_index"][-1]
    # event_start = events["event_index"][0]
    # event_end = events["event_index"][-1]

    ref_start = min(events["reference_index"])
    ref_end = max(events["reference_index"])
    event_start = min(events["event_index"])
    event_end = max(events["event_index"])

    ref_length = ref_end - ref_start + 1
    event_length = event_end - event_start + 1
    print(ref_length, event_length)
    # posterior_matrix = [[0 for x in range(event_length)] for x in range(ref_length)]
    # # print(posterior_matrix[ref_length - 1][event_length - 1])
    # event_matrix = [[0 for x in range(event_length)] for x in range(ref_length)]
    posterior_matrix = [[0 for x in range(ref_length)] for x in range(event_length)]
    # print(posterior_matrix[ref_length - 1][event_length - 1])
    event_matrix = [[0 for x in range(ref_length)] for x in range(event_length)]

    for event in events:
        ref_indx = event["reference_index"] - ref_start
        event_indx = event["event_index"] - event_start
        # using full event so we can use the same matrix to assemble the training data later
        # posterior_matrix[ref_indx][event_indx] = event['posterior_probability']
        # event_matrix[ref_indx][event_indx] = event
        posterior_matrix[event_indx][ref_indx] = event['posterior_probability']
        event_matrix[event_indx][ref_indx] = event

    sparse_matrix = sparse.coo_matrix(posterior_matrix)
    mea_alignemnt = maximum_expected_accuracy_alignment_sparse(sparse_matrix)
    # print(mea_alignemnt)
    print(mea_alignemnt[-1][3])
    probs = []
    for x in mea_alignemnt:
        probs.append(x[3])
    print(np.mean(probs))
    print(np.std(probs))
    print(max(probs))

# best_path = get_events_from_best_path_sparse(mea_alignemnt)
    # print(best_path)
    # best_path = maximum_expected_accuracy_alignment(posterior_matrix)
    # events = get_events_from_best_path(best_path)


    # # print(best_path)
    # prev_ref = events[0][0]
    # prev_event = events[0][1]
    # for event in events[1:]:
    #     assert prev_ref <= events[0][0]
    #     assert prev_event <= events[0][1]
    #     prev_ref = events[0][0]
    #     prev_event = events[0][1]
    #     # print(event)

    # print(events["posterior_probability"])
    # print(events["event_index"])
    #
    # print(type(events))
    # a = events[events["event_index"].argsort()]
    # print(a)
    # print(a[a["reference_index"].argsort()])


    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
