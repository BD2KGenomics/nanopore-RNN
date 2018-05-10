#!/usr/bin/env python
"""Plot speeds of maximum expected accuracy methods"""
########################################################################
# File: plot_mea_speeds.py
#  executable: plot_mea_speeds.py
#
# Author: Andrew Bailey
# History: Created 02/24/18
########################################################################

from __future__ import print_function
import sys
import os
from timeit import default_timer as timer
import pysam
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import numpy as np
import scipy.stats as stats
from py3helpers.utils import list_dir, time_it

from signalalign.mea_algorithm import maximum_expected_accuracy_alignment, mea_slow, \
    mea_slower, create_random_prob_matrix


def compare_mea_speeds():
    """Compare the speeds of the three mea implementations"""
    plt.figure(figsize=(14, 4))
    panel1 = plt.axes([0.07, 0.11, .4, .8])
    panel2 = plt.axes([0.55, 0.11, .4, .8])

    # longest = max(data[0]) + data[1])

    fast_times = []
    slow_times = []
    slower_times = []
    num_points = []
    for x in range(10, 200, 10):
        size = x
        prob_matrix, shortest_future_col_per_row = create_random_prob_matrix(col=size, row=size)
        points = np.count_nonzero(prob_matrix)
        num_points.append(points)
        _, time = time_it(mea_slow, prob_matrix, shortest_future_col_per_row)
        slow_times.append(time)
        print(time, time/points)
        _, time = time_it(mea_slower, prob_matrix, shortest_future_col_per_row)
        slower_times.append(time)
        print(time, time/points)
        _, time = time_it(maximum_expected_accuracy_alignment, prob_matrix, shortest_future_col_per_row)
        fast_times.append(time)
        print(time, time/points)

    # plot number of points vs time
    handle1, = panel1.plot(num_points, fast_times, color='black')
    handle2, = panel1.plot(num_points, slow_times, color='blue')
    handle3, = panel1.plot(num_points, slower_times, color='red')

    slowest_time = max(slower_times + slow_times)
    most_points = max(num_points)

    panel1.set_xlim(0, most_points)
    panel1.set_ylim(0, slowest_time)

    panel1.set_xlabel('Number of Points')
    panel1.set_ylabel('Time (s)')
    panel1.legend([handle1, handle2, handle3], ["Fast", "Slow", "Slower"], loc='upper left')
    panel1.set_title('Time (s) vs Number of points ')

    fast_ratio = [x/y for x, y in zip(fast_times, num_points)]
    slow_ratio = [x/y for x, y in zip(slow_times, num_points)]
    slower_ratio = [x/y for x, y in zip(slower_times, num_points)]

    handle1, = panel2.plot(num_points, fast_ratio, color='black')
    handle2, = panel2.plot(num_points, slow_ratio, color='blue')
    handle3, = panel2.plot(num_points, slower_ratio, color='red')

    panel2.set_xlim(0, max(num_points))
    panel2.set_ylim(0, max(slower_ratio))

    panel2.set_xlabel('Number of Points')
    panel2.set_ylabel('Time/ number of points (s/point)')
    panel2.legend([handle1, handle2, handle3], ["Fast", "Slow", "Slower"], loc='upper left')
    panel2.set_title('Time(s)/Data Points vs Number of points ')

    plt.show()




def main():
    """Main docstring"""
    start = timer()

    compare_mea_speeds()

    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
