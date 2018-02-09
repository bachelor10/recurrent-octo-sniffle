# This is an exploration of time segments and relation between them
# The important part here is to research if time between traces can be taken into helping classifying different segments
#

import numpy as np


class time_segmenter():
    def __init__(self, delta_mode=True, raw_time_mode=False):
        self.delta_mode = delta_mode
        self.raw_time_mode = raw_time_mode
        self.buffer = []
        self.avg_deltat = 0
        self.avg_time_between = 0

    # Buffer is now
    def add_time(self, coordinates):
        if self.buffer is not None:  # use coordinates
            pass

    # find average time between traces
    def find_avg_time_between(self, buffer):
        average = 0
        num_traces = len(buffer)
        for i, trace in enumerate(buffer[:-1]):
            # print(buffer[i+1][0][2])
            average += (buffer[i + 1][0][2] - buffer[i][-1][2])  # last element in current trace minus the first elements time in next trace

        average = average / (num_traces - 1)
        print("Average time between traces: ", average)

        return average

    '''
    :returns nxn matrix where n = len(buffer)
    constructs a time matrix.
    time between trace #0 and #1 can be found at [0][1].
    time between trace #1 and #2 can be found at [1][2].
    '''

    def find_time_between_shit(self, buffer):
        size = len(buffer)
        time_matrix = np.zeros([size, size])  # rows, cols

        for i, trace in enumerate(buffer):
            for k, trace in enumerate(buffer[:]):
                print("i: ", i, " k: ", k)
                if not (i == k) and k >= i:
                    time_matrix[i][k] = (buffer[k][0][2] - buffer[i][-1][2])

        print(time_matrix)
        return time_matrix

    def find_time_between(self, buffer):
        size = len(buffer)
        time_arr = np.zeros(size)
        print(buffer)
        print(buffer[0][0][2])

        # time at element is time between traces
        # time_arr[0] is time between trace [0] and [1]
        for i, trace in enumerate(buffer[:-1]):
            time_arr[i] = (buffer[i+1][0][2] - buffer[i][-1][2])

        print(time_arr)
        return time_arr

    def propability_traces(self):
        # an algorithm to determine by time if two slices are most likely a part of the same tracegroup.
        # a combination of checking avg time distance and a lookup to determine if that is correct should be implemented.

        '''
        pseudo:
        avg = find_avg_time(buff)
        for times
            if time between < avg
                high propability of same group
            else:
                low propability of same group.
                save that aswell!
        '''
        raise NotImplementedError()