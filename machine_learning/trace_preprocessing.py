import xml.etree.ElementTree as ET
import uuid, math, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from itertools import cycle
import random
import os
import sys

def format_trace(text):
    l = []

    for coord in text.split(','):
        c = coord.strip().split(" ")
        l.append(c)

    return l


def find_trace2(root):

    traces = []

    for child in root.findall('{http://www.w3.org/2003/InkML}trace'):
        traces.append(format_trace(child.text))

    return traces


def find_trace(root, id):
    for child in root.findall('{http://www.w3.org/2003/InkML}trace'):
        if child.attrib['id'] == id:
            return format_trace(child.text)


def find_segments(root):
    segments = []
    for group in root.findall('{http://www.w3.org/2003/InkML}traceGroup'):
        for item in group.findall('{http://www.w3.org/2003/InkML}traceGroup'):

            id = uuid.uuid4()

            truth = item.find('{http://www.w3.org/2003/InkML}annotation').text

            segment = Segment(id, truth)

            for trace in item.findall('{http://www.w3.org/2003/InkML}traceView'):
                traceId = trace.attrib['traceDataRef']

                segment.add_trace(find_trace(root, traceId))

            segments.append(segment)

    return segments



class Trace:
    GOAL_AMOUNT = 48

    def __init__(self, points):
        self.points = points
    
    def normalize(self):
        if len(self.points) > Trace.GOAL_AMOUNT:

            # Remove points

            print('More!')
        elif len(self.points) < Trace.GOAL_AMOUNT:

            # Add points

            print('Less!')

class Segment:
    TRACE_POINTS_AMOUNT_GOAL = 48
    RESOLUTION = 200

    def __init__(self, id, truth):
        self.traces = []
        self.truth = truth

    def add_trace(self, trace):
        self.traces.append(np.asarray(trace).astype('float32'))


def find_single_trace_distances(trace):
    trace_cycle = cycle(trace)
    next(trace_cycle)

    distances = []

    for point in trace[:-1]:
        next_point = next(trace_cycle)
        dist = math.hypot(next_point[0] - point[0], next_point[1] - point[1])

        distances.append(dist)
    return distances
        


def normalize_trace(trace):
    if len(trace) > Segment.TRACE_POINTS_AMOUNT_GOAL:

        # Remove points
        # Does not remove first or last point
        
        to_remove = len(trace) - Segment.TRACE_POINTS_AMOUNT_GOAL
        # find distance between all points

        distances = find_single_trace_distances(trace)

        # find median distance to surrounding points for each point

        median_distances = []
        distances_cycle = cycle(distances)
        next(distances_cycle)   

        # put this value and index in trace in a list

        for i, distance in enumerate(distances[:-1]):
            next_distance = next(distances_cycle)
            median_distance = np.median([distance, next_distance])
            median_distances.append([median_distance, (i+1)])

        # sort list
        sorted_medians = sorted(median_distances)
        
        sorted_medians_nparray = np.asarray(sorted_medians)

        to_delete = sorted_medians_nparray[0:to_remove, 1]

        # remove x indexes from original trace, with x index from list

        new_trace = np.asarray([i for j, i in enumerate(trace) if j not in to_delete])
        return new_trace

    elif len(trace) < Segment.TRACE_POINTS_AMOUNT_GOAL:

        while len(trace) < Segment.TRACE_POINTS_AMOUNT_GOAL:
            to_add = Segment.TRACE_POINTS_AMOUNT_GOAL - len(trace)
            
            if to_add > len(trace):
                to_add = len(trace) - 1
            
            distances = find_single_trace_distances(trace)
            distances_index = [[j, i] for i, j in enumerate(distances)]
            sorted_distances_index = np.asarray(sorted(distances_index, reverse=True))


            for i in sorted_distances_index[0:to_add, 1]:
                index = int(i)

                new_x = (trace[index, 0] + trace[index + 1, 0]) / 2
                new_y = (trace[index, 1] + trace[index + 1, 1]) / 2

                trace = np.insert(trace, index+1, np.array((new_x, new_y)), axis=0)
    return trace

def calculate_distances(trace1, trace2):
    distances_full = []

    for coord1 in trace1:
        distances = []
        for coord2 in trace2:
            dist = math.hypot(coord2[0] - coord1[0], coord2[1] - coord1[1])
            distances_full.append(dist)

        #distances_full.append(distances)
    return distances_full

#def scale_trace(trace):
    # Find max and min
    # 

def get_training_pairs():

    overlapping_traces = []
    separate_traces = []

    count = 0

    max_x = 0
    min_x = math.inf
    max_y = 0
    min_y = math.inf


    for file in os.listdir(os.getcwd() + '/data')[0:3]:
        if count % 100 == 0: 
            print(count)
            print(max_x, min_x, max_y, min_y)

        count += 1
        full_filename = os.getcwd() + '/data/' + file
        try:
            tree = ET.parse(full_filename)
        except:
            print("Failed to parse tree")
            continue

        root = tree.getroot()
        
        segments = find_segments(root)

        segment_cycle = cycle(segments)
        next(segment_cycle)




        for segment in segments:

            # Scale the traces to be the same format
            #for i, trace in enumerate(segment.traces):
            #    trace = scale_trace(trace)
            #    segment.traces[i] = trace
            
            # Normalize the traces
            for i, trace in enumerate(segment.traces):
                trace = normalize_trace(trace)
                segment.traces[i] = trace
        

        for segment in segments[:-1]:
            if len(segment.traces) > 1:

                trace_cycle = cycle(segment.traces)
                next(trace_cycle)

                for trace in segment.traces[:-1]:
                    next_trace = next(trace_cycle)
                    overlapping_traces.append((trace, next_trace))

            else:
                next_segment = next(segment_cycle)
                
                separate_traces.append((segment.traces[0], next_segment.traces[0]))


    training_and_validation_list = []

    print('Calculating distances')

    #print('Drawing overlap')
    #draw_scenario(overlapping_traces[3][0], overlapping_traces[3][1], 'overlap')

    #print('Drawing separate')
    #draw_scenario(separate_traces[7][0], separate_traces[7][1], 'separate')



    for traces in overlapping_traces:
        distances_array = calculate_distances(traces[0], traces[1])
        training_and_validation_list.append([distances_array, 1])

    for traces in separate_traces:
        distances_array = calculate_distances(traces[0], traces[1])
        training_and_validation_list.append([distances_array, 0])


    training_and_validation_arr = np.asarray(training_and_validation_list)
    np.random.shuffle(training_and_validation_arr)
    
    training_arr = np.zeros((len(training_and_validation_arr), Segment.TRACE_POINTS_AMOUNT_GOAL**2))
    validation_arr = np.zeros((len(training_and_validation_arr), 1))

    for i, val in enumerate(training_and_validation_arr[:, 0]):
        training_arr[i] = val

    for i, val in enumerate(training_and_validation_arr[:, 1]):
        validation_arr[i] = val

    return training_arr, validation_arr

def scale_trace(trace):
    print(trace)


def scale_linear_bycolumn(rawpoints, high=24, low=0, ma=0, mi=0, printt=False):  # , maximum=None, minimum=None):
    mins = mi  # np.min(rawpoints, axis=0)
    maxs = ma  # np.max(rawpoints, axis=0)

    rng = maxs - mins

    output = high - (((high - low) * (maxs - rawpoints)) / rng)

    if(printt):
        print("raw", rawpoints)
        print("high", high)
        print("low", low)
        print("ma", ma)
        print("mi", mi)
        print("out", output)
    
    return output


def draw_scenario(trace1, trace2, fn):
    # draw two traces


    #print('trace1', trace1)
    #print('trace2', trace2)
    image_resolution = 200

    image = Image.new('L', (image_resolution, image_resolution), "white")
    draw = ImageDraw.Draw(image)

    max_x = 0
    min_x = math.inf
    max_y = 0
    min_y = math.inf

    for trace in [trace1, trace2]:
        y = np.array(trace).astype("float32")

        x, y = y.T

        if max_x < x.max():
            max_x = x.max()

        if max_y < y.max():
            max_y = y.max()

        if min_x > x.min():
            min_x = x.min()

        if min_y > y.min():
            min_y = y.min()

    width = max_x - min_x
    height = max_y - min_y
    scale = width / height

    width_scale = 0
    height_scale = 0

    if scale > 1:
        # width > height
        height_scale = image_resolution / scale
    else:
        # width < height
        width_scale = image_resolution * scale

    for trace in [trace1, trace2]:
        y = np.array(trace).astype("float32")

        x, y = y.T

        new_x = []
        new_y = []

        if width_scale > 0:
            # add padding in x-direction
            new_y = scale_linear_bycolumn(y, high=image_resolution, low=0, ma=max_y, mi=min_y)
            side = (image_resolution - width_scale) / 2
            new_x = scale_linear_bycolumn(x, high=(image_resolution - side), low=(side), ma=max_x, mi=min_x)
        else:
            # add padding in y-direction
            new_x = scale_linear_bycolumn(x, high=image_resolution, low=0, ma=max_x,
                                        mi=min_x)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))
            side = (image_resolution - height_scale) / 2
            new_y = scale_linear_bycolumn(y, high=(image_resolution - side), low=(side), ma=max_y,
                                        mi=min_y)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))
        coordinates = list(zip(new_x, new_y))
        xy_cycle = cycle(coordinates)

        next(xy_cycle)

        for x_coord, y_coord in coordinates[:-1]:
            next_coord = next(xy_cycle)
            draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="black", width=1)

    directory = os.getcwd() + "/"

    filename = directory + fn + ".bmp"

    image.save(filename)



def semi_parallell(A,B): # A and B are traces
    a_1 = (A[-1][1] - A[0][1]) / (A[-1][0] - A[0][0])
    a_2 = (B[-1][1] - B[0][1]) / (B[-1][0] - B[0][0])

    print("a_1: ", a_1)
    print("a_2: ", a_2)
    print(a_1 - a_2)

    if np.abs(a_1 - a_2) < 0.1:
        return True
    return False



if __name__ == '__main__':

    #t, v = get_training_pairs()
    

    

        
    '''     for file in os.listdir(os.getcwd() + '/data'):
        full_filename = os.getcwd() + '/data/' + file
        try:
            tree = ET.parse(full_filename)
        except:
            print("Failed to parse tree")
            continue

        root = tree.getroot() '''

    ''' test_traces = find_trace(root)

        for coordinates in test_traces:
            trace = Trace(points=coordinates)
            trace.normalize()

        break '''

