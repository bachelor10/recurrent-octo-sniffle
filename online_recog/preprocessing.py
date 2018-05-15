import os 
import xml.etree.ElementTree as ET
from xml_parse import find_segments, scale_linear_bycolumn
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from rdp import rdp
from keras.preprocessing import sequence
import math
from PIL import Image, ImageDraw
from itertools import cycle
import rpd_test
import random
import time


"""
    This file includes different methods for preprocessing data before
    prediction. It also has a method to generate a full dataset of specified symbols.
"""
def get_inkml_root(file):
	return ET.parse(file).getroot()


# Continously generates images from the InkML files
def segment_generator(directory):
    count = 0
    for file in os.listdir(directory):
        try:

            if file.endswith(".inkml"):

                root = get_inkml_root(directory + file)
                
                segments = find_segments(root)

                for i, segment in enumerate(segments):
                    yield segment

        except GeneratorExit: 
            break
        except Exception as e:
            count += 1
            print("Error in segment generator", e)
            print("Current number of exceptions", count)
            continue

# Find the first segment from with a specified truth
def get_single_segment(truth, num=0):
    curr = 0
    for segment in segment_generator(os.getcwd() + '/data/xml/'):
        if segment.truth == truth:
            if curr == num:
                    return segment.traces
            curr += 1

def plot_trace(trace):  # trace is x,y coordinates
	plt.plot(trace[:, 0], trace[:, 1])
	axes = plt.gca()
	axes.invert_yaxis()
	
	plt.show()

# Scale traces between to within a specified range [-recolution, resolution].
# Will keep the same proportions as the original trace
def scale_traces(trace, resolution=1):

    trace = np.array(trace)

    traceX = trace[:, 0]
    traceY = trace[:, 1]

    max_x = np.max(traceX)
    min_x = np.min(traceX)
    max_y = np.max(traceY)
    min_y = np.min(traceY)

    width = max_x - min_x
    height = max_y - min_y
    scale = width / height

    width_scale = 0
    height_scale = 0

    if scale > 1:
        # width > height
        height_scale = resolution / scale
        side = height_scale
    else:
        # width < height
        width_scale = resolution * scale
        side = width_scale


    if scale < 1:
        # add padding in x-direction

        trace[:,1] = scale_linear_bycolumn(trace[:,1], high=resolution, low=-resolution, ma=max_y, mi=min_y)
        trace[:,0] = scale_linear_bycolumn(trace[:,0], high=(side), low=(-side), ma=max_x, mi=min_x)

    else:

        # add padding in y-direction
        trace[:,0] = scale_linear_bycolumn(trace[:,0], high=resolution, low=-resolution, ma=max_x,
                                        mi=min_x) 
        trace[:,1] = scale_linear_bycolumn(trace[:,1], high=(-side), low=(side), ma=max_y,
                                        mi=min_y) 

    return trace

# Combines a list of traces to a single numpy array.
# The end of each trace will be marked as a 1 in index 2 of the first axis.
# Outputs an array with shape [x_coord, y_coord, end_of_trace(0 | 1)]

def combine_segment(traces):
    combined_segment = []

    max_len = -math.inf
    for trace in traces:
        if(len(trace) > max_len):
            max_len = len(trace) 

    for trace in traces:
        for i, coords in enumerate(trace):
            if i == len(trace) - 1:
                combined_segment.append([coords[0], coords[1], 1])
            else:
                combined_segment.append([coords[0], coords[1], 0])

    return np.array(combined_segment)

# Runs the fixed RDP algorithm on all traces. Each trace will be reduced to 40 datapoints.
def run_rdp_on_traces(traces):
    traces_after_rdp = []

    for trace in traces:
        traces_after_rdp.append(rpd_test.rdp_fixed_num(trace, fixed_num=40))

    return traces_after_rdp


# Create a bitmap (image) from the traces
IMG_RESOLUTION = 26
def generate_bitmap(traces):
    resolution = IMG_RESOLUTION
    image_resolution = IMG_RESOLUTION

    image = Image.new('L', (image_resolution, image_resolution), "black")
    draw = ImageDraw.Draw(image)

    max_x = 0
    min_x = math.inf
    max_y = 0
    min_y = math.inf

    for trace in traces:
        coords = np.array(trace).astype("float32")

        x = coords[:, 0]
        y  = coords[:, 1]

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
        height_scale = resolution / scale
    else:
        # width < height
        width_scale = resolution * scale

    for trace in traces:
        coords = np.array(trace).astype("float32")

        x = coords[:, 0]
        y  = coords[:, 1]

        new_x = []
        new_y = []

        if width_scale > 0:
            # add padding in x-direction
            new_y = scale_linear_bycolumn(y, high=resolution, low=0, ma=max_y, mi=min_y)
            side = (resolution - width_scale) / 2
            new_x = scale_linear_bycolumn(x, high=(resolution - side), low=(side), ma=max_x, mi=min_x)
        else:
            # add padding in y-direction
            new_x = scale_linear_bycolumn(x, high=resolution, low=0, ma=max_x,
                                            mi=min_x)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))
            side = (resolution - height_scale) / 2
            new_y = scale_linear_bycolumn(y, high=(resolution - side), low=(side), ma=max_y,
                                            mi=min_y)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))

        coordinates = list(zip(new_x, new_y))
        xy_cycle = cycle(coordinates)

        next(xy_cycle)

        for x_coord, y_coord in coordinates[:-1]:
            next_coord = next(xy_cycle)
            draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="white", width=1)

    return image
"""
    Generates the full dataset. 

    input:
    numb_symbols - number of symbols to create
    include - A dict with the truths to include in the dataset. If None, all truths available will be included
    returnType - Array of types to include. Can be 'TRACE', 'IMAGE' or both.

    
    returns 
    padded traces, 
    images, 
    truths, 
    original traces
"""
def generate_dataset(numb_symbols=100, include=None, returnType=['TRACE', 'IMAGE']):
    segments = []
    images = []
    truths = []
    original_traces = []

    count = 0

    for segment in segment_generator(os.getcwd() + '/data/xml/'):

        if include:
            try: 
                include[segment.truth]
            except:
                continue

            if 'TRACE' in returnType:
                processed = run_rdp_on_traces(segment.traces)
                processed = combine_segment(processed)

                processed = scale_traces(processed)


                if np.isnan(processed).any(): 
                    print(segment.traces, segment.truth)
                    continue
                
                segments.append(processed)

            if 'IMAGE' in returnType:

                image = generate_bitmap(segment.traces)

                images.append(image)

            truths.append(segment.truth)
            original_traces.append(segment.traces)

            if(len(segments) % 1000 == 0): print(len(segments))
            

        count += 1
        if count > numb_symbols: break


    segments = np.array(segments)
    truths = np.array(truths)

    padded = sequence.pad_sequences(segments, dtype='float32', maxlen=40)

    return padded, images, truths, original_traces


