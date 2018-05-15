import xml.etree.ElementTree as ET
import uuid, math, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from itertools import cycle
import random
import os
import sys

"""
    This file is used to read InkML files, parse each file into different segments and create 
    python dictionaries with each symbols including their traces and truth.
"""

IMG_RESOLUTION = 26
    

# Scales a list of traces to a given range. See https://gist.github.com/perrygeo/4512375
def scale_linear_bycolumn(rawpoints, high=24, low=0, ma=0, mi=0): 
    mins = mi 
    maxs = ma

    rng = maxs - mins

    if rng == 0:
        print("Rawpoints", rawpoints, "MAX", maxs, "MIN", mins)
        rng = 0.001
    output = high - (((high - low) * (maxs - rawpoints)) / rng)
    return output

# Parse a single single trace from InkML trace string
def format_trace(text):
    l = []

    for coord in text.split(','):
        c = coord.strip().split(" ")
        l.append(c)

    return l

# Search for a trace in the InkML file from a given id
def find_trace(root, id):
    for child in root.findall('{http://www.w3.org/2003/InkML}trace'):
        if child.attrib['id'] == id:
            return format_trace(child.text)


# Find a combination all symbols and their truths from a InkML file
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


"""
    A class used to represent a group of traces, that combined makes up a symbol.
    This class also has methods for calculating bounding boxes, scaling etc.
    These methods are no longer in use, however, they may be useful for another 
    attempt at predicting bounding boxes around symbols.
 """
class Segment:

    def __init__(self, id, truth):
        self.traces = []
        self.truth = truth
        self.id = id
        self.x = None
        self.y = None
        self.w = None
        self.h = None

    def add_trace(self, trace):
        self.traces.append(np.asarray(trace).astype('float32'))

    def calculate_bounding_box(self):

        right = 0
        left = math.inf
        top = math.inf
        bottom = 0

        t = []
        for i, trace in enumerate(self.traces):
            t.append(np.asarray(trace).astype('float32'))
            right = max(right, np.max(t[i][:, 0]))
            left = min(left, np.min(t[i][:, 0]))
            top = min(top, np.min(t[i][:, 1]))
            bottom = max(bottom, np.max(t[i][:, 1]))

        self.x = (right + left) / 2
        self.y = (top + bottom) / 2

        self.h = np.abs(top - bottom)
        self.w = np.abs(right - left)

        return right, bottom, left, top

    def normalize(self, img_high, img_width, max_x, max_y, min_x, min_y, width_scale, height_scale):
        
        if width_scale > 0:
            # add padding in x-direction
            for i, trace in enumerate(self.traces):
                
                side = (Equation.IMG_WIDTH - width_scale) / 2
                trace[:, 0] = scale_linear_bycolumn(trace[:, 0], high=(Equation.IMG_WIDTH - side), low=side, ma=max_x, mi=min_x)
                trace[:, 1] = scale_linear_bycolumn(trace[:, 1], high=Equation.IMG_HEIGHT, low=0, ma=max_y, mi=min_y)

        else:
            # add padding in y-direction

            for i, trace in enumerate(self.traces):
                
                side = (Equation.IMG_HEIGHT - height_scale) / 2
                trace[:, 0] = scale_linear_bycolumn(trace[:, 0], high=Equation.IMG_WIDTH, low=0, ma=max_x, mi=min_x)
                trace[:, 1] = scale_linear_bycolumn(trace[:, 1], high=(Equation.IMG_HEIGHT - side), low=side, ma=max_y, mi=min_y)



    def draw_symbol(self, draw):
        for trace in self.traces:
            y = np.array(trace).astype("float32")

            coordinates = list(zip(trace[:, 0], trace[:, 1]))
            xy_cycle = cycle(coordinates)

            next(xy_cycle)


            for x_coord, y_coord in coordinates[:-1]:
                next_coord = next(xy_cycle)
                draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="white", width=1)
    
    def draw_bounding_box(self, draw):

        draw.rectangle(((self.x - self.w/2, self.y - self.h/2), (self.x + self.w/2, self.y + self.h/2)), outline="red")

    def generate_bitmap(self):
        try:
            resolution = IMG_RESOLUTION - 4
            image_resolution = IMG_RESOLUTION

            image = Image.new('L', (image_resolution, image_resolution), "white")
            draw = ImageDraw.Draw(image)

            max_x = 0
            min_x = math.inf
            max_y = 0
            min_y = math.inf

            for trace in self.traces:
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
                height_scale = resolution / scale
            else:
                # width < height
                width_scale = resolution * scale

            for trace in self.traces:
                y = np.array(trace).astype("float32")

                x, y = y.T

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
                    draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="black", width=1)

            return image, self.truth, self.id
        except:
            print("Item failed")
            return None, None, None


"""
    This class has methods for creating full images from the whole InkML file. 
    The class is not currently in use, however may be useful for end to end predictions.
"""
class Equation:
    IMG_HEIGHT = 64
    IMG_WIDTH = 128

    def __init__(self, segments):
        self.segments = segments
        self.glob_max_x = 0
        self.glob_min_x = math.inf
        self.glob_max_y = 0
        self.glob_min_y = math.inf
        self.width_scale = 0
        self.height_scale = 0

    # Calculates  boundries for all segments in the equation
    def compute_global_boundaries(self):

        for segment in self.segments:
            max_x, max_y, min_x, min_y = segment.calculate_bounding_box()


            self.glob_max_x = max(self.glob_max_x, max_x)

            self.glob_max_y = max(self.glob_max_y, max_y)
            self.glob_min_x = min(self.glob_min_x, min_x)
            self.glob_min_y = min(self.glob_min_y, min_y)
        
        width = self.glob_max_x - self.glob_min_x
        height = self.glob_max_y - self.glob_min_y


        img_ratio = Equation.IMG_WIDTH / Equation.IMG_HEIGHT

        scale = width / (height * img_ratio)


        bounding_boxes = []

        for segment in self.segments:
            pass
        if scale > 1:
            self.height_scale = Equation.IMG_HEIGHT / scale

        else:
            self.width_scale = Equation.IMG_WIDTH * scale



    # Converts the equation to a scaled image
    def create_image_and_scale(self):

        image = Image.new('LA', (Equation.IMG_WIDTH, Equation.IMG_HEIGHT), "black")

        draw = ImageDraw.Draw(image)

        bounding_boxes = []

        for segment in self.segments:
            #segment.calculate_bounding_box() # No longer in use


            segment.normalize(Equation.IMG_HEIGHT, Equation.IMG_WIDTH, self.glob_max_x, self.glob_max_y,
                              self.glob_min_x, self.glob_min_y, self.width_scale, self.height_scale)
            segment.draw_symbol(draw)
            segment.calculate_bounding_box()
            #segment.draw_bounding_box(draw) # No longer in use

            bounding_boxes.append([segment.x, segment.y, segment.w, segment.h])

        return (image, bounding_boxes)


