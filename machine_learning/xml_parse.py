import xml.etree.ElementTree as ET
import uuid, math, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from itertools import cycle
import random
import os
import sys


IMG_RESOLUTION = 26
    
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '=', '+']


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


def format_trace(text):
    l = []

    for coord in text.split(','):
        c = coord.strip().split(" ")
        l.append(c)

    return l


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

        #return self.x, self.y, self.h, self.w
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

    """def save(self):
        for segment in self.segments:
            x, y, h, w = segment.compute_bounding_box()

            image, truth, segment_id = segment.generate_bitmap()

            if image is None: continue

            directory = os.getcwd() + "/" + dirs[1]

            if random.random() > 0.8:
                directory = os.getcwd() + "/" + dirs[0]

            if truth.isalpha():
                if truth.isupper():
                    truth = truth + "_"

            if truth == '|':
                truth = '_|'

            subdir = directory + "/" + truth

            filename = subdir + '/' + truth + "_" + segment_id + ".bmp"

            if not os.path.exists(subdir):
                os.makedirs(subdir)

            image.save(filename)"""


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




    def create_image_and_scale(self):

        image = Image.new('LA', (Equation.IMG_WIDTH, Equation.IMG_HEIGHT), "black")

        draw = ImageDraw.Draw(image)

        bounding_boxes = []

        for segment in self.segments:
            #segment.calculate_bounding_box()


            segment.normalize(Equation.IMG_HEIGHT, Equation.IMG_WIDTH, self.glob_max_x, self.glob_max_y,
                              self.glob_min_x, self.glob_min_y, self.width_scale, self.height_scale)
            segment.draw_symbol(draw)
            segment.calculate_bounding_box()
            #segment.draw_bounding_box(draw)

            bounding_boxes.append([segment.x, segment.y, segment.w, segment.h])

        return (image, bounding_boxes)


def continous_symbol_generator(limit=0):
    count = 0
    for file in os.listdir(os.getcwd() + '/train'):
        if count > limit: break
        if count%100 == 0: print('Count', count)
        
        full_filename = os.getcwd() + '/train/' + file
        try:
            tree = ET.parse(full_filename)
        except:
            print("Failed to parse tree")
            continue

        root = tree.getroot()

        segments = find_segments(root)
        full_truth = root.find('{http://www.w3.org/2003/InkML}annotation').text

        images_with_truth = []
        processed = dict()
        for segment in segments:
            image, truth, segment_id = segment.generate_bitmap()

            start_index = 0
            try:
                start_index = processed[truth]
            except:
                pass
            try: 
                truth_index = full_truth.index(truth, start_index)
            except:
                continue

            images_with_truth.append((image, truth, truth_index))

            processed[truth] = truth_index + 1
    
        sorted_returnvalues = sorted(images_with_truth, key=lambda t: t[2])

        for val in sorted_returnvalues:
            yield (val[0], val[1])

        count += 1




def model_data_generator(limit=10000):
    count = 0
    for file in os.listdir(os.getcwd() + '/data'):
        if count > limit: break
        if count%100 == 0: print('Count', count)
        full_filename = os.getcwd() + '/data/' + file
        try:
            tree = ET.parse(full_filename)
        except:
            print("Failed to parse tree")
            continue

        root = tree.getroot()

        segments = find_segments(root)

        equation = Equation(segments)

        equation.compute_global_boundaries()

        count += 1

        

        yield equation.create_image_and_scale()

        # equation.save()

    # generate_bitmaps(segments)

    # generate_bitmap(segments[0])

if __name__ == '__main__':
    for val in continous_symbol_generator(limit=10):
        print(val[1])