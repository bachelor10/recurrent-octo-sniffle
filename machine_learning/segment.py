import numpy as np
import machine_learning.xml_parse as

from itertools import cycle
from PIL import Image, ImageDraw
import math

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
        left = 0
        top = 0
        bottom = 0

        t = []
        # TODO Rework this. This shit makes no sense. Max of some value and infinity is infinity.
        for i, trace in enumerate(self.traces):
            t.append(np.asarray(trace).astype('float32'))
            right = max(right, np.max(t[i][:, 0]))
            left = min(left, np.min(t[i][:, 0]))
            top = min(top, np.min(t[i][:, 1]))
            bottom = max(bottom, np.max(t[i][: 1]))
            # print("VAFFELRØRE:", right, left, top, bottom)

        self.x = (right + left) / 2
        self.y = (top + bottom) / 2

        self.h = np.abs(top - bottom)
        self.w = np.abs(right - left)

        return self.x, self.y, self.h, self.w

    def normalize(self, img_high, img_width, max_x, max_y, min_x, min_y):

        # Normalize bounding box
        # print("IN normalize: ", max_x, max_y)
        self.x = scale_linear_bycolumn(self.x, high=img_width, low=0, ma=max_x, mi=min_x)
        self.y = scale_linear_bycolumn(self.y, high=img_high, low=0, ma=max_y, mi=min_y)
        self.h = scale_linear_bycolumn(self.h, high=img_high, low=0, ma=max_y, mi=min_y)
        self.w = scale_linear_bycolumn(self.w, high=img_width, low=0, ma=max_x, mi=min_x)

        # Normalize traces

        for i, trace in enumerate(self.traces):
            trace[:, 0] = scale_linear_bycolumn(trace[i, 0], high=img_width, low=0, ma=max_x,
                                                mi=min_x)
            trace[:, 1] = scale_linear_bycolumn(trace[:, 1], high=img_high, low=0, ma=max_y, mi=min_y)

    def draw_symbol(self, draw):

        for trace in self.traces:
            y = np.array(trace).astype("float32")

            coordinates = list(zip(trace[:, 0], trace[:, 1]))
            # print(coordinates)
            xy_cycle = cycle(coordinates)

            next(xy_cycle)

            for x_coord, y_coord in coordinates[:-1]:
                next_coord = next(xy_cycle)
                draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="black", width=1)

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