import numpy as np
import math
from machine_learning import xml_parse
from PIL import Image, ImageDraw
from itertools import cycle

class Preprocessor:

    def pre_process(self, traces):
        resolution = 24
        image_resolution = 26

        image = Image.new('L', (image_resolution, image_resolution), "white")
        draw = ImageDraw.Draw(image)

        max_x = 0
        min_x = math.inf
        max_y = 0
        min_y = math.inf

        for trace in traces:
            y = np.array(trace).astype(np.float)
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

        for trace in traces:

            y = np.array(trace).astype(np.float)

            x, y = y.T

            if width_scale > 0:
                # add padding in x-direction
                new_y = xml_parse.scale_linear_bycolumn(y, high=resolution, low=0, ma=max_y, mi=min_y)
                side = (resolution - width_scale) / 2
                new_x = xml_parse.scale_linear_bycolumn(x, high=(resolution - side), low=(side), ma=max_x, mi=min_x)
            else:
                # add padding in y-direction
                new_x = xml_parse.scale_linear_bycolumn(x, high=resolution, low=0, ma=max_x,
                                                        mi=min_x)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))
                side = (resolution - height_scale) / 2
                new_y = xml_parse.scale_linear_bycolumn(y, high=(resolution - side), low=(side), ma=max_y,
                                                        mi=min_y)  # , maximum=(max_x, max_y), minimum=(min_x, min_y))

            coordinates = list(zip(new_x, new_y))
            xy_cycle = cycle(coordinates)

            next(xy_cycle)

            for x_coord, y_coord in coordinates[:-1]:
                next_coord = next(xy_cycle)
                draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="black", width=1)

        i = image.convert('LA')

        arr = np.asarray(i)

        formatted = []
        for row in arr:
            new_row = []
            for col in row:
                new_row.append(col[0])

            formatted.append(new_row)
        # print(np.asarray([np.asarray(i)]))
        return np.asarray([np.asarray(formatted).reshape((26, 26, 1))])
