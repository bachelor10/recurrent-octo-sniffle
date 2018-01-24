import xml.etree.ElementTree as ET
import uuid, math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from itertools import cycle

tree = ET.parse('trace_ex.inkml')

root = tree.getroot()


def scale_linear_bycolumn(rawpoints, high=24, low=0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)


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
            segments.append(dict())
            i = len(segments) - 1

            segments[i]["truth"] = truth
            segments[i]["traces"] = []
            segments[i]["id"] = str(id)

            for trace in item.findall('{http://www.w3.org/2003/InkML}traceView'):
                traceId = trace.attrib['traceDataRef']

                segments[i]["traces"].append(find_trace(root, traceId))
    
    return segments



def generate_bitmap(segment):

    resolution = 24

    image = Image.new('L', (resolution, resolution), "white")
    draw = ImageDraw.Draw(image)
    
    max_x = 0
    min_x = math.inf
    max_y = 0
    min_y = math.inf

    for trace in segment["traces"]:
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

    print("Truth ", segment["truth"])
    if scale > 1:
        # width > height
        height_scale = 24 / scale
    else:
        # width < height
        width_scale = 24 * scale
    
    print("width", width_scale)
    print("height", height_scale)

    for trace in segment["traces"]:
        y = np.array(trace).astype(np.float)

        x, y = y.T

        new_x = []
        new_y = []

        if width_scale > 0:
            # add padding in x-direction
            new_y = scale_linear_bycolumn(y, high=resolution, low=0)
            side = (24 - width_scale)/2
            print("side", side)
            new_x = scale_linear_bycolumn(x, high=(resolution-side), low=(side))
            print(new_x)
        else:
            # add padding in y-direction
            new_x = scale_linear_bycolumn(x, high=resolution, low=0)
            side = (24 - height_scale)/2
            new_y = scale_linear_bycolumn(y, high=(resolution-side), low=(side))


        coordinates = list(zip(new_x, new_y))
        xy_cycle = cycle(coordinates)

        next(xy_cycle)

        for x_coord, y_coord in coordinates[:-1]:
            next_coord = next(xy_cycle)
            draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="black", width=1)

    filename = segment["truth"] + "_" + segment["id"] + ".bmp"
    image.save(filename)

def generate_bitmaps(segments):
    for segment in segments:
        generate_bitmap(segment)

segments = find_segments(root)
generate_bitmaps(segments)



