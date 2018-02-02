import xml.etree.ElementTree as ET
import uuid, math, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from itertools import cycle
import random
import os

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '=', '+']
def scale_linear_bycolumn(rawpoints, high=24, low=0, ma=0, mi=0):#, maximum=None, minimum=None):
    mins = mi#np.min(rawpoints, axis=0)
    maxs = ma#np.max(rawpoints, axis=0)
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

    try:

        if segment["truth"] == '|': return None, None, None

        resolution = 32
        image_resolution = 36

        image = Image.new('L', (image_resolution, image_resolution), "white")
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

        if scale > 1:
            # width > height
            height_scale = resolution / scale
        else:
            # width < height
            width_scale = resolution * scale


        for trace in segment["traces"]:
            y = np.array(trace).astype(np.float)

            x, y = y.T

            new_x = []
            new_y = []

            if width_scale > 0:
                # add padding in x-direction
                new_y = scale_linear_bycolumn(y, high=resolution, low=0, ma=max_y, mi=min_y)
                side = (resolution - width_scale)/2
                new_x = scale_linear_bycolumn(x, high=(resolution-side), low=(side), ma=max_x, mi=min_x)
            else:
                # add padding in y-direction
                new_x = scale_linear_bycolumn(x, high=resolution, low=0, ma=max_x, mi=min_x)#, maximum=(max_x, max_y), minimum=(min_x, min_y))
                side = (resolution - height_scale)/2
                new_y = scale_linear_bycolumn(y, high=(resolution-side), low=(side), ma=max_y, mi=min_y)#, maximum=(max_x, max_y), minimum=(min_x, min_y))


            coordinates = list(zip(new_x, new_y))
            xy_cycle = cycle(coordinates)

            next(xy_cycle)

            for x_coord, y_coord in coordinates[:-1]:
                next_coord = next(xy_cycle)
                draw.line([x_coord, y_coord, next_coord[0], next_coord[1]], fill="black", width=1)

        filename = segment["truth"] + "_" + segment["id"] + ".jpg"
        segment_id = segment["id"]
        truth = segment["truth"]
        return image, truth, segment_id
    except:
        print("Item failed")
        return None, None, None

def generate_bitmaps(segments):
    for segment in segments:
        yield generate_bitmap(segment)

if __name__ == '__main__':
    count = 0
    dirs = ['validation', 'train']
    random.seed(100)
    for file in os.listdir(os.getcwd() + '/data'):
        full_filename = os.getcwd() + '/data/' + file
        try:
            tree = ET.parse(full_filename)
        except:
            print("Failed to parse tree")
            continue

        root = tree.getroot()

        segments = find_segments(root)

        for segment in segments:
            count += 1
            if count < 20000: continue
            if count % 1000 == 0: print("Processing number", count)
            image, truth, segment_id = generate_bitmap(segment)

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

            image.save(filename)




    #generate_bitmaps(segments)

    #generate_bitmap(segments[0])

