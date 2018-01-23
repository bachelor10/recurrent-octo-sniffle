import xml.etree.ElementTree as ET
import uuid
import numpy as np
import matplotlib.pyplot as plt

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


segments = []


for group in root.findall('{http://www.w3.org/2003/InkML}traceGroup'):
    for item in group.findall('{http://www.w3.org/2003/InkML}traceGroup'):

        id = uuid.uuid4()

        truth = item.find('{http://www.w3.org/2003/InkML}annotation').text
        segments.append(dict())
        i = len(segments) - 1

        segments[i]["truth"] = truth
        segments[i]["traces"] = []

        for trace in item.findall('{http://www.w3.org/2003/InkML}traceView'):
            traceId = trace.attrib['traceDataRef']

            segments[i]["traces"].append(find_trace(root, traceId))


# +
y = np.array(segments[0]["traces"][0]).astype(np.float)

x, y = y.T

width = x.max() - x.min()
height = y.max() - y.min()

goal = 24

scale = width / height

new_y = scale_linear_bycolumn(y, high=24, low=0)
side = (scale/2)*24
new_x = scale_linear_bycolumn(x, high=(24-side), low=(side))


plt.scatter(new_y, new_x)

plt.xlim(0, 24)
plt.ylim(0, 24)

plt.show()
