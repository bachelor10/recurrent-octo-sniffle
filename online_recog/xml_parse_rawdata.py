import xml.etree.ElementTree as ET
import uuid, math, time
import numpy as np
from PIL import Image, ImageDraw
from itertools import cycle
import random
import os
import matplotlib.pyplot as plt
import matplotlib.quiver as quiv
from rdp import rdp, rdp_iter  # https://pypi.python.org/pypi/rdp

from machine_learning.xml_parse import format_trace, find_segments

class SeqParser:
	def __init__(self):
		print("s")
	


def plot_trace(trace): # trace is x,y coordinates
	plt.plot(trace[:, 0], trace[:, 1])
	axes = plt.gca()
	axes.set_xlim([np.min(trace[:, 0]) - 1.0, np.max(trace[:, 0]) + 1.0])
	axes.set_ylim([np.min(trace[:, 1]) - 1.0, np.max(trace[:, 1]) + 1.0])
	axes.invert_yaxis()
	
	plt.show()

def plot_sequential(trace):
	raise NotImplementedError()
	

def consecutive_segments(segm):
	#print("SEGM", segm)
	#print(type(segm))
	for i, t in enumerate(segm):
		print(t.truth)
		if True:
			parse_single_segment(t)


def find_tracelength(trace):
	lengths = []
	for i, t in enumerate(trace):
		lengths.append(len(t))
	return lengths

# this method is meant to parse a segment, sequentially return data and truth
# https://www.tensorflow.org/versions/master/tutorials/recurrent_quickdraw
def parse_single_segment(segment): # segment object
	truth = segment.truth
	traces = segment.traces
	stroke_lengths = find_tracelength(traces)
	total_points = sum(stroke_lengths)
	np_ink = np.zeros((total_points, 3), dtype=np.float32)
	it = 0
	
	for trace in traces:
		trace = np.array(trace).astype(np.float32)
		np_ink[it:(it + len(trace)), 0:2] = trace
		it += len(trace)
		np_ink[it - 1, 2] = 1 # stroke end
		
	
	
	lower_x = np.min(np_ink[:, 0:2], axis=0)
	lower_y = np.min(np_ink[:, 0:2], axis=0)
	upper_x = np.max(np_ink[:, 0:2], axis=0)
	upper_y = np.max(np_ink[:, 0:2], axis=0)
	
	plot_trace(np_ink)
	
	relation_between = upper_x - lower_x
	relation_between[relation_between == 0] = 1
	np_ink[:, 0:2] = (np_ink[:, 0:2] - lower_x) / relation_between
	np_ink = np_ink[1:, 0:2] - np_ink[0:-1, 0:2]
	#print (np_ink)
	return np_ink, truth
	

if __name__ == '__main__':
	tree = ET.parse('01.inkml')
	root = tree.getroot()
	segments = find_segments(root) # gets a list of Segment objects

	# print(segments)
	consecutive_segments(segments)
