from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import tensorflow as tf
import h5py
from scipy import signal

from machine_learning.xml_parse import format_trace, find_segments


class SeqParser:
	def __init__(self):
		print("s")


def plot_trace(trace):  # trace is x,y coordinates
	plt.plot(trace[:, 0], trace[:, 1])
	axes = plt.gca()
	axes.set_xlim([np.min(trace[:, 0]) - 1.0, np.max(trace[:, 0]) + 1.0])
	axes.set_ylim([np.min(trace[:, 1]) - 1.0, np.max(trace[:, 1]) + 1.0])
	axes.invert_yaxis()
	
	plt.show()


def plot_sequential(trace):
	raise NotImplementedError()


def consecutive_segments(segm):
	# print("SEGM", segm)
	# print(type(segm))
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
def parse_single_segment(segment):  # segment object
	truth = segment.truth
	traces = segment.traces
	stroke_lengths = find_tracelength(traces)
	total_points = sum(stroke_lengths)
	np_ink = np.zeros((total_points, 3), dtype=np.float32)
	it = 0
	
	for i, trace in enumerate(traces):
		print(i)
		trace = np.array(trace).astype(np.float32)
		np_ink[it:(it + len(trace)), 0:2] = trace
		it += len(trace)
		np_ink[it - 1, 2] = 1  # stroke end
	
	lower = np.min(np_ink[:, 0:2], axis=0)
	upper = np.max(np_ink[:, 0:2], axis=0)
	
	# plot_trace(np_ink)
	
	relation_between = upper - lower
	relation_between[relation_between == 0] = 1
	np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / relation_between
	
	np_ink = np_ink[1:, 0:2] - np_ink[0:-1, 0:2]
	np_ink = np_ink[1:, :]
	return np_ink, truth


def get_inkml_root(file):
	return ET.parse(file).getroot()


def seg_to_tfexample(filename, writer = None, directory=None):
	truths = []
	print("Trying to write segments to file.")
	try:
		for file in os.listdir(directory):
			
			if file.endswith(".inkml"):
				root = get_inkml_root(directory + file)
				segm = find_segments(root)
				for i, t in enumerate(segm):
					if i < 1:
						tf_ex, truth = parse_single_segment(t)
						print(truth)
						if truth not in truths:
							truths.append(truth)
						ftore = tf.train.Example(features=tf.train.Features(feature={
							'truth_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[truths.index(truth)])),
							'ink': tf.train.Feature(float_list=tf.train.FloatList(value=tf_ex.flatten())),
							'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=tf_ex.shape))
						}))
						writer.write(ftore.SerializeToString())
		with tf.gfile.GFile(filename + ".txt", "w",) as f:
			for t in truths:
				f.write(t + "\n")
		return truths
	except ET.ParseError:
		print("Error handling file: ", file)
		


datafile = 'data.h5'
classfile = 'classes.h5'


def seg_to_npr(directory=None, file=None):
	print("Trying to write segments to file.")
	
	truths = []
	data = []
	# for files in directory
	# parse inkml
	# find segments
	# for each file, we get an array of segments.
	'''

	'''
	if not (file is None):
		root = get_inkml_root(file)
		segm = find_segments(root)
	else:
		return None
	
	'''
		t here are segment objects.
	'''
	for i, t in enumerate(segm):
		#print(t.truth)
		if i < 1:
			tf_ex, truth = parse_single_segment(t)
			if truth not in truths:
				truths.append(truth)
				#with h5py.File(datafile, "w") as f:
					#data = f.create_dataset("data",data=tf_ex)
	


# send to handler here
# need to have shape somehow ?
# 1. read
# 2. format and scale
# 3. write to correct file
# 4. test writing and reading from file
# 5. ready for training


if __name__ == '__main__':
	root = get_inkml_root('01.inkml')
	segments = find_segments(root)  # gets a list of Segment objects
	
	# print(segments)
	# consecutive_segments(segments)
	writer = tf.python_io.TFRecordWriter('GT.tfrecords')
	trt = seg_to_tfexample("GT", writer=writer, directory=os.curdir+"/BACHELOR_DATA/GT/")
	arr = np.array(trt)
	np.savez('truth_index.npz',data=arr)
	writer.close()
	
