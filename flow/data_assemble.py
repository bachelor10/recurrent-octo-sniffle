import json
import codecs
import os
import xml.etree.ElementTree as ET
import numpy as np
from machine_learning.xml_parse import find_segments
from online_recog.create_dataset import find_tracelength
from rdp import rdp
import newlinejson as nlj

remove_punctuation_map = dict((ord(char), None) for char in '\/*?:"<>|')


def get_inkml_root(file):
	return ET.parse(file).getroot()


'''
writeclass_ndjson
Writes traces to an ndjson file
'''


def __writeclass_ndjson(trace_data, class_name, out_path = "DATA"):
	out_data = list()
	for trace in trace_data:
		x = trace[:,0]
		y = trace[:,1]
		tmp = list()
		tmp.append(x.tolist())
		tmp.append(y.tolist())
		out_data.append(tmp)
	# creaing map
	data = {}
	data['class'] = class_name
	data['drawing'] = out_data
	with nlj.open("DATA/" + class_name.strip('\\') + ".ndjson", "a") as dst:
		dst.write(data)
	


'''
scale_uniform
Scales data into a 256x256 region
Uses RDP to reduce number of strokes

'''


def __scale_uniform(data, class_name, max=255):
	if len(data) > 1:
		for t in data:
			low = np.min(t[:])
			upp = np.max(t[:])
	else:
		low = np.min(data[:])
		upp = np.max(data[:])
	# print("Low: ", low, "\nUpper: ", upp)
	scale = max/upp
	
	if len(data) > 1:
		data = np.asarray(data[:]) * scale
		for i, d in enumerate(data):
			data[i] = d.astype(int)
	else:
		data = np.asarray(data) * scale
		data = data.astype(int)

	output_data = []
	for trace in data:
		newtrace = rdp(trace, epsilon=1.0)
		output_data.append(newtrace)
	__writeclass_ndjson(output_data, class_name)


def convert_segment_tojson(segment):  # segment object
	truth = segment.truth
	traces = segment.traces
	
	stroke_lengths = find_tracelength(traces)
	total_points = sum(stroke_lengths)
	np_ink = np.zeros((total_points, 3), dtype=np.float32)
	it = 0
	if not traces:
		print("No trace provided.")
		return None, None
	
	for i, trace in enumerate(traces):
		trace = np.array(trace).astype(np.float32)
		np_ink[it:(it + len(trace)), 0:2] = trace
		it += len(trace)
		np_ink[it - 1, 2] = 1  # stroke end
	return traces, truth
	lower = np.min(np_ink[:, 0:2], axis=0)
	upper = np.max(np_ink[:, 0:2], axis=0)
	
	# plot_trace(np_ink)
	
	relation_between = upper - lower
	relation_between[relation_between == 0] = 1
	np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / relation_between
	
	np_ink = np_ink[1:, 0:2] - np_ink[0:-1, 0:2]
	np_ink = np_ink[1:, :]


if __name__ == '__main__':
	data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../online_recog/GT"
	
	for file in os.listdir(data_dir):
		if file.endswith(".inkml"):
			root = get_inkml_root(os.path.join(data_dir, file))
			segments = find_segments(root)  # gets a list of Segment objects
			for s in segments:
				traces = s.traces
				truth = s.truth
				if traces is not None:
					if len(traces) > 0:
						if len(traces[0]) > 1:
							__scale_uniform(traces, truth)
						else:
							print("Skipping trace from file %s, because of trace size. " % file)
					else:
						print("Skipping trace from file %s, because of missing trace. " % file)
		else:
			print("Skipping: ", file)
