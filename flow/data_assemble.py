import json
import xml.etree.ElementTree as ET
import numpy as np
from machine_learning.xml_parse import find_segments
from online_recog.create_dataset import find_tracelength

def get_inkml_root(file):
	return ET.parse(file).getroot()

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
		
if __name__ == '__main__':
	root = get_inkml_root('01.inkml')
	segments = find_segments(root)  # gets a list of Segment objects
	for s in segments:
		trace, trt = convert_segment_tojson(s)
		print("TRACE: ", trace)
		print("TRUTH: ", trt)