import os
import numpy as np
from machine_learning.xml_parse import format_trace, find_segments
from online_recog.xml_parse_rawdata import get_inkml_root


#TODO close filstream somehow (?)
def max_segmentlength():
	longest = []
	directory = os.curdir + "/BACHELOR_DATA" + "/GT"
	for filename in os.listdir(directory):
		if filename.endswith(".inkml"):
			if not (filename is None):
				root = get_inkml_root(directory + "/" + filename)
				segm = find_segments(root)
				for i, t in enumerate(segm):
					for k, traces in enumerate(t.traces):
						# print(k)
						# print(traces, "len", len(traces), "len0", len(traces[0]))
						longest.append(len(traces) * len(traces[0]))
	
	return np.max(longest)


print(max_segmentlength())
