import os
import numpy as np
from machine_learning.xml_parse import format_trace, find_segments
from online_recog.xml_parse_rawdata import get_inkml_root

def max_segmentlength():
	longest = []
	directory = os.curdir + "/BACHELOR_DATA" + "/11_TESTGT"
	for filename in os.listdir(directory):
		if filename.endswith(".inkml"):
			if not (filename is None):
				root = get_inkml_root(directory + "/" + filename)
				segm = find_segments(root)
				for i, t in enumerate(segm):
					for traces in t.traces:
						print(len(traces))
						longest.append(len(traces))
			
	return np.max(longest)

print(max_segmentlength())