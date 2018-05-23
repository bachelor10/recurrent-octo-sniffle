### Sources and externals for recurrent-octo-sniffle

- File: /online_recog/rdp_test.py
    - RDP algorithm adjusted to our need.
	- F. Hirschmann, rdp: Python/Numpy implementation of the Ramer-Douglas-Peucker algorithm. 2018.
	- https://github.com/fhirschmann/rdp [MIT]
- File: /online_recog/xml_parse.py
    - scale_linear_bycolumn
	- M. Perry, Normalize a 2D numpy array so that each “column” is on the same scale (Linear stretch from lowest value = 0 to highest value = 100). 2013.
	- https://gist.github.com/perrygeo/4512375
- File: visualization/activation_functions.py
    - Used activation functions from this gist.
    - Yusuke Sugomori, Dropout Neural Networks (with ReLU), 2015
    - https://gist.github.com/yusugomori/cf7bce19b8e16d57488a

- File: online_recog/keras_lstm.py
    - Storage callback inspiration
    - Joel (https://github.com/joelthchao)
    - https://github.com/keras-team/keras/issues/2548