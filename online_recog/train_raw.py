import functools

import keras
from keras import backend
import tensorflow as tf
import tensorflow
from keras import backend as K
from keras import layers
from keras.callbacks import Callback
from keras.utils import Sequence
import numpy as np

from online_recog.xml_parse_rawdata import *

'''
class EvaluateInputTensor(Callback):
	def __init__(self, model, steps, metrics_prefix='val', verbose=1):
		super(EvaluateInputTensor, self).__init__()
		self.val_model = model
		self.num_steps = steps
		self.verbose = verbose
		self.metrics_prefix = metrics_prefix
	
	def on_epoch_begin(self, epoch, logs=None):
		if logs is None:
			logs = {}
		results = self.val_model.evaluate(None, None, steps=int(self.num_steps), verbose=self.verbose)
		metrics_str = "\n"
		
		for res, name in zip(results, self.val_model.metrics_names):
			metric_name = self.metrics_prefix + "_" + name
			logs[metric_name] = res
			if self.verbose > 0:
				metrics_str = metrics_str + metric_name + ": " + str(res) + " "
		
		if self.verbose > 0:
			print(metrics_str)
'''

'''
	operation = True => predict
	operation = False => train
'''


def in_func(mode, directory, batch_size):
	def parse_tf_ex(file, operation=False):
		ft_to_ink = {
			'ink': tf.VarLenFeature(dtype=tf.float32),
			'shape': tf.FixedLenFeature([2], dtype=tf.int64)  # this is rows,cols of ink data
		}
		if not operation:
			ft_to_ink['truth_index'] = tf.FixedLenFeature([1], dtype=tf.int64)
		
		res = tf.parse_single_example(file, ft_to_ink)
		labels = None
		if not operation:
			labels = res['truth_index']
		
		res['ink'] = tf.sparse_tensor_to_dense(res['ink'])
		# print(tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10])))
		return res, labels
	
	def in_func():
		data = tf.data.TFRecordDataset.list_files(directory)
		if not mode:
			data = data.shuffle(buffer_size=10)
		data = data.repeat()
		data = data.interleave(tf.data.TFRecordDataset, cycle_length=10, block_length=1)
		data = data.map(functools.partial(parse_tf_ex, mode=mode), num_parallel_calls=10)
		data = data.prefetch(100)
		if not mode:
			data = data.shuffle(buffer_size=10000)
		
		data = data.padded_batch(batch_size, padded_shapes=data.output_shapes)
		features, labels = data.make_one_shot_iterator().get_next()
		return features, labels
	
	return in_func()

# https://stackoverflow.com/questions/42184863/how-do-you-make-tensorflow-keras-fast-with-a-tfrecord-dataset
# https://www.tensorflow.org/versions/master/tutorials/recurrent_quickdraw
# https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
# FUCK !

class InkMLSequence(Sequence):
	
	def __init__(self, x_set, y_set, batch_size):
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size
		self.directory = ''
	
	def __len__(self):
		return np.ceil(len(self.x) / float(self.batch_size))
	
	def __getitem__(self, ix):
		batch_x = self.x[ix * self.batch_size:(ix + 1) * self.batch_size]
		batch_y = self.y[ix * self.batch_size:(ix + 1) * self.batch_size]


class SeqTrainer:
	def __init__(self):
		self.truths = []
		self.data = []
	
	def get_classes(self):
		raise NotImplementedError()
	
	def preprocess(self, directory):
		pass


if __name__ == '__main__':
	print("")
