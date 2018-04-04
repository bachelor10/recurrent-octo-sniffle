import random

import keras
import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.python.ops import data_flow_ops

from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras import callbacks as cbks
from keras import optimizers, objectives
from keras import metrics as metrics_module

if K.backend() != 'tensorflow':
	raise RuntimeError("TfRecords does not support something other than Tensorflow.")
random.seed(198123647582)


def parse_ink(ex, is_predict):
	feature_to_type = {
		"ink": tf.VarLenFeature(dtype=tf.float32),
		"shape": tf.FixedLenFeature([2], dtype=tf.int64)
	}
	if not is_predict:
		feature_to_type['truth_index'] = tf.FixedLenFeature([1], dtype=tf.int64)
	parsed_fts = tf.parse_single_example(ex, feature_to_type)
	labels = None
	if not is_predict:
		labels = parsed_fts['truth_index']
	parsed_fts['ink'] = tf.sparse_tensor_to_dense(parsed_fts['ink'])
	return parsed_fts, labels


def model_fn(features, labels, mode, params):
	def get_in_tensors(fts, labels):
		shapes = fts['shape']
		lengths = tf.squeeze(tf.slice(shapes, begin=[0, 0], size=[params.batch_size, 1]))
		inks = tf.reshape(fts['ink'], [params.batch_size, -1, 3])
		if labels is not None:
			labels = tf.squeeze(labels)
		return inks, lengths, labels
	
	def add_conv_layers(inks):
		cnved = inks
		for i in range(len(params.num_conv)):
			x = cnved
			x = keras.layers.BatchNormalization(x)
			x = keras.layers.Conv1D(filters=params.num_conv[i], kernel_size=params.conv_len[i], activation=None,
									padding='same')(x)
			x = Dropout(0.3)(x)
			cnved = x
		return x
	
	def add_rnn_layers(cnved, lengths):
		x = keras.layers.Gru()


def read_and_decode_rec(tf_glob, one_hot=True, classes=None, is_predict=False,
						batch_shape=(64, 270, 2, 1), parallelism=1):
	print("Creating graph for %s TFRecords..." % tf_glob)
	with tf.variable_scope("TFRecords"):
		record_input = data_flow_ops.RecordInput(
			tf_glob, batch_size=batch_shape[0], parallelism=parallelism
		)
		records_op = record_input.get_yield_op()
		records_op = tf.split(records_op, batch_shape[0], 0)
		records_op = [tf.reshape(record, []) for record in records_op]
		progbar = Progbar(len(records_op))
		
		data = []
		labels = []
		
		for i, serialized_ex in enumerate(records_op):
			progbar.update(i)
			with tf.variable_scope("parse_ink", reuse=True):
				ft, lab = parse_ink(serialized_ex, is_predict)
				label = None
				if one_hot and classes is not None and lab is not None:
					label = tf.one_hot(lab, classes)
				data.append(ft['ink'])
				labels.append(label)
		
		print("\n")
		
		return data, labels


def split_data(data, labels, split=0.3):
	print(len(data))
	print(len(labels))
	train_size = int(len(data) * (1 - 0.3))
	train_x, test_x = data[0:train_size, :], data[train_size:len(data), :]
	train_y, test_y = labels[0:train_size, :], labels[train_size:len(data), :]


# if random.uniform(0,1) > (1-split):
#

# Get array of all objects
# Parse data to np array
# REMEMBER TO SPLIT TEST AND TRAIN DATA!
# Use np array and create LSTM network
# Train
#

def find_classes(filename=None):
	if not filename:
		filename = 'GT.txt'
	classes = []
	with open(filename) as f:
		for line in f:
			classes.append(line.rstrip())
	return classes, len(classes)

def add_cnn_layers(x_train):
	x = keras.layers.BatchNormalization()(x_train)
	x = keras.layers.Conv1D(filter=48, kernel_size=5, activation='relu', padding='valid')(x)
	x = keras.layers.Dropout(0.3)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Conv1D(filter=64, kernel_size=5)(x)
	x = keras.layers.Dropout(0.2)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Conv1D(filter=96, kernel_size=3)(x)
	x = keras.layers.Flatten()(x)
	return x
	
def add_rnn_layers(x_train):
	print("HW")


if __name__ == '__main__':
	# classes, num_classes = find_classes('GT_Kopi.txt')
	# print(classes, num_classes)
	classes = ['y', 'n', '0', 'A', 'x', 'a', 'Y', '(', ',', '\\gamma', 'd', '2', 'e', 'w', 'T', 'R', 'q', 'B', '1', 'r',
			   '\\Delta', 'G', 'm', 'l', '\\mu', '\\int', 'S', '|', '-', 'u', '5', 'M', 'P', 'o', 's', 'L', '\\{',
			   '\\sigma', 'c', '\\sqrt', '\\alpha', 'v', 'V', 'C', 't', 'f', 'g', 'H', 'p', 'E', '4', 'F', 'N', '3',
			   'b', 'I', '[', 'X', '\\theta', '\\infty', '\\times', '\\phi', '+', '6', '\\sum', '/', 'z', '8', '9',
			   '\\beta', '7', 'k', 'i', '\\cos', '\\log', '\\sin', '\\pi', '\\lim', '\\lambda', '\\tan', '\\pm',
			   '\\exists', '\\forall', '\\geq', 'h']
	num_classes = len(classes)
	epochs = 5
	x_batch_shape = [64, 270, 2, 1]
	
	x_train_batch, y_train_batch = read_and_decode_rec(
		'GT.tfrecords',
		one_hot=True,
		classes=num_classes,
		batch_shape=x_batch_shape
	)
	x_test_batch, y_test_batch = read_and_decode_rec(
		'GT_T.tfrecords',
		one_hot=True,
		classes=num_classes,
		batch_shape=x_batch_shape
	)
	print(x_train_batch)
	print(type(x_batch_shape))
	x_train_input = Input(tensor=x_train_batch, batch_shape=x_batch_shape)
	#x_train_out =