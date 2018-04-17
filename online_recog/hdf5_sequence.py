import functools

import h5py
import keras
from keras import backend, Sequential
import tensorflow as tf
import tensorflow
from keras import backend as K
from keras import layers
from keras.callbacks import Callback
from keras.layers import LSTM, Bidirectional, BatchNormalization, Activation, Embedding, Masking, Conv1D, TimeDistributed
from keras.utils import Sequence
import numpy as np
import os
from scipy.optimize import curve_fit


from create_trdata import *
from tf_sequence import *


def populate_data_names(fil):
	list = []
	print("Populating data names.")
	with h5py.File(fil, "r") as f:
		for d in f:
			list.append(d)
	return list


# Parse data into self.x and self.y, and split them into train, validation.Testdata is avalable in another subdirectory

# The
class InkMLSequence(Sequence):  # hint: https://gist.github.com/alxndrkalinin/6cc4228e9178ec4af7b2696a0d1ad5a1
	
	def __init__(self, batch_size=1):  # batch size is one because one matrix is several traces
		self.batch_size = batch_size
		self.file = h5py.File(os.getcwd() + '/data/raw_train.hdf5', 'r')
		self.class_file = h5py.File(os.getcwd() + '/data/raw_class.hdf5', 'r')
		self.classes, self.num_classes = find_classes(os.getcwd() + '/data/raw_data_classes.txt')
		self.x_name, self.y_name = populate_data_names(self.file.filename), populate_data_names(
			self.class_file.filename)
	
	def __len__(self):
		return int(np.ceil(len(self.x_name) / float(self.batch_size)))
	
	def __getitem__(self, ix):
		batch_x = self.file[self.x_name[ix]][
			()]  # self.file[self.x_name[ix * self.batch_size:(ix + 1) * self.batch_size]]
		batch_y = self.class_file[self.y_name[ix]][
			()]  # trailing [()] https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
		batch_y = keras.utils.to_categorical(batch_y, self.num_classes)
		print("Batch y", batch_y)
		batch_x = pad_sequences(batch_x, dtype=np.float32, padding='pre')
		print("Batch shape after pad", batch_x.shape)

		return np.array([batch_x]), np.array([batch_y])


if __name__ == '__main__':

	file = h5py.File(os.getcwd() + '/data/raw_train.hdf5', 'r')
	class_file = h5py.File(os.getcwd() + '/data/raw_class.hdf5', 'r')
	classes, num_classes = find_classes(os.getcwd() + '/data/raw_data_classes.txt')
	x_name_arr, y_name_arr = populate_data_names(file.filename), populate_data_names(
		class_file.filename)

	print(y_name_arr)
	
	def gaussian(x, amp, cen, wid):
	    return amp * np.exp(-(x-cen)**2 / wid)

	for y_name, x_name in zip(y_name_arr, x_name_arr):
		batch_x = file[x_name]
		batch_y = class_file[y_name]


		print(classes[batch_y[0]]])
		single_symbol = batch_x[0]
		x_vals = single_symbol[:, 0]
		x_vals = single_symbol[:, 0]


		print("Before", single_symbol)
		
		best_vals, covar = curve_fit(gaussian, , y)




	




	"""
	epochs = 5
	verbose = 1
	x_batch_shape = [64, 550, 1, 1]
	
	_num_classes = find_classes(os.getcwd() + '/data/raw_data_classes.txt')
	print("Num classes", _num_classes)
	
	train_generator = InkMLSequence()
	print(len(train_generator.x_name))
	model = Sequential()
	model.add(TimeDistributed(Conv1D(48, kernel_size=5, activation='relu'),  input_shape=(13, 106)))
	model.add(Dropout(0.3))
	model.add(BatchNormalization())
	model.add(TimeDistributed(Conv1D(64, kernel_size=5)))
	model.add(Dropout(0.2))
	model.add(TimeDistributed(Conv1D(96, kernel_size=3)))
	
	model.add(LSTM(64, input_shape=(13, 106), return_sequences=True))
	model.add(Dense(_num_classes[1]))
	
	print("Summary", model.summary())
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	
	print(len(train_generator.x_name))
	print(type(train_generator))
	
	model.fit_generator(
		train_generator,
		epochs=epochs
	)
"""