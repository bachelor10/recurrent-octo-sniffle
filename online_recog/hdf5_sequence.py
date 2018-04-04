import functools

import h5py
import keras
from keras import backend, Sequential
import tensorflow as tf
import tensorflow
from keras import backend as K
from keras import layers
from keras.callbacks import Callback
from keras.layers import LSTM, Bidirectional, BatchNormalization, Activation, Embedding, Masking, Conv1D
from keras.utils import Sequence
import numpy as np

from online_recog.create_trdata import *
from online_recog.tf_sequence import *


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
		self.file = h5py.File('raw_train.hdf5', 'r')
		self.class_file = h5py.File('raw_class.hdf5', 'r')
		self.classes, self.num_classes = find_classes('raw_data_classes.txt')
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
		batch_x = pad_sequences(batch_x, dtype=np.float32, padding='pre')
		
		return np.array([batch_x]), np.array(batch_y)


if __name__ == '__main__':
	epochs = 5
	verbose = 1
	x_batch_shape = [64, 550, 1, 1]
	
	_num_classes = find_classes('raw_data_classes.txt')
	
	train_generator = InkMLSequence()
	print(len(train_generator.x_name))
	model = Sequential()
	model.add(BatchNormalization())
	model.add(Conv1D(filter=48, kernel_size=5, activation='relu'))
	model.add(Dropout(0.3))
	model.add(BatchNormalization())
	model.add(Conv1D(filter=64, kernel_size=5))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Conv1D(filter=96, kernel_size=3))
	model.add(Flatten())
	
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	
	print(len(train_generator.x_name))
	print(type(train_generator))
	
	model.fit_generator(
		train_generator,
		epochs=epochs
	)
