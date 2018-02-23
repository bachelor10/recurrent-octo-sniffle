import keras
from keras import backend
import tensorflow as tf


class SeqTrainer:
	def __init__(self):
		self.classes = []
	
	def get_classes(self):
		raise NotImplementedError()


'''
operation = True => predict
operation = False => train
'''


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
	#print(tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10])))
	return res, labels


if __name__ == '__main__':
	parse_tf_ex(file='test.tfrecords')
