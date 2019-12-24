import scipy
import skimage
import numpy as np

class Pooling:
	def __init__(self, filterSize, stride):
		self.filterSize = int(filterSize)
		self.stride = int(stride)

	def __repr__(self):
		return 'PoolingLayer => filter size: {0}x{0}  stride: {1}'.format(self.filterSize, self.stride)

	def __str__(self):
		return self.__repr__()

	def feedForward(self, x):
		y = []
		if isinstance(x, list):
			for el in x:
				y.append(skimage.measure.block_reduce(x, block_size = (self.filterSize, self.filterSize), func = np.max, cval = 255))
		elif isinstance(x, np.ndarray):
			# y = scipy.ndimage.maximum_filter(x, size = (self.filterSize, self.filterSize), mode = 'constant', cval = 255)
			y = skimage.measure.block_reduce(x, block_size = (self.filterSize, self.filterSize), func = np.max, cval = 255)
		else:
			raise Exception('Invalid type: ' + str(type(x)))
		return y

	def backpropagation(self, trainData, testData):
		pass