import scipy
import numpy as np

class Convolution:
	def __init__(self, filterSize, filersCount):
		self.filterSize = int(filterSize)
		self.filters = []
		for i in range(int(filersCount)):
			self.filters = np.random.randint(-2, 2, (int(filterSize), int(filterSize)))

	def __repr__(self):
		return 'ConvolutionLayer => filter size: {0}x{0} filters count: {1}'.format(self.filterSize, len(self.filters))

	def __str__(self):
		return self.__repr__()

	def feedForward(self, x):
		y = []
		if isinstance(x, list):
			# list of np arrays
			for el in x:
				for kernel in self.filters:
					y.append(scipy.signal.convolve2d(el, (kernel, kernel), boundary = 'fill', fillvalue = 255))
		elif isinstance(x, np.ndarray):
			for kernel in self.filters:
				y.append(scipy.signal.convolve2d(x, (kernel, kernel), boundary = 'fill', fillvalue = 255))
		else:
			raise Exception('Invalid type: ' + str(type(x)))
		return y

	def backpropagation(self, trainData, testData):
		pass