import scipy.signal
import numpy as np

class Convolution:
	def __init__(self, filterSize, filtersCount):
		self.filterSize = int(filterSize)
		self.filters = []
		for i in range(int(filtersCount)):
			self.filters.append(np.random.randint(-2, 2, (int(filterSize), int(filterSize))))

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
					y.append(scipy.signal.convolve2d(el, kernel, mode = 'same'))
		elif isinstance(x, np.ndarray):
			for kernel in self.filters:
				y.append(scipy.signal.convolve2d(x, kernel, mode = 'same'))
		else:
			raise Exception('Invalid type: ' + str(type(x)))
		return y

	def backpropagation(self, trainData, testData):
		pass