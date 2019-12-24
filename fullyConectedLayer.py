import numpy as np

class FullyConected:
	def __init__(self, inputSize, outputSize):
		self.inputSize = int(inputSize)
		self.outputSize = int(outputSize)
		self.weights = np.random.rand(outputSize, inputSize) - 0.5

	def __repr__(self):
		return 'FullyConectedLayer => input size: {0}  ouput size: {1}'.format(self.inputSize, self.outputSize)

	def __str__(self):
		return self.__repr__()

	def feedForward(self, x):
		# todo: flatten x to 1 dimensional array, matrix dot product between x and weights
		pass

	def backpropagation(self, trainData, testData):
		pass