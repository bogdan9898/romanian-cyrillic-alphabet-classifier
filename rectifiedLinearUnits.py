class ReLU:
	def __init__(self):
		pass

	def __repr__(self):
		return 'ReLULayer'

	def __str__(self):
		return self.__repr__()

	def feedForward(self, x):
		y = []
		if isinstance(x, list):
			for el in x:
				y.append(el.clip(0))
		elif isinstance(x, np.ndarray):
			y = x.clip(0) # all negative values become 0
		else:
			raise Exception('Invalid type: ' + str(type(x)))
		return y

	def backpropagation(self, trainData, testData):
		pass