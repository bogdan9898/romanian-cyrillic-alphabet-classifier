from convolutionLayer import *
from rectifiedLinearUnits import *
from poolingLayer import *
from fullyConectedLayer import *
import math

class CNN:
	def __init__(self, imageResolutin, layers):
		# layers must be formated: C2,3-R-P2,2-C3,3-R-F4...
		# Cn1,n2 -> convolutio layer, n2 = filter size, n1 = number of filter
		# R -> RELU layer
		# Pn1,n2 -> polling layer, n1 = filter size, n2 = stride
		# Fn -> fully conected layer, n = number of output neurons
		self.layers = []
		layers = layers.upper()
		layers = layers.split('-')
		print(layers)
		currentLayerShape = [100, 100, 1]
		for layer in layers:
			if layer.startswith('C'):
				n1, n2 = layer[1:].split(',')
				self.layers.append(Convolution(n1, n2))
				currentLayerShape[2] *= int(n2)
			elif layer.startswith('R'):
				self.layers.append(ReLU())
			elif layer.startswith('P'):
				n1, n2 = layer[1:].split(',')
				self.layers.append(Pooling(n1, n2))
				currentLayerShape[0] = int(math.ceil(currentLayerShape[0] / int(n2)))
				currentLayerShape[1] = int(math.ceil(currentLayerShape[1] / int(n2)))
			elif layer.startswith('F'):
				n1 = int(layer[1:])
				self.layers.append(FullyConected(currentLayerShape[0] * currentLayerShape[1] * currentLayerShape[2], n1))
				currentLayerShape[0] = int(n1)
				currentLayerShape[1] = 1
				currentLayerShape[2] = 1
			else:
				raise Exception('Invalid argument: {0}'.format(layer))

			# print(currentLayerShape)

	def __repr__(self):
		t = 'CNN:\n'
		for layer in self.layers:
			t += layer.__repr__() + '\n'
		return t

	def __str__(self):
		return self.__repr__()

	def feedForward(self, data):
		pass

	def train(self, trainData, testData):
		pass