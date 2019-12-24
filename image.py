import numpy as np

class MonoImage:
	def __init__(self, width, height, data):
		self.width = width
		self.height = height

		# reduce the color space to grayscale
		self.data = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])
		# self.data = data

	def __repr__(self):
		return repr(self.data)

	def __str__(self):
		return self.__repr__()

	def __getitem__(self, key):
		return self.data[key[0]][key[1]]

	def __setitem__(self, key, item):
		self.data[key[0]][key[1]] = item
		return self.data[key[0]][key[1]]
		
	def render(self, canvas):
		for j in range(self.height):
			for i in range(self.width):
				canvas.create_oval(i, j, i + 1, j + 1, outline = '#ffffff' if self.data[j, i] else '#000000', fill = '')
		canvas.update()
		# todo: optimize image drawing
		# canvas.create_image(0, 0, image = self.data, anchor = NW)

def rgbToHex(color, lpadding = '#'):
	return lpadding + ''.join('{:02X}'.format((c) % 256) for c in color)