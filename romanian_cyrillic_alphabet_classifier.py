import os
from tkinter import *
import imageio

# custom modules
from image import *
from cnn import CNN

index = 0
letter = 'a'
def showImage(self):
	global index
	global letter
	global canvas
	dataSet[letter][index].render(canvas)
	index += 1


def readDirectory(path):
	with os.scandir(path) as folder:
		for entry in folder:
			if not entry.name.startswith('.') and entry.is_dir():
				print(entry.path)
				with os.scandir(entry.path) as folder2:
					for entry2 in folder2:
						if not entry2.name.startswith('.'):
							file_path = os.path.join(entry.path, entry2.name)
							# print(file_path)
							data = imageio.imread(file_path)
							width = data.shape[1]
							height = data.shape[0]
							if not isinstance(dataSet.get(entry.name), list):
								dataSet[entry.name] = []
							dataSet[entry.name].append(MonoImage(width, height, data))

def loadData(path):
	readDirectory(path + '/train/')
	readDirectory(path + '/test/')

root = Tk()
canvas = Canvas(root, width = 100, height = 100)
canvas.pack()
canvas.bind('<Button-1>', showImage)

dataSet = {}

loadData('char_trainable_split')

root.mainloop()