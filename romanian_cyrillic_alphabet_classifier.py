import os
from tkinter import *
import imageio

# custom modules
from image import *
from cnn import CNN

index = 0
letter = 'a'
def showImages(event):
	global index
	global letter
	global canvas
	# dataSet[letter][index].render(canvas)
	renderImage(dataSet[letter][index])
	index += 1

def renderImage(imageData):
	for j in range(imageData.shape[0]):
		for i in range(imageData.shape[1]):
			canvas.create_oval(i, j, i + 1, j + 1, outline = '#ffffff' if imageData[j, i] else '#000000', fill = '')
	canvas.update()

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
							# dataSet[entry.name].append(MonoImage(width, height, data))
							dataSet[entry.name].append(np.dot(data[...,:3], [0.2989, 0.5870, 0.1140]))#.astype(np.float128))


def loadData(path):
	print('loading data...')
	readDirectory(path + '/train/')
	readDirectory(path + '/test/')
	print('data loaded')

def train():
	print('todo...')

root = Tk()
canvas = Canvas(root, width = 100, height = 100)
canvas.pack()
canvas.bind('<Button-1>', showImages)

dataSet = {}
loadData('char_trainable_split')

# cnn = CNN((100, 100), 'C10,4-R-C4,2-R-P3,3-C5,2-R-P4,4-F32')
# cnn = CNN((100, 100), 'C10,4-R-C4,2-R-P2,2-C5,2-R-P2,2-F32')
cnn = CNN((100, 100), 'C5,4-R-P2,2-C5,4-R-P2,2-F32')
# cnn = CNN((100, 100), 'C10,7-R-P3,3-C5,5-R-P2,2-F32')

print(cnn)

trainButton = Button(root, text = 'Train', command = train)
trainButton.pack()

result = cnn.feedForward(dataSet['a'][0])
print(result.shape)
print(result)
print(np.argmax(result))

root.mainloop()

