import os
from tkinter import *
import imageio
from multiprocessing import Process
import json
from datetime import datetime
import math
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.utils import to_categorical
from keras import models
# pip3 install tensorflow==1.5
# pip3 install keras==2.1.5

index = 0
letter = 'a'
def showImages(event): # DELETE ME!
	global index, letter, canvas, dataSet
	# dataSet[letter][index].render(canvas)
	renderImage(dataSet['train'][letter][index])
	index += 1

def renderImage(imageData): # only grayspace images
	global dataSet
	for j in range(imageData.shape[0]):
		for i in range(imageData.shape[1]):
			canvas.create_oval(i, j, i + 1, j + 1, outline = '#ffffff' if imageData[j, i] else '#000000', fill = '')
	canvas.update()

def readDirectory(path):
	dirData = {}
	dirLabels = []
	with os.scandir(path) as folder:
		for entry in folder:
			if not entry.name.startswith('.') and entry.is_dir():
				print(entry.path)
				dirLabels.append(entry.name)
				with os.scandir(entry.path) as folder2:
					for entry2 in folder2:
						if not entry2.name.startswith('.'):
							file_path = os.path.join(entry.path, entry2.name)
							# print(file_path)
							data = imageio.imread(file_path)
							width = data.shape[1]
							height = data.shape[0]
							if not isinstance(dirData.get(entry.name), list):
								dirData[entry.name] = []
							dirData[entry.name].append(np.dot(data[...,:3], [0.2989, 0.5870, 0.1140]))#.astype(np.float128))
							# dirData[entry.name].append(data)

	# converts labels indexes to binary lists
	# length = len(dirLabels)
	# paddingWidth = math.ceil(math.log(length, 2))
	# tmp = {}
	# for labelIndex in range(length):
	# 	tmp[dirLabels[labelIndex]] = list(int(x) for x in '{number:{fill}>{width}b}'.format(number=labelIndex, fill=0, width=paddingWidth))
	# dirLabels = tmp
	return dirData, dirLabels

def loadData(path):
	print('loading data...')
	global dataSet, labelsOrder
	# p1 = Process(target=readDirectory, args=(path + '/train/', ))
	# p1.start()
	# p2 = Process(target=readDirectory, args=(path + '/test/', ))
	# p2.start()
	# p1.join()
	# p2.join()
	dataSet['train'], labelsOrder = readDirectory(path + '/train/')
	dataSet['test'],_ = readDirectory(path + '/test/')
	print('data loaded')
	# print(dataSet.keys())
	# print(labelsOrder)
	# 
	# for el in dataSet:
	# 	print(el, ':\n', list(dataSet[el].keys()), '\n', labelsOrder)
	# 	print(list(dataSet[el].keys()) == labelsOrder)
	# print(labelsOrder)

def train():
	global dataSet, labelsOrder, model
	# process training data
	input_train_data = []
	train_data_labels = []
	# j = 0
	for letter in dataSet['train']:
		# print(j, letter)
		# j += 1
		# for el in letter:
		for i in range(min(len(dataSet['train'][letter]), 999999)):
			# print('\t', i)
			el = dataSet['train'][letter][i]
			input_train_data.append(el)
			tmp = [0 for x in range(len(labelsOrder))]
			tmp[labelsOrder.index(letter)] = 1
			train_data_labels.append(tmp)

	input_train_data = np.array(input_train_data)
	input_train_data = input_train_data.reshape(*input_train_data.shape, 1) # to stop keras from complainig about: expected conv2d_1_input to have 4 dimensions
	train_data_labels = np.array(train_data_labels)
	print(input_train_data.shape)
	print(train_data_labels.shape)

	# process test data
	input_test_data = []
	test_data_labels = []

	for letter in dataSet['test']:
		for i in range(min(len(dataSet['test'][letter]), 999999)):
			el = dataSet['test'][letter][i]
			input_test_data.append(el)
			tmp = [0 for x in range(len(labelsOrder))]
			tmp[labelsOrder.index(letter)] = 1
			test_data_labels.append(tmp)

	input_test_data = np.array(input_test_data)
	input_test_data = input_test_data.reshape(*input_test_data.shape, 1)
	test_data_labels = np.array(test_data_labels)
	print(input_test_data.shape)
	print(test_data_labels.shape)

	model.fit(input_train_data, train_data_labels, validation_data=(input_test_data, test_data_labels), epochs=8, batch_size=16)
	# print('todo...')

def predict():
	global dataSet, labelsOrder, model, predictedOutput
	# print(to_categorical(labelsOrder, num_classes=len(labelsOrder)))
	# return
	input_data = dataSet['test']['a2'][0]
	renderImage(input_data)
	input_data = np.array([input_data])
	input_data = input_data.reshape(*input_data.shape, 1)
	print(input_data.shape)
	prediction = model.predict(input_data)
	print(prediction)
	predictedLetter = labelsOrder[np.argmax(prediction)]
	print(np.argmax(prediction), predictedLetter)
	predictedOutput.set(predictedLetter)

def saveModel():
	global model
	if not os.path.isdir('./models/'):
		os.mkdir('./models/')
	model.save('./models/' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.h5')

def loadModel():
	global model
	fileName = loadModelSelectedFile.get()
	model = models.load_model('./models/' + fileName)

# setup GUI
root = Tk()
root.title('Romanian cyrillic alphabet classifier')
root.resizable(0, 0)

frame1 = Frame(root, width=500, height=500, bg='#264653')
frame1.pack()
frame1.pack_propagate(0) # dont allow widgets to change size

predictedOutput = StringVar()
predictedOutput.set('Predicted output will appear here')
Label(frame1, textvariable=predictedOutput, bg='#264653', fg='#FFFFFF', font=(None, 20)).pack()

canvas = Canvas(frame1, width=100, height=100, bg='#264653')
canvas.pack()
canvas.bind('<Button-1>', showImages)

trainButton = Button(frame1, text='Train', command=train, bg='#2A9D8F', highlightthickness=0, pady=10)
trainButton.pack()

predictButton = Button(frame1, text='Predict', command=predict, bg='#2A9D8F', highlightthickness=0, pady=10)
predictButton.pack()

modelsDirectoryPath = './models/'
saveModelButton = Button(frame1, text='Save model', command=saveModel, bg='#2A9D8F', highlightthickness=0, pady=10)
saveModelButton.pack()

loadModelButton = Button(frame1, text='Load model', command=loadModel, bg='#2A9D8F', highlightthickness=0, pady=10)
loadModelButton.pack()

modelFileOptions = [
	'Choose a file to load an existing model',
]
with os.scandir(modelsDirectoryPath) as folder:
	for entry in folder:
		if not entry.name.startswith('.') and not entry.is_dir():
			modelFileOptions.append(entry.name)
print(modelFileOptions)

loadModelSelectedFile = StringVar()
loadModelSelectedFile.set(modelFileOptions[0])
OptionMenu(frame1, loadModelSelectedFile, *modelFileOptions).pack()
# end setup GUI

'''
dataSet: {
	'train': {	
		'a': ndarray,
		'b': ndarray,
		...
	},
	'test': {	
		'a': ndarray,
		'b': ndarray,
		...
	},
}
labelsOrder: [
	'a',
	...
]
'''
dataSet = {}
labelsOrder = []
loadData('char_trainable_split')
print(labelsOrder)

# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=5, activation='relu', input_shape=(100, 100, 1)))
# model.add(MaxPool2D(pool_size=(3, 3)))
# model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
# model.add(MaxPool2D(pool_size=(3,3)))
# model.add(Flatten())
# model.add(Dense(32, activation='softmax'))
# model.compile(optimizer='adam', loss='mean_squared_error')
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=8, padding='same', activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Conv2D(filters=3, kernel_size=4, padding='same', activation='relu'))
#
# model.add(MaxPool2D(pool_size=(4, 4)))
model.add(Conv2D(filters=5, kernel_size=4, padding='same', activation='relu'))
#
model.add(Flatten())
model.add(Dense(32, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

root.mainloop()

