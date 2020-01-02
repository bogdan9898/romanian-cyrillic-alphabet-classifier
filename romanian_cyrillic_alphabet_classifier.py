import os
from tkinter import *
import imageio
# from multiprocessing import Process
import json
import random
from copy import copy
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

def train(dataSet, labelsOrder, model):
	# process training data
	input_train_data = []
	train_data_labels = []
	# j = 0
	for letter in dataSet['train']:
		# print(j, letter)
		# j += 1
		# for el in letter:
		nr_elements = min(len(dataSet['train'][letter]), 80)
		samples = random.sample(dataSet2['train'][letter], nr_elements)
		input_train_data.extend(samples)
		for i in range(nr_elements):
			# print('\t', i)
			# el = dataSet['train'][letter][i]
			# input_train_data.append(el)
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
	if not fileName.startswith('--'):
		model = models.load_model('./models/' + fileName)

def saveModel2():
	global model2
	if not os.path.isdir('./models/'):
		os.mkdir('./models/')
	model2.save('./models/' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.h5')

def loadModel2():
	global model2
	fileName = loadModelSelectedFile.get()
	if not fileName.startswith('--'):
		model2 = models.load_model('./models/' + fileName)

def calcFScore(dataSet, labelsOrder, model):
	stats = {}
	for letter in labelsOrder:
		stats[letter] = {
			'right predictions': 0,
			'total predictions': 0,
		}
	for letter in dataSet['test']:
		for sample in dataSet['test'][letter]:
			input_data = np.array([sample])
			input_data = input_data.reshape(*input_data.shape, 1)
			prediction = model.predict(input_data)
			output_letter = labelsOrder[np.argmax(prediction)]
			if output_letter == letter:
				stats[letter]['right predictions'] += 1
				stats[letter]['total predictions'] += 1
			else:
				stats[output_letter]['total predictions'] += 1
	print(json.dumps(stats, indent=2))
	fscores = {}
	for letter in stats:
		precision = calcPrecision(stats, letter)
		recall = calcRecall(stats, letter, dataSet)
		if precision <= 0 and recall <= 0:
			fscores[letter] = 0
		else:
			fscores[letter] = 2 * ((precision * recall) / (precision + recall))
	return fscores

def calcPrecision(stats, letter):
	if stats[letter]['total predictions'] <= 0:
		return 0
	return stats[letter]['right predictions'] / stats[letter]['total predictions']

def calcRecall(stats, letter, dataSet):
	if len(dataSet['test'][letter]) <= 0:
		return 0
	return stats[letter]['right predictions'] / len(dataSet['test'][letter])

def printFScore():
	global dataSet, labelsOrder, model
	fscore = calcFScore(dataSet, labelsOrder, model)
	print('Model 1:\n\tF-Score:')
	print(json.dumps(fscore, indent=2))
	print('\tAverage f-score:', sum(fscore.values()) / len(fscore.values()))

	global dataSet2, model2
	fscore = calcFScore(dataSet2, labelsOrder, model2)
	print('Model 2:\n\tF-Score:')
	print(json.dumps(fscore, indent=2))
	print('\tAverage f-score:', sum(fscore.values()) / len(fscore.values()))

def trainM1():
	global dataSet, labelsOrder, model
	train(dataSet, labelsOrder, model)

def trainM2():
	global dataSet2, labelsOrder, model2
	train(dataSet2, labelsOrder, model2)

def genModel():
	global model2
	model2 = Sequential()
	model2.add(Conv2D(filters=10, kernel_size=8, padding='same', activation='relu', input_shape=(100, 100, 1)))
	model2.add(MaxPool2D(pool_size=(3, 3)))
	model2.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu'))
	model2.add(MaxPool2D(pool_size=(3, 3)))
	model2.add(Conv2D(filters=3, kernel_size=4, padding='same', activation='relu'))
	model2.add(Conv2D(filters=5, kernel_size=4, padding='same', activation='relu'))
	model2.add(Flatten())
	model2.add(Dense(32, activation='softmax'))
	model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# GUI setup
root = Tk()
root.title('Romanian cyrillic alphabet classifier')
root.resizable(0, 0)

bkgFrame = Frame(root, width=500, height=500, bg='#264653')
bkgFrame.pack()
bkgFrame.grid_propagate(0)
bkgFrame.pack_propagate(0)

frame1 = Frame(bkgFrame, bg='#264653')
frame1.place(relx=0.5, rely=0.5, anchor=CENTER)
# frame1.pack()
# frame1.pack_propagate(0) # dont allow widgets to change size
# frame1.grid_propagate(0)

genModel = Button(frame1, text='gen model', command=genModel)
genModel.grid()

predictedOutput = StringVar()
predictedOutput.set('No output')
Label(frame1, textvariable=predictedOutput, bg='#264653', fg='#FFFFFF', font=(None, 20)).grid(row=0, columnspan=2)#.pack()

canvas = Canvas(frame1, width=100, height=100, bg='#264653')
# canvas.pack()
canvas.grid(row=1, columnspan=2)
canvas.bind('<Button-1>', showImages)

trainM1Button = Button(frame1, text='Train model 1', command=trainM1, bg='#2A9D8F', highlightthickness=0, pady=10)
trainM1Button.grid(row=2, column=0)
# trainM1Button.pack()

trainM2Button = Button(frame1, text='Train model 2', command=trainM2, bg='#2A9D8F', highlightthickness=0, pady=10)
trainM2Button.grid(row=2, column=1)
# trainM2Button.pack()

predictButton = Button(frame1, text='Predict', command=predict, bg='#2A9D8F', highlightthickness=0, pady=10)
predictButton.grid(row=3, columnspan=2)
# predictButton.pack()

printFScore = Button(frame1, text='Print F-Score', command=printFScore, bg='#2A9D8F', highlightthickness=0, pady=10)
printFScore.grid(row=4, columnspan=2)
# printFScore.pack()

modelsDirectoryPath = './models/'
modelFileOptions = [
	'-- Already existing models --',
]
with os.scandir(modelsDirectoryPath) as folder:
	for entry in folder:
		if not entry.name.startswith('.') and not entry.is_dir():
			modelFileOptions.append(entry.name)
print(modelFileOptions)

loadModelSelectedFile = StringVar()
loadModelSelectedFile.set(modelFileOptions[0])
OptionMenu(frame1, loadModelSelectedFile, *modelFileOptions).grid(row=5, columnspan=2)#.pack()

loadModelButton = Button(frame1, text='Load model 1', command=loadModel, bg='#2A9D8F', highlightthickness=0, pady=10)
loadModelButton.grid(row=6, column=0)
# loadModelButton.pack()

saveModelButton = Button(frame1, text='Save model 1', command=saveModel, bg='#2A9D8F', highlightthickness=0, pady=10)
saveModelButton.grid(row=7, column=0)
# saveModelButton.pack()

loadModelButton2 = Button(frame1, text='Load model 2', command=loadModel2, bg='#2A9D8F', highlightthickness=0, pady=10)
loadModelButton2.grid(row=6, column=1)
# loadModelButton2.pack()

saveModelButton2 = Button(frame1, text='Save model 2', command=saveModel2, bg='#2A9D8F', highlightthickness=0, pady=10)
saveModelButton2.grid(row=7, column=1)
# saveModelButton2.pack()
# end GUI setup

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

######
model2 = Sequential()
model2.add(Conv2D(filters=10, kernel_size=8, padding='same', activation='relu', input_shape=(100, 100, 1)))
model2.add(MaxPool2D(pool_size=(3, 3)))
model2.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu'))
model2.add(MaxPool2D(pool_size=(3, 3)))
model2.add(Conv2D(filters=3, kernel_size=4, padding='same', activation='relu'))
model2.add(Conv2D(filters=5, kernel_size=4, padding='same', activation='relu'))
model2.add(Flatten())
model2.add(Dense(32, activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

dataSet2 = copy(dataSet)
tmp = ['dz', 'ia']
for t in tmp:
	# print(len(dataSet2['train'][t]))
	for i in range(3):
		dataSet2['train'][t].extend(dataSet2['train'][t])
		dataSet2['train'][t].extend(dataSet2['test'][t])
	# print(len(dataSet2['train'][t]))
######

root.mainloop()

