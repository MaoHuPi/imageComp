'''
2024 (c) MaoHuPi
imageComp/main.py
'''
VERSION = '1.0.0'

# parse arguments
import os
def str2bool(s):
	return str.lower(s) in ['true', '1', 't', 'y']
argDescription = {
	'SHOW_HELP':           [['bool'],         'Show this arguments description.'], 
	'SHOW_VERSION':        [['bool'],         'Show the current version.'], 
	'SOURCE_DIR':          [['str'],          'Set the directory of source images. Selected image must in this directory.'], 
	'TARGET_DIR':          [['str', 'False'], 'Set the directory to which the output file will be copy.'], 
	'SELECTED_IMAGE_NAME': [['str'],          'Set the image that the other images should be compare with.'], 
	'HSV_W':               [['list[3]'],      'Set the degree of influence of h, s and v on the compare score which used for comparison.'], 
	'UPDATE_CATCH':        [['bool'],         'Update catch data after the weight fitted.'], 
	'USE_CATCH':           [['bool'],         'Use catch data when it is exist.'], 
	'COMPARE_MODE':        [['str'],          'Can be head|last|all.'], 
	'OUTPUT_NUMBER_LIMIT': [['int'],          'Set the maxima output number.'], 
	'MIN_VAL_ACCURACY':    [['float'],        'Set the \'val_accuracy\' value to early stopping the training process.']
}
argKeys = {
	'SHOW_HELP':           ['-h', '--help'], 
	'SHOW_VERSION':        ['-v', '--version'], 
	'SOURCE_DIR':          ['-s', '--source_dir'], 
	'TARGET_DIR':          ['-t', '--target_dir'], 
	'SELECTED_IMAGE_NAME': ['-i', '--selected_image'], 
	'HSV_W':               ['-w', '--hsv_w'], 
	'UPDATE_CATCH':        ['-u', '--update_catch'], 
	'USE_CATCH':           ['-c', '--use_catch'], 
	'COMPARE_MODE':        ['-m', '--compare_mode'], 
	'OUTPUT_NUMBER_LIMIT': ['-l', '--output_number_limit'], 
	'MIN_VAL_ACCURACY':    ['-a', '--min_val_accuracy']
}
defaultConstData = {
	'SHOW_HELP':           'False', 
	'SHOW_VERSION':        'False', 
	'SOURCE_DIR':          None, 
	'TARGET_DIR':          'False', 
	'SELECTED_IMAGE_NAME': None, 
	'HSV_W':               '[1, 0.2, 0.2]', 
	'UPDATE_CATCH':        'True', 
	'USE_CATCH':           'True', 
	'COMPARE_MODE':        'head', 
	'OUTPUT_NUMBER_LIMIT': '5', 
	'MIN_VAL_ACCURACY':    '0.7', 
}
constData = {}
currentConstKey = None
for argv in os.sys.argv:
	isArgKey = False
	for constKey in argKeys:
		if argv in argKeys[constKey]:
			isArgKey = True
			currentConstKey = constKey
			constData[constKey] = 'True'
	if (not isArgKey) and currentConstKey is not None:
		constData[currentConstKey] = argv
constData = {**defaultConstData, **constData}
hasShown = False
if str2bool(constData['SHOW_HELP']):
	helpDescription = '\n'.join([''.join([
		', '.join(argKeys[constKey]) + ' (' + '|'.join(argDescription.get(constKey, [['Undefined'], 0])[0]) + ')', 
		f'\n\tDefault value: {defaultConstData[constKey]}.' if defaultConstData.get(constKey) else '', 
		'\n\t', 
		argDescription.get(constKey, [0, 'No description.'])[1]
	]) for constKey in argKeys])
	print(helpDescription)
	hasShown = True
if str2bool(constData['SHOW_VERSION']):
	print('imageComp version: ' + VERSION)
	hasShown = True
for constKey in constData:
	if constData[constKey] is None:
		if hasShown: exit()
		else: raise Exception(f'The value of parameter {constKey} is not specified. Use ' + ' or '.join([f"'{argKey}'" for argKey in argKeys[constKey]]) + ' to set the value.')

# basic import
import json
from pathlib import Path
import cv2
import numpy as np
import shutil

# basic method
def prepareImage(url:str):
	image = cv2.imread(url, cv2.IMREAD_COLOR)
	if image is None:
		extension = str.lower(url.split('.')[-1])
		supportedFormats = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm', 'sr', 'ras', 'tiff', 'tif', 'exr', 'hdr', 'pic']
		if extension not in supportedFormats:
			raise Exception(f'Can not read image({url}) of unsupported format({extension}).')
		else:
			raise Exception(f'Can not read image correctly({url}).')
	image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	return image
def getDifferenceList(selectedData, compDataList, w = False):
	differenceList = []
	for data in compDataList:
		if w: differenceList.append(np.sum(np.abs(selectedData - data)[:,:]*w, axis=None, dtype=float))
		else: differenceList.append(np.sum(np.abs(selectedData - data), axis=None, dtype=float))
	differenceMin = 0 # 因為選擇的圖片與其自身之差異為 0 且相似度為 100%
	differenceMax = np.max(differenceList)
	differenceList = np.array(differenceList)
	differenceList = 1 - (differenceList - differenceMin)/(differenceMax - differenceMin)
	return differenceList

# load and prepare training data
sourceDir = os.path.abspath(constData['SOURCE_DIR'])
if not os.path.exists(sourceDir): raise Exception(f'Can not found source directory({sourceDir}).')
if constData['TARGET_DIR'] != 'False':
	targetDir = os.path.abspath(constData['TARGET_DIR'])
	if not os.path.exists(targetDir): os.makedirs(targetDir)
selectedImage = os.path.join(sourceDir, constData['SELECTED_IMAGE_NAME'])
compareImages = [os.path.join(sourceDir, name) for name in os.listdir(sourceDir) if name not in [constData['SELECTED_IMAGE_NAME'], 'catchData']]
if not os.path.isfile(selectedImage): raise Exception(f'Can not found selected image({selectedImage}).')
xTrain = np.array([prepareImage(compareImage) for compareImage in [selectedImage, *compareImages]])
xTrain = xTrain/255
yTrain = np.array(list(range(1+len(compareImages))))

# training import
from tensorflow.keras.backend import eval as TFKEval
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import Callback
from PIL import Image
import matplotlib.pyplot as plt

# create model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(256, 256, 3), padding='same', activation='relu', name='conv2d_1'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1+len(compareImages), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
yTrain = to_categorical(yTrain)
class CustomCallback(Callback):
	def on_test_batch_end(self, batch, logs=None):
		if logs.get('accuracy') >= float(constData['MIN_VAL_ACCURACY']):
			self.model.stop_training = True
			print(f'\nval_accuracy >= {constData["MIN_VAL_ACCURACY"]}\n')
catchLoaded = False
catchDataDir = os.path.join(sourceDir, 'catchData')
currentLabel = [os.path.basename(path) for path in [selectedImage, *compareImages]]
labelIndexMap = list(range(len(currentLabel)))
if os.path.isdir(catchDataDir) and os.path.isfile(os.path.join(catchDataDir, 'situation')) and str2bool(constData['USE_CATCH']):
	situationCatchFile = open(os.path.join(catchDataDir, 'situation'), 'r+')
	situation = situationCatchFile.read()
	situationCatchFile.close()
	if situation:
		situation = json.loads(situation)
		label = situation['label']
		currentLabel.sort()
		label.sort()
		if ''.join(label) == ''.join(currentLabel):
			model.load_weights(os.path.join(catchDataDir, 'weights'))
			labelIndexMap = [currentLabel.index(label[i]) for i in range(len(label))]
			catchLoaded = True
if not catchLoaded:
	model.fit(xTrain, yTrain,
		batch_size=64,
		epochs=50,
		verbose=1,
		validation_data=(xTrain, yTrain),
		callbacks=[CustomCallback()])
	if str2bool(constData['UPDATE_CATCH']):
		if os.path.isdir(catchDataDir): shutil.rmtree(catchDataDir)
		if os.path.isfile(catchDataDir): os.unlink(catchDataDir)
		os.mkdir(catchDataDir)
		model.save_weights(os.path.join(catchDataDir, 'weights').format(epoch=0))
		situationCatchFile = open(os.path.join(catchDataDir, 'situation'), 'w+')
		situationCatchFile.write(json.dumps({'label': currentLabel, 'min_val_accuracy': constData['MIN_VAL_ACCURACY']}))
		situationCatchFile.close()

# calculate difference
selectedImageFeature = None
compareImagesFeature = []
selectedImageOutput = None
compareImagesOutput = []
conv2d_1 = model.get_layer('conv2d_1')
for i in range(len(xTrain)):
	feature = conv2d_1(np.array([xTrain[i]])).numpy()
	if i == 0: selectedImageFeature = feature
	else: compareImagesFeature.append(feature)
	output = model.predict(np.array([xTrain[i]]))
	if i == 0: selectedImageOutput = output
	else: compareImagesOutput.append(output)
differenceList = [
	getDifferenceList(xTrain[0], xTrain[1:], w=json.loads(constData['HSV_W'])), 
	getDifferenceList(selectedImageFeature, compareImagesFeature), 
	getDifferenceList(selectedImageOutput, compareImagesOutput)
]
differenceList = np.sum(differenceList, axis=0) / len(differenceList)
rank = sorted([[labelIndexMap[i], differenceList[i]] for i in range(len(differenceList))], key=lambda ISPair: ISPair[1], reverse=True)

# output
limit = np.abs(int(constData['OUTPUT_NUMBER_LIMIT']))
if constData['COMPARE_MODE'] == 'head': rank = rank[:limit]
elif constData['COMPARE_MODE'] == 'last': rank = rank[-limit:]
elif constData['COMPARE_MODE'] == 'all': rank = rank
else: raise Exception('COMPARE_MODE must be \'head\', \'last\' or \'all\'.')
print(*['{similarity}% :\n{path}'.format(path=compareImages[ISPair[0]], similarity=str(int(ISPair[1]*100)).rjust(3, ' ')) for ISPair in rank], sep='\n')
if constData['TARGET_DIR'] != 'False':
	for i in [ISPair[0] for ISPair in rank]:
		shutil.copyfile(compareImages[i], os.path.join(targetDir, os.path.basename(compareImages[i])))