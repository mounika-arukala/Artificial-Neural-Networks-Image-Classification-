"""
#!/usr/bin/python

#===============================================
# image_manip.py
#
# some helpful hints for those of you
# who'll do the final project in Py
#
# bugs to vladimir dot kulyukin at usu dot edu
#===============================================
from __future__ import division, print_function, absolute_import
import cv2
import numpy as np
import os

# two dictionaries that map integers to images, i.e.,
# 2D numpy array.
TRAIN_IMAGE_DATA = {}
TEST_IMAGE_DATA  = {}

# the train target is an array of 1's
TRAIN_TARGET = []
# the set target is an array of 0's.
TEST_TARGET  = []

### Global counters for train and test samples
NUM_TRAIN_SAMPLES = 0
NUM_TEST_SAMPLES  = 0

## define the root directory
ROOT_DIR = 'C:/Users/mouni/Documents/TOC_Proj/nn_train/'

## read the single bee train images
YES_BEE_TRAIN = ROOT_DIR + 'single_bee_train'

for root, dirs, files in os.walk(YES_BEE_TRAIN):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TRAIN_IMAGE_DATA[NUM_TRAIN_SAMPLES] = img
            TRAIN_TARGET.append(int(1))
        NUM_TRAIN_SAMPLES +=1


## read the single bee test images
YES_BEE_TEST = ROOT_DIR + 'single_bee_test'

for root, dirs, files in os.walk(YES_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            # print img.shape
            TEST_IMAGE_DATA[NUM_TEST_SAMPLES] = img
            TEST_TARGET.append(int(1))
        NUM_TEST_SAMPLES += 1

## read the no-bee train images
NO_BEE_TRAIN = ROOT_DIR + 'no_bee_train'

for root, dirs, files in os.walk(NO_BEE_TRAIN):
    for item in files:
        if item.endswith('.png')::
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TRAIN_IMAGE_DATA[NUM_TRAIN_SAMPLES] = img
            TRAIN_TARGET.append(int(0))
        NUM_TRAIN_SAMPLES += 1
        
# read the no-bee test images
NO_BEE_TEST = ROOT_DIR + 'no_bee_test'

for root, dirs, files in os.walk(NO_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TEST_IMAGE_DATA[NUM_TEST_SAMPLES] = img
            TEST_TARGET.append(int(0))
        NUM_TEST_SAMPLES += 1

print (NUM_TRAIN_SAMPLES)
print (NUM_TEST_SAMPLES)
TRAIN_IMAGE_CLASSIFICATIONS = zip([k for k in TRAIN_IMAGE_DATA.keys()], TRAIN_TARGET)
TEST_IMAGE_CLASSIFICATIONS = zip([k for k in TEST_IMAGE_DATA.keys()], TEST_TARGET)

TRAIN_IMAGE_DATA=list(TRAIN_IMAGE_DATA.values())
TRAIN_IMAGE_DATA=np.array(TRAIN_IMAGE_DATA)
TEST_IMAGE_DATA=list(TEST_IMAGE_DATA.values())
TEST_IMAGE_DATA=np.array(TEST_IMAGE_DATA)
TRAIN_TARGET=np.array(TRAIN_TARGET)
TEST_TARGET=np.array(TEST_TARGET)

#TRAIN_IMAGE_DATA=TRAIN_IMAGE_DATA.astype(np.float32)
#TEST_IMAGE_DATA=TEST_IMAGE_DATA.astype(np.float32)
#TRAIN_TARGET=TRAIN_TARGET.astype(np.float32)
#TRAIN_TARGET=TRAIN_TARGET.astype(np.float32)
######################################


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from keras.utils import np_utils

X = TRAIN_IMAGE_DATA
Y = TRAIN_TARGET
X_test = TEST_IMAGE_DATA
Y_test = TEST_TARGET
Y = np_utils.to_categorical(Y)
Y_test = np_utils.to_categorical(Y_test)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)


# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],data_preprocessing=img_prep,data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.002)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=200, run_id='mounika')
model.save("C:/Users/mouni/Documents/TOC_Proj/mounika1.model")
model.load("C:/Users/mouni/Documents/TOC_Proj/mounika1.model")
predictions = model.predict(X_test)

single_bees = []
no_bees = []

no_bees_count = 0       
for i in Y_test:
    if i[1] < 0.5:
        no_bees_count = no_bees_count+1        
single_bees_count = len(Y_test)-no_bees_count

for i in predictions[0:single_bees_count]:
    if i[1] < 0.5:
        no_bees.append(0)
    else:
        single_bees.append(1)

print("Single bees data accuracy is",(len(single_bees)/single_bees_count)*100)

single_bees = []
no_bees = []

for i in predictions[single_bees_count:len(Y_test)]:
    if i[1] < 0.5:
        no_bees.append(0)
    else:
        single_bees.append(1)

print("No bees data accuracy is",(len(no_bees)/no_bees_count)*100)
"""

import cv2
import numpy as np
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from keras.utils import np_utils

def testNet(netpath,dirpath):
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    
    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    
    
    # Convolutional network building
    network = input_data(shape=[None, 32, 32, 3],data_preprocessing=img_prep,data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.002)
    
    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.load(netpath)
    
    NO_BEE_TEST = dirpath
    NUM_TEST_SAMPLES = 0
    TEST_IMAGE_DATA={}
    TEST_TARGET = []
    for root, dirs, files in os.walk(NO_BEE_TEST):
        for item in files:
            if item.endswith('.png'):
                ip = os.path.join(root, item)
                img = (cv2.imread(ip)/float(255))
                TEST_IMAGE_DATA[NUM_TEST_SAMPLES] = img
                TEST_TARGET.append(int(0))
            NUM_TEST_SAMPLES += 1
    TEST_IMAGE_DATA=list(TEST_IMAGE_DATA.values())
    test_X=np.array(TEST_IMAGE_DATA)
    test_Y=np.array(TEST_TARGET)
    predictions = model.predict(test_X)
    for i in predictions:
        if i[1] < 0.5:
            print("NO BEES")
        else:
            print("SINGLE BEE")

testNet("C:/Users/mouni/Documents/TOC_Proj/mounika1.model",
        "C:/Users/mouni/Documents/TOC_Proj/nn_train/test_images")
