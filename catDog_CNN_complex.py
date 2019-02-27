import cv2
import numpy as np
import pandas as pd
import os         
from random import shuffle 
from tqdm import tqdm      
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from CNN_defines import *

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 64
LR = 1e-3
MODEL_NAME = 'dogs-vs-cats-convnet'

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
#train_data = create_train_data(TRAIN_DIR, IMG_SIZE)
#test_data = create_test_data(TEST_DIR, IMG_SIZE)
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')
test_data = test_data[test_data[:,1].argsort()]

#######################
#####Train Dataset#####
#######################
X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = [i[1] for i in train_data]

X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)
convnet = dropout(convnet, 0.75)
convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)
convnet = conv_2d(convnet, 256, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)
convnet = dropout(convnet, 0.75)
convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)
convnet = dropout(convnet, 0.75)
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.75)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=50,
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

y_pred = []
for test_d in X_test:
    y_predict = model.predict([test_d])[0]
    if y_predict[0]>y_predict[1]:
        y_pred.append(0)
    else:
        y_pred.append(1)

    
#y_pred = [y_pred[i][0]<y_pred[i][1] for i in y_pred]
ImageId = [i[1] for i in test_data]
result = pd.DataFrame({
    'id' : ImageId,
    'label' : y_pred
    })
result.to_csv(SCRIPT_PATH + "/submission_cnn_complex.csv", index=False)