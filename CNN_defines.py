from tqdm import tqdm
import cv2
import numpy as np
import os         
from random import shuffle 
def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])
    
def create_train_data(train_dir, img_size=50):
    TRAIN_DIR = train_dir
    IMG_SIZE = img_size
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
def create_test_data(test_dir, img_size=50):
    TEST_DIR = test_dir
    IMG_SIZE = img_size
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        testing_data.append([np.array(img_data), int(img_num)])
        
    #shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data