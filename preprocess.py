import time
import glob
import os
import cv2
import numpy as np

from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
def resize_img(img,width=720, height=1280):
    return cv2.resize(img,(width,height),interpolation = cv2.INTER_LINEAR)
def readimg(path, width = 720, height = 1280):
    img = cv2.imread(path)
    return resize_img(img, width, height)

def normalize_data(X,y = None):
    X = np.array(X,dtype=np.uint8)
    y = np.array(y,dtype=np.uint8)

    print('Convert to float...')
    X = X.astype('float32')
    X = X / 255
    y = np_utils.to_categorical(y, 8)

    print('Train shape:', X.shape)
    print(X.shape[0], 'train samples')
    return X, y


def load_train(img_per_cat = 10):
    X = []
    X_id = []
    y = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('.', 'train', fld, '*.jpg')
        files = glob.glob(path)
        i = 0
        for fl in files:
            if i > img_per_cat : break
            i+=1
            flbase = os.path.basename(fl)
            img = readimg(fl)
            X.append(img)
            X_id.append(flbase)
            y.append(index)
    # X,y = normalize_data(X,y)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X,y,X_id

def train_test_split(X,y,split_ratio = 5, ind = 0):
    skf = StratifiedKFold(split_ratio)
    k=0
    for train_index, test_index in skf.split(X,y) :
        if k < ind:
            k+=1

        else :
            print train_index, test_index
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            X_train,y_train= normalize_data(X_train, y_train)
            X_test,y_test= normalize_data(X_test, y_test)

            return X_train,X_test,y_train,y_test
