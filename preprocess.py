import time
import glob
import os
import cv2
import numpy as np

from sklearn.model_selection import StratifiedKfold

def load_train(img_per_cat = 10):
    X_train = []
    X_train_id = []
    y_train = []
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
            img = cv2.imread(fl)
            X.append(img)
            X_id.append(flbase)
            y.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return np.array(X), np.array(y), np.array(X_id)

def train_test_split(X,y,kratio = 5, ind = 0):
    skf = StratifiedKFold(kratio)
    k=0
    for train_index, test_index in skf.split(X,y) :
        if k < ind:
            k+=1
        else :
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            return X_train,X_test,y_train,y_test
