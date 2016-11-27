from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss


class NN(BaseEstimator):
    def __init__(self, layers =[{'type': Dense, 'params':{'output_dim':8,'activation':'relu'}}],
                optimizer=SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True), loss="categorical_crossentropy",
                batch_size =16, nb_epoch=30, verbose = 1, shuffle = True,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0),]):
        self.layers = layers
        self.optimizer = optimizer
        self.loss= loss
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.shuffle = shuffle
        self.callbacks = callbacks
        self._on = False

    def fit(self,X,y,sample_weight = None):

        self.model = Sequential()
        for layer in self.layers :
            if self.verbose >= 1 : print("Adding "+ layer['type'] +"(" +str(layer['params']) +") ...")
            self.model.add(layer["type"](**layer["params"]))

        self.model.compile(optimizer=self.optimizer, loss = self.loss)
        self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
              shuffle=self.shuffle, verbose=self.verbose, validation_data=(X_valid, Y_valid),
              callbacks=self.callbacks)
        self._on = True
        
        return self
    def predict(self,X,sample_weight = None):
        if self._on == True :
            return self.model.predict(X, batch_size=self.batch_size, verbose = self.verbose)
        else :
            return 0

    def score(self, X,y,sample_weight = None):
        result = self.predict(X)
        score = log_loss(result, y)
        if self.verbose >= 1 : print("Score : {}".format(score))
        return score
