from sklearn.model_selection import GridSearchCV
import preprocess as pre
from nn_wrapper import NN

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss

X,y, ids =  pre.load_train(20)
# X_train,X_test,y_train,y_test = pre.train_test_split(X,y,split_ratio = 5)

model = NN()

param_grid= dict(layers = [[
                    {"type" : ZeroPadding2D, "params" : {'padding': (1,1), 'input_shape' : (1280,720,3)}},
                    {"type" : Convolution2D, "params" : {'nb_filter': 4, 'nb_row': 3, 'nb_col': 3, 'activation': 'relu'}},
                    {"type" : ZeroPadding2D, "params" : {'padding': (1,1)}},
                    {"type" : Convolution2D, "params" : {'nb_filter': 4, 'nb_row': 3, 'nb_col': 3, 'activation': 'relu'}},
                    {"type" : MaxPooling2D, "params" : {'pool_size': (2,2), 'strides': (2,2)}},
                    {"type" : ZeroPadding2D, "params" : {'padding': (1,1)}},
                    {"type" : Convolution2D, "params" : {'nb_filter': 8, 'nb_row': 3, 'nb_col': 3, 'activation': 'relu'}},
                    {"type" : ZeroPadding2D, "params" : {'padding': (1,1)}},
                    {"type" : Convolution2D, "params" : {'nb_filter': 8, 'nb_row': 3, 'nb_col': 3, 'activation': 'relu'}},
                    {"type" : MaxPooling2D, "params" : {'pool_size': (2,2), 'strides': (2,2)}},

                    {"type" : Flatten, "params" : {}},
                    {"type" : Dense, "params" : {'output_dim': 32, 'activation': 'relu'}},
                    {"type" : Dropout, "params" : {'p': 0.5}},
                    {"type" : Dense, "params" : {'output_dim': 32, 'activation': 'relu'}},
                    {"type" : Dropout, "params" : {'p': 0.5}},
                    {"type" : Dense, "params" : {'output_dim': 8, 'activation': 'softmax'}},
                    ],
                    [
                    {"type" : ZeroPadding2D, "params" : {'padding': (1,1), 'input_shape' : (1280,720,3)}},
                    {"type" : Convolution2D, "params" : {'nb_filter': 8, 'nb_row': 3, 'nb_col': 3, 'activation': 'relu'}},
                    {"type" : ZeroPadding2D, "params" : {'padding': (1,1)}},
                    {"type" : Convolution2D, "params" : {'nb_filter': 8, 'nb_row': 3, 'nb_col': 3, 'activation': 'relu'}},
                    {"type" : MaxPooling2D, "params" : {'pool_size': (2,2), 'strides': (2,2)}},

                    {"type" : Flatten, "params" : {}},
                    {"type" : Dense, "params" : {'output_dim': 32, 'activation': 'relu'}},
                    {"type" : Dropout, "params" : {'p': 0.5}},
                    {"type" : Dense, "params" : {'output_dim': 32, 'activation': 'relu'}},
                    {"type" : Dropout, "params" : {'p': 0.5}},
                    {"type" : Dense, "params" : {'output_dim': 8, 'activation': 'softmax'}},
                    ]]
                    )

gridsearch = GridSearchCV(model, param_grid = param_grid)
gridsearch.fit(X,y)


print gridsearch.best_params_
print gridsearch.best_score_
