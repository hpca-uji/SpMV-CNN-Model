""" SpMV-CNN: Convolutional neural nets for estimating the run time and 
energy consumption of the sparse matrix-vector product

SpMV-CNN is a set of Convolutional Neural Networks (CNNs) that provide 
accurate estimations of the performance and energy consumption of the SPMV 
kernel. The proposed CNN-based models use a block-wise approach to make the 
CNN architecture independent of the matrix size. These models cat be trained 
to estimate run time as well as total, package and DRAM energy consumption at 
different processor frequencies. 

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

__author__ = "Manuel F. Dolz, Maria Barreda, M. Asunción Castaño"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Maria Barreda, M. Asunción Castaño"]
__date__ = "2020/05/28"

__email__ =  "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.0.0"


from __future__ import print_function

import tensorflow as tf
from hyperopt import Trials, STATUS_OK, tpe
import keras, sys, h5py
import pandas as pd
import numpy as np
import sys
import json

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from hyperas import optim
from hyperas.distributions import choice, uniform
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 24}, \
        intra_op_parallelism_threads = 48, \
        inter_op_parallelism_threads = 1) 
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def model(x_train, y_train, x_test, y_test):
  outputs = 1
  # block_size minus 1 due to the computation of 
  # differeneces between consecutive elements of vpos 
  block_size = 249  
  epochs = 1
  
  model = Sequential()

  # Two choices for the number of filters in the 2 conv layers: (16,32), (32,64)
  #    (16,32) -> 16 filters for the first layer and 32 for the second layer
  # Four choices for the filter size in conv layers: {3,5,7,9} x 1
  # Five choices for the pool size in MaxPool layers: {2,3,5,7,9}x1
  model_choice = {{choice(['one', 'two'])}}
  if model_choice == 'one': 
    model.add(Conv1D(16, kernel_size={{choice([3,5,7,9])}},
                     activation='relu',
                     input_shape=(block_size,1)))
    model.add(MaxPooling1D(pool_size={{choice([2,3,5,7,9])}}))
    model.add(Conv1D(32, kernel_size={{choice([3,5,7,9])}}, 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size={{choice([2,3,5,7,9])}}))
  elif model_choice == 'two':
    model.add(Conv1D(32, kernel_size={{choice([3,5,7,9])}},
                     activation='relu',
                     input_shape=(block_size,1)))
    model.add(MaxPooling1D(pool_size={{choice([2,3,5,7,9])}}))
    model.add(Conv1D(64, kernel_size={{choice([3,5,7,9])}}, 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size={{choice([2,3,5,7,9])}}))
  model.add(Flatten())

  # Three choices for the number of fully connected layers: 1, 2, 3 layers
  # Three choices for the number of neurons in each fully connected layer: 10, 100, 1000
  dense_choice = {{choice(['one', 'two', 'three'])}}
  if dense_choice == 'one':
    model.add(Dense({{choice([10,100,1000])}}, activation='relu'))
    model.add(Dropout({{uniform(0,1)}}))
  elif dense_choice == 'two':
    model.add(Dense({{choice([10,100,1000])}}, activation='relu'))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense({{choice([10,100,1000])}}, activation='relu'))
    model.add(Dropout({{uniform(0,1)}}))
  elif dense_choice == 'three':
    model.add(Dense({{choice([10,100,1000])}}, activation='relu'))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense({{choice([10,100,1000])}}, activation='relu'))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense({{choice([10,100,1000])}}, activation='relu'))
    model.add(Dropout({{uniform(0,1)}}))
  model.add(Dense(outputs))
 
  # Three choices for the optimizer algorithm: adam, rmsprop and sgd
  # Three choices for the initial learning rate: 10**-3, 10**-2, 10**-1
  adam    = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
  rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
  sgd     = keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
  
  optim = {"adam": adam, 
           "rmsprop": rmsprop, 
           "sgd": sgd}[{{choice(['adam', 'sgd', 'rmsprop'])}}]

  model.compile(loss='mse', metrics=['mse'], optimizer=optim) 
 
  # Four choices for the batch size: 32, 64, 128, 256 
  model.fit(x_train, y_train,
            batch_size={{choice([32, 64, 128, 256])}},
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
  
  score, loss = model.evaluate(x_test, y_test, verbose=0)
  
  print('Test loss:', loss)
  return {'loss': loss, 'status': STATUS_OK, 'model': model}


def data():
  with open("train_file.txt", "r") as f:
    train_file = f.readline()[:-1]
    metric_key = f.readline()[:-1]

  print("Loading HDF5 file...")
  with h5py.File(train_file, "r") as d:
    # File with the training and validation dataset for all metrics
    print("Loading X...")
    X = d["MB"].value
    print("Loading Y...")
    Y = d[metric_key].value # * 1e3 # To nanoseconds

  # Normalize the differenece between two consecutive elements of vpos
  X = np.log(np.absolute(np.diff(X, axis=1))+1)  
  
  # 20% of the training dataset for validation
  # 80% of the training dataset for training
  size_test  = 0.2 
  size_train = 1 - size_test
 
  # random_state=1 is required in order to use the same samples for 
  # training and validation in every hyperparameter search, so the 
  # validation results could be compared later
  (x_train, x_test, y_train, y_test) = train_test_split(X, Y, \
      test_size=size_test, train_size=size_train, random_state=1)

  x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

  size_test  = size_test  * 100
  size_train = size_train * 100 
  return x_train, y_train, x_test, y_test, size_test, size_train

# Parameters
frequency = sys.argv[1] # 2400000 
metric    = sys.argv[2] # Time, Energy, EPKG, EDRAM

metric_key = {"Time": "B", 
              "Energy": "E", 
              "EPKG": "EPKG", 
              "EDRAM": "EDRAM"}[metric] # Time, Energy, EPKG, EDRAM

train_file = "./dataset/train/merged_energy_train_shuffle_f%s_250.h5" % frequency
f=open("train_file.txt", "w")
f.write("%s\n" % train_file)
f.write("%s\n" % metric_key)
f.close()

trials = Trials()

best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=100,
                                      trials=trials)

# This value should be consistent with that for the variable block_size
bsize=250 

case = "%s_f%s_b%d_%d_%d_diff_logInput" % (metric, frequency, bsize, size_test, size_train)

with open("./results/models/best_model_%s.json" % case, "w") as f:
   f.write(best_model.to_json())

print("Best performing model chosen hyper-parameters:")
print(best_run)

with open("./results/models/best_run_%s.json" % (case), "w") as f:
   json.dump(best_run, f)
