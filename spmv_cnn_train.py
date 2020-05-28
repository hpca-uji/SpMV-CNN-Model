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


import keras, sys, h5py
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 24}, \
        intra_op_parallelism_threads = 48, \
        inter_op_parallelism_threads = 1) 
sess = tf.Session(config=config)
keras.backend.set_session(sess)

bsize=250

# Parameters
frequency = sys.argv[1]
metric    = sys.argv[2] # Time, Energy, EPKG, EDRAM

metric_key = {"Time": "B", 
              "Energy": "E", 
              "EPKG": "EPKG", 
              "EDRAM": "EDRAM"}[metric] # Time, Energy, EPKG, EDRAM

# 20% of the training set for validation
# 80% of the training set for training
size_test  = 20
size_train = 100 - size_test

case = "%s_f%s_b%d_%d_%d_diff_logInput" % (metric, frequency, bsize, size_test, size_train)

# Time metric and frequencies use best model/run for time at 2.4GHz
# Energy metrics and frequencies use best model/run for energy at 2.4GHz
if metric == "Time": 
  metric_inherited = "Time"
elif metric in ["Energy", "EPKG", "EDRAM"]: 
  metric_inherited = "Energy"

frequency_inherited = "2400000"
case_inherited = "%s_f%s_b%d_%d_%d_diff_logInput" % \
  (metric_inherited, frequency_inherited, bsize, size_test, size_train)

# Options considered for the hyperparamethers
batch_size_options = [32, 64, 128, 256]
optim_options      = ['adam', 'sgd', 'rmsprop']
lr_options         = [10**-3, 10**-2, 10**-1]

# Maximum number of epochs to train the model
epochs = 100 
  
# Read the values of the hyperparamethers provided by Hyperas
with open('./results/models/best_run_%s.json' % (case_inherited) as f:
  best_run = json.loads(f)

batch_size = batch_size_options[best_run["batch_size"]]
optim = optim_options[best_run["choiceval"]]
learning_rate = lr_options[best_run[{"adam": "lr", "rmsprop": "lr_1", "sgd": "lr_2"}[optim]]]

# Load the keras model
with open("./results/models/best_model_%s.json" % (case_inherited),"r") as f:
  json_string = f.readline()
  model = model_from_json(json_string)

# If you just want to show the hyperparamethers values, 
# but skip the model train uncomment these two lines
# print("size_test=%d size_train=%d" % (size_test, size_train))
# print("batch_size=%d optim=%s learning_rate=%0.4f" % (batch_size, optim, learning_rate))
# sys.exit()  

# Load trainig and validation dataset
with h5py.File("./dataset/train/merged_energy_train_shuffle_f%s_%d.h5" % (frequency, bsize), 'r') as d:
  X = d["MB"].value
  Y = d[metric_key].value

# Normalize the differenece between two consecutive elements of vpos
X = np.log(np.absolute(np.diff(X, axis=1))+1)

# Early stopping is used to prevent model overfitting
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min') 

# Learning rate is reduced when the val_loss stops improving
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                              patience=5, min_lr=0.0001)

# File for the weights of the trained model
filepath="./results/weights/best_weights_%s.hdf5" % (case)

# Save the latest best model
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                             save_best_only=True, mode='min')

# Specify the training and validation data
size_test_2 = size_test / 100
size_train_2 = size_train / 100
(x_train, x_test, y_train, y_test) = \
   train_test_split(X, Y, test_size=size_test_2, train_size=size_train_2, random_state=1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) 
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 
print(x_train.shape, 'train samples')

# Configure the model for training
optimizer = {"adam": "Adam", "rmsprop": "RMSProp", "sgd": "SGD"}[optim]
model.compile(getattr(keras.optimizers, optimizer)(lr=learning_rate), loss="mse", metrics=["mse"])

# Train the model
history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks=[early_stop, reduce_lr, checkpoint])

# History visualization
print (history.history)
with open("history_%s.txt" % (case),"w") as f:
  output = sys.stdout
  sys.stdout = f
  print (history.history)
  sys.stdout = output

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
plt.plot(history.history['loss'][1:])
plt.plot(history.history['val_loss'][1:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
out_png = 'Training_%s.png' % (case) 
plt.savefig(out_png, dpi=150)
