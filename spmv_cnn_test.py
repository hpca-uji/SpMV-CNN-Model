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
import pandas as pd
import numpy as np
import os
import sys
import json

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 0, 'CPU': 24}, \
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

size_test  = 20
size_train = 80

case = "%s_f%s_b%d_%d_%d_diff_logInput" % (metric, frequency, bsize, size_test, size_train)

# Read the values of the hyperparamethers provided by Hyperas  
with open('./results/models/best_run_%s.json' % (case), "r") as f:
   best_run = json.loads(f.read())

# Options considered for the hyperparamethers
batch_size_options = [32, 64, 128, 256]
optim_options      = ['adam', 'sgd', 'rmsprop']
lr_options         = [10**-3, 10**-2, 10**-1]

batch_size    = batch_size_options[best_run["batch_size"]]
optim         = optim_options[best_run["choiceval"]]
learning_rate = lr_options[best_run[{"adam": "lr", "rmsprop": "lr_1", "sgd": "lr_2"}[optim]]]

with open("./results/models/best_model_%s.json" % (case),"r") as f:
    model = model_from_json(f.readline())

# File with the weights of the trained model
model.load_weights("./results/weights/best_weights_%s.hdf5" % (case))

# Configure the model
optimizer = {"adam": "Adam", "rmsprop": "RMSProp", "sgd": "SGD"}[optim]
model.compile(getattr(keras.optimizers, optimizer)(lr=learning_rate), loss="mse", metrics=["mse"])

avg_loss = 0.0
error = 0.0
count = 0

# Files to store the output predictions and the test metrics
test=open("./results/tests/Test_%s.txt" % (case) ,"w")
pred=open("./results/tests/Pred_%s.txt" % (case) ,"w")

# Specify the directory where the files with test samples are. A file per test matrix
path = "./dataset/test/f_%s_b%s" % (frequency, bsize)

l = os.listdir(path)
l.sort()

# Compute inference time 
import time
before = time.time()

loss      = np.zeros(len(l))
rel_error = np.zeros(len(l))

# Repeat the test process for each test matrix
for mat in range(len(l)):
    print (l[mat])
    dat = h5py.File("%s/%s" % (path, l[mat]), 'r')
    X = dat["MB"].value
    Y = dat["%s_%d" % (metric_key, bsize)].value 
    X = X[:(X.shape[0]//bsize)*bsize].reshape(-1, bsize, 1) 
    X = np.log(np.absolute(np.diff(X, axis=1))+1) # log + diff
    Y = Y[:X.shape[0]]

    score = model.evaluate(X, Y, verbose=0)
    predictions = model.predict(X)
  
    rel_error_bl = np.zeros(len(predictions))

    for i in range(len(predictions)):
        epred_bl = predictions[i]
        ereal_bl = Y[i]  
        rel_error_bl[i] = abs((epred_bl - ereal_bl) / ereal_bl)
         
        pred.write("%s ; %s ; %s ; %.8f ; %.8f ; %.8f ;\n" % \
          (l[mat].replace("output_", "").replace("_%s.h5" % frequency,""), \
             metric_key, frequency, epred_bl, ereal_bl, rel_error_bl[i]))

    loss[mat]  = score[1]
    rel_error[mat] = np.mean(rel_error_bl)
    test.write("%s rel_error_tot_blk: %.8f rel_error: %.8f\n" % \
                  (l[mat], rel_error[mat], abs(sum(predictions) - sum(Y))/sum(Y) ) )

print("--- %s seconds ---" % (time.time() - before))

test.write("TOTAL --- error: %.7f loss: %.7f\n" % (np.mean(rel_error), np.mean(loss)))
test.close()
pred.close()
