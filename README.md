# Convolutional neural nets for estimating the run time and energy consumption of the sparse matrix-vector product - SpMV-CNN

## Introduction

**SpMV-CNN** is a set of Convolutional Neural Networks (CNNs) that provide 
accurate estimations of the performance and energy consumption of the SPMV 
kernel. The proposed CNN-based models use a block-wise approach to make the 
CNN architecture independent of the matrix size. These models cat be trained 
to estimate run time as well as total, package and DRAM energy consumption at 
different processor frequencies. 

## Prerequisites

**SpMV-CNN** requires Python3 with the following packages:
```
  keras==2.1.6
  tensorflow==1.8.0
  h5py==2.7.1
  matplotlib==2.1.1
  scikit-learn==0.19.1
```

## Obtaining the dataset

The SpMV dataset can be downloaded from: https://bit.ly/2ZHsuVI
...

## Hypeparameter search

The script `spmv_cnn_hyperas.py` performs the hypeparameter search via the Hyperas tool.
This script requires the hdf5 file dataset in the directory dataset/train and produces
both a best_model_*.json and best_run_*.json files containing the model structure and
hyperparameters of the best performing configuration.

This script can be invoked in the following way:

`python3 spmv_cnn_hyperas.py 2400000 Time`

where `2400000` is the operating processor frequency (2.4 GHz) and `Time` the modeled metric.
According to the labels in the dataset, the hyperparameter search can also be 
performed with the `Energy`, `EPKG` and `EDRAM` metrics, corresponding to the energy
measured by the Intel RAPL counters from our Intel Xeon Haswell core.

## Training search


Metric: Execution time; Frequency: 2.4GHz; block_size: 250

FILES & DIRECTORIES
===================
merged_energy_train_shuffle_f2400000_250.h5      Training and validation corpora 
test_matrices_f_2400000_b250                     Directory with the samples for each test matrix
spmv_dnn_hyperas.py                              Code to run the hyperparmethers search          
spmv_dnn_training.py                             Code to run the training                        
spmv_dnn_test.py                                 Code to run the test                            

HYPERPARAMETHER SEARCH, TRAINING AND TEST
=========================================
nohup python3 spmv_dnn_hyperas.py  > out_hyperas &
nohup python3 spmv_dnn_training.py > out_training &
python3 spmv_dnn_test.py  



