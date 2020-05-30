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

The execution time and energy consumption data corresponding to the SpMV operation
on a set of sparse matrices from the SuiteSparse Collection have been obtained on 
an Intel Xeon E5-2630 core running at frequencies 1.2, 1.6, 2.0, 2.4 GHz. The energy 
consumption measurements are obtained via the Intel RAPL interface and gathered at 
three different levels (total, package and DRAM, where total = package + DRAM) 
for this specific processor.

The training and testing dataset along with the obtained results can be downloaded from: https://bit.ly/2ZHsuVI

## Hyperparameter search

The script `spmv_cnn_hyperas.py` performs the hypeparameter search via the Hyperas tool.
This script requires the hdf5 file dataset in the directory dataset/train and produces
both a `best_model_*.json` and `best_run_*.json` files in the results/models/ directory
containing the model structure and hyperparameters of the best performing configuration.

This script can be invoked in the following way:

`python3 spmv_cnn_hyper.py 2400000 Time`

where `2400000` is the operating processor frequency (2.4 GHz) and `Time` the modeled metric.
According to the labels in the dataset, the hyperparameter search can also be 
performed with the `Energy`, `EPKG` and `EDRAM` metrics, corresponding to the energy
measured by the Intel RAPL counters from our Intel Xeon Haswell core. In our case, however
we only search hyperparameters for the `Time` and `Energy` metrics at 2.4 GHz. Other metrics
and frequencies inherint the best performing model and settings from the previous
configuration.

## Training

The script `spmv_cnn_train.py` performs the training on the best performing models obtained
on the previous step. For that it uses both the `best_model_*.json` and `best_run_*.json` files
obtained in the second step. 

This script can be invoked in the following way:

`python3 spmv_cnn_train.py 2400000 Time`

where `2400000` is the operating processor frequency (2.4 GHz) and `Time` the modeled metric.
The training should be performed per metric and frequency. The training produces a file that
contains the trained weights, so the model is ready for performing inference.

## Test

The script `spmv_cnn_test.py` performs the test on the the set of testing matrices involved in
the SpMV operation.

This script can be invoked in the following way:

`python3 spmv_cnn_test.py 2400000 Time`

where `2400000` is the operating processor frequency (2.4 GHz) and `Time` the modeled metric.
The test should be performed per metric and frequency. The training produces two files in the 
results/tests/ directory:

* `Pred_*.txt`: This file contains the real measurements and the predictions performed by 
the CNN for the individual vpos blocks of the testing matrices.
* `Test_*.txt`: This file summarizes the information of `Pred_*.txt` file, showing the average
relative error among the blocks of a same matrix and the total relative error, which is computed
by summing up the real measurements and the predicitions for all the blocks of a same matrix and
computing the relative error upon those values.

# Summary

All previous steps can be performed at once via the `run.sh` bash script provided in this repository.

## References

Publications describing **SpMV-CNN-Model**:

* Barreda, M., Dolz, M.F., Castaño, M.A. et al. Performance modeling of the sparse matrix–vector product 
 via convolutional neural networks. J Supercomput (2020). https://doi.org/10.1007/s11227-020-03186-1

## Acknowledgments

The **SpMV-CNN-Model** research has been partially supported by:

* Project TIN2017-82972-R **"Agorithmic Techniques for Energy-Aware and Error-Resilient High Performance Computing"** funded by the Spanish Ministry of Economy and Competitiveness (2018-2020).

* Project CDEIGENT/2017/04 **"High Performance Computing for Neural Networks"** funded by the Valencian Government.

* Project UJI-A2019-11 **"Energy-Aware High Performance Computing for Deep Neural Networks"** funded by the Universitat Jaume I.

