#!/bin/bash

export PATH=$PATH:/home/mvaya/hdf5/bin:/home/mvaya/papi_install/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mvaya/hdf5/lib/:/home/mvaya/papi_install/lib/

numactl --membind 0 taskset -c 0 ./driver $1 10000 250 250 250 0 $2 $3 
