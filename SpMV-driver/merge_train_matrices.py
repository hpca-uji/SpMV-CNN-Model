#from __future__ import print_function

import sys, h5py, os
import pandas as pd
import numpy as np
from math import ceil, floor

def merge_data(path, outpath):
    l= os.listdir(path) 

    for bsize in [250]: #xrange(250, 1000, 250):
      count = 0
      alld = {}
      hf = h5py.File("%s/merged_energy_train_shuffle.h5" % (outpath), 'w')

      for mat in l:
        count+=1

        print mat
        dat = h5py.File("%s/%s" % (path, mat), 'r')
        try:
          bt= dat["B_%d" % bsize]
        except:
          continue
        
        be     = dat["E_%d" % bsize]
        bepkg  = dat["EPKG_%d" % bsize]
        bedram = dat["EDRAM_%d" % bsize]

        f  = dat["FREC"]
        r  = dat["HEAD"][0]
        c  = dat["HEAD"][1]
        z  = dat["HEAD"][2]
        mb = dat["MB"]

        bt    = bt[:(mb.shape[0]/bsize)]
        be    = be[:(mb.shape[0]/bsize)]
        bepkg = bepkg[:(mb.shape[0]/bsize)]
        bedram= bedram[:(mb.shape[0]/bsize)]
        mb    = mb[:(mb.shape[0]/bsize)*bsize]

        cols= np.array([c]*(mb.shape[0]/bsize))
        rows= np.array([r]*(mb.shape[0]/bsize))
        nnz = np.array([z]*(mb.shape[0]/bsize))
        frec= np.array([f]*(mb.shape[0]/bsize))
	
        mb = mb.reshape(mb.shape[0]/bsize, bsize)

        d={"MB"   : mb,     "B"    : bt,
           "E"    : be,     "EPKG" : bepkg,
           "EDRAM": bedram, "NCOLS": cols,
           "NROWS": rows,   "NNZ"  : nnz}

        if alld == {}:
          for k, v in d.items(): alld[k]= v
        else:
          for k, v in d.items():
             alld[k] = np.concatenate((alld[k], v), axis=0)

      perm = np.random.permutation(alld["MB"].shape[0])

      for k, v in d.items(): 
         alld[k]= alld[k][perm]
         hf.create_dataset(k, data=v, compression="gzip")

      hf.close()

#for f in [1200000, 1600000, 2000000, 2400000]: 
#  merge_data("dataset/train/f_%d_b250/" % f, "merge_data/merged_energy_train_shuffle_f%d" % f)

merge_data(sys.argv[1], sys.argv[2])

