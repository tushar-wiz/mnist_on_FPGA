#!/usr/bin/env python3

import h5py
import pandas as pd 
import numpy as np

f = h5py.File('my_model.h5','r')
print(list(f.keys()))

dset = f['model_weights']
print(list(dset.keys()))

layer1_bias = dset['dense']['dense']['bias:0']
layer1_kernel = dset['dense']['dense']['kernel:0']

layer2_bias = dset['dense_1']['dense_1']['bias:0']
layer2_kernel = dset['dense_1']['dense_1']['kernel:0']

print(layer1_bias.shape)
print(layer1_bias[:])
arr1 = np.asarray(layer1_bias)
pd.DataFrame(arr1).to_csv('sample.csv',index_label = "Index",header  = ['Bias_0'])
print(layer1_kernel.shape)

print(layer2_bias.shape)
print(layer2_kernel.shape)
