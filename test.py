#!/bin/python
import numpy as np
# import logging
import os
# import scipy.io as sio
# from ridge import ridge, ridge_corr, bootstrap_ridge
from ridge import *
from mvpa2.suite import *
# from hdf5 import *

# logging.basicConfig(level=logging.DEBUG)

sub = 'sub-foo'
train_dst = 'foo'
test_dst = 'foo'
this_model = 'foo'

# Create some test data
N = 200 # features
M = 1000 # response sources (voxels, whatever)
TR = 1000 # regression timepoints
TP = 200 # prediction timepoints

snrs = np.linspace(0, 0.2, M)
realwt = np.random.randn(N, M)
features = np.random.randn(TR+TP, N)
realresponses = np.dot(features, realwt) # shape (TR+TP, M)
noise = np.random.randn(TR+TP, M)
responses = (realresponses * snrs) + noise

Rresp = responses[:TR]
Presp = responses[TR:]
Rstim = features[:TR]
Pstim = features[TR:]

# make mask for test dst beta value: 1 = not include
nuis_len = 0
train_model_len = Rstim.shape[1]/5
# print train_model_len
feat_beta_mask = np.ones([Rstim.shape[1], Rresp.shape[1]], dtype=bool)
use_nuis = False
# # print feat_beta_mask.shape

for ntps in range(5):
    start_inx = ntps*train_model_len
    end_inx = (train_model_len-nuis_len)+ntps*train_model_len
    # print test_model, 'model start from ', start_inx, ' to ', end_inx
    feat_beta_mask[start_inx:end_inx, :] = False # keep the model 
    # print test_model, 'model has ', sum(feat_beta_mask[:, 1]==False), ' features'
# # print test_model, nuis, use_nuis

train_stim = Rstim 
train_brain_resp = Rresp
test_stim = Pstim
test_brain_resp = Presp
alphas = np.logspace(-2, 2, 20)
nboots = 5
chunklen = 10
nchunks = 15

wt, corr, p, valpha, corr_all, valinds = bootstrap_ridge(train_stim, train_brain_resp, 
                                                      test_stim, test_brain_resp, alphas, nboots, 
                                                      chunklen, nchunks, feat_beta_mask, bootstrap=False, 
                                                      exclude_feature=use_nuis)

# Corr should increase quickly across "voxels". Last corr should be large (>0.9-ish).
# wt should be very similar to realwt for last few voxels.
project_path = '/home/zhengang/brainSLAM'
out_path = os.path.join(project_path, 'RESULTS/%s'%sub)
print wt.shape
print corr.shape
print p.shape
h5save("%s/%s_%s_%s_%s_corrs"%(out_path, sub, train_dst, test_dst, this_model), corr, compression='gzip')
h5save("%s/%s_%s_%s_%s_ps"%(out_path, sub, train_dst, test_dst, this_model), p, compression='gzip')
