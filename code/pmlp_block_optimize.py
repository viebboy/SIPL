#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

import pickle
import dill
import sys
import os
import pmlp_utility
from time import time

def main(argv):

    log_dir = os.environ['pmlp_log_dir']

    with open(os.path.join(log_dir, 'hyperparameters.pickle'), 'rb') as f:
        hyperparameters = pickle.load(f)

    with open(os.path.join(log_dir, 'tmp_train_states.pickle'), 'rb') as f:
        train_states = pickle.load(f)


    start_time = time()
    block_optimization_result = pmlp_utility.optimize_block(train_states, hyperparameters)
    stop_time = time()
    block_optimization_time = stop_time - start_time
    
    start_time = time()
    subset_indices = pmlp_utility.subset_sampling(train_states, hyperparameters, block_optimization_result)
    stop_time = time()
    subset_sampling_time = stop_time - start_time
    
    new_block = {'dense_weights': block_optimization_result['dense_weights'],
                 'bn_weights': block_optimization_result['bn_weights'],
                 'output_weights': block_optimization_result['output_weights'],
                 'measure': block_optimization_result['measure'],
                 'history': block_optimization_result['history'],
                 'conf_index': block_optimization_result['conf_index'],
                 'subset_indices': subset_indices,
                 'block_optimization_time': block_optimization_time,
                 'subset_sampling_time': subset_sampling_time}
    
    with open(os.path.join(log_dir, 'new_block.pickle'), 'wb') as fid:
        pickle.dump(new_block, fid)

    
    with open(os.path.join(log_dir, 'new_block_complete.txt'), 'w') as fid:
        fid.write('x')
    
    
if __name__ == "__main__":
    main(sys.argv[1:])