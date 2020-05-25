#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

import pickle
import sys
import os
import pmlp_utility
from time import time

def main(argv):

    log_dir = os.environ['pmlp_log_dir']

    with open(os.path.join(log_dir, 'hyperparameters.pickle'), 'rb') as f:
        hyperparameters = pickle.load(f)

    with open(os.path.join(log_dir, 'train_states.pickle'), 'rb') as f:
        train_states = pickle.load(f)
        
    finetune_params_file = os.path.join(log_dir, 'finetune_parameters.pickle')
    if os.path.exists(finetune_params_file):
        fid = open(finetune_params_file, 'rb')
        finetune_params = pickle.load(fid)
        fid.close()
        hyperparameters['lr'] = finetune_params['lr']
        hyperparameters['epochs'] = finetune_params['epochs']
        hyperparameters['regularizer'] = finetune_params['regularizer']
        hyperparameters['dropout'] = finetune_params['dropout']
        if 'subset_percentage' in finetune_params.keys():
            hyperparameters['subset_percentage'] = finetune_params['subset_percentage']



    start_time = time()
    finetune_result = pmlp_utility.finetune(train_states, hyperparameters)
    stop_time = time()
    finetune_time = stop_time - start_time
    
    new_block = {'weights': finetune_result['weights'],
                 'measure': finetune_result['measure'],
                 'history': finetune_result['history'],
                 'conf_index': finetune_result['conf_index'],
                 'finetune_time': finetune_time}
    
    fid = open(os.path.join(log_dir, 'finetune.pickle'), 'wb')
    pickle.dump(new_block, fid)
    fid.close()
    
    fid = open(os.path.join(log_dir, 'finetune_complete.txt'), 'w')
    fid.write('x')
    fid.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])