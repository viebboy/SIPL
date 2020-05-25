#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

import os

data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')

if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'output')):
    os.mkdir(os.path.join(os.path.dirname(os.getcwd()), 'output'))

if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'log')):
    os.mkdir(os.path.join(os.path.dirname(os.getcwd()), 'log'))
    
if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'tmp')):
    os.mkdir(os.path.join(os.path.dirname(os.getcwd()), 'tmp'))
    
output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

log_dir = os.path.join(os.path.dirname(os.getcwd()), 'log')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    
tmp_dir = os.path.join(os.path.dirname(os.getcwd()), 'tmp')
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
    
BATCH_SIZE = 128
LR = (1e-3, 1e-4, 1e-5)
EPOCH = (20, 30, 30)
FINETUNE_LR = (1e-3, 1e-4, 1e-5)
FINETUNE_EPOCH = (10, 30, 30)

StackedELM_names = ['prefix', 
                    'dataset', 
                    'layer_size', 
                    'pca_dimension', 
                    'max_layer', 
                    'regularizer', 
                    'exp']

StackedELM_values = {'prefix': ['StackedELM'],
                     'dataset': ['caltech256', 'mit_indoor', 'cfw'],
                     'layer_size': [200],
                     'pca_dimension': [100],
                     'max_layer': [12],
                     'regularizer': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3, 1e4],
                     'exp': [0, 1, 2]}

PLN_names = ['prefix', 'dataset', 'block_size', 'max_block', 'max_layer', 'regularizer', 'muy', 'alpha', 'exp']
PLN_values = {'prefix': ['PLN'],
              'dataset': ['caltech256', 'mit_indoor', 'cfw'],
              'block_size': [40],
              'max_block': [20],
              'max_layer': [8],
              'regularizer': [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3],
              'alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
              'muy': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
              'exp': [0, 1, 2]}

PMLP_names = ['prefix', 'dataset', 'block_size', 'max_block', 
              'max_layer', 'regularizer', 'dropout', 'subset_percentage', 
              'sampling_strategy', 'exp']

PMLP_values_proposal = {'prefix': ['PMLP'],
               'dataset': ['caltech256', 'mit_indoor', 'cfw', ],
               'block_size': [40],
               'max_block': [-1,],
               'max_layer': [8],
               'regularizer': [[(None, 2.0), (None, 3.0), (None, 4.0), (1e-4, None)]],
               'dropout': [[None, 0.1, 0.2, 0.3, 0.4, 0.5]],
               'subset_percentage': [0.1, 0.2, 0.3],
               'sampling_strategy': ['random-without-class', 'top_loss', 'cluster_loss'],
               'exp': [0, 1, 2]}

PMLP_values_original = {'prefix': ['PMLP'],
                        'dataset': ['caltech256', 'mit_indoor', 'cfw'],
                        'block_size': [40],
                        'max_block': [-1,],
                        'max_layer': [8],
                        'regularizer': [(None, 2.0), (None, 3.0), (None, 4.0), (1e-4, None)],
                        'dropout': [None, 0.1, 0.2, 0.3, 0.4, 0.5],
                        'subset_percentage': [1.0],
                        'sampling_strategy': ['random-without-class'],
                        'exp': [0, 1, 2]}

