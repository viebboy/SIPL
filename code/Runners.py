#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

import exp_configurations as Conf
from pmlp import PMLP
from PLN import PLN
import os
import Utility
import numpy as np
from StackedELM import StackedELM
from time import time
import pickle
from sklearn.neighbors import KNeighborsClassifier

OUTPUT_DIR = Conf.output_dir
LOG_DIR = Conf.log_dir
DATA_DIR = Conf.data_dir
BATCH_SIZE = Conf.BATCH_SIZE



def train_StackedELM(args):
    prefix = args[0]
    dataset = args[1]
    layer_size = args[2]
    pca_dimension = args[3]
    max_layer = args[4]
    regularizer = args[5]
    exp = args[6]
    
    result = Utility.check_result(args, OUTPUT_DIR)
    
    if result is not None:
        return result

    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    
    model = StackedELM(layer_size=layer_size, 
                       pca_dimension=pca_dimension, 
                       max_layer=max_layer, 
                       regularizer=regularizer, 
                       metrics=['acc',],
                       convergence_measure='val_acc',
                       direction='higher')
    
    start_time = time()
    weights, train_performance, val_performance, test_performance = model.fit([x_train, y_train], [x_val, y_val], [x_test, y_test])
    stop_time = time()
    
    
    result = {'weights': weights,
              'train_performance': train_performance,
              'val_performance': val_performance,
              'test_performance': test_performance,
              'time_taken': stop_time - start_time}
    
    for metric in train_performance.keys():
        result['train_' + metric] = train_performance[metric][-1]
        result['val_' + metric] = val_performance[metric][-1]
        result['test_' + metric] = test_performance[metric][-1]
        
    metrics = list(train_performance.keys())
    nb_block = len(train_performance[metrics[0]])
    result['optimize_time_per_block'] = result['time_taken'] / float(nb_block)
    
    return result

def train_PLN(args):
    prefix = args[0]
    dataset = args[1]
    block_size = args[2]
    max_block = args[3]
    max_layer = args[4]
    regularizer = args[5]
    muy = args[6]
    alpha = args[7]   
    exp = args[8]
    
    result = Utility.check_result(args, OUTPUT_DIR)
    
    if result is not None:
        return result

    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    
    model = PLN()
    params = model.get_default_parameters()
    params['block_size'] = block_size
    params['max_block'] = max_block
    params['max_layer'] = max_layer
    params['regularizer'] = regularizer
    params['muy'] = muy
    params['alpha'] = alpha
    params['direction'] = 'higher'
    params['metrics'] = ['acc',]
    params['convergence_measure'] = 'val_acc'
    
    
    start_time = time()
    model_data, train_performance, val_performance, test_performance = model.fit(params, [x_train, y_train], [x_val, y_val], [x_test, y_test])
    stop_time = time()
    
    
    result = {'model_data': model_data,
              'train_performance': train_performance,
              'val_performance': val_performance,
              'test_performance': test_performance,
              'time_taken': stop_time - start_time}
    
    for metric in train_performance.keys():
        result['train_' + metric] = train_performance[metric][-1]
        result['val_' + metric] = val_performance[metric][-1]
        result['test_' + metric] = test_performance[metric][-1]
        
    metrics = list(train_performance.keys())
    nb_block = len(train_performance[metrics[0]])
    result['optimize_time_per_block'] = result['time_taken'] / float(nb_block)
    
    return result

def process_model_name(name):
    name = name.replace(' ', '')
    name = name.replace('(', '-')
    name = name.replace(')', '-')
    name = name.replace('[', '-')
    name = name.replace(']', '-')
    return name


def calculate_nb_configuration(regularizer, dropout):
    if not isinstance(regularizer, list):
        regularizer = [regularizer]
    if not isinstance(dropout, list):
        dropout = [dropout]
    
    return len(regularizer)*len(dropout)


def train_PMLP(args):
    prefix = args[0]
    dataset = args[1]
    block_size = args[2]
    max_block = args[3]
    max_layer = args[4]
    regularizer = args[5]
    dropout = args[6]
    subset_percentage = args[7]
    sampling_strategy = args[8]
    exp = args[9]
    
    result = Utility.check_result(args, OUTPUT_DIR)
    if result is not None:
        return result
    
    x_train_file = os.path.join(Conf.data_dir, dataset + '_x_train.npy')
    y_train_file = os.path.join(Conf.data_dir, dataset + '_y_train.npy')
    
    x_val_file = os.path.join(Conf.data_dir, dataset + '_x_val.npy')
    y_val_file = os.path.join(Conf.data_dir, dataset + '_y_val.npy')
    
    x_test_file = os.path.join(Conf.data_dir, dataset + '_x_test.npy')
    y_test_file = os.path.join(Conf.data_dir, dataset + '_y_test.npy')
    
    
    model_name = '_'.join([str(v) for v in args])
    model_name = process_model_name(model_name)
    input_dim, output_dim = Utility.get_data_dimension(dataset)
    
    model = PMLP(tmp_dir=Conf.tmp_dir,
                 model_name=model_name,
                 input_dim=input_dim,
                 output_dim=output_dim,
                 block_size=block_size,
                 max_block=max_block,
                 max_layer=max_layer,
                 loss='categorical_crossentropy',
                 direction='higher',
                 output_activation='softmax',
                 metrics=['acc', 'categorical_crossentropy'],
                 subset_percentage=subset_percentage,
                 sampling_strategy=sampling_strategy,
                 convergence_measure='val_acc',
                 regularizer=regularizer,
                 dropout=dropout,
                 lr=Conf.LR,
                 epochs=Conf.EPOCH,
                 start_seed=exp,
                 computation=('cpu', 4))
    
    start_time = time()
    statistics, topology, weights = model.fit(x_train_file, 
                                              y_train_file, 
                                              x_val_file, 
                                              y_val_file, 
                                              batch_size=Conf.BATCH_SIZE)
    stop_time = time()
    optimize_time = stop_time - start_time
    
    finetune_data = model.finetune(finetune_params={'lr': Conf.FINETUNE_LR,
                                                    'epochs': Conf.EPOCH,
                                                    'regularizer': regularizer,
                                                    'dropout': dropout,
                                                    'subset_percentage': 1.0})
        
    val_performance = model.evaluate(x_val_file, y_val_file, Conf.BATCH_SIZE)
    test_performance = model.evaluate(x_test_file, y_test_file, Conf.BATCH_SIZE)

    
    history = model.process_history()
    
    result = {'statistics': statistics, 
              'topology': topology,
              'weights': weights,
              'finetune_data': finetune_data,
              'history': history,
              'optimize_time': optimize_time,
              'time_taken': optimize_time}
    
    for metric in val_performance.keys():
        result['val_' + metric] = val_performance[metric]
        result['test_' + metric] = test_performance[metric]
        
    """
    compute statistics
    """
    nb_sample = np.load(y_train_file).shape[0]
    index_histogram = model._compute_subset_index_histogram(statistics['subset_indices'], nb_sample)
    result['number_of_selected_sample'] = np.where(index_histogram > 0)[0].size
    result['max_frequency'] = np.max(index_histogram)
    result['min_frequency'] = np.min(index_histogram)
    result['index_histogram'] = index_histogram
    result['std_frequency'] = np.std(index_histogram)
    
    nb_block_optimize = 0
    nb_layer_optimize = len(statistics['measure'])
    for layer in statistics['measure']:
        for block in layer:
            nb_block_optimize += 1
    
    result['nb_layer_optimize'] = nb_layer_optimize
    result['nb_block_optimize'] = nb_block_optimize
    
    nb_block = 0
    nb_layer = len(topology)-2
    for layer in topology[1:-1]:
        for block in layer:
            nb_block += 1
    
    result['nb_layer'] = nb_layer
    result['nb_block'] = nb_block
    result['optimize_time_per_block'] = optimize_time / (float(nb_block_optimize) * calculate_nb_configuration(regularizer, dropout))
    
    return result

