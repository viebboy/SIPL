#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""


import numpy as np
import os
import pickle
import dill
import subprocess
import pmlp_utility

class PMLP:
    def __init__(self, 
                 tmp_dir,
                 model_name,
                 input_dim,
                 output_dim,
                 block_size=20,
                 max_block=5,
                 max_layer=5, 
                 threshold=1e-4, 
                 lr=(1e-3, 1e-4, 1e-4),
                 epochs=(1, 1, 1),
                 loss='categorical_crossentropy',
                 output_activation='softmax',
                 convergence_measure='train_acc',
                 metrics=['acc', 'categorical_crossentropy'],
                 direction='higher',
                 regularizer = (1e-4, None),
                 dropout = 0.1,
                 sampling_strategy='random',
                 subset_percentage=0.1,
                 cluster_algo='kmean',
                 cluster_scale=3, 
                 sampling_routine='iterative',
                 max_patient_factor=1,
                 computation=('cpu', 4),
                 start_seed=0):
        
        
        if not os.path.exists(tmp_dir):
            raise RuntimeError('tmp_dir (%s) doesnt exist' % str(tmp_dir))
        
        self._tmp_dir = tmp_dir
        self._model_name = model_name
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._log_dir = os.path.join(tmp_dir, model_name)
        if not os.path.exists(self._log_dir):
            os.mkdir(self._log_dir)
            
        self._train_state_file = os.path.join(self._log_dir, 'train_states.pickle')
        self._tmp_train_state_file = os.path.join(self._log_dir, 'tmp_train_states.pickle')
        self._parameter_file = os.path.join(self._log_dir, 'hyperparameters.pickle')
        self._block_size = block_size
        self._max_block = max_block
        self._max_layer = max_layer
        self._threshold = threshold
        self._lr = lr
        self._epochs = epochs
        self._loss = loss
        self._output_activation = output_activation
        self._convergence_measure = convergence_measure
        self._metrics = metrics
        self._direction = direction
        self._regularizer = regularizer
        self._sampling_strategy = sampling_strategy
        self._subset_percentage = subset_percentage
        self._cluster_algo = cluster_algo
        self._sampling_routine = sampling_routine
        self._max_patient_factor = max_patient_factor
        self._computation = computation
        self._patient_factor = 0
        self._topology = []
        self._weights = {}
        self._progression_terminate = False
        self._last_layer_iter = 0
        self._last_block_iter = 0
        self._dropout = dropout
        self._cluster_scale = cluster_scale
        self._start_seed = start_seed
        self._statistics = {'measure': [],
                            'history': [],
                            'conf_index': [],
                            'subset_indices': [],
                            'block_optimization_time': [],
                            'subset_sampling_time': []}

        self._initialize_train_state()
        
        
    def _initialize_train_state(self,):
        if os.path.exists(self._train_state_file):
            fid = open(self._train_state_file, 'rb')
            data = pickle.load(fid)
            fid.close()
            
            if data['topology'][0] == self._input_dim and data['topology'][-1][0] == self._output_dim:
                self._topology = data['topology']
                self._weights = data['weights']
                self._progression_terminate = data['progression_terminate']
                self._last_layer_iter = data['layer_iter']
                self._last_block_iter = data['block_iter']
                self._statistics = data['statistics']
            else:
                self._topology.append(self._input_dim)
                self._topology.append([])
                self._topology.append((self._output_dim, self._output_activation))
                
        else:
            self._topology.append(self._input_dim)
            self._topology.append([])
            self._topology.append((self._output_dim, self._output_activation))
            
    def _dump_tmp_train_state(self, subset_indices, block_number):
        data = {'topology': self._topology,
                'weights': self._weights,
                'block_number': block_number,
                'layer_iter': self._layer_iter,
                'block_iter': self._block_iter,
                'subset_indices': subset_indices}
        
        fid = open(self._tmp_train_state_file, 'wb')
        pickle.dump(data, fid)
        fid.close()
        
    def _dump_train_state(self, layer_iter, block_iter, block_number):
        
        data = {'topology': self._topology,
                'weights': self._weights,
                'layer_iter': layer_iter,
                'block_iter': block_iter,
                'block_number': block_number,
                'progression_terminate': self._progression_terminate,
                'statistics': self._statistics}
        
        fid = open(self._train_state_file, 'wb')
        pickle.dump(data, fid)
        fid.close()
        
        
    def _dump_hyperparameters(self, 
                              x_train_file, 
                              y_train_file, 
                              x_val_file, 
                              y_val_file, 
                              batch_size):
        
        hyperparameters = {'x_train_file': x_train_file,
                           'y_train_file': y_train_file,
                           'x_val_file': x_val_file,
                           'y_val_file': y_val_file,
                           'batch_size': batch_size,
                           'dropout': self._dropout,
                           'log_dir': self._log_dir,
                           'block_size': self._block_size,
                           'lr': self._lr,
                           'epochs': self._epochs,
                           'loss': self._loss,
                           'convergence_measure': self._convergence_measure,
                           'direction': self._direction,
                           'metrics': self._metrics,
                           'regularizer': self._regularizer,
                           'cluster_scale': self._cluster_scale,
                           'start_seed': self._start_seed,
                           'sampling_strategy': self._sampling_strategy,
                           'subset_percentage': self._subset_percentage,
                           'cluster_algo': self._cluster_algo,
                           'sampling_routine': self._sampling_routine,
                           'computation': self._computation}
        
        fid = open(self._parameter_file, 'wb')
        pickle.dump(hyperparameters, fid)
        fid.close()
        
    def _optimize_block(self, subset_indices, block_number):
        self._dump_tmp_train_state(subset_indices, block_number)
        runnable = 'pmlp_block_optimize.py'
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), runnable)
        cmd = 'pmlp_log_dir=%s python %s' % (self._log_dir, filename)
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        
        status_file = os.path.join(self._log_dir, 'new_block_complete.txt')
        output_file = os.path.join(self._log_dir, 'new_block.pickle')
        
        if not os.path.exists(status_file):
            raise RuntimeError('Block optimization fails')
        else:
             
            fid = open(output_file, 'rb')
            block_data = pickle.load(fid)
            fid.close()
            
            os.remove(status_file)
            os.remove(output_file)
            
        return block_data
    

    def evaluate(self, x_file, y_file, batch_size=64):
        print('*** Evaluating ***')
        performance = pmlp_utility.evaluate(x_file, y_file, self._topology, self._weights, self._loss, self._metrics, batch_size)
        for metric in performance.keys():
            print('--- %s: %.6f' % (metric, performance[metric]))
        return performance
    
    def _finetune(self, finetune_params):
        # finetune_params if not None should contain: regularizer, dropout, lr, epoch
        if finetune_params is not None:
            file = os.path.join(self._log_dir, 'finetune_parameters.pickle')
            fid = open(file, 'wb')
            pickle.dump(finetune_params, fid)
            fid.close()
            
        self._print_architecture()
        
        runnable = 'pmlp_finetune.py'
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), runnable)
        cmd = 'pmlp_log_dir=%s python %s' % (self._log_dir, filename)
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        
        status_file = os.path.join(self._log_dir, 'finetune_complete.txt')
        output_file = os.path.join(self._log_dir, 'finetune.pickle')
        
        if not os.path.exists(status_file):
            raise RuntimeError('Network finetune fails')
        else:
             
            fid = open(output_file, 'rb')
            finetune_data = pickle.load(fid)
            fid.close()
            
            os.remove(status_file)
            os.remove(output_file)
            
        return finetune_data
    
    def _print_performance(self,):
        print('--- %s = %.6f' % (self._convergence_measure, self._statistics['measure'][-1][-1]))
        
    def _print_speed(self,):
        print('--- block optimization takes %.1f' % self._statistics['block_optimization_time'][-1][-1])
        print('--- subset sampling    takes %.1f' % self._statistics['subset_sampling_time'][-1][-1])
        
    def _print_architecture(self,):
        architecture = '%d-> ' % self._input_dim
        
        for layer_iter in range(len(self._topology)-2):
            hidden_nodes = int(np.sum(self._topology[layer_iter + 1]))
            architecture += '%d-> ' % hidden_nodes
        
        architecture += '%d' % self._output_dim
        
        print('************************')
        print('*** Network Topology ***')
        print(architecture)
        print('************************')
        
    def _compute_subset_index_histogram(self, subset_indices, nb_sample):
        histogram = np.zeros((nb_sample,), dtype=np.int32)
        for layer in subset_indices:
            for block in layer:
                histogram[block] += 1
        
        return histogram
    
    def top_k_selected_sample_index(self, subset_indices, k, nb_sample):
        histogram = self._compute_subset_index_histogram(subset_indices, nb_sample)
        return np.argsort(-histogram)[:k]

    def top_k_selected_configuration(self, regularizer, dropout, conf_indices, k):
        configurations = pmlp_utility.create_hyperparameter_configuration(regularizer, dropout)
        histogram = np.zeros((len(configurations),), dtype=np.int32)
        for layer in conf_indices:
            for block in layer:
                histogram[block] += 1
        
        top_k = []
        sorted_indices = np.argsort(-histogram)[:min(k, len(configurations))]
        for idx in sorted_indices:
            top_k.append((configurations[idx][0], configurations[idx][1], configurations[idx][2], histogram[idx]))
            
        return top_k
    
    def process_history(self,):
        history = self._statistics['history']
        metrics = history[0][0].keys()
        
        flatten_history = {}
        
        for metric in self._metrics:
            for layer in history:
                for block in layer:
                    idx = np.argmax(block[self._convergence_measure]) if self._direction == 'higher' else np.argmin(block[self._convergence_measure]) 
                    if metric not in flatten_history.keys():
                        flatten_history[metric] = [[block['train_' + metric][idx]]]
                    else:
                        flatten_history[metric][0].append(block['train_' + metric][idx])
                        
                    if 'val_' + metric in metrics and block['val_' + metric] is not None:
                        if len(flatten_history[metric]) == 1:
                            flatten_history[metric].append([block['val_' + metric][idx]])
                        else:
                            flatten_history[metric][-1].append(block['val_' + metric][idx])

        return flatten_history        
        
    def fit(self, 
            x_train_file, 
            y_train_file, 
            x_val_file=None, 
            y_val_file=None,
            batch_size=64):
        
        if not self._progression_terminate:
            self._dump_hyperparameters(x_train_file,
                                       y_train_file,
                                       x_val_file,
                                       y_val_file,
                                       batch_size)
            
            subset_indices = None
            sign = 1.0 if self._direction=='higher' else -1.0
            
            
            for self._layer_iter in range(self._last_layer_iter, self._max_layer):
                if self._max_block == -1 or self._max_block is None:
                    if self._layer_iter == 0:
                        nb_max_block = int(np.ceil(self._input_dim / self._block_size)) + 1
                    else:
                        nb_max_block = len(self._topology[-2]) + 1
                else:
                    nb_max_block = self._max_block
                    
                for self._block_iter in range(self._last_block_iter, nb_max_block):
                    print('****************************************')
                    print('***** layer_iter=%d, block_iter=%d *****' % (self._layer_iter, self._block_iter))
                    print('****************************************')
                    print('--- subset sampling strategy: %s' % self._sampling_strategy)
                    print('--- subset percentage: %s' % self._subset_percentage)
                    if self._block_iter == 0:
                        self._statistics['measure'].append([])
                        self._statistics['history'].append([])
                        self._statistics['conf_index'].append([])
                        self._statistics['subset_indices'].append([])
                        self._statistics['block_optimization_time'].append([])
                        self._statistics['subset_sampling_time'].append([])
                        
                    # optimize new block
                    block_number = 0
                    for layer in self._topology[1:-1]:
                        for block in layer:
                            block_number += 1
                    
                    print('--- block number: %d ---------' % block_number)
                    block_data = self._optimize_block(subset_indices, block_number)
                    subset_indices = block_data['subset_indices']
                    
                    # save statistics
                    self._statistics['measure'][-1].append(block_data['measure'])
                    self._statistics['history'][-1].append(block_data['history'])
                    self._statistics['conf_index'][-1].append(block_data['conf_index'])
                    self._statistics['subset_indices'][-1].append(block_data['subset_indices'])
                    self._statistics['block_optimization_time'][-1].append(block_data['block_optimization_time'])
                    self._statistics['subset_sampling_time'][-1].append(block_data['subset_sampling_time'])
                    
                    self._print_performance()
                    self._print_speed()
                    
                    # update topology
                    if self._layer_iter == 0 and self._block_iter == 0:
                        self._topology[1].append(self._block_size)
                    elif self._layer_iter > 0 and self._block_iter == 0:
                        self._topology.pop(-1)
                        self._topology.append([self._block_size])
                        self._topology.append((self._output_dim, self._output_activation))
                    else:
                        self._topology[-2].append(self._block_size)
                        
                    self._print_architecture()
                    # update weights
                    self._weights['dense%d_%d' % (self._layer_iter, self._block_iter)] = block_data['dense_weights']
                    self._weights['bn%d_%d' % (self._layer_iter, self._block_iter)] = block_data['bn_weights']
                    self._weights['output%d_%d' % (self._layer_iter, self._block_iter)] = block_data['output_weights']
                    
                    # check convergence condition
                    if self._block_iter > 0:
                        is_deteriorate = sign*(self._statistics['measure'][-1][-1] -\
                                               self._statistics['measure'][-1][-2])/self._statistics['measure'][-1][-2] < self._threshold
                                               
                        if is_deteriorate:
                            self._patient_factor += 1
                        else:
                            self._patient_factor = 0
                            
                        if self._patient_factor >= self._max_patient_factor:
                            block_converge = True
                        else:
                            block_converge = False
                    else:
                        block_converge = False
                        
                    
                    if block_converge:
                        layer_iter = self._layer_iter +1
                        block_iter = 0
                        # remove blocks that corresponds to max_patient factor
                        for i in range(self._max_patient_factor):
                            self._topology[-2].pop(-1)
                            
                        self._dump_train_state(layer_iter, block_iter, block_number + 1)
                        break
                    else:
                        block_iter = self._block_iter+1 if self._block_iter < self._max_block -1 else 0
                        layer_iter = self._layer_iter if block_iter > 0 else self._layer_iter + 1
                        self._dump_train_state(layer_iter, block_iter, block_number + 1)
                        
                # check convergence condition of layer
                if self._layer_iter > 0:
                    if sign*(self._statistics['measure'][-1][-1] -\
                                               self._statistics['measure'][-2][-1])/self._statistics['measure'][-2][-1] < self._threshold:
                        layer_converge = True
                    else:
                        layer_converge = False
                else:
                    layer_converge = False
                
                if layer_converge:
                    self._progression_terminate = True
                    # remove the last hidden layer
                    self._topology.pop(-2)
                    self._dump_train_state(self._layer_iter, self._block_iter, block_number + 1)
                    break
            
        return self._statistics, self._topology, self._weights
    
    def finetune(self, finetune_params=None):
        print('*************************************')
        print('*** Finetuning Final Architecture ***')
        print('*************************************')
        finetune_data = self._finetune(finetune_params)
        print('--- %s = %.6f' % (self._convergence_measure, finetune_data['measure']))
        for layer_name in finetune_data['weights'].keys():
            self._weights[layer_name] = finetune_data['weights'][layer_name]
            
        return finetune_data