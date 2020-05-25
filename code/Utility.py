#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

from sklearn import cluster as Cluster
import numpy as np
from joblib import Parallel, delayed
import random
import threading
import os
import exp_configurations as Conf
from keras.utils import to_categorical
import pickle

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)

    def next(self):     # Py2
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def calculate_entropy(pred_prob, labels):
    if pred_prob is None and labels is None:
        return None
    nb_sample = pred_prob.shape[0]
    entropy = -np.log(pred_prob[list(range(nb_sample)), np.argmax(labels, axis=1)])
    return entropy


def cluster_sampling_runner(cluster_label, entropy, cluster_index):
    """
    given the cluster assignment for all samples, return the index of sample
    that has highest entropy within cluster_index
    """
    
    sample_indices = np.where(cluster_label == cluster_index)
    if entropy is None:
        return random.choice(sample_indices[0])
    else:
        highest_entropy_idx = np.argmax(entropy[sample_indices])
        return sample_indices[0][highest_entropy_idx]

def cluster_sampling(cluster_label, nb_cluster, entropy, sampling_routine):
    assert sampling_routine in ['iterative', 'parallel']
    
    if sampling_routine == 'parallel':
        indices = Parallel(n_jobs=-1, prefer='processes')(delayed(cluster_sampling_runner)(cluster_label, entropy, cluster_index) for cluster_index in range(nb_cluster))
    else:
        indices = []
        for i in range(nb_cluster):
            cluster_indices = np.where(cluster_label == i)
            if entropy is None:
                idx = random.choice(cluster_indices[0])
            else:
                idx = cluster_indices[0][np.argmax(entropy[cluster_indices])]
            indices.append(idx)
            
    return indices

def subset_sampling(x_hidden, 
                    pred_prob, 
                    labels, 
                    percentage, 
                    sampling_strategy,
                    cluster_algo='kmean', 
                    normalize=True, 
                    sampling_routine='parallel',
                    loss_type='mse'):
    """
    algorithm to perform subset sampling based on clustering and labels
    
    Arguments
        x_hidden: NxD
        pred_prob: NxO
        labels: Nx0
        percentage: 0 < p <= 1
        cluster_algo: 'kmean', 'agglomerative'
        
    Return
        indices of selected samples
    """
    
    assert cluster_algo in ['kmean', 'agglomerative'], 'Unsupported clustering algorithm'
    assert loss_type in ['mse', 'entropy']
    
    nb_sample = x_hidden.shape[0]
    if percentage == 1.0:
        return list(range(nb_sample))
    
    nb_select = round(percentage * nb_sample)
    
    if sampling_strategy == 'cluster_loss':
    
        # normalize hidden representation 
        if normalize:
            x_norm = np.linalg.norm(x_hidden, axis=1, keepdims=True)
            x_hidden /= (x_norm + 1e-4)
        
        # create features to perform clustering
        if pred_prob is not None and labels is not None:
            x = np.concatenate((x_hidden, pred_prob), axis=1)
        else:
            x = x_hidden
            
        if cluster_algo == 'kmean':
            clusterer = Cluster.KMeans(n_clusters=nb_select, n_init=1, n_jobs=-1, max_iter=100, algorithm='elkan')
        else:
            clusterer = Cluster.AgglomerativeClustering(n_clusters=nb_select)
        
        # perform clustering & get cluster labels
        
        print('shape of data to cluster: %s' % str(x.shape))
        cluster_label = clusterer.fit_predict(x)
        
        if loss_type == 'entropy':
            losses = calculate_entropy(pred_prob, labels)
        else:
            losses = np.sum((pred_prob - labels)**2, axis=-1)
            
        indices = cluster_sampling(cluster_label, nb_select, losses, sampling_routine)
    
    elif sampling_strategy == 'random':
        indices = list(range(nb_sample))
        random.shuffle(indices)
        indices = indices[:nb_select]
    
    elif sampling_strategy == 'top_loss':
        if loss_type == 'entropy':
            losses = calculate_entropy(pred_prob, labels)
        else:
            losses = np.sum((pred_prob - labels)**2, axis=-1)
            
        indices = np.argsort(losses)[-nb_select:]
        
    else:
        raise RuntimeError('Unknown sampling strategy (%s)' % str(sampling_strategy))
        
    return indices
    
def load_raw_data(name):
    prefix = Conf.data_dir
    x_train = np.load(os.path.join(prefix, name + '_x_train.npy'))
    y_train = np.load(os.path.join(prefix, name + '_y_train.npy'))
    x_val = np.load(os.path.join(prefix, name + '_x_val.npy'))
    y_val = np.load(os.path.join(prefix, name + '_y_val.npy'))
    x_test = np.load(os.path.join(prefix, name + '_x_test.npy'))
    y_test = np.load(os.path.join(prefix, name + '_y_test.npy'))

    
    n_class = np.unique(y_train).size
    
    y_train = to_categorical(y_train, n_class)
    y_val = to_categorical(y_val, n_class)
    y_test = to_categorical(y_test, n_class)
    
    scale = 255.0 if np.max(x_train.flatten()) > 1 else 1.0
    
    x_train = x_train.astype('float32') / scale
    y_train = y_train.astype('float32') / scale
    x_val = x_val.astype('float32') / scale
    y_val = y_val.astype('float32') / scale
    x_test = x_test.astype('float32') / scale
    y_test = y_test.astype('float32') / scale
    
    return x_train, y_train, x_val, y_val, x_test, y_test



def load_data(name):
    suffix = ''
        
    # load data, if autoencoder features not exist, generate them
    x_train_file = os.path.join(Conf.data_dir, name + suffix + '_x_train.npy')
    x_val_file = os.path.join(Conf.data_dir, name + suffix + '_x_val.npy')
    x_test_file = os.path.join(Conf.data_dir, name + suffix + '_x_test.npy')

    x_train = np.load(x_train_file)
    x_val = np.load(x_val_file)
    x_test = np.load(x_test_file)
        
    y_train = np.load(os.path.join(Conf.data_dir, name + '_y_train.npy'))
    y_val = np.load(os.path.join(Conf.data_dir, name + '_y_val.npy'))
    y_test = np.load(os.path.join(Conf.data_dir, name + '_y_test.npy'))
        
    return x_train, y_train, x_val, y_val, x_test, y_test

def get_data_dimension(dataset):
    datasets = ['caltech256', 'mit_indoor', 'cfw']
    assert dataset in datasets
    inputs = [512, 512, 512]
    outputs = [257, 67, 500]
    
    input_dim = inputs[datasets.index(dataset)]
    output_dim = outputs[datasets.index(dataset)]
    
    
    return input_dim, output_dim

def check_result(args, output_dir):
    
    result_filename = '_'.join([str(v) for v in args]) + '.pickle'
    result_filename = os.path.join(output_dir, result_filename)
    
    if os.path.exists(result_filename):
        fid = open(result_filename, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result
    else:
        return None


