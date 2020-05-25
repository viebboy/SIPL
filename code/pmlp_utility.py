#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""
from sklearn import cluster as Cluster
from keras.layers import Input, Dropout, Activation, Dense, Add, BatchNormalization as BN, Concatenate
from keras import Model, regularizers, constraints
from keras import backend as K
import numpy as np
import os
import pickle
import random
import copy
from keras import losses as keras_losses
from joblib import Parallel, delayed

def softmax(X, theta = 1.0, axis = -1):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def finetune_worker(train_states, hyperparameters, conf_index, save_result=True, subset_indices=None):
    result_file = os.path.join(hyperparameters['log_dir'], 'finetune_result_%d.pickle' % conf_index)
    if save_result and os.path.exists(result_file):
        return
    
    weight_decay, weight_constraint, dropout = create_hyperparameter_configuration(hyperparameters['regularizer'], 
                                                                                   hyperparameters['dropout'])[conf_index]
    
    topology = train_states['topology']
    weights = train_states['weights']
     
    # build the model
    model = network_builder(topology, weight_decay, weight_constraint, dropout, stage='finetune')
    model.compile('adam', hyperparameters['loss'], hyperparameters['metrics'])
    
    # set the weights
    for layer in model.layers:
        if layer.name in train_states['weights'].keys():
            layer.set_weights(train_states['weights'][layer.name])
    
    
    trainable_layers = [layer.name for layer in model.layers\
                        if layer.name.startswith('dense') or\
                        layer.name.startswith('bn') or\
                        layer.name.startswith('output')]

            
    # load data
    x_train, y_train, x_val, y_val = load_data(hyperparameters['x_train_file'],
                                               hyperparameters['y_train_file'],
                                               hyperparameters['x_val_file'],
                                               hyperparameters['y_val_file'])
    
    if subset_indices is not None:
        x_train = x_train[subset_indices]
        y_train = y_train[subset_indices]
        
    model, measure, history = network_trainer(model, x_train, y_train, x_val, y_val, hyperparameters, trainable_layers)
    
    weights = {layer.name: layer.get_weights() for layer in model.layers if layer.name in trainable_layers}
    
    result = {'weights': weights,
              'measure': measure,
              'history': history,
              'conf_index': conf_index}
    
    K.clear_session()
    
    if save_result:
        fid = open(result_file, 'wb')
        pickle.dump(result, fid)
        fid.close()
        
        summary_file = os.path.join(hyperparameters['log_dir'], 'finetune_result_%d.txt' % conf_index)
        fid = open(summary_file, 'w')
        fid.write('%.7f' % measure)
        fid.close()
        return
    else:
        return result
    
def result_collector(log_dir, nb_configuration, direction, stage='block_optimization'):
    sign = 1.0 if direction == 'higher' else -1.0
    best_measure = np.inf if sign == -1.0 else -np.inf
    best_conf_index = -1
    removed_files = []
    
    prefix = 'search' if stage == 'block_optimization' else 'finetune'
    
    # loop through summary file to find the best configuration index
    for conf_index in range(nb_configuration):
        summary_file = os.path.join(log_dir, '%s_result_%d.txt' % (prefix, conf_index))
        result_file = os.path.join(log_dir, '%s_result_%d.pickle' % (prefix, conf_index))
        assert os.path.exists(summary_file), 'Result doesnt exist for index (%d) during %s result collection process' % (prefix, conf_index)
        assert os.path.exists(result_file), 'Result doesnt exist for index (%d) during %s result collection process' % (prefix, conf_index)
        
        removed_files.append(summary_file)
        removed_files.append(result_file)
        
        fid = open(summary_file, 'r')
        measure = float(fid.read())
        if sign * (measure - best_measure) > 0:
            best_measure = measure
            best_conf_index = conf_index

    # load the full result
    fid = open(os.path.join(log_dir, '%s_result_%d.pickle' % (prefix, best_conf_index)), 'rb')
    result = pickle.load(fid)
    fid.close()
    
    # remove intermediate files after collection
    for f in removed_files:
        os.remove(f)
    
    return result

def finetune(train_states, hyperparameters):

    nb_configuration = len(create_hyperparameter_configuration(hyperparameters['regularizer'], 
                                                               hyperparameters['dropout']))
    
    subset_indices = subset_sampling_for_finetune(train_states, hyperparameters)
    
    print('--- finetune with %d configurations' % nb_configuration)
    if nb_configuration == 1:
        result = finetune_worker(train_states, hyperparameters, conf_index=0, save_result=False, subset_indices=subset_indices)
    else:
        # perform parallel evaluation of different regularization configuration
        Parallel(n_jobs=hyperparameters['computation'][-1])(
        delayed(finetune_worker)(train_states, hyperparameters, conf_index, True, subset_indices=subset_indices) for conf_index in range(nb_configuration))
        
        result = result_collector(hyperparameters['log_dir'], nb_configuration, hyperparameters['direction'], 'finetune')
        
    return result    
    
def evaluate(x_file, y_file, topology, model_weights, loss, metrics, batch_size):
    x = np.load(x_file)
    y = np.load(y_file)
    
    model = network_builder(topology, None, None, None)
    model.compile('adam', loss, metrics)
    
    # set weights
    for layer in model.layers:
        if layer.name in model_weights.keys():
            layer.set_weights(model_weights[layer.name])
    
    model_metrics = model.metrics_names
    
    output = model.evaluate(x, y, batch_size=batch_size, verbose=0)
    
    performance = {m: output[idx] for idx, m in enumerate(model_metrics)}
    
    K.clear_session()
    
    return performance
    
    
def network_builder(topology, 
                    weight_decay, 
                    weight_constraint,
                    dropout,
                    stage='block_optimization'):
    
    if weight_decay is not None:
        weight_decay = regularizers.l2(weight_decay)
    
    if weight_constraint is not None:
        weight_constraint = constraints.max_norm(weight_constraint, axis=0)
        
    input_dim = topology[0]
    output_dim = topology[-1][0]
    
    nb_hidden_layer = len(topology)-2
    inputs = Input((input_dim,), name='inputs')
    
    hiddens = inputs
    
    for layer_iter in range(nb_hidden_layer):
        hidden_blocks = []
        
        for block_iter in range(len(topology[1+layer_iter])):
            dense_output = Dense(topology[1+layer_iter][block_iter], 
                                 kernel_initializer='he_normal', 
                                 kernel_regularizer=weight_decay, 
                                 kernel_constraint=weight_constraint,
                                 name = 'dense%d_%d' % (layer_iter, block_iter))(hiddens)
            
            bn_output = BN(name='bn%d_%d' % (layer_iter, block_iter))(dense_output)
            
            act_output = Activation('relu')(bn_output)
            
            if (stage=='block_optimization' and\
                layer_iter == nb_hidden_layer -1 and\
                block_iter == len(topology[1+layer_iter])-1 and\
                dropout is not None) or\
                (stage=='finetune' and dropout is not None):
                    
                dropout_output = Dropout(dropout)(act_output)
                hidden_blocks.append(dropout_output)
            else:
                hidden_blocks.append(act_output)
            
        # if last layer, attach output prediction block
        if layer_iter == nb_hidden_layer - 1:
            output_blocks = []
            for block_idx, hidden_block in enumerate(hidden_blocks):
                output_blocks.append(Dense(output_dim,
                                           kernel_initializer='he_normal', 
                                           kernel_regularizer=weight_decay, 
                                           kernel_constraint=weight_constraint,
                                           name = 'output%d_%d' % (nb_hidden_layer-1, block_idx))(hidden_block))
                
        # if not last hidden layer, concatenate all hidden blocks
        else:
            if len(hidden_blocks) > 1:
                hiddens = Concatenate(axis=-1)(hidden_blocks)
            else:
                hiddens = hidden_blocks[0]
                
    if len(output_blocks) > 1:
        outputs = Add()(output_blocks)
    else:
        outputs = output_blocks[0]
        
    output_activation = topology[-1][-1]
    
    pre_predictions = Activation('linear', name='pre_predictions')(outputs)
    
    if output_activation is not None:
        predictions = Activation(output_activation, name='predictions')(pre_predictions)
    else:
        predictions = Activation('linear', name='predictions')(pre_predictions)
        
    model = Model(inputs=inputs, outputs=predictions)
        
    return model

def create_hyperparameter_configuration(regularizer, dropout):
    
    if isinstance(regularizer, list):
        regularizers_ = regularizer
    else:
        regularizers_ = [regularizer]
    
    if isinstance(dropout, list):
        dropout_ = dropout
    else:
        dropout_ = [dropout]
    
    conf_list = []
    for regu in regularizers_:
        for d in dropout_:
            # configuration: weight decay, weight constraint, dropout
            conf = (regu[0], regu[1], d)
            conf_list.append(conf)
            
    return conf_list

def load_data(x_train_file, y_train_file, x_val_file, y_val_file):
    x_train = np.load(x_train_file)
    y_train = np.load(y_train_file)
    x_val = np.load(x_val_file) if x_val_file is not None else None
    y_val = np.load(y_val_file) if y_val_file is not None else None
    
    return x_train, y_train, x_val, y_val


def network_trainer(model, x_train, y_train, x_val, y_val, hyperparameters, trainable_layers):
    
    current_weights = model.get_weights()
    optimal_weights = model.get_weights()
    
    sign = 1.0 if hyperparameters['direction'] == 'higher' else -1.0
    history = {}
    
    train_p = model.evaluate(x_train, y_train, batch_size=hyperparameters['batch_size'], verbose=0)
    val_p = model.evaluate(x_train, y_train, batch_size=hyperparameters['batch_size'], verbose=0) if x_val is not None else None
    
    for idx, metric in enumerate(model.metrics_names):
        history['train_' + metric] = [train_p[idx]]
        history['val_' + metric] = [val_p[idx]] if x_val is not None else None
        
    measure = history[hyperparameters['convergence_measure']][-1]
    
    for lr, epoch in zip(hyperparameters['lr'], hyperparameters['epochs']):
        model.compile('adam', hyperparameters['loss'], hyperparameters['metrics'])
        model.set_weights(current_weights)
        for layer in model.layers:
            if layer.name in trainable_layers:
                layer.trainable = True
            else:
                layer.trainable = False
                
        model.optimizer.lr = lr
        h = model.fit(x_train, 
                      y_train, 
                      epochs=epoch, 
                      batch_size=hyperparameters['batch_size'],
                      validation_data=(x_val, y_val) if x_val is not None else None,
                      verbose=0)
        
        for metric in hyperparameters['metrics']:
            history['train_' + metric] += h.history[metric]
            if x_val is not None:
                history['val_' + metric] += h.history['val_' + metric]
        
        current_weights = model.get_weights()
        
        if sign*(history[hyperparameters['convergence_measure']][-1] - measure) > 0:
            optimal_weights = model.get_weights()
            measure = history[hyperparameters['convergence_measure']][-1]
        
    model.set_weights(optimal_weights)
    
    return model, measure, history

def block_optimizer(train_states, hyperparameters, conf_index, save_result=True):
    """ runner to feed into parallel for loop
    
    perform block training given the configuration (weight decay, weight constraint, dropout) index
    
    """
    result_file = os.path.join(hyperparameters['log_dir'], 'search_result_%d.pickle' % conf_index)
    if save_result and os.path.exists(result_file):
        return
    
    weight_decay, weight_constraint, dropout = create_hyperparameter_configuration(hyperparameters['regularizer'], 
                                                                                   hyperparameters['dropout'])[conf_index]
    
    topology = copy.copy(train_states['topology'])
    layer_iter = train_states['layer_iter']
    block_iter = train_states['block_iter']
    
    # add new block to topology
    if layer_iter == 0 and block_iter == 0:
        topology[1].append(hyperparameters['block_size'])
    elif layer_iter > 0 and block_iter == 0:
        output_layer = topology.pop(-1)
        topology.append([hyperparameters['block_size']])
        topology.append(output_layer)
    else:
        topology[-2].append(hyperparameters['block_size'])
     
    # build the model
    model = network_builder(topology, weight_decay, weight_constraint, dropout, stage='block_optimization')
    model.compile('adam', hyperparameters['loss'], hyperparameters['metrics'])
    
    # set the weights
    for layer in model.layers:
        if layer.name in train_states['weights'].keys():
            layer.set_weights(train_states['weights'][layer.name])
    
    # free all layers, except current hidden block and output block
    trainable_layers = ['dense%d_%d' % (layer_iter, block_iter), 'bn%d_%d' % (layer_iter, block_iter), 'output%d_%d' % (layer_iter, block_iter)]
    for layer in model.layers:
        if layer.name in trainable_layers:
            layer.trainable = True
        else:
            layer.trainable = False
            
    # load data
    x_train, y_train, x_val, y_val = load_data(hyperparameters['x_train_file'],
                                               hyperparameters['y_train_file'],
                                               hyperparameters['x_val_file'],
                                               hyperparameters['y_val_file'])
    
    if train_states['subset_indices'] is None:
        subset_indices = list(range(x_train.shape[0]))
        random.shuffle(subset_indices)
        subset_indices = subset_indices[:int(np.floor(hyperparameters['subset_percentage']*x_train.shape[0]))]
    else:
        subset_indices = train_states['subset_indices']
        
    x_train = x_train[subset_indices]
    y_train = y_train[subset_indices]
    
    model, measure, history = network_trainer(model, x_train, y_train, x_val, y_val, hyperparameters, trainable_layers)
    
    result = {'dense_weights': model.get_layer('dense%d_%d' %(layer_iter, block_iter)).get_weights(),
              'bn_weights': model.get_layer('bn%d_%d' % (layer_iter, block_iter)).get_weights(),
              'output_weights': model.get_layer('output%d_%d' % (layer_iter, block_iter)).get_weights(),
              'measure': measure,
              'history': history,
              'conf_index': conf_index}
    
    K.clear_session()
    
    if save_result:
        fid = open(result_file, 'wb')
        pickle.dump(result, fid)
        fid.close()
        
        summary_file = os.path.join(hyperparameters['log_dir'], 'search_result_%d.txt' % conf_index)
        fid = open(summary_file, 'w')
        fid.write('%.7f' % measure)
        fid.close()
        return
    else:
        return result
    
def optimize_block(train_states, hyperparameters):
    
    nb_configuration = len(create_hyperparameter_configuration(hyperparameters['regularizer'], 
                                                               hyperparameters['dropout']))
    
    print('--- block optimization with %d configurations' % nb_configuration)
    if nb_configuration == 1:
        result = block_optimizer(train_states, hyperparameters, conf_index=0, save_result=False)
    else:
        # perform parallel evaluation of different regularization configuration
        Parallel(n_jobs=hyperparameters['computation'][-1])(
        delayed(block_optimizer)(train_states, hyperparameters, conf_index, True) for conf_index in range(nb_configuration))
        
        result = result_collector(hyperparameters['log_dir'], nb_configuration, hyperparameters['direction'], 'block_optimization')
        
    return result


def cluster_sampling_runner(cluster_labels, subset_percentage, losses, cluster_index):
    """
    given the cluster assignment for all samples, return the index of sample
    that has highest loss within cluster_index
    """
    
    sample_indices = np.where(cluster_labels == cluster_index)
    nb_select = int(np.ceil(sample_indices[0].size * subset_percentage))
    
    if losses is None:
        return random.sample(sample_indices[0], nb_select)
    else:
        highest_loss_indices = np.argsort(losses[sample_indices])[-nb_select:]
        return sample_indices[0][highest_loss_indices]

def cluster_sampling(cluster_labels, nb_cluster, subset_percentage, losses, sampling_routine):
    assert sampling_routine in ['iterative', 'parallel']
    
    if sampling_routine == 'parallel':
        indices_list = Parallel(n_jobs=-1)(delayed(cluster_sampling_runner)(cluster_labels, subset_percentage, losses, cluster_index) for cluster_index in range(nb_cluster))
        indices = [idx for idx_list in indices_list for idx in idx_list]
    else:
        indices = []
        for i in range(nb_cluster):
            cluster_indices = np.where(cluster_labels == i)
            nb_select = int(np.ceil(cluster_indices[0].size * subset_percentage))
            if losses is None:
                indices_ = random.sample(cluster_indices[0], nb_select)
            else:
                indices_ = cluster_indices[0][np.argsort(losses[cluster_indices])[-nb_select:]]
            indices += indices_.tolist()
            
    print('Sampling %d from %d samples' % (len(indices), len(cluster_labels)))
    return indices

def subset_sampling_for_finetune(train_states, hyperparameters):
    random.seed(hyperparameters['start_seed']*1000 + train_states['block_number'])
    y_train = np.load(hyperparameters['y_train_file'])
    nb_sample = y_train.shape[0]
    nb_select = int(np.floor(nb_sample * hyperparameters['subset_percentage']))
    
    if nb_select == nb_sample:
        return list(range(nb_sample))
    
    if hyperparameters['sampling_strategy'] == 'random-with-class':
        
        subset_indices = list(range(nb_sample))
        random.shuffle(subset_indices)
        subset_indices = subset_indices[:nb_select]
        
        subset_indices = []
        y_train_lb = np.argmax(y_train, axis=-1)
        for lb in np.unique(y_train_lb):
            indices = np.where(y_train_lb == lb)[0].tolist()
            random.shuffle(indices)
            nb_subset_select = int(np.floor(len(indices)*hyperparameters['subset_percentage']))
            subset_indices += indices[:nb_subset_select]
        return subset_indices
    
    if hyperparameters['sampling_strategy'] == 'random-without-class':
        subset_indices = list(range(nb_sample))
        random.shuffle(subset_indices)
        subset_indices = subset_indices[:nb_select]
        return subset_indices
    
    topology = copy.copy(train_states['topology'])     
    # build the model
    model = network_builder(topology, None, None, None)
    model.compile('adam', hyperparameters['loss'], hyperparameters['metrics'])
    
    # set the weights
    for layer in model.layers:
        if layer.name in train_states['weights'].keys():
            layer.set_weights(train_states['weights'][layer.name])
    
    # load full train data
    x_train = np.load(hyperparameters['x_train_file'])
    feature_model = Model(inputs=model.input, outputs=model.get_layer('pre_predictions').output)
    features = feature_model.predict(x_train, batch_size=hyperparameters['batch_size'], verbose=0)
    if topology[-1][-1] is not None:
        y_pred = softmax(features)
    else:
        y_pred = features
    # compute the loss induced by each sample
    losses = K.eval(keras_losses.get(hyperparameters['loss'])(K.variable(y_train), K.variable(y_pred)))
    
    K.clear_session()
    # 
    if hyperparameters['sampling_strategy'] == 'top_loss':
        subset_indices = np.argsort(losses)[-nb_select:]
        return subset_indices
    else:
        nb_cluster = min(hyperparameters['cluster_scale'] * topology[-1][0], nb_select)
        if hyperparameters['cluster_algo'] == 'kmean':
            clusterer = Cluster.KMeans(n_clusters=nb_cluster, n_init=1, n_jobs=-1, max_iter=100, algorithm='elkan')
        else:
            clusterer = Cluster.AgglomerativeClustering(n_clusters=nb_cluster)
            
        # perform clustering on the prediction
        cluster_labels = clusterer.fit_predict(features)
        
        subset_indices = cluster_sampling(cluster_labels, nb_cluster, hyperparameters['subset_percentage'], losses, hyperparameters['sampling_routine'])
        return subset_indices

def subset_sampling(train_states, hyperparameters, new_block_result):
    random.seed(hyperparameters['start_seed']*1000 + train_states['block_number'])
    y_train = np.load(hyperparameters['y_train_file'])
    nb_sample = y_train.shape[0]
    nb_select = int(np.floor(nb_sample * hyperparameters['subset_percentage']))
    
    if nb_select == nb_sample:
        return list(range(nb_sample))
    
    if hyperparameters['sampling_strategy'] == 'random-with-class':
        subset_indices = list(range(nb_sample))
        random.shuffle(subset_indices)
        subset_indices = subset_indices[:nb_select]
        
        subset_indices = []
        y_train_lb = np.argmax(y_train, axis=-1)
        for lb in np.unique(y_train_lb):
            indices = np.where(y_train_lb == lb)[0].tolist()
            random.shuffle(indices)
            nb_subset_select = int(np.floor(len(indices)*hyperparameters['subset_percentage']))
            subset_indices += indices[:nb_subset_select]
        return subset_indices
    
    if hyperparameters['sampling_strategy'] == 'random-without-class':
        subset_indices = list(range(nb_sample))
        random.shuffle(subset_indices)
        subset_indices = subset_indices[:nb_select]
        return subset_indices
    
    topology = copy.copy(train_states['topology'])
    layer_iter = train_states['layer_iter']
    block_iter = train_states['block_iter']
    weights = train_states['weights']
    
    # add new block to topology
    if layer_iter == 0 and block_iter == 0:
        topology[1].append(hyperparameters['block_size'])
    elif layer_iter > 0 and block_iter == 0:
        output_layer = topology.pop(-1)
        topology.append([hyperparameters['block_size']])
        topology.append(output_layer)
    else:
        topology[-2].append(hyperparameters['block_size'])
     
    # build the model
    model = network_builder(topology, None, None, None)
    model.compile('adam', hyperparameters['loss'], hyperparameters['metrics'])
    
    # set the weights of old blocks
    for layer in model.layers:
        if layer.name in weights.keys():
            layer.set_weights(weights[layer.name])
        
    # set the weights of new block
    model.get_layer('dense%d_%d' % (layer_iter, block_iter)).set_weights(new_block_result['dense_weights'])
    model.get_layer('bn%d_%d' % (layer_iter, block_iter)).set_weights(new_block_result['bn_weights'])
    model.get_layer('output%d_%d' % (layer_iter, block_iter)).set_weights(new_block_result['output_weights'])
    
    # load full train data
    x_train = np.load(hyperparameters['x_train_file'])
    feature_model = Model(inputs=model.input, outputs=model.get_layer('pre_predictions').output)
    features = feature_model.predict(x_train, batch_size=hyperparameters['batch_size'], verbose=0)
    if topology[-1][-1] is not None:
        y_pred = softmax(features)
    else:
        y_pred = features
    # compute the loss induced by each sample
    losses = K.eval(keras_losses.get(hyperparameters['loss'])(K.variable(y_train), K.variable(y_pred)))
    
    # 
    if hyperparameters['sampling_strategy'] == 'top_loss':
        subset_indices = np.argsort(losses)[-nb_select:]
        return subset_indices
    else:
        nb_cluster = min(hyperparameters['cluster_scale'] * topology[-1][0], nb_select)
        if hyperparameters['cluster_algo'] == 'kmean':
            clusterer = Cluster.KMeans(n_clusters=nb_cluster, n_init=1, n_jobs=-1, max_iter=100, algorithm='elkan')
        else:
            clusterer = Cluster.AgglomerativeClustering(n_clusters=nb_cluster)
            
        # perform clustering on the prediction
        cluster_labels = clusterer.fit_predict(features)
        
        subset_indices = cluster_sampling(cluster_labels, nb_cluster, hyperparameters['subset_percentage'], losses, hyperparameters['sampling_routine'])
        return subset_indices