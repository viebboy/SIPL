#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

from time import time
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import categorical_crossentropy as entropy_loss


def standardize(X, stat=None, epsilon=1e-6):
    if stat is None:
        x_mean = np.mean(X, axis=1, keepdims=True)
        x_std = np.std(X, axis=1, keepdims=True)
        stat = [x_mean, x_std]
    else:
        x_mean = stat[0]
        x_std = stat[1]
    
    X = (X - x_mean)/(x_std + epsilon)
    return X, stat

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



class PLN:

    def get_default_parameters(self,):
        params = {}
        params['block_size'] = 40
        params['max_block'] = 5
        params['max_layer'] = 6
        params['epsilon'] = 1e-6
        params['metrics'] = ['mse',]
        params['convergence_measure'] = 'train_mse'
        params['direction'] = 'lower'
        params['threshold'] = 1e-4
        params['max_iter'] = 500
        params['muy'] = 1.0
        params['alpha'] = 1.0
        params['regularizer'] = 1.0
        params['max_patient_factor'] = 1
        params['output_activation'] = None
        
        return params
        
    def check_convergence(self, performance, convergence_measure, direction, threshold):
        sign = 1.0 if direction=='higher' else -1.0
        if len(performance[convergence_measure]) >=2 and \
            (performance[convergence_measure][-1] - performance[convergence_measure][-2])*sign /performance[convergence_measure][-2]  < threshold:
            return True
        else:
            return False
    
    def debug(self, x):
        print(x)
        return
    
    def V_(self, N):
        tmp = np.eye(N)
        V = np.concatenate((tmp, -tmp), axis=0)
        return V
    
    def U_(self, N):
        tmp = np.eye(N)
        U = np.concatenate((tmp, -tmp), axis=1)
        return U
    
    def mse(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype='float32')
        y_pred = tf.convert_to_tensor(y_pred, dtype='float32')
        
        return K.eval(tf.losses.mean_squared_error(y_true, y_pred))
    
    def accuracy(self, y_true, y_pred):
        y_pred_lb = np.argmax(y_pred, axis=0)
        y_true_lb = np.argmax(y_true, axis=0)
    
        return np.sum(y_pred_lb == y_true_lb) / float(len(y_pred_lb))
    
    def categorical_crossentropy(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true.T, dtype='float32')
        y_pred = tf.convert_to_tensor(y_pred.T, dtype='float32')
        
        tmp = K.eval(entropy_loss(y_true, y_pred))
        return np.mean(tmp)
        
    def RR(self, X,Y,c):
        D, N = X.shape
        W = np.linalg.pinv(np.dot(X,X.T) + c*np.eye(D))
        W = np.dot(Y, np.dot(W,X).T)
        return W
    
    def get_loss_func(self, loss_type):
        if loss_type == 'mse':
            return self.mse
        if loss_type == 'nme':
            return self.nme
    
    def admm(self, Y, T, alpha, max_iter, muy, fuzzy):
        D, N =Y.shape
        D_prime = T.shape[0]
        U = self.U_(D_prime)
        epsilon = np.sqrt(alpha) * np.linalg.norm(U.flatten())
        
        C = np.linalg.pinv(np.dot(Y,Y.T) + 1.0/muy*np.eye(D))
        
        Q = np.zeros((D_prime, D))
        O = np.zeros((D_prime, D))
        Lambda = np.zeros((D_prime, D))
        
        for k in range(max_iter):
            O = np.dot((np.dot(T,Y.T) + (Q + Lambda)/muy), C)
            norm = np.linalg.norm(O-Lambda) + fuzzy
            if  norm > epsilon:
                Q = (O - Lambda)*epsilon/norm
            else:
                Q = O - Lambda
            Lambda = Lambda + Q - O
        return O
    
        
    def column_normalize(self, X):
        col_norm = np.linalg.norm(X, axis=0, keepdims = True)
        return X/(col_norm + 1e-6)
        
    def relu(self, x):
        mask = x > 0.0
        x = mask * x
        return x

    def evaluate(self, y_true, y_pred, metric):
        if metric == 'mse' or metric == 'mean_squared_error':
            return self.mse(y_true, y_pred)
        elif metric == 'accuracy' or metric == 'acc':
            return self.accuracy(y_true, y_pred)
        elif metric == 'categorical_crossentropy':
            return self.categorical_crossentropy(y_true, y_pred)
        else:
            raise Exception('unknown metric "%s"' % metric)
    
    def print_performance(self, train_performance, val_performance, test_performance):
        print('#############################')
        for metric in train_performance.keys():
            print('train_%s: %.7f' %(metric, train_performance[metric][-1]))
            
        if val_performance is not None:
            for metric in val_performance.keys():
                print('val_%s: %.7f' %(metric, val_performance[metric][-1]))
                
        if test_performance is not None:
            for metric in test_performance.keys():
                print('test_%s: %.7f' %(metric, test_performance[metric][-1]))
        return
    
    def fit(self, 
            params,
            train_data, 
            val_data,
            test_data):

        patient_factor = 0
        train_performance = {}
        val_performance = {} if val_data is not None else None
        test_performance = {} if test_data is not None else None
        
        train_data[0] = train_data[0].T
        train_data[1] = train_data[1].T
        if val_data is not None:
            val_data[0] = val_data[0].T
            val_data[1] = val_data[1].T
        if test_data is not None:
            test_data[0] = test_data[0].T
            test_data[1] = test_data[1].T
        
        for metric in params['metrics']:
            train_performance[metric] = []
            if val_performance is not None:
                val_performance[metric] = []
            if test_performance is not None:
                test_performance[metric] = []
        
        O = self.RR(train_data[0], train_data[1], params['regularizer'])
        
        model_O = [O,]
        model_R = []
        
        
        Q = train_data[1].shape[0]
        V = self.V_(Q)
        
        layer_performance = []
        
        for layer_iter in range(params['max_layer']):
            print('layer ' + str(layer_iter))
            
            P, N = train_data[0].shape
            W = np.dot(V,model_O[-1])

            x_train_p1 = np.dot(W, train_data[0])
            x_val_p1 = np.dot(W, val_data[0]) if val_data is not None else None
            x_test_p1 = np.dot(W, test_data[0]) if test_data is not None else None
            
            
            layer_O = None
            layer_R = None
            R = None
            
            for neuron in range(params['max_block']):
                print('block ' + str(neuron))
                if R is None:
                    R = np.random.uniform(-1.0, 1.0, (params['block_size'],P))
                else:
                    R = np.concatenate((R,np.random.uniform(-1.0, 1.0, (params['block_size'],P))), axis=0)
                    
                x_train_p2 = self.column_normalize(np.dot(R, train_data[0]))
                x_val_p2 = self.column_normalize(np.dot(R, val_data[0])) if val_data is not None else None
                x_test_p2 = self.column_normalize(np.dot(R, test_data[0])) if test_data is not None else None

                x_train = np.concatenate((x_train_p1, x_train_p2), axis=0)
                x_val = np.concatenate((x_val_p1, x_val_p2), axis=0) if val_data is not None else None
                x_test = np.concatenate((x_test_p1, x_test_p2), axis=0) if test_data is not None else None
                
                x_train = self.relu(x_train)
                x_val = self.relu(x_val) if x_val is not None else None
                x_test = self.relu(x_test) if x_test is not None else None
                
                
                O = self.admm(x_train, train_data[1], params['alpha'], \
                              params['max_iter'], params['muy'], params['epsilon'])
                
                y_train_pred = np.dot(O, x_train) 
                y_val_pred = np.dot(O, x_val) if x_val is not None else None
                y_test_pred = np.dot(O, x_test) if x_test is not None else None
                
                if params['output_activation'] is not None:
                    y_train_pred = softmax(y_train_pred, axis=0)
                    y_val_pred = softmax(y_val_pred, axis=0) if y_val_pred is not None else None
                    y_test_pred = softmax(y_test_pred, axis=0) if y_test_pred is not None else None
                
                for metric in params['metrics']:
                    train_performance[metric].append(self.evaluate(train_data[1], y_train_pred, metric))
                    if val_performance is not None:
                        val_performance[metric].append(self.evaluate(val_data[1], y_val_pred, metric))
                    if test_performance is not None:
                        test_performance[metric].append(self.evaluate(test_data[1], y_test_pred, metric))   
                        
                self.print_performance(train_performance, val_performance, test_performance)
                
                layer_O = O
                layer_R = R
                
                measure = '_'.join(params['convergence_measure'].split('_')[1:])
                if params['convergence_measure'].startswith('train'):
                    performance = train_performance
                else:
                    performance = val_performance
                    
                if neuron > 0:
                    if self.check_convergence(performance, measure, params['direction'], params['threshold']):
                        patient_factor += 1
                    else:
                        patient_factor = 0
                    
                    print('patient factor: %d' % patient_factor)
                    if patient_factor >= params['max_patient_factor']:
                        break
                    
            
            sign = 1.0 if params['direction'] == 'higher' else -1.0
            
            model_R.append(layer_R)
            model_O.append(layer_O)
            
            layer_performance.append(performance[measure][-1])
            if len(layer_performance) > 1 and \
                sign*(layer_performance[-1] - layer_performance[-2])/ layer_performance[-2] < params['threshold']:
                break
            
            """ transform input"""
        

            train_data[0] = x_train
            if val_data is not None:
                val_data[0] = x_val
            if test_data is not None:
                test_data[0] = x_test
            

        model = {}
        model['O'] = model_O
        model['R'] = model_R
        model['V'] = V

        return model, train_performance, val_performance, test_performance
    
    def predict(self, X):
        if self.is_optimize == False:
            return
        
        if self.model['input_normalize']:
            X, input_stat = standardize(X, self.model['input_stat'])
        
        O = self.model['O']
        R = self.model['R']
        V = self.model['V']
        L = len(R) # number of hidden layer
        
        for l in range(L):
            W = np.dot(V, O[l])
            part1 = np.dot(W, X)
            part2 = self.column_normalize(np.dot(R[l], X))
            X = np.concatenate((part1, part2), axis=0)
            X = self.relu(X)
            
        Y_pred = np.dot(O[l+1], X)

        return Y_pred
            
        
            
            
            
                
                
            
        
        