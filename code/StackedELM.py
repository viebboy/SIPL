import numpy as np
from time import time
from keras import backend as K
from keras import metrics as keras_metrics
from keras import losses as keras_loss
import Utility
import random


class StackedELM:
    
    def __init__(self,
                 layer_size=40,
                 pca_dimension=30,
                 max_layer=30,
                 threshold=1e-4,
                 convergence_measure='train_mse',
                 metrics=['acc',],
                 direction='higher',
                 regularizer=1e-1,
                 max_patient_factor=1):
    
        
        self._layer_size = layer_size
        self._pca_dimension = pca_dimension
        self._max_layer = max_layer
        self._threshold = threshold
        self._convergence_measure = convergence_measure
        self._metrics = metrics
        self._direction = direction
        self._regularizer = regularizer if regularizer is not None else 0.0
        self._patient_factor = 0
        self._max_patient_factor = max_patient_factor
        
    def _mse(self, y_true, y_pred):
        y_pred = K.cast_to_floatx(y_pred)
        y_true = K.cast_to_floatx(y_true)
        return np.mean(K.eval(keras_metrics.mean_squared_error(y_true, y_pred)))
    
    def _categorical_crossentropy(self, y_true, y_pred):
        return np.mean((-y_true*np.log(y_pred)).flatten())
    
    def _accuracy(self, y_true, y_pred):
        y_pred = K.cast_to_floatx(y_pred)
        y_true = K.cast_to_floatx(y_true)
        return np.sum(K.eval(keras_metrics.categorical_accuracy(y_true, y_pred)))/ float(y_pred.shape[0])
    
    
    def _evaluate(self,y_true, y_pred, metric):
        # C x N
        if metric == 'acc' or metric == 'accuracy':
            return self._accuracy(y_true, y_pred)
        elif metric == 'mse' or metric == 'mean_squared_error':
            return self._mse(y_true, y_pred)
        elif metric == 'categorical_crossentropy':
            return self._categorical_crossentropy(y_true, y_pred)
        else:
            raise Exception('unknown metric "%s"' % metric)
    
    def _sigmoid(self,x):
        return 1./(1. + np.exp(-x))
        
    def _ELM_AE(self, X):
        
        # X: NxD
        D = X.shape[-1]
        W = np.random.uniform(-1.0, 1.0, size=(D, self._layer_size))
        Bias = np.random.uniform(-1.0, 1.0, size=(1, self._layer_size))
        
        X_hidden = self._sigmoid(np.dot(X, W) + Bias)
    
        # W_reconstruct : hidden_size x D
        W_reconstruct = self._least_square(X_hidden, X)
        
        return W_reconstruct.T, Bias
    
    def _least_square(self, X,Y):
        # X: NxD, Y: NxC
        hidden_size = X.shape[-1]
        W = np.dot(np.dot(np.linalg.pinv(self._regularizer * np.eye(hidden_size) + np.dot(X.T, X)), X.T), Y)
        return W
    
    def _PCA(self, W):
        # W: hidden_size x output_size     
        cov = np.dot(W, W.T) + self._regularizer * np.eye(W.shape[0])
        U,S,V = np.linalg.svd(cov)
        # U: hidden_size x number_of_principal_axes
        if self._pca_dimension <= U.shape[1]:
            P = U[:,:self._pca_dimension]
        else:
            P = U
        
        return P
    
    def _check_convergence(self, performance):
        sign = 1.0 if self._direction=='higher' else -1.0
        measure = '_'.join(self._convergence_measure.split('_')[1:])
        
        is_deteriorate = len(performance[measure]) >=2 and\
             (performance[measure][-1] - performance[measure][-2])*sign / performance[measure][-2]  < self._threshold
        
        if is_deteriorate:
            self._patient_factor += 1
        else:
            self._patient_factor = 0
            
        if self._patient_factor >= self._max_patient_factor:
            return True
        else:
            return False
        
    def _print_performance(self, train_performance, val_performance, test_performance):

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
            train_data, 
            val_data, 
            test_data):
        
        
        train_performance = {}
        val_performance = {} if val_data is not None else None
        test_performance = {} if test_data is not None else None
        
        for metric in self._metrics:
            train_performance[metric] = []
            if val_performance is not None:
                val_performance[metric] = []
            if test_performance is not None:
                test_performance[metric] = []
            
        biases = []
        hidden_weights = []
        pca_weights = []
        output_weights = None
            
        """ 1st layer """
        
        """ solve Autoencoder ELM to initialize the weights"""
        
        print('Initialize hidden weights with ELM AE')
        W_hidden, Bias = self._ELM_AE(train_data[0])
        X_train_hidden = self._sigmoid(np.dot(train_data[0], W_hidden) + Bias)
        X_val_hidden = self._sigmoid(np.dot(val_data[0], W_hidden) + Bias) if val_data is not None else None
        X_test_hidden = self._sigmoid(np.dot(test_data[0], W_hidden) + Bias) if test_data is not None else None
        
        """ compute output layer weights based on subset of data """
        print('Solve the output layer weights')
        W_out = self._least_square(X_train_hidden, train_data[1])
        
        """ compute the prediction """
        Y_train_pred = np.dot(X_train_hidden, W_out)
        Y_val_pred = np.dot(X_val_hidden, W_out) if X_val_hidden is not None else None
        Y_test_pred = np.dot(X_test_hidden, W_out) if X_test_hidden is not None else None
                    
        for metric in self._metrics:
            train_performance[metric].append(self._evaluate(train_data[1], Y_train_pred, metric))
            if val_performance is not None:
                val_performance[metric].append(self._evaluate(val_data[1], Y_val_pred, metric))
            if test_performance is not None:
                test_performance[metric].append(self._evaluate(test_data[1], Y_test_pred, metric))
                
        self._print_performance(train_performance, val_performance, test_performance)
        
        """ apply PCA """
        
        W_pca = self._PCA(W_out)
        
        X_train_hidden = np.dot(X_train_hidden, W_pca)
        X_val_hidden = np.dot(X_val_hidden, W_pca) if X_val_hidden is not None else None
        X_test_hidden = np.dot(X_test_hidden, W_pca) if X_test_hidden is not None else None
        
        """ log data """
        biases.append(Bias)
        hidden_weights.append(W_hidden)
        pca_weights.append(W_pca)
        output_weights = W_out
        
        for layer_iter in range(self._max_layer - 1):
            print('------- layer %d --------------' % layer_iter)
            
            """ generate new hidden features based on subset of data"""
            # initialize hidden weights of new block
            print('Initialize hidden weights with ELM AE')
            W_hidden, Bias = self._ELM_AE(train_data[0])
            # calculate hidden representation by concatenating PCA of old ones, and new one
            X_train_hidden = np.concatenate((self._sigmoid(np.dot(train_data[0], W_hidden) + Bias), X_train_hidden), axis=-1)
            X_val_hidden = np.concatenate((self._sigmoid(np.dot(val_data[0], W_hidden) + Bias), X_val_hidden), axis=-1) if X_val_hidden is not None else None
            X_test_hidden = np.concatenate((self._sigmoid(np.dot(test_data[0], W_hidden) + Bias), X_test_hidden), axis=-1) if X_test_hidden is not None else None
            
            """ compute the output layer weights """
            print('Solve the output layer weights')
            W_out = self._least_square(X_train_hidden, train_data[1])
    
            """ compute the prediction """
            Y_train_pred = np.dot(X_train_hidden, W_out)
            Y_val_pred = np.dot(X_val_hidden, W_out) if X_val_hidden is not None else None
            Y_test_pred = np.dot(X_test_hidden, W_out) if X_test_hidden is not None else None
            
                
            for metric in self._metrics:
                train_performance[metric].append(self._evaluate(train_data[1], Y_train_pred, metric))
                if val_performance is not None:
                    val_performance[metric].append(self._evaluate(val_data[1], Y_val_pred, metric))
                if test_performance is not None:
                    test_performance[metric].append(self._evaluate(test_data[1], Y_test_pred, metric))
                    
            self._print_performance(train_performance, val_performance, test_performance)
                    
            """ apply PCA """
            
            W_pca = self._PCA(W_out)
            
            X_train_hidden = np.dot(X_train_hidden, W_pca)
            X_val_hidden = np.dot(X_val_hidden, W_pca) if X_val_hidden is not None else None
            X_test_hidden = np.dot(X_test_hidden, W_pca) if X_test_hidden is not None else None
            
            """ log data """
            biases.append(Bias)
            hidden_weights.append(W_hidden)
            pca_weights.append(W_pca)
            output_weights = W_out
            
            
            if self._convergence_measure.startswith('train'):
                performance = train_performance
            else:
                performance = val_performance
                
            """ check convergence """
            if self._check_convergence(performance):
                break
            
        weights = {'hidden_weights': hidden_weights,
                   'biases': biases,
                   'pca_weights':pca_weights,
                   'output_weights': output_weights}
            
        return weights, train_performance, val_performance, test_performance