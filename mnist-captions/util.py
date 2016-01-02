import numpy as np
import theano

def shared_normal(num_rows, num_cols, scale=0.01):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(np.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))

def shared_normal_vector(num_rows, scale=0.01):
    '''Initialize a vector shared variable with normally distributed
    elements.'''
    return theano.shared(np.random.normal(
        scale=scale, size=(num_rows)).astype(theano.config.floatX))

def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX))

def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s