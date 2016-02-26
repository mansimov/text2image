import h5py
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np
import copy

if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from collections import OrderedDict
import datetime
import pylab
import scipy

"""
architecture of gan trained to sharpen blurry images
K = 32
factors = [1,1,1]
kernel_sizes = [7,7,5]
num_filts = [128,128,3]
"""

def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX))

def load_weights(params, path, num_conv):
    print 'Loading gan weights from ' + path
    with h5py.File(path, 'r') as hdf5:
        params['skipthought2image'] = theano.shared(np.copy(hdf5['skipthought2image']))
        params['skipthought2image-bias'] = theano.shared(np.copy(hdf5['skipthought2image-bias']))

        for i in xrange(num_conv):
            params['W_conv{}'.format(i)] = theano.shared(np.copy(hdf5['W_conv{}'.format(i)]))
            params['b_conv{}'.format(i)] = theano.shared(np.copy(hdf5['b_conv{}'.format(i)]))

            # Flip w,h axes
            params['W_conv{}'.format(i)] = params['W_conv{}'.format(i)][:,:,::-1,::-1]

            w = np.abs(np.copy(hdf5['W_conv{}'.format(i)]))
            print 'W_conv{}'.format(i), np.min(w), np.mean(w), np.max(w)
            b = np.abs(np.copy(hdf5['b_conv{}'.format(i)]))
            print 'b_conv{}'.format(i), np.min(b), np.mean(b), np.max(b)

    return params

def rectifier(x):
    return x * (x > 0)

def build_gan(K, batch_size, factors, kernel_sizes, num_filts, pathToGANWeights):

    assert len(factors) == len(kernel_sizes)
    assert len(factors) == len(num_filts)

    x_blurry = T.matrix() # batch_size x (3*K*K)
    skipthought = T.matrix() # batch_size x dimSkip

    #####################################################################
    # Define the network parameters

    params = OrderedDict()
    params = load_weights(params, pathToGANWeights, len(factors))

    #####################################################################
    # Define the network input

    noise_input = shared_zeros(batch_size, 1, K, K) # deterministic (works best)

    skipthought_img = T.dot(skipthought, params['skipthought2image']) + params['skipthought2image-bias']
    skipthought_img = rectifier(skipthought_img)
    skipthought_img = skipthought_img.reshape(shape=(batch_size, 1, K, K))

    x_blurry_img = x_blurry.reshape(shape=(batch_size, 3, K, K))
   
    x_combined = T.concatenate((noise_input, skipthought_img, x_blurry_img), axis=1)

    #####################################################################
    # Define the convolutions
    
    def pad(layer, kernel_size):
        pad_leftright = T.alloc(0., layer.shape[0], layer.shape[1], layer.shape[2], (kernel_size-1)/2)
        layer = T.concatenate((pad_leftright, layer, pad_leftright), axis=3)
        pad_topbottom = T.alloc(0., layer.shape[0], layer.shape[1], (kernel_size-1)/2, layer.shape[3])
        layer = T.concatenate((pad_topbottom, layer, pad_topbottom), axis=2)
        return layer
    
    last_layer = x_combined
    for i in xrange(len(factors)):
        h = last_layer.shape[2]
        w = last_layer.shape[3]

        # zero-pad the previous layer so the convolution output is the same size as the input
        last_layer = pad(last_layer, kernel_sizes[i])
        last_layer = conv.conv2d(last_layer, params['W_conv{}'.format(i)],
                                subsample=(1,1), 
                                border_mode='valid')
        last_layer = last_layer + params['b_conv{}'.format(i)].dimshuffle('x',0,'x','x')

        assert factors[i] == 1 or factors[i] == 2 # only take care of 1x or 2x factor for now
        if factors[i] == 2:
            # Reshape and concatenate to create an image factor^2 bigger (nearest-neighbor upsampling)
            last_layer = last_layer.reshape(shape=(last_layer.shape[0], num_filts[i], h, 1, w, 1))
            last_layer = T.concatenate((last_layer, last_layer), axis=5)
            last_layer = T.concatenate((last_layer, last_layer), axis=3)
            last_layer = last_layer.reshape(shape=(last_layer.shape[0], num_filts[i], 2*h, 2*w))

        # Last layer is rectified
        if i < len(factors)-1:
            last_layer = rectifier(last_layer)
    
    return x_blurry, skipthought, last_layer, params

def gan(K, batch_size, factors, kernel_sizes, num_filts, pathToGANWeights):
    x_blurry, skipthought, edges, params = build_gan(K, batch_size, factors, kernel_sizes, num_filts, pathToGANWeights)

    print 'building gan sharpening function'
    t1 = datetime.datetime.now()
    generate_edges_function = theano.function(inputs=[x_blurry, skipthought], outputs=edges)
    t2 = datetime.datetime.now()
    print(t2-t1)

    return generate_edges_function
