# RMSProp converges faster than Adam

import h5py
import theano
import theano.tensor as T
import numpy as np

if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams 

from util import shared_zeros, shared_normal, shared_normal_vector
from attention import SelectiveAttentionModel
from collections import OrderedDict
import datetime
import sys
import json
import math
import cPickle as pickle

assert(theano.config.scan.allow_gc == True), "set scan.allow_gc to True ; otherwise you will run out of gpu memory"
assert(theano.config.allow_gc == True), "set allow_gc to True ; otherwise you will run out of gpu memory"

sys.stdout.flush()

homogeneous = __import__('homogeneous-data')
HomogeneousData = homogeneous.HomogeneousData

np.random.seed(np.random.randint(1 << 30))
rng = RandomStreams(seed=np.random.randint(1 << 30)) #np.random.randint(1 << 30)

params_names = ['W_y_hLangEnc', 'W_hLangEnc_hLangEnc', 'b_hLangEnc', 'W_yRev_hLangEncRev', 'W_hLangEncRev_hLangEncRev', 'b_hLangEncRev', 'W_lang_align', 'W_hdec_align', 'b_align', 'v_align', 'W_s_hdec', 'W_hdec_read_attent', 'b_read_attent', 'W_henc_henc', 'W_inp_henc', 'b_henc', 'W_henc_mu', 'W_henc_logsigma', 'b_mu', 'b_logsigma', 'W_hdec_hdec', 'W_z_hdec', 'b_hdec', 'W_hdec_write_attent', 'b_write_attent', 'W_hdec_c', 'b_c', 'W_hdec_mu_and_logsigma_prior', 'b_mu_and_logsigma_prior', 'h0_lang', 'h0_enc', 'h0_dec', 'c0']


def load_weights(path):
    '''Path is a an absolute path to the hdf5 file containing all the weights including their history for AdaGrad'''
    params = [0 for i in xrange(len(params_names))]

    for i in xrange(len(params)):
        params[i] = theano.shared(np.copy(h5py.File(path, 'r')[params_names[i]]))

    return params

def create_lstm_weights(dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ):
    W_hdec_read_attent = shared_normal(dimRNNDec, 5)
    b_read_attent = shared_zeros(5)

    W_henc_henc = shared_normal(dimRNNEnc, 4 * dimRNNEnc)
    W_inp_henc = shared_normal(dimReadAttent*dimReadAttent*3 + dimReadAttent*dimReadAttent*3 + dimRNNDec, 4 * dimRNNEnc)
    b_henc = shared_zeros(4 * dimRNNEnc)

    W_henc_mu = shared_normal(dimRNNEnc, dimZ)
    W_henc_logsigma = shared_normal(dimRNNEnc, dimZ)
    b_mu = shared_zeros(dimZ)
    b_logsigma = shared_zeros(dimZ)

    W_hdec_hdec = shared_normal(dimRNNDec, 4 * dimRNNDec)
    W_z_hdec = shared_normal(dimZ, 4 * dimRNNDec)
    b_hdec = shared_zeros(4 * dimRNNDec)

    W_hdec_write_attent = shared_normal(dimRNNDec, 5)
    b_write_attent = shared_zeros(5)

    W_hdec_c = shared_normal(dimRNNDec, 3*dimWriteAttent*dimWriteAttent)
    b_c = shared_zeros(3*dimWriteAttent*dimWriteAttent)

    return W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c

def create_lang_encoder_weights(dimY, dimLangRNN):
    W_y_hLangEnc = shared_normal(dimY, 4 * dimLangRNN)
    W_hLangEnc_hLangEnc = shared_normal(dimLangRNN, 4 * dimLangRNN)
    b_hLangEnc = shared_zeros(4 * dimLangRNN)

    W_yRev_hLangEncRev = shared_normal(dimY, 4 * dimLangRNN)
    W_hLangEncRev_hLangEncRev = shared_normal(dimLangRNN, 4 * dimLangRNN)
    b_hLangEncRev = shared_zeros(4 * dimLangRNN)

    return W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc, W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev

def create_align_weights(dimLangRNN, dimAlign, dimRNNEnc, dimRNNDec):
    W_lang_align = shared_normal(2*dimLangRNN, dimAlign)
    W_hdec_align = shared_normal(dimRNNDec, dimAlign)
    b_align = shared_zeros(dimAlign)
    v_align = shared_normal_vector(dimAlign)

    W_s_hdec = shared_normal(2*dimLangRNN, 4 * dimRNNDec)

    return W_lang_align, W_hdec_align, b_align, v_align, W_s_hdec

def build_lang_encoder_and_attention_vae_decoder(dimY, dimLangRNN, dimAlign, dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runStepsInt, pathToWeights=None):
    x = T.matrix() # has dimension batch_size x dimX
    y = T.matrix(dtype="int32") # matrix (sentence itself) batch_size x words_in_sentence
    y_reverse = y[::-1]
    tol = 1e-04

    if pathToWeights != None:
        W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc, W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev, W_lang_align, W_hdec_align, b_align, v_align, W_s_hdec, W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c, W_hdec_mu_and_logsigma_prior, b_mu_and_logsigma_prior, h0_lang, h0_enc, h0_dec, c0 = load_weights(pathToWeights)
    else:
        W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc, W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev = create_lang_encoder_weights(dimY, dimLangRNN)
        W_lang_align, W_hdec_align, b_align, v_align, W_s_hdec = create_align_weights(dimLangRNN, dimAlign, dimRNNEnc, dimRNNDec)
        W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c = create_lstm_weights(dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ)
        W_hdec_mu_and_logsigma_prior = shared_normal(dimRNNDec, 2*dimZ)
        b_mu_and_logsigma_prior = shared_zeros(2*dimZ)

        h0_lang = theano.shared(np.zeros((1,dimLangRNN)).astype(theano.config.floatX))
        h0_enc = theano.shared(np.zeros((1,dimRNNEnc)).astype(theano.config.floatX))
        h0_dec = theano.shared(np.zeros((1,dimRNNDec)).astype(theano.config.floatX))
        c0 = theano.shared(np.zeros((1,dimX)).astype(theano.config.floatX))

    params = [W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc, W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev, W_lang_align, W_hdec_align, b_align, v_align, W_s_hdec, W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c, W_hdec_mu_and_logsigma_prior, b_mu_and_logsigma_prior, h0_lang, h0_enc, h0_dec, c0]

    h0_lang = T.extra_ops.repeat(h0_lang, repeats=y.shape[0], axis=0)
    h0_enc = T.extra_ops.repeat(h0_enc, repeats=y.shape[0], axis=0)
    h0_dec = T.extra_ops.repeat(h0_dec, repeats=y.shape[0], axis=0)
    c0 = T.extra_ops.repeat(c0, repeats=y.shape[0], axis=0)

    cell0_lang = T.zeros((y.shape[0],dimLangRNN))
    cell0_enc = T.zeros((y.shape[0],dimRNNEnc))
    cell0_dec = T.zeros((y.shape[0],dimRNNDec))

    kl_0 = T.zeros(())
    mu_prior_0 = T.zeros((y.shape[0],dimZ))
    log_sigma_prior_0 =  T.zeros((y.shape[0],dimZ))

    run_steps = T.scalar(dtype='int32')
    eps = rng.normal(size=(runStepsInt,y.shape[0],dimZ), avg=0.0, std=1.0, dtype=theano.config.floatX)

    def recurrence_lang(y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h):
        # Transform y_t into correct representation
        # y_t is 1 x batch_size

        temp_1hot = T.zeros((y_t.shape[0], dimY))
        y_t_1hot = T.set_subtensor(temp_1hot[T.arange(y_t.shape[0]), y_t], 1) # batch_size x dimY

        lstm_t = T.dot(y_t_1hot, W_y_h) + T.dot(h_tm1, W_h_h) + b_h
        i_t = T.nnet.sigmoid(lstm_t[:, 0*dimLangRNN:1*dimLangRNN])
        f_t = T.nnet.sigmoid(lstm_t[:, 1*dimLangRNN:2*dimLangRNN])
        cell_t = f_t * cell_tm1 + i_t * T.tanh(lstm_t[:, 2*dimLangRNN:3*dimLangRNN])
        o_t = T.nnet.sigmoid(lstm_t[:, 3*dimLangRNN:4*dimLangRNN])
        h_t = o_t * T.tanh(cell_t)

        return [h_t, cell_t]

    (h_t_forward, _), updates_forward_lstm = theano.scan(lambda y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h: recurrence_lang(y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h),
                                                        sequences=y.T, outputs_info=[h0_lang, cell0_lang], non_sequences=[W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc])

    (h_t_backward, _), updates_backward_lstm = theano.scan(lambda y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h: recurrence_lang(y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h),
                                                        sequences=y_reverse.T, outputs_info=[h0_lang, cell0_lang], non_sequences=[W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev])

    
    # h_t_forward is sentence_length x batch_size x dimLangRNN (example 6 x 100 x 128)
    h_t_lang = T.concatenate([h_t_forward, h_t_backward], axis=2) # was -1
    hid_align = h_t_lang.dimshuffle([0,1,2,'x']) * W_lang_align.dimshuffle(['x','x',0,1])      
    hid_align = hid_align.sum(axis=2) # sentence_length x batch_size x dimAlign # was -2

    read_attention_model = SelectiveAttentionModel(height, width, dimReadAttent)
    write_attention_model = SelectiveAttentionModel(height, width, dimWriteAttent)

    def recurrence(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, h_tm1_enc, cell_tm1_enc, kl_tm1, mu_prior_tm1, log_sigma_prior_tm1):

        # Step 1
        x_t_hat = x - T.nnet.sigmoid(c_tm1)

        # Step 2
        #rt = T.concatenate([x, xt_hat], axis=1)
        read_attent_params = T.dot(h_tm1_dec, W_hdec_read_attent) + b_read_attent # dimension batch_size x 5
        g_y_read, g_x_read, delta_read, sigma_read, gamma_read = read_attention_model.matrix2att(read_attent_params)
        
        x_read_R = read_attention_model.read(x[:,0:height*width], g_y_read, g_x_read, delta_read, sigma_read)
        x_read_G = read_attention_model.read(x[:,height*width:2*height*width], g_y_read, g_x_read, delta_read, sigma_read)
        x_read_B = read_attention_model.read(x[:,2*height*width:3*height*width], g_y_read, g_x_read, delta_read, sigma_read)
        x_read = T.concatenate([x_read_R, x_read_G, x_read_B], axis=1)

        x_t_hat_R = read_attention_model.read(x_t_hat[:,0:height*width], g_y_read, g_x_read, delta_read, sigma_read)
        x_t_hat_G = read_attention_model.read(x_t_hat[:,height*width:2*height*width], g_y_read, g_x_read, delta_read, sigma_read)
        x_t_hat_B = read_attention_model.read(x_t_hat[:,2*height*width:3*height*width], g_y_read, g_x_read, delta_read, sigma_read)
        x_t_hat_read = T.concatenate([x_t_hat_R, x_t_hat_G, x_t_hat_B], axis=1)

        r_t = gamma_read * T.concatenate([x_read, x_t_hat_read], axis=1)
            
        # Step 3

        # Step new calculate alignments
        hdec_align = T.dot(h_tm1_dec, W_hdec_align) # batch_size x dimAlign
        all_align = T.tanh(hid_align + hdec_align.dimshuffle(['x', 0, 1]) + b_align.dimshuffle(['x','x',0])) # sentence_length x batch_size x dimAlign

        e = all_align * v_align.dimshuffle(['x','x',0])
        e = e.sum(axis=2) # sentence_length x batch_size # was -1

        # normalize
        alpha = (T.nnet.softmax(e.T)).T # sentence_length x batch_size

        # sentence representation at time T
        s_t = alpha.dimshuffle([0, 1, 'x']) * h_t_lang
        s_t = s_t.sum(axis=0) # batch_size x (dimLangRNN * 2)

        # no peepholes for lstm
        lstm_t_enc = T.dot(h_tm1_enc, W_henc_henc) + T.dot(T.concatenate([r_t, h_tm1_dec], axis=1), W_inp_henc) + b_henc
        i_t_enc = T.nnet.sigmoid(lstm_t_enc[:, 0*dimRNNEnc:1*dimRNNEnc])
        f_t_enc = T.nnet.sigmoid(lstm_t_enc[:, 1*dimRNNEnc:2*dimRNNEnc])
        cell_t_enc = f_t_enc * cell_tm1_enc + i_t_enc * T.tanh(lstm_t_enc[:, 2*dimRNNEnc:3*dimRNNEnc])
        o_t_enc = T.nnet.sigmoid(lstm_t_enc[:, 3*dimRNNEnc:4*dimRNNEnc])
        h_t_enc = o_t_enc * T.tanh(cell_t_enc)

        # Step 4
        mu_enc = T.dot(h_t_enc, W_henc_mu) + b_mu
        log_sigma_enc = 0.5 * (T.dot(h_t_enc, W_henc_logsigma) + b_logsigma)
        z_t = mu_enc + T.exp(log_sigma_enc) * eps_t

        kl_t = kl_tm1 + T.sum(-1 + ((mu_enc - mu_prior_tm1)**2  + T.exp(2*log_sigma_enc)) / (T.exp(2 * log_sigma_prior_tm1)) - 2*log_sigma_enc + 2*log_sigma_prior_tm1)
        #kl_t = kl_tm1 + T.sum(-log_sigma_enc + 0.5 * (T.exp(2*log_sigma_enc) + mu_enc**2) - 0.5)

        # Step 5
        lstm_t_dec = T.dot(h_tm1_dec, W_hdec_hdec) + T.dot(z_t, W_z_hdec) + T.dot(s_t, W_s_hdec) + b_hdec
        i_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 0*dimRNNDec:1*dimRNNDec])
        f_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 1*dimRNNDec:2*dimRNNDec])
        cell_t_dec = f_t_dec * cell_tm1_dec + i_t_dec * T.tanh(lstm_t_dec[:, 2*dimRNNDec:3*dimRNNDec])
        o_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 3*dimRNNDec:4*dimRNNDec])
        h_t_dec = o_t_dec * T.tanh(cell_t_dec)

        # mu and logsigma depend on the activations of the hidden states of the decoder # reference to the Philip Bachman's paper (Data generation as sequential decision making)
        mu_and_logsigma_prior_t = T.tanh(T.dot(h_t_dec, W_hdec_mu_and_logsigma_prior) + b_mu_and_logsigma_prior)
        mu_prior_t = mu_and_logsigma_prior_t[:, 0:dimZ]
        log_sigma_prior_t = mu_and_logsigma_prior_t[:, dimZ:2*dimZ]

        # Step 6
        write_attent_params = T.dot(h_t_dec, W_hdec_write_attent) + b_write_attent
        window_t = T.dot(h_t_dec, W_hdec_c) + b_c

        g_y_write, g_x_write, delta_write, sigma_write, gamma_write = write_attention_model.matrix2att(write_attent_params)
    
        x_t_write_R = write_attention_model.write(window_t[:,0:dimWriteAttent*dimWriteAttent], g_y_write, g_x_write, delta_write, sigma_write)
        x_t_write_G = write_attention_model.write(window_t[:,dimWriteAttent*dimWriteAttent:2*dimWriteAttent*dimWriteAttent], g_y_write, g_x_write, delta_write, sigma_write)
        x_t_write_B = write_attention_model.write(window_t[:,2*dimWriteAttent*dimWriteAttent:3*dimWriteAttent*dimWriteAttent], g_y_write, g_x_write, delta_write, sigma_write)

        x_t_write = T.concatenate([x_t_write_R, x_t_write_G, x_t_write_B], axis=1)

        c_t = c_tm1 + 1.0/gamma_write * x_t_write
        return [c_t, h_t_dec, cell_t_dec, h_t_enc, cell_t_enc, kl_t, mu_prior_t, log_sigma_prior_t, read_attent_params, write_attent_params]

    def recurrence_from_prior(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, mu_prior_tm1, log_sigma_prior_tm1):
        z_t = mu_prior_tm1 + T.exp(log_sigma_prior_tm1) * eps_t

        # Step New (calculate alignment)
        hdec_align = T.dot(h_tm1_dec, W_hdec_align) # batch_size x dimAlign
        all_align = T.tanh(hid_align + hdec_align.dimshuffle(['x', 0, 1]) + b_align.dimshuffle(['x','x',0])) # sentence_length x batch_size x dimAlign

        e = all_align * v_align.dimshuffle(['x','x',0])
        e = e.sum(axis=2) # sentence_length x batch_size # was -1

        # normalize
        alpha = (T.nnet.softmax(e.T)).T # sentence_length x batch_size

        # sentence representation at time T
        s_t = alpha.dimshuffle([0, 1, 'x']) * h_t_lang
        s_t = s_t.sum(axis=0) # batch_size x (dimLangRNN * 2)

        # Step 5
        lstm_t_dec = T.dot(h_tm1_dec, W_hdec_hdec) + T.dot(z_t, W_z_hdec) + T.dot(s_t, W_s_hdec) + b_hdec
        i_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 0*dimRNNDec:1*dimRNNDec])
        f_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 1*dimRNNDec:2*dimRNNDec])
        cell_t_dec = f_t_dec * cell_tm1_dec + i_t_dec * T.tanh(lstm_t_dec[:, 2*dimRNNDec:3*dimRNNDec])
        o_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 3*dimRNNDec:4*dimRNNDec])
        h_t_dec = o_t_dec * T.tanh(cell_t_dec)

        # mu and logsigma depend on the activations of the hidden states of the decoder # reference to the Philip Bachman's paper (Data generation as sequential decision making)
        mu_and_logsigma_prior_t = T.tanh(T.dot(h_t_dec, W_hdec_mu_and_logsigma_prior) + b_mu_and_logsigma_prior)
        mu_prior_t = mu_and_logsigma_prior_t[:, 0:dimZ]
        log_sigma_prior_t = mu_and_logsigma_prior_t[:, dimZ:2*dimZ]

        # Step 6
        write_attent_params = T.dot(h_t_dec, W_hdec_write_attent) + b_write_attent
        window_t = T.dot(h_t_dec, W_hdec_c) + b_c

        g_y_write, g_x_write, delta_write, sigma_write, gamma_write = write_attention_model.matrix2att(write_attent_params)
    
        x_t_write_R = write_attention_model.write(window_t[:,0:dimWriteAttent*dimWriteAttent], g_y_write, g_x_write, delta_write, sigma_write)
        x_t_write_G = write_attention_model.write(window_t[:,dimWriteAttent*dimWriteAttent:2*dimWriteAttent*dimWriteAttent], g_y_write, g_x_write, delta_write, sigma_write)
        x_t_write_B = write_attention_model.write(window_t[:,2*dimWriteAttent*dimWriteAttent:3*dimWriteAttent*dimWriteAttent], g_y_write, g_x_write, delta_write, sigma_write)

        x_t_write = T.concatenate([x_t_write_R, x_t_write_G, x_t_write_B], axis=1)

        c_t = c_tm1 + 1.0/gamma_write * x_t_write
        return [c_t, h_t_dec, cell_t_dec, mu_prior_t, log_sigma_prior_t, write_attent_params, alpha.T]

    all_params = params[:]
    all_params.append(x)
    all_params.append(hid_align)
    all_params.append(h_t_lang)

    (c_t, h_t_dec, cell_t_dec, h_t_enc, cell_t_enc, kl_t, mu_prior_t, log_sigma_prior_t, read_attent_params, write_attent_params), updates_train = theano.scan(lambda eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, h_tm1_enc, cell_tm1_enc, kl_tm1, mu_prior_tm1, log_sigma_prior_tm1, *_: recurrence(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, h_tm1_enc, cell_tm1_enc, kl_tm1, mu_prior_tm1, log_sigma_prior_tm1),
                                                        sequences=eps, outputs_info=[c0, h0_dec, cell0_dec, h0_enc, cell0_enc, kl_0, mu_prior_0, log_sigma_prior_0, None, None], non_sequences=all_params, n_steps=run_steps)

    all_gener_params = params[:]
    all_gener_params.append(hid_align)
    all_gener_params.append(h_t_lang)

    (c_t_gener, h_t_dec_gener, cell_t_dec_gener, mu_prior_t_gener, log_sigma_prior_t_gener, write_attent_params_gener, alphas_gener), updates_gener = theano.scan(lambda eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, mu_prior_tm1, log_sigma_prior_tm1, *_: recurrence_from_prior(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, mu_prior_tm1, log_sigma_prior_tm1),
                                                        sequences=eps, outputs_info=[c0, h0_dec, cell0_dec, mu_prior_0, log_sigma_prior_0, None, None], non_sequences=all_gener_params, n_steps=run_steps)

    c_t_final = T.nnet.sigmoid(c_t[-1])
    kl_final = 0.5 * kl_t[-1]

    logpxz = T.nnet.binary_crossentropy(c_t_final,x).sum()
    log_likelihood = kl_final + logpxz

    log_likelihood = log_likelihood.sum() / y.shape[0]
    kl_final = kl_final.sum() / y.shape[0]
    logpxz = logpxz.sum() / y.shape[0]

    return [kl_final, logpxz, log_likelihood, c_t, c_t_gener, x, y, run_steps, updates_train, updates_gener, read_attent_params, write_attent_params, write_attent_params_gener, alphas_gener, params, mu_prior_t_gener, log_sigma_prior_t_gener]

class ReccurentAttentionVAE():

    def __init__(self, dimY, dimLangRNN, dimAlign, dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runSteps, batch_size, reduceLRAfter, inputData, inputDataCaptions, valData=None, valDataCaptions=None, pathToWeights=None):
        self.dimY = dimY
        self.dimLangRNN = dimLangRNN
        self.dimAlign = dimAlign
        self.dimX = dimX
        self.dimReadAttent = dimReadAttent
        self.dimWriteAttent = dimWriteAttent
        self.dimRNNEnc = dimRNNEnc
        self.dimRNNDec = dimRNNDec
        self.dimZ = dimZ
        self.runSteps = runSteps
        self.batch_size = batch_size
        self.reduceLRAfter = reduceLRAfter
        self.pathToWeights = pathToWeights
    
        self.train_data = theano.shared(inputData)
        self.train_captions = theano.shared(inputDataCaptions)
        self.train_cap2im = pickle.load(open(train_paths["cap2im"], 'r'))
        self.train_captions_len = np.load(train_paths["caplen"])
        self.input_shape = inputData.shape
        del inputData
        del inputDataCaptions

        self.train_iter = HomogeneousData(self.train_captions_len, self.train_cap2im, batch_size=self.batch_size, minlen=7, maxlen=30)
        self.train_iter.prepare()

        if valData != None:
            self.val_data = theano.shared(valData)
            self.val_data_captions = theano.shared(valDataCaptions)
            self.val_cap2im = pickle.load(open(dev_paths["cap2im"], 'r'))
            self.val_captions_len = np.load(dev_paths["caplen"])
            self.val_shape = valData.shape
            del valData
            del valDataCaptions

            self.val_iter = HomogeneousData(self.val_captions_len, self.val_cap2im, batch_size=self.batch_size)
            self.val_iter.prepare()

        self._kl_final, self._logpxz, self._log_likelihood, self._c_ts, self._c_ts_gener, self._x, self._y, self._run_steps, self._updates_train, self._updates_gener, self._read_attent_params, self._write_attent_params, self._write_attent_params_gener, self._alphas_gener, self._params, self._mu_prior_t_gener, self._log_sigma_prior_t_gener = build_lang_encoder_and_attention_vae_decoder(self.dimY, self.dimLangRNN, self.dimAlign, self.dimX, self.dimReadAttent, self.dimWriteAttent, self.dimRNNEnc, self.dimRNNDec, self.dimZ, self.runSteps, self.pathToWeights)

    def _build_sample_from_prior_function(self):
        print 'building sample from prior function'
        t1 = datetime.datetime.now()
        self._sample_from_prior = theano.function(inputs=[self._run_steps, self._y], 
                                                    outputs=[self._c_ts_gener, self._write_attent_params_gener, self._alphas_gener, self._mu_prior_t_gener, self._log_sigma_prior_t_gener],
                                                    updates=self._updates_gener)
        t2 = datetime.datetime.now()
        print(t2-t1)

    def _build_sample_from_input_function(self):
        print 'building sample from input function'
        t1 = datetime.datetime.now()
        self._sample_from_input = theano.function(inputs=[self._x, self._run_steps, self._y], 
                                                    outputs=[self._c_ts, self._write_attent_params, self._log_likelihood],
                                                    updates=self._updates_train)
        t2 = datetime.datetime.now()
        print(t2-t1)

    def _build_validate_function(self):
        print 'building validate function'
        t1 = datetime.datetime.now()
        data = self.val_data
        captions = self.val_data_captions

        self._index_im_val = T.vector(dtype='int32') # index to the minibatch
        self._index_cap_val = T.vector(dtype='int32')
        self._cap_len_val = T.scalar(dtype='int32')
        self._validate_function = theano.function(inputs=[self._index_im_val, self._index_cap_val, self._cap_len_val, self._run_steps], 
                                                outputs=[self._kl_final, self._logpxz, self._log_likelihood],
                                                updates=self._updates_train,
                                                givens={
                                                    self._x: data[self._index_im_val],
                                                    self._y: captions[self._index_cap_val,0:self._cap_len_val]
                                                })
        t2 = datetime.datetime.now()
        print (t2-t1)

    def _build_train_function(self):
        print 'building gradient function'
        t1 = datetime.datetime.now()
        gradients = T.grad(self._log_likelihood, self._params)
        t2 = datetime.datetime.now()
        print(t2-t1)

        self._index_cap = T.vector(dtype='int32') # index to the minibatch
        self._index_im = T.vector(dtype='int32')
        self._cap_len = T.scalar(dtype='int32')
        self._lr = T.scalar('lr', dtype=theano.config.floatX)

        # Currently use AdaGrad & threshold gradients
        his = []
        for param in self._params:
            param_value_zeros = param.get_value() * 0
            his.append(theano.shared(param_value_zeros))

        threshold = 10.0
        decay_rate = 0.9

        self._updates_train_and_params = OrderedDict()
        self._updates_train_and_params.update(self._updates_train)
        
        for param, param_his, grad in zip(self._params, his, gradients):
            l2_norm_grad = T.sqrt(T.sqr(grad).sum())
            multiplier = T.switch(l2_norm_grad < threshold, 1, threshold / l2_norm_grad)
            grad = multiplier * grad

            param_his_new = decay_rate * param_his + (1 - decay_rate) * grad**2

            self._updates_train_and_params[param_his] = param_his_new
            self._updates_train_and_params[param] = param - (self._lr / T.sqrt(param_his_new + 1e-6)) * grad

        print 'building train function'
        t1 = datetime.datetime.now()
        self._train_function = theano.function(inputs=[self._index_im, self._index_cap, self._cap_len, self._lr, self._run_steps], 
                                                outputs=[self._kl_final, self._logpxz, self._log_likelihood, self._c_ts, self._read_attent_params, self._write_attent_params],
                                                updates=self._updates_train_and_params,
                                                givens={
                                                    self._x: self.train_data[self._index_im],
                                                    self._y: self.train_captions[self._index_cap, 0:self._cap_len]
                                                })
        t2 = datetime.datetime.now()
        print (t2-t1)

    def sample_from_prior(self, run_steps, y):
        self._build_sample_from_prior_function()
        sys.stdout.flush()
        return self._sample_from_prior(run_steps, y)

    def sample_from_input(self, x, run_steps, y):
        self._build_sample_from_input_function()
        sys.stdout.flush()
        return self._sample_from_input(x, run_steps, y)

    def sample_from_input_no_theano_build(self, x, run_steps, y):
        return self._sample_from_input(x, run_steps, y)
        
    def sample_from_prior_no_theano_build(self, run_steps, y):
        return self._sample_from_prior(run_steps, y)

    def validate(self):

        self._build_validate_function()
        sys.stdout.flush()

        all_outputs = np.array([0.0,0.0,0.0])
        self.val_iter.reset()
        while True:
            index_val_cap, index_val_im, val_cap_len = self.val_iter.next()
            if type(index_val_cap) == int:
                break
            [kl, logpxz, log_likelihood] = self._validate_function(index_val_im, index_val_cap, val_cap_len, self.runSteps)
            all_outputs[0] = all_outputs[0] + kl * index_val_im.shape[0]
            all_outputs[1] = all_outputs[1] + logpxz * index_val_im.shape[0]
            all_outputs[2] = all_outputs[2] + log_likelihood * index_val_im.shape[0]

        all_outputs = all_outputs / (self.val_shape[0]*5)
        return all_outputs

    def save_weights(self, path, c_ts, read_attent_params, write_attent_params):
        weights_f = h5py.File(path, 'w')
        
        for i in xrange(len(self._params)):
            dset = weights_f.create_dataset(params_names[i], self._params[i].shape.eval(), dtype='f')
            dset[:] = np.copy(self._params[i].eval())

        weights_f.close()

    def train(self, lr, epochs, save=False, validateAfter=0):
        self._build_train_function()
        sys.stdout.flush()

        if save == True:
            curr_time = datetime.datetime.now()
            weights_f_name = ("./attention-vae-%s-%s-%s-%s-%s-%s.h5" % (curr_time.year, curr_time.month, curr_time.day, curr_time.hour, curr_time.minute, curr_time.second))
            print weights_f_name

        all_outputs = np.array([0.0,0.0,0.0])
        iter_outputs = np.array([0.0,0.0,0.0])
        curr_iter = 0
        print_after = 100
        seen_examples = 0
        total_seen_examples = 0
        prev_outputs = np.array([float("inf"),float("inf"),float("inf")])
        prev_val_results = np.array([float("inf"),float("inf"),float("inf")])

        for epoch in xrange(0, epochs):
            a = datetime.datetime.now()
            
            self.train_iter.reset()
            while True:
                index_cap, index_im, cap_len = self.train_iter.next()
                if type(index_cap) == int:
                    break
                [kl, logpxz, log_likelihood, c_ts, read_attent_params, write_attent_params] = self._train_function(index_im, index_cap, cap_len, lr, self.runSteps)
                kl_total = kl * index_im.shape[0]
                logpxz_total = logpxz * index_im.shape[0]
                log_likelihood_total = log_likelihood * index_im.shape[0]
                all_outputs[0] = all_outputs[0] + kl_total
                all_outputs[1] = all_outputs[1] + logpxz_total
                all_outputs[2] = all_outputs[2] + log_likelihood_total
                iter_outputs[0] = iter_outputs[0] + kl_total
                iter_outputs[1] = iter_outputs[1] + logpxz_total
                iter_outputs[2] = iter_outputs[2] + log_likelihood_total
                seen_examples = seen_examples + index_im.shape[0]
                total_seen_examples = total_seen_examples + index_im.shape[0]

                if curr_iter % print_after == 0 and curr_iter != 0:
                    print 'Iteration %d ; Processed %d entries' % (curr_iter, total_seen_examples)
                    iter_outputs = iter_outputs / seen_examples
                    print float(iter_outputs[0]), float(iter_outputs[1]), float(iter_outputs[2])
                    print '\n'
                    iter_outputs = np.array([0.0,0.0,0.0])
                    seen_examples = 0
                    sys.stdout.flush()

                if curr_iter % (print_after*10) == 0 and curr_iter != 0:
                    self.save_weights(weights_f_name, c_ts, read_attent_params, write_attent_params)
                    print 'Done Saving Weights'
                    print '\n'
                    sys.stdout.flush()
                
                curr_iter = curr_iter + 1
            b = datetime.datetime.now()
            print("Epoch %d took %s" % (epoch, (b-a)))

            if save == True:
                self.save_weights(weights_f_name, c_ts, read_attent_params, write_attent_params)
                print 'Done Saving Weights'

            all_outputs = all_outputs / (self.input_shape[0] * 5) # 5 captions per image
            print 'Train Results'
            print float(all_outputs[0]), float(all_outputs[1]), float(all_outputs[2])

            if validateAfter != 0:
                if epoch % validateAfter == 0:
                    print 'Validation Results'
                    val_results = self.validate()
                    print float(val_results[0]), float(val_results[1]), float(val_results[2])
                    print '\n'

            if float(val_results[-1]) > float(prev_val_results[-1]):
                print("Learning Rate Decreased")
                lr = lr * 0.1
            elif self.reduceLRAfter != 0:
                if epoch == self.reduceLRAfter:
                    print ("Learning Rate Manually Decreased")
                    lr = lr * 0.1
            else:
                prev_val_results = np.copy(val_results)

            print '\n'
            all_outputs = np.array([0.0,0.0,0.0])
            sys.stdout.flush()

if __name__ == '__main__':

    model_name = sys.argv[1]
    with open(model_name) as model_file:
        model = json.load(model_file)

    dimY = int(model["model"][0]["dimY"])
    dimLangRNN = int(model["model"][0]["dimLangRNN"])
    dimAlign = int(model["model"][0]["dimAlign"])
    
    dimX = int(model["model"][0]["dimX"])
    dimReadAttent = int(model["model"][0]["dimReadAttent"])
    dimWriteAttent = int(model["model"][0]["dimWriteAttent"])
    dimRNNEnc = int(model["model"][0]["dimRNNEnc"])
    dimRNNDec = int(model["model"][0]["dimRNNDec"])
    dimZ = int(model["model"][0]["dimZ"])
    runSteps = int(model["model"][0]["runSteps"])

    global height
    global width

    height = int(math.sqrt(dimX/3))
    width = int(math.sqrt(dimX/3))

    validateAfter = int(model["validateAfter"])
    save = bool(model["save"])
    lr = float(model["lr"])
    epochs = int(model["epochs"])
    batch_size = int(model["batch_size"])
    reduceLRAfter = int(model["reduceLRAfter"])
    pathToWeights = str(model["pathToWeights"])
    if pathToWeights == "None":
        pathToWeights = None

    global train_paths
    global dev_paths

    train_paths = model["data"]["train"]
    dev_paths = model["data"]["dev"]
    
    data = np.float32(np.load(train_paths["images"]))
    data_captions = np.int32(np.load(train_paths["captions"]))
    
    val_data = np.float32(np.load(dev_paths["images"]))
    val_data_captions = np.int32(np.load(dev_paths["captions"]))

    rvae = ReccurentAttentionVAE(dimY, dimLangRNN, dimAlign, dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runSteps, batch_size, reduceLRAfter, data, data_captions, valData=val_data, valDataCaptions=val_data_captions, pathToWeights=pathToWeights)
    rvae.train(lr, epochs, save=save, validateAfter=validateAfter)
