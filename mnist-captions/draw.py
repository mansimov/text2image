"""Theano implementation of the original DRAW model. Kept as a reference"""

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

floatX = theano.config.floatX

np.random.seed(np.random.randint(1 << 30))
rng = RandomStreams(seed=np.random.randint(1 << 30)) #np.random.randint(1 << 30)

# Hardcoded for batch size = 100
batch_size = 100
params_names = ['W_hdec_read_attent', 'b_read_attent', 'W_henc_henc', 'W_inp_henc', 'b_henc', 'W_henc_mu', 'W_henc_logsigma', 'b_mu', 'b_logsigma', 'W_hdec_hdec', 'W_z_hdec', 'b_hdec', 'W_hdec_write_attent', 'b_write_attent', 'W_hdec_c', 'b_c']

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
    W_inp_henc = shared_normal(dimReadAttent*dimReadAttent + dimReadAttent*dimReadAttent + dimRNNDec, 4 * dimRNNEnc)
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

    W_hdec_c = shared_normal(dimRNNDec, dimWriteAttent*dimWriteAttent)
    b_c = shared_zeros(dimWriteAttent*dimWriteAttent)

    return W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c


def build_lstm_attention_vae(dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runStepsInt, pathToWeights=None):
    x = T.matrix() # has dimension batch_size x dimX
    tol = 1e-04

    if pathToWeights != None:
        W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c = load_weights(pathToWeights)
    else:
        W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c = create_lstm_weights(dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ)

    h0_enc = T.zeros((batch_size,dimRNNEnc))
    h0_dec = T.zeros((batch_size,dimRNNDec))
    cell0_enc = T.zeros((batch_size,dimRNNEnc))
    cell0_dec = T.zeros((batch_size,dimRNNDec))

    c0 = T.zeros((batch_size, dimX))
    kl_0 = T.zeros(())

    run_steps = T.scalar(dtype='int32')
    eps = rng.normal(size=(runStepsInt,batch_size,dimZ), avg=0.0, std=1.0, dtype=floatX)

    params = [W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c]

    read_attention_model = SelectiveAttentionModel(28, 28, dimReadAttent)
    write_attention_model = SelectiveAttentionModel(28, 28, dimWriteAttent)

    def recurrence(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, h_tm1_enc, cell_tm1_enc, kl_tm1):

        # Step 1
        x_t_hat = x - T.nnet.sigmoid(c_tm1)

        # Step 2
        #rt = T.concatenate([x, xt_hat], axis=1)
        read_attent_params = T.dot(h_tm1_dec, W_hdec_read_attent) + b_read_attent # dimension batch_size x 5
        g_y_read, g_x_read, delta_read, sigma_read, gamma_read = read_attention_model.matrix2att(read_attent_params)
        r_t = gamma_read * T.concatenate([read_attention_model.read(x, g_y_read, g_x_read, delta_read, sigma_read), read_attention_model.read(x_t_hat, g_y_read, g_x_read, delta_read, sigma_read)], axis=1)
            
        # Step 3
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

        kl_t = kl_tm1 + T.sum(-1 + mu_enc**2 + T.exp(2*log_sigma_enc) - 2*log_sigma_enc)
        # Step 5
        lstm_t_dec = T.dot(h_tm1_dec, W_hdec_hdec) + T.dot(z_t, W_z_hdec) + b_hdec
        i_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 0*dimRNNDec:1*dimRNNDec])
        f_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 1*dimRNNDec:2*dimRNNDec])
        cell_t_dec = f_t_dec * cell_tm1_dec + i_t_dec * T.tanh(lstm_t_dec[:, 2*dimRNNDec:3*dimRNNDec])
        o_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 3*dimRNNDec:4*dimRNNDec])
        h_t_dec = o_t_dec * T.tanh(cell_t_dec)

        # Step 6
        write_attent_params = T.dot(h_t_dec, W_hdec_write_attent) + b_write_attent
        window_t = T.dot(h_t_dec, W_hdec_c) + b_c
        g_y_write, g_x_write, delta_write, sigma_write, gamma_write = write_attention_model.matrix2att(write_attent_params)
        c_t = c_tm1 + 1.0/gamma_write * write_attention_model.write(window_t, g_y_write, g_x_write, delta_write, sigma_write)
        return [c_t.astype(floatX), h_t_dec.astype(floatX), cell_t_dec.astype(floatX), h_t_enc.astype(floatX), cell_t_enc.astype(floatX), kl_t.astype(floatX), read_attent_params, write_attent_params]

    def recurrence_from_prior(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec):
        z_t = eps_t

        # Step 5
        lstm_t_dec = T.dot(h_tm1_dec, W_hdec_hdec) + T.dot(z_t, W_z_hdec) + b_hdec
        i_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 0*dimRNNDec:1*dimRNNDec])
        f_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 1*dimRNNDec:2*dimRNNDec])
        cell_t_dec = f_t_dec * cell_tm1_dec + i_t_dec * T.tanh(lstm_t_dec[:, 2*dimRNNDec:3*dimRNNDec])
        o_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 3*dimRNNDec:4*dimRNNDec])
        h_t_dec = o_t_dec * T.tanh(cell_t_dec)

        # Step 6
        write_attent_params = T.dot(h_t_dec, W_hdec_write_attent) + b_write_attent
        window_t = T.dot(h_t_dec, W_hdec_c) + b_c
        g_y_write, g_x_write, delta_write, sigma_write, gamma_write = write_attention_model.matrix2att(write_attent_params)
        c_t = c_tm1 + 1.0/gamma_write * write_attention_model.write(window_t, g_y_write, g_x_write, delta_write, sigma_write)
        return [c_t.astype(floatX), h_t_dec.astype(floatX), cell_t_dec.astype(floatX), write_attent_params]


    all_params = params[:]
    all_params.append(x)

    (c_t, h_t_dec, cell_t_dec, h_t_enc, cell_t_enc, kl_t, read_attent_params, write_attent_params), updates_train = theano.scan(lambda eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, h_tm1_enc, cell_tm1_enc, kl_tm1, *_: recurrence(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, h_tm1_enc, cell_tm1_enc, kl_tm1),
                                                        sequences=eps, outputs_info=[c0, h0_dec, cell0_dec, h0_enc, cell0_enc, kl_0, None, None], non_sequences=all_params, n_steps=run_steps)

    all_gener_params = params[:]

    (c_t_gener, h_t_dec_gener, cell_t_dec_gener, write_attent_params_gener), updates_gener = theano.scan(lambda eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, *_: recurrence_from_prior(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec),
                                                        sequences=eps, outputs_info=[c0, h0_dec, cell0_dec, None], non_sequences=all_gener_params, n_steps=run_steps)


    c_t_final = T.nnet.sigmoid(c_t[-1])
    kl_final = 0.5 * kl_t[-1]
    logpxz = T.nnet.binary_crossentropy(c_t_final,x).sum()
    log_likelihood = kl_final + logpxz
    
    log_likelihood = log_likelihood.sum() / x.shape[0]
    kl_final = kl_final.sum() / x.shape[0]
    logpxz = logpxz.sum() / x.shape[0]

    return [kl_final, logpxz, log_likelihood, c_t, c_t_gener, x, run_steps, updates_train, updates_gener, read_attent_params, write_attent_params, write_attent_params_gener, params]

class ReccurentAttentionVAE():

    def __init__(self, dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runSteps, inputData, valData=None, testData=None, pathToWeights=None):
        self.dimX = dimX
        self.dimReadAttent = dimReadAttent
        self.dimWriteAttent = dimWriteAttent
        self.dimRNNEnc = dimRNNEnc
        self.dimRNNDec = dimRNNDec
        self.dimZ = dimZ
        self.runSteps = runSteps
        self.pathToWeights = pathToWeights

        self.n_batches = inputData.shape[0] / batch_size
        self.train_data = theano.shared(inputData)
        del inputData

        if valData != None:
            self.n_val_batches = valData.shape[0] / batch_size
            self.val_data = theano.shared(valData)
            del valData

        if testData != None:
            self.n_test_batches = testData.shape[0] / batch_size
            self.test_data = theano.shared(testData)
            del testData

        self._kl_final, self._logpxz, self._log_likelihood, self._c_ts, self._c_ts_gener, self._x, self._run_steps, self._updates_train, self._updates_gener, self._read_attent_params, self._write_attent_params, self._write_attent_params_gener, self._params = build_lstm_attention_vae(self.dimX, self.dimReadAttent, self.dimWriteAttent, self.dimRNNEnc, self.dimRNNDec, self.dimZ, self.runSteps, self.pathToWeights)

    def _build_sample_from_prior_function(self):
        print 'building sample from prior function'
        t1 = datetime.datetime.now()
        self._sample_from_prior = theano.function(inputs=[self._run_steps], 
                                                    outputs=[self._c_ts_gener, self._write_attent_params_gener],
                                                    updates=self._updates_gener)
        t2 = datetime.datetime.now()
        print(t2-t1)

    def _build_validate_function(self, isVal=True):
        print 'building validate function'
        t1 = datetime.datetime.now()
        if isVal:
            data = self.val_data
        else:
            data = self.test_data

        self._index_val = T.scalar(dtype='int32') # index to the minibatch
        self._validate_function = theano.function(inputs=[self._index_val, self._run_steps], 
                                                outputs=[self._kl_final, self._logpxz, self._log_likelihood],
                                                updates=self._updates_train,
                                                givens={
                                                    self._x: data[(self._index_val * batch_size):((self._index_val + 1) * batch_size)].astype(floatX)
                                                })
        t2 = datetime.datetime.now()
        print (t2-t1)

    def _build_train_function(self):
        print 'building gradient function'
        t1 = datetime.datetime.now()
        gradients = T.grad(self._log_likelihood, self._params)
        t2 = datetime.datetime.now()
        print(t2-t1)

        self._index = T.scalar(dtype='int32') # index to the minibatch
        self._lr = T.scalar('lr', dtype=floatX)

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
        self._train_function = theano.function(inputs=[self._index, self._lr, self._run_steps], 
                                                outputs=[self._kl_final, self._logpxz, self._log_likelihood, self._c_ts, self._read_attent_params, self._write_attent_params],
                                                updates=self._updates_train_and_params,
                                                givens={
                                                    self._x: self.train_data[(self._index * batch_size):((self._index + 1) * batch_size)].astype(floatX)
                                                })
        t2 = datetime.datetime.now()
        print (t2-t1)

    def sample_from_prior(self, run_steps):
        self._build_sample_from_prior_function()
        sys.stdout.flush()
        return self._sample_from_prior(run_steps)

    def validate(self, isValValue=True):

        self._build_validate_function(isVal=isValValue)
        sys.stdout.flush()

        n_local_batches = self.n_val_batches
        if isValValue == False:
            n_local_batches = self.n_test_batches

        all_outputs = np.array([0.0,0.0,0.0])
        for i_batch in xrange(n_local_batches):
            [kl, logpxz, log_likelihood] = self._validate_function(i_batch, self.runSteps)
            all_outputs[0] = all_outputs[0] + kl
            all_outputs[1] = all_outputs[1] + logpxz
            all_outputs[2] = all_outputs[2] + log_likelihood

        all_outputs = all_outputs / n_local_batches
        return all_outputs

    def save_weights(self, path, c_ts, read_attent_params, write_attent_params):
        weights_f = h5py.File(path, 'w')
        
        for i in xrange(len(self._params)):
            dset = weights_f.create_dataset(params_names[i], self._params[i].shape.eval(), dtype='f')
            dset[:] = np.copy(self._params[i].eval())

        weights_f.close()

    def train(self, lr, epochs, save=False, savedir=None, validateAfter=0):
        self._build_train_function()
        sys.stdout.flush()

        if save == True:
            curr_time = datetime.datetime.now()
            if savedir == None:
                savedir = "."
            weights_f_name = ("%s/attention-vae-%s-%s-%s-%s-%s-%s.h5" % (savedir, curr_time.year, curr_time.month, curr_time.day, curr_time.hour, curr_time.minute, curr_time.second))
            print weights_f_name

        all_outputs = np.array([0.0,0.0,0.0])
        prev_outputs = np.array([float("inf"),float("inf"),float("inf")])

        for epoch in xrange(0, epochs):
            a = datetime.datetime.now()
            for i_batch in xrange(self.n_batches):
                [kl, logpxz, log_likelihood, c_ts, read_attent_params, write_attent_params] = self._train_function(i_batch, lr, self.runSteps)
                all_outputs[0] = all_outputs[0] + kl
                all_outputs[1] = all_outputs[1] + logpxz
                all_outputs[2] = all_outputs[2] + log_likelihood
                #print kl, logpxz, log_likelihood
            b = datetime.datetime.now()
            print("Epoch %d took %s" % (epoch, (b-a)))

            if save == True:
                self.save_weights(weights_f_name, c_ts, read_attent_params, write_attent_params)
                print 'Done Saving Weights'

            all_outputs = all_outputs / self.n_batches
            print 'Train Results'
            print float(all_outputs[0]), float(all_outputs[1]), float(all_outputs[2])

            if float(all_outputs[-1]) > float(prev_outputs[-1]):
                print("Learning Rate Decreased")
                lr = lr * 0.1
            else:
                prev_outputs = np.copy(all_outputs)

            if validateAfter != 0:
                if epoch % validateAfter == 0 and epoch != 0:
                    print 'Validation Results'
                    val_results = self.validate()
                    print float(val_results[0]), float(val_results[1]), float(val_results[2])
                    print '\n'

            all_outputs = np.array([0.0,0.0,0.0])
            sys.stdout.flush()

if __name__ == '__main__':

    model_name = sys.argv[1]
    with open(model_name) as model_file:
        model = json.load(model_file)

    dimX = int(model["model"][0]["dimX"])
    dimReadAttent = int(model["model"][0]["dimReadAttent"])
    dimWriteAttent = int(model["model"][0]["dimWriteAttent"])
    dimRNNEnc = int(model["model"][0]["dimRNNEnc"])
    dimRNNDec = int(model["model"][0]["dimRNNDec"])
    dimZ = int(model["model"][0]["dimZ"])
    runSteps = int(model["model"][0]["runSteps"])

    validateAfter = int(model["validateAfter"])
    save = bool(model["save"])
    lr = float(model["lr"])
    epochs = int(model["epochs"])

    if "data" in model:
        train_key = model["data"]["train"]["key"]
        train_file = model["data"]["train"]["file"]
        val_key = model["data"]["validation"]["key"]
        val_file = model["data"]["validation"]["file"]
    else:
        train_file = "/ais/gobi3/u/nitish/mnist/mnist.h5"
        train_key = "train"
        val_file = "/ais/gobi3/u/nitish/mnist/mnist.h5"
        val_key = "validation"

    train_data = np.copy(h5py.File(train_file, 'r')[train_key])
    val_data = np.copy(h5py.File(val_file, 'r')[val_key])

    savedir = None
    if "savedir" in model:
        savedir = model["savedir"]

    pathToWeights = None
    if "pathToWeights" in model:
        pathToWeights = model["pathToWeights"]

    rvae = ReccurentAttentionVAE(dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runSteps, train_data, valData=val_data, pathToWeights=pathToWeights)
    rvae.train(lr, epochs, save=save, savedir=savedir, validateAfter=validateAfter)
