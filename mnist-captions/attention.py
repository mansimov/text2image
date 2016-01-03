import numpy as np
import theano
import theano.tensor as T

"""Jorg Bornschein's implementation of attention was used as a reference"""
class SelectiveAttentionModel(object):
    """Haven't tested the cases when A != B"""
    def __init__(self, A, B, N):
        self.A = A
        self.B = B
        self.N = N

    def internal_dot_product(self, matrix1, matrix2):
        '''Matrix1 has dimension dim1 x dim2 x dim3 ; Matrix2 has dimension dim1 x dim3 x dim4
        Result is dim1 x dim2 x dim4'''

        matrix3 = matrix1.dimshuffle([0,1,2,'x']) * matrix2.dimshuffle([0,'x',1,2])      
        return matrix3.sum(axis=2)

    def matrix2att(self, matrix):
        '''Input is vector of size (batch_size,5) in theano terms'''
        g_hat_x = matrix[:,0]
        g_hat_y = matrix[:,1]
        log_delta = matrix[:,2]
        log_sigma_sqr = matrix[:,3]
        log_gamma = matrix[:,4]

        g_x = (self.A + 1.0) / 2.0 * (g_hat_x + 1.0)
        g_y = (self.B + 1.0) / 2.0 * (g_hat_y + 1.0)

        delta = (max(self.A,self.B) - 1.0) / (self.N - 1) * T.exp(log_delta)
        gamma = T.exp(log_gamma).dimshuffle(0, 'x')
        sigma = T.exp(log_sigma_sqr/2.0)

        return g_y, g_x, delta, sigma, gamma

    def matrix2att_cpu(self, matrix):
        '''Input is vector of size (batch_size,5) in numpy terms'''
        g_hat_x = matrix[:,0]
        g_hat_y = matrix[:,1]
        log_delta = matrix[:,2]
        log_sigma_sqr = matrix[:,3]
        log_gamma = matrix[:,4]

        g_x = (self.A + 1.0) / 2.0 * (g_hat_x + 1.0)
        g_y = (self.B + 1.0) / 2.0 * (g_hat_y + 1.0)

        delta = (max(self.A,self.B) - 1.0) / (self.N - 1) * np.exp(log_delta)
        gamma = np.exp(log_gamma)
        sigma = np.exp(log_sigma_sqr/2.0)

        return g_y, g_x, delta, sigma, gamma

    def get_filterbank_matrices(self, g_y, g_x, delta, sigma):

        tol = 1e-04
        mu_x = g_x.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(self.N)-self.N/2-0.5) # dimension (batch_size, N)
        mu_y = g_y.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(self.N)-self.N/2-0.5)

        a = T.arange(self.A)
        b = T.arange(self.B)

        f_x = T.exp( -(a-mu_x.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 ) # dimension (batch_size, N, A)
        f_y = T.exp( -(b-mu_y.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )

        f_x = f_x / (f_x.sum(axis=2).dimshuffle(0, 1, 'x') + tol) # dimension (batch_size, N, A)
        f_y = f_y / (f_y.sum(axis=2).dimshuffle(0, 1, 'x') + tol)
        return f_y, f_x

    def get_mean_filters_cpu(self, g_y, g_x, delta, sigma):
        mu_x = g_x + delta * (np.arange(self.N) - self.N/2 - 0.5)
        mu_y = g_y + delta * (np.arange(self.N) - self.N/2 - 0.5)

        return mu_y, mu_x

    def read(self, images, g_y, g_x, delta, sigma):
        f_y, f_x = self.get_filterbank_matrices(g_y, g_x, delta, sigma)
        batch_size = images.shape[0]
        reshaped_images = images.reshape( (batch_size, self.A, self.B) ) # dimension (batch_size, A, B)

        w = self.internal_dot_product(self.internal_dot_product(f_y, reshaped_images), f_x.transpose([0,2,1]))
        return w.reshape((batch_size, self.N*self.N))

    def write(self, windows, g_y, g_x, delta, sigma):
        f_y, f_x = self.get_filterbank_matrices(g_y, g_x, delta, sigma)
        batch_size = windows.shape[0]
        reshaped_windows = windows.reshape( (batch_size, self.N, self.N) ) # dimension (batch_size, N, N)

        im = self.internal_dot_product(self.internal_dot_product(f_y.transpose([0,2,1]), reshaped_windows), f_x)
        return im.reshape((batch_size, self.A * self.B))