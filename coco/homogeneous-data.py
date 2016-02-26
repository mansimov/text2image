"The class is similar to to https://github.com/kelvinxu/arctic-captions/blob/master/homogeneous_data.py"

import numpy as np
import copy
from random import randint

class HomogeneousData():
    # prepare -> reset -> next
    def __init__(self, captions_len, cap2im, batch_size=64, minlen=None, maxlen=None):
        self.batch_size = batch_size
        self.captions_len = captions_len
        self.cap2im = cap2im
        self.minlen = minlen
        self.maxlen = maxlen

    def prepare(self):
        self.lengths = [self.captions_len[i,-1] for i in xrange(self.captions_len.shape[0])]
        self.len_unique = np.unique(self.lengths)
        if self.minlen != None and self.maxlen != None:
            self.len_unique = [ll for ll in self.len_unique if ll >= self.minlen and ll < self.maxlen]
        else:
            self.len_unique = [ll for ll in self.len_unique]

        self.len2cap = dict()

        for ll in self.len_unique:
            self.len2cap[ll] = np.where(self.lengths == ll)[0]

    def reset(self):
        self.len2ind = dict()
        # shuffle it
        for ll in self.len2cap:
            self.len2cap[ll] = np.random.permutation(self.len2cap[ll])
            self.len2ind[ll] = 0

        self.len_unique_copy = copy.copy(self.len_unique)

    def next(self):
        if self.len_unique_copy == []:
            return -1, -1, -1 # epoch is done

        i = randint(0, len(self.len_unique_copy)-1)
        len_i = self.len_unique_copy[i] # current len
        ind_i = self.len2ind[len_i] # index of first not read caption of that len

        cap_i = self.len2cap[len_i][ind_i:ind_i+self.batch_size]
        im_i = []
        for curr_cap_i in cap_i:
            im_i.append(self.cap2im[curr_cap_i])

        self.len2ind[len_i] = self.len2ind[len_i] + self.batch_size
        if self.len2ind[len_i] > self.len2cap[len_i].shape[0]:#len(self.len2cap[len_i]):
            del self.len_unique_copy[i]
        
        return np.int32(np.array(cap_i)), np.int32(np.array(im_i)), np.int32(len_i)

    def __iter__(self):
        return self
