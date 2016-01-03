""" Generating simple MNIST digits using the original DRAW model. Kept as a reference"""

import h5py
import numpy as np
import json
import sys
from random import randint
import pylab
from util import sigmoid
import scipy
from attention import SelectiveAttentionModel

draw = __import__('draw')
ReccurentAttentionVAE = draw.ReccurentAttentionVAE

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

    # dummy data just for the sake of passing something as a parameter
    data = np.copy(h5py.File(train_file, 'r')[train_key])

    pathToWeights = None
    if "pathToWeights" in model:
        pathToWeights = model["pathToWeights"]

    displayEachStep = False

    rvae = ReccurentAttentionVAE(dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runSteps, data, pathToWeights=pathToWeights)
    # number of generated images is hardcoded to be equal to batch_size which equals to 100
    ct_s, _  = rvae.sample_from_prior(runSteps)
    ct_s = sigmoid(ct_s)

    for i in xrange(runSteps):
        total_image = np.zeros((280,280))

        for j in xrange(100):
            c = ct_s[i,j,:].reshape([28, 28])
            row = j/10
            column = j%10
            total_image[(row*28):((row+1)*28),(column*28):((column+1)*28)] = c[:][:]
        
        if displayEachStep:
            pylab.figure()
            pylab.gray()
            pylab.imshow(total_image)
            pylab.show(block=True)
            pylab.close()

    pylab.figure()
    pylab.gray()
    pylab.imshow(total_image)
    pylab.show(block=True)
    pylab.close()    
