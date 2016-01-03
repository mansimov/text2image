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
from argparse import ArgumentParser
from PIL import Image
import os

draw = __import__('draw')
ReccurentAttentionVAE = draw.ReccurentAttentionVAE

# main code (not in a main function to be able to run in IPython as well).
def in_ipython():
  try:
    __IPYTHON__
  except NameError:
    return False
  else:
    return True

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str,
                   help='json settings for model')
    parser.add_argument("--weights", type=str, dest="weights_file",
                default=None, help="Filename with weights")
    parser.add_argument('--subdir', dest='subdir', default="sample")
    args = parser.parse_args()

    model_name = args.model_name
    subdir = args.subdir

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
    if args.weights_file:
        pathToWeights = args.weights_file

    ipython_mode = in_ipython()
    if (not ipython_mode) and (not os.path.exists(subdir)):
        os.makedirs(subdir)

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
        
        if ipython_mode:
            if displayEachStep:
                pylab.figure()
                pylab.gray()
                pylab.imshow(total_image)
                pylab.show(block=True)
                pylab.close()
        else:
            img = Image.fromarray((255*total_image).astype(np.uint8))
            img.save("{0}/time-{1:03d}.png".format(subdir, i))

    if ipython_mode:
        pylab.figure()
        pylab.gray()
        pylab.imshow(total_image)
        pylab.show(block=True)
        pylab.close()
    else:
        img = Image.fromarray((255*total_image).astype(np.uint8))
        img.save("{0}/sample.png".format(subdir))
        os.system("convert -delay 5 {0}/time-*.png -delay 300 {0}/sample.png {0}/sequence.gif".format(subdir))

