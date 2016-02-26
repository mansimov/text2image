# Generate Images from Captions using this file

import h5py
import numpy as np
import json
import sys
from random import randint
import pylab
from util import sigmoid
import scipy
import math
import cPickle as pickle
from PIL import Image
import math
import os
from sharpen import gan
from argparse import ArgumentParser

# select the model from which you want to sample from
draw = __import__('alignDraw')
ReccurentAttentionVAE = draw.ReccurentAttentionVAE

"""Converts sentence into a list of vector representation of each word"""
def sent2matrix(sentence, dictionary):
    words = sentence.split()
    m = np.int32(np.zeros((1, len(words)))) 

    for i in xrange(len(words)):
        if words[i] in dictionary:
            m[0,i] = dictionary[words[i]]
        else:
            print words[i] + ' not in dictionary'
            m[0,i] = dictionary['UNK']
    return m, words

def find_max(array, excluding):
    best = -1
    best_i = -1
    i = 0
    for element in array:
        if element > best and i != excluding:
            best_i = i
            best = element
        i = i + 1
    return best_i

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='json settings for model')
    parser.add_argument('--weights', type=str, help="path to hdf5 file of trained weights")
    parser.add_argument('--dictionary', type=str, help='path to dictionary')
    parser.add_argument('--gan_path', type=str, help='path to gan to sharpen your images')
    parser.add_argument('--skipthought_path', type=str, help='path to skipthought.py file')
    args = parser.parse_args()
        
    model_name = args.model
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
    runSteps = int(model["model"][0]["runSteps"])# + 20

    validateAfter = int(model["validateAfter"])
    save = bool(model["save"])
    lr = float(model["lr"])
    epochs = int(model["epochs"])
    batch_size = int(model["batch_size"])
    reduceLRAfter = int(model["reduceLRAfter"])
    costFunctionType = str(model["model"][0]["costFunction"]["type"])
    samplePixels = str(model["model"][0]["costFunction"]["samplePixels"]) == "True"
    
    train_paths = model["data"]["train"]
    dev_paths = model["data"]["dev"]

    draw.train_paths = train_paths
    draw.dev_paths = dev_paths

    data = np.float32(np.load(train_paths["images"]))
    data_captions = np.int32(np.load(train_paths["captions"]))
    themean = np.mean(data, axis=0)
    thestd = np.std(data, axis=0)

    height = int(math.sqrt(dimX/3))
    width = int(math.sqrt(dimX/3))
    draw.height = height
    draw.width = width

    rvae = ReccurentAttentionVAE(dimY, dimLangRNN, dimAlign, dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runSteps, batch_size, reduceLRAfter, data, data_captions, pathToWeights=args.weights)

    dictionary = pickle.load(open(args.dictionary, "r" ))
    num_samples = 100

    sentences = ["a large commercial airplane is flying in clear skies ."]

    for sentence in sentences:
        y, words = sent2matrix(sentence, dictionary)
        y = np.int32(np.repeat(y, num_samples, axis=0))
        print y.shape

        ct_s, write_attent_params, alphas, mu, logsigma = rvae.sample_from_prior(runSteps, y)
        ct_s = sigmoid(ct_s)
        
        #ct_s[ct_s > 1] = 1
        #ct_s[ct_s < 0] = 0

        alphas = np.mean(alphas,axis=1)
        alphas = np.mean(alphas,axis=0)
        
        for i in xrange(len(sentence.split(' '))):
            print sentence.split(' ')[i] + ' ' + str(alphas[i])

        dimension = int(math.sqrt(dimX/3))

        generated_imgs = np.float32(np.zeros((num_samples,dimension*dimension*3)))
        most_used = np.float32(np.zeros((y.shape[1])))

        sentence_towrite = sentence[0:-2]
        sentence_towrite = sentence_towrite.replace(' ', '-')

        path_towrite_images = './gen-imgs/'
        total_image = np.zeros((dimension*10,dimension*10,3))
        
        for i in xrange(num_samples):
            generated_imgs[i,:] = ct_s[-1,i,:]
            c = ct_s[-1,i,:].reshape([3,dimension,dimension])
            c = c.transpose(1, 2, 0)
            row = i/10
            column = i%10
            total_image[(row*dimension):((row+1)*dimension),(column*dimension):((column+1)*dimension),:] = c[:][:][:]

        # Note that this is architecture is hardcoded for now
        # After generating blurry images ; sharpen them
        K = 32
        factors = [1,1,1]
        kernel_sizes = [7,7,5]
        num_filts = [128,128,3]
        pathToGANWeights = args.gan_path

        sys.path.append(args.skipthought_path)
        import skipthoughts
        model = skipthoughts.load_model()
        y_skipthought = skipthoughts.encode(model, [sentence])
        y_skipthought = np.float32(np.repeat(y_skipthought, 100, axis=0))

        batch_size = generated_imgs.shape[0]
        print generated_imgs.shape, y_skipthought.shape

        generate_edges_func = gan(K, batch_size, factors, kernel_sizes, num_filts, pathToGANWeights)
        edges = generate_edges_func(generated_imgs, y_skipthought)

        generated_imgs = generated_imgs.reshape([generated_imgs.shape[0], 3, K, K])
        sharp_imgs = generated_imgs + edges
        sharp_imgs[sharp_imgs>1]=1
        sharp_imgs[sharp_imgs<0]=0

        total_image_sharp = np.zeros((dimension*int(math.sqrt(num_samples)),dimension*int(math.sqrt(num_samples)),3))
        for j in xrange(num_samples):
            row = j/int(math.sqrt(num_samples))
            col = j%int(math.sqrt(num_samples))
            theimage = sharp_imgs[j]
            theimage = theimage.transpose(1,2,0)
            total_image_sharp[(row*K):((row+1)*K),(col*K):((col+1)*K),:] = theimage

        # save images instead
        print ('done generating images ; saving them to gen-imgs folder')
        scipy.misc.toimage(total_image, cmin=0.0, cmax=1.0).save(path_towrite_images + 'total_image.png')
        scipy.misc.toimage(total_image_sharp, cmin=0.0, cmax=1.0).save(path_towrite_images + 'total_image_sharp.png')
