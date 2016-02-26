import h5py
import numpy as np
import json
import sys
from random import randint
import pylab
from util import sigmoid
import scipy
from attention import SelectiveAttentionModel
import math
import cPickle as pickle
from PIL import Image
import math

#np.random.seed(np.random.randint(1 << 30))
#rng = RandomStreams(seed=np.random.randint(1 << 30))

draw = __import__('alignDraw')
ReccurentAttentionVAE = draw.ReccurentAttentionVAE

create_captions = __import__('create-captions')
create_mnist_captions_dataset = create_captions.create_mnist_captions_dataset

dictionary = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'the': 10, 'digit': 11, 'is': 12, 'on': 13, 'at': 14, 'left': 15, 'right': 16, 'bottom': 17, 'top': 18, 'of': 19, 'image': 20, '.': 21}

def sent2matrix(sentence, dictionary):
    words = sentence.split()
    m = np.int32(np.zeros((1, len(words)))) 

    for i in xrange(len(words)):
        m[0,i] = dictionary[words[i]]

    return m, words

def find_max(array, index):
    neg_array = -array
    sorted_i = np.argsort(neg_array)
    return sorted_i[index]

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='json settings for model')
    parser.add_argument('--weights', type=str, help="path to hdf5 file of trained weights")
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
    
    data = np.copy(h5py.File(model["data"]["train_data"]["file"], 'r')[model["data"]["train_data"]["key"]])
    labels = np.copy(h5py.File(model["data"]["train_labels"]["file"], 'r')[model["data"]["train_labels"]["key"]])

    sentence = "the digit 1 is on the left of the digit 0 ."
    
    y, words = sent2matrix(sentence, dictionary)
    y = np.int32(np.repeat(y, 100, axis=0))
    print y.shape

    height = int(math.sqrt(dimX))
    width = int(math.sqrt(dimX))
    draw.height = height
    draw.width = width

    rvae = ReccurentAttentionVAE(dimY, dimLangRNN, dimAlign, dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runSteps, batch_size, reduceLRAfter, data, labels, pathToWeights=args.weights)

    ct_s, write_attent_params, alphas, _, _ = rvae.sample_from_prior(runSteps, y)
    ct_s = sigmoid(ct_s)

    dimension = int(math.sqrt(dimX))
    print dimension, dimension
    
    most_used = np.float32(np.zeros((y.shape[1])))

    for i in xrange(runSteps):
        total_image = np.zeros((dimension*10,dimension*10))

        for j in xrange(100):
            c = ct_s[i,j,:].reshape([dimension,dimension])
            row = j/10
            column = j%10
            
            total_image[(row*dimension):((row+1)*dimension),(column*dimension):((column+1)*dimension)] = c[:][:]
        
        most_used = most_used / 100.
        best_i = find_max(most_used, 0)
        second_best_i = find_max(most_used, 1)
        third_best_i = find_max(most_used, 2)
        forth_best_i = find_max(most_used, 3)
        fifth_best_i = find_max(most_used, 4)
        sixth_best_i = find_max(most_used, 5)
        seventh_best_i = find_max(most_used, 6)

        text = words[best_i] + ' ' + words[second_best_i] + ' ' + words[third_best_i] + ' ' + words[forth_best_i] + ' ' + words[fifth_best_i] + ' ' + words[sixth_best_i] + ' ' + words[seventh_best_i]
        probs = str(most_used[best_i]) + ' ' + str(most_used[second_best_i]) + ' ' + str(most_used[third_best_i]) + ' ' + str(most_used[forth_best_i]) + ' ' + str(most_used[fifth_best_i]) + ' ' + str(most_used[sixth_best_i]) + ' ' + str(most_used[seventh_best_i])

        most_used = np.float32(np.zeros((y.shape[1])))
        print text, probs

        sentence = sentence.replace(' ', '-')

        scipy.misc.toimage(total_image, cmin=0.0, cmax=1.0).save("./%s-%03d.png" % (sentence, i))
