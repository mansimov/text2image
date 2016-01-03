import h5py
import numpy as np
from random import randint
import pylab
import datetime
import scipy
from scipy.misc import toimage
np.random.seed(np.random.randint(1 << 30))

def create_reverse_dictionary(dictionary):
    dictionary_reverse = {}

    for word in dictionary:
        index = dictionary[word]
        dictionary_reverse[index] = word
    return dictionary_reverse

dictionary = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'the': 10, 'digit': 11, 'is': 12, 'on': 13, 'at': 14, 'left': 15, 'right': 16, 'bottom': 17, 'top': 18, 'of': 19, 'image': 20, '.': 21}
reverse_dictionary = create_reverse_dictionary(dictionary)

def sent2matrix(sentence, dictionary):
    words = sentence.split()
    m = np.int32(np.zeros((1, len(words)))) 

    for i in xrange(len(words)):
        m[0,i] = dictionary[words[i]]

    return m

def matrix2sent(matrix, reverse_dictionary):
    text = ""
    for i in xrange(matrix.shape[0]):
        text = text + " " + reverse_dictionary[matrix[i]]

    return text

def create_2digit_mnist_image_leftright(digit1, digit2):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))
    digit1 = digit1.reshape(28,28)
    digit2 = digit2.reshape(28,28)

    w = randint(16,18)
    h = randint(0,4)
    image[w:w+28,h:h+28] = digit1

    h = randint(28,32)
    image[w:w+28,h:h+28] = digit2

    image = image.reshape(-1)

    return image

def create_2digit_mnist_image_topbottom(digit1, digit2):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))
    digit1 = digit1.reshape(28,28)
    digit2 = digit2.reshape(28,28)

    h = randint(16,18)
    w = randint(0,2)
    image[w:w+28,h:h+28] = digit1

    w = randint(30,32)
    image[w:w+28,h:h+28] = digit2

    image = image.reshape(-1)

    return image

def create_1digit_mnist_image_topleft(digit1):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))
    digit1 = digit1.reshape(28,28)

    w = randint(0,2)
    h = randint(0,4)
    image[w:w+28,h:h+28] = digit1

    image = image.reshape(-1)

    return image

def create_1digit_mnist_image_topright(digit1):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))
    digit1 = digit1.reshape(28,28)

    w = randint(0,2)
    h = randint(28,32)
    image[w:w+28,h:h+28] = digit1

    image = image.reshape(-1)

    return image

def create_1digit_mnist_image_bottomright(digit1):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))
    digit1 = digit1.reshape(28,28)

    w = randint(30,32)
    h = randint(28,32)
    image[w:w+28,h:h+28] = digit1

    image = image.reshape(-1)

    return image

def create_1digit_mnist_image_bottomleft(digit1):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))
    digit1 = digit1.reshape(28,28)

    w = randint(30,32)
    h = randint(0,4)
    image[w:w+28,h:h+28] = digit1

    image = image.reshape(-1)

    return image


def create_mnist_captions_dataset(data, labels, banned, num=10000):
    images = np.zeros((num,60*60))
    captions = np.zeros((num,12))
    
    counts = [0, 0, 0, 0, 0, 0, 0, 0]

    curr_num = 0
    while True:
        # only left/right case for now
        k = randint(0,7)

        # Select 2 random digits
        i = randint(0,data.shape[0]-1)
        j = randint(0,data.shape[0]-1)

        # some cases are hidden from training set
        if k <= 3:
            if labels[i] == banned[k*2] or labels[j] == banned[k*2+1]:
                continue
        else:
            if labels[i] == banned[k+4]:
                continue

        if k == 0:
            sentence = 'the digit %d is on the left of the digit %d .' % (labels[i], labels[j])
        elif k == 1:
            sentence = 'the digit %d is on the right of the digit %d .' % (labels[j], labels[i])
        elif k == 2:
            sentence = 'the digit %d is at the top of the digit %d .' % (labels[i], labels[j])
        elif k == 3:
            sentence = 'the digit %d is at the bottom of the digit %d .' % (labels[j], labels[i])
        elif k == 4:
            sentence = 'the digit %d is at the top left of the image .' % (labels[i])
        elif k == 5:
            sentence = 'the digit %d is at the bottom right of the image .' % (labels[i])
        elif k == 6:
            sentence = 'the digit %d is at the top right of the image .' % (labels[i])
        elif k == 7:
            sentence = 'the digit %d is at the bottom left of the image .' % (labels[i])

        counts[k] = counts[k] + 1

        sentence_matrix = sent2matrix(sentence, dictionary)
        captions[curr_num,:] = sentence_matrix

        if k == 0 or k == 1:
            images[curr_num,:] = create_2digit_mnist_image_leftright(data[i,:], data[j,:])
        if k == 2 or k == 3:
            images[curr_num,:] = create_2digit_mnist_image_topbottom(data[i,:], data[j,:])
        if k == 4:
            images[curr_num,:] = create_1digit_mnist_image_topleft(data[i,:])
        if k == 5:
            images[curr_num,:] = create_1digit_mnist_image_bottomright(data[i,:])
        if k == 6:
            images[curr_num,:] = create_1digit_mnist_image_topright(data[i,:])
        if k == 7:
            images[curr_num,:] = create_1digit_mnist_image_bottomleft(data[i,:])

        curr_num += 1
        #print curr_num
        if curr_num == num:
            break

    return np.float32(images), np.int32(captions), counts

if __name__ == '__main__':
    data = np.copy(h5py.File('/ais/gobi3/u/nitish/mnist/mnist.h5', 'r')["train"])
    labels = np.copy(h5py.File('/ais/gobi3/u/nitish/mnist/mnist.h5', 'r')["train_labels"])

    image = create_1digit_mnist_image_topright(data[327,:])
    pylab.figure()
    pylab.gray()
    pylab.imshow(image.reshape((60,60)), interpolation='nearest')
    pylab.show(block=True)
