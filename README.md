## Generating Images from Captions with Attention

Code for paper [Generating Images from Captions with Attention](http://arxiv.org/abs/1511.02793) by Elman Mansimov, Emilio Parisotto, Jimmy Ba and Ruslan Salakhutdinov; ICLR 2016.

We introduce a model that generates image blobs from natural language descriptions. The proposed model iteratively draws patches on a canvas, while attending to the relevant words in the description.

![theimage](https://pbs.twimg.com/media/CTfsgHYXIAEXEOb.png)

### Getting Started

The code is written in python. To use it you will need:

* Python 2.7
* Theano 0.7 (mostly tested using commit from June/July 2015)
* numpy and scipy
* h5py (HDF5 (>= 1.8.11))
* [skip-thoughts](https://github.com/ryankiros/skip-thoughts)

Before running the code make sure that you set floatX to float32 in Theano settings.

Additionally, depending on the tasks you will probably need to download these files by running:

```
wget http://www.cs.toronto.edu/~emansim/datasets/mnist.h5
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-images-32x32.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-images-56x56.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-captions.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-captions-len.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-cap2im.pkl
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-images-32x32.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-images-56x56.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-captions.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-captions-len.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-cap2im.pkl
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/test-images-32x32.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/test-captions.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/test-captions-len.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/test-cap2im.pkl
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/gan.hdf5
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dictionary.pkl
```

### MNIST with Captions

To train the model simply go to mnist-captions folder and run

```
python alignDraw.py models/mnist-captions.json
```

To generate 60x60 MNIST images from captions as specified in appendix of the paper run

```
python sample-captions.py --model models/mnist-captions.json --weights /path/to/trained-weights
```

**Note**: I have also provided implementation of simple draw model in files draw.py and sample.py

### Microsoft COCO

To train the model simply go to coco folder and run

```
python alignDraw.py models/coco-captions-32x32.json
```

To generate images from captions after training run

```
python sample-captions.py --model models/coco-captions-32x32.json --weights /path/to/trained-weights --dictionary dictionary.pkl --gan_path gan.hdf5 --skipthought_path /path/to/skipthoughts-folder
```

**Note**: I have been caught up with other non-research stuff, so I will add baseline model files like noAlignDraw and others during the week of Feb 29 - Mar 6.

Feel free to email me if you have some questions or if you are uncertain about some parts of the code.

### Acknowledgments

I would like to acknowledge the help of [Tom White](https://github.com/dribnet) for some suggestion on cleaning and organizing the code.

### Reference

If you found this code or our paper useful, please consider citing the following paper:

```
@inproceedings{mansimov16_text2image,
  author    = {Elman Mansimov and Emilio Parisotto and Jimmy Ba and Ruslan Salakhutdinov},
  title     = {Generating Images from Captions with Attention},
  booktitle = {ICLR},
  year      = {2016}
}
```

You would probably also need to cite some of the papers that we have referred to ;)