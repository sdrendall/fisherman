import caffe
import numpy
import json
from itertools import izip
from numpy import random
from os import path
from fisherman import data_io, detection

class FishFovDataLayer(caffe.Layer):
    """
    Load (training_image, label_image) pairs from the FISH_net training db

    This is designed to feed data to a fully convolutional network

    Heavily based on nyud_layers.py from the 
        J. Long, E. Shelhamer, T. Darrell Fully Convolutional Networks for Semantic Segmentation

        Source code which can be found here: shelhamer/fcn.berkeleyvision.org
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - data_dir: path to training images dir
        - split: train / test: data set to load image names from
        - tops: list of tops to output from {image, label}
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        - channels_of_interest: tuple denoting the channels of interest to extract from the 
          source image. (default: (1,2))

        example: params = dict(data_dir="/path/to/fish_training_data", split="train",
                                tops=['image', 'label'])
        """
        params = eval(self.param_str)
        self.data_dir = path.expanduser(params['data_dir'])
        self.split = params['split']
        self.tops = params['tops']
        #self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.n_samples = params.get('n_samples', 1)
        self.channels_of_interest = params.get('channels_of_interest', (1,2))


        # store top data for reshape + forward
        self.data = {}

        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.".format(len(self.tops)))
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        index_path = path.join(self.data_dir, '{}_index.json'.format(self.split))
        self.index = json.load(open(index_path))
        self.keys = list(self.index.iterkeys())

        if self.n_samples > len(self.keys):
            raise Exception("n_samples must not exceed the number of possible inputs.")

        self.csr = random.choice(range(0, len(self.keys)-1), self.n_samples, replace=False)


    def get_keys(self):
        return [self.keys[i] for i in self.csr]

    
    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, self.get_keys())
            top[i].reshape(*self.data[t].shape)


    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

        # pick next inputs
        self.csr = random.choice(range(0, len(self.keys)-1), self.n_samples, replace=False)


    def backward(self, top, propagate_down, bottom):
        pass


    def load(self, top, keys):
        if top == 'image':
            return self.load_images(keys)
        elif top == 'label':
            return self.load_labels(keys)
        else:
            raise Exception("Unknown output type: {}".format(top))


    def load_images(self, keys):
        return numpy.stack((self.load_image(key) for key in keys))


    def load_image(self, key):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - transpose to channel x height x width order
        """
        elem = self.index[key]
        importer = data_io.SourceImageImporter(path.join(self.data_dir, elem['image']))
        importer.set_channels_of_interest(self.channels_of_interest)
        importer.set_transpose(2,0,1)
        in_ = importer.import_image().astype(numpy.float32)
        # TODO: Mean/std normalize here
        return in_
        

    def load_labels(self, keys):
        return numpy.stack((self.load_label(key) for key in keys))


    def load_label(self, key):
        """
        Load label image as 1 x height x width integer array of label indices.
        Set all "cell" classes to 1 (2 includes uncertain cells).
        The leading singleton dimension is required by loss layers.
        """
        elem = self.index[key]
        importer = data_io.SourceImageImporter(path.join(self.data_dir, elem['label']))
        label = importer.import_image()
        label = (label > 0).astype(numpy.uint8)
        label = label[numpy.newaxis, ...]
        return label


class ChunkingFishFovDataLayer(FishFovDataLayer):
    """
    This Data Layer allows for subsampling of input images through systematic cropping.
    The ImageChunker handles subsampling, providing a series of 'chunks' or tiled views into
    the source image and label. 

    Only randomly ordered loading is currently supported by this layer
    """

    def setup(self, bottom, top):
        super(ChunkingFishFovDataLayer, self).setup(bottom, top)

        self.chunker_params = {
            'chunk_size': 125,
            'window_size': 101,
            'stride': 1,
            'num_classes': 1
        }

        self.reset_chunk_cache()


    def reset_chunk_cache(self):
        # Cached images that are taken from the chunker
        self._chunkers = None
        self._chunk_lists = None
        self._chunk_csr = None


    def refresh_chunkers(self, keys):
        """
        Reload the image and label that correspond to the given keys. Create a new image 
         chunker as well as a new chunker_key
        """
        images = self.load_images(keys)
        labels = self.load_labels(keys)

        # The chunker creates a zeroed output buffer consistent with the image size and 
        #  specified kernel size (window size) and stride. The loaded label data is dumped
        #  into the output buffer so that it can be iterated over
        self._chunkers = [
            detection.ImageChunkerWithOutput(image, **self.chunker_params) for image in images
        ]
        for chunker, label in izip(self._chunkers, labels):
            chunker.allocate_output(dtype=label.dtype)
            chunker._output[...] = label[...]

        # The image chunker is an iterable that returns (image, label) pairs
        # We will extract views from this list by randomly indexing into it
        self._chunk_lists = map(list, self._chunkers) 

        self._chunk_csrs = [random.randint(0, len(chunk_list)-1) for chunk_list in self._chunk_lists]


    def forward(self, bottom, top):
        super(ChunkingFishFovDataLayer, self).forward(bottom, top)
        self.reset_chunk_cache()


    def load(self, top, keys):
        if self._chunk_lists is None or self._chunkers is None:
            self.refresh_chunkers(keys)

        # self._chunk_lists is a list of (image, label) tuples
        if top == 'image':
            return numpy.stack(chunk_list[chunk_csr][0] for 
                chunk_list, chunk_csr in izip(self._chunk_lists, self._chunk_csrs)
            )
        elif top == 'label':
            return numpy.stack(chunk_list[chunk_csr][1] for 
                chunk_list, chunk_csr in izip(self._chunk_lists, self._chunk_csrs)
            )
        else:
            raise Exception("Unknown output type: {}".format(top))

