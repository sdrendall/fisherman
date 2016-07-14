import json
import caffe
import numpy
from itertools import izip, imap, chain
from numpy import random
from os import path
from fisherman import data_io, detection, math
from cProfile import Profile

SCALE = (0.00325218, 0.00021881)
OFFSET = (469.376, 4183.8239)

def rescale_image(image, scale=SCALE, offset=OFFSET):
    """
    Performs mean and std normalization on the given image, on a channel by channel basis.
    Hardcoded to match the original_and_partial_qc_2_channel_k149_scaled dataset.
    """
    for i in range(0, 2):
        image[i, ...] -= offset[i]
        image[i, ...] *= scale[i]

    return image

def subsample_list(input_list, indicies):
    return [input_list[i] for i in indicies]


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
        - chunker_params: params to pass to the image chunker. See detection.ImageChunkerWithOutput

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
        self.chunker_params = {
            'chunk_size': 125,
            'window_size': 101,
            'stride': 1,
            'num_classes': 1
        }
        self.chunker_params.update(params.get('chunker_params', {}))

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
        # load data for tops and reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, self.get_keys())
            top[i].reshape(*self.data[t].shape)


    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

        # pick next inputs
        self.csr = random.choice(range(0, len(self.keys)), self.n_samples, replace=False)


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
        image = importer.import_image().astype(numpy.float32)
        # TODO: Mean/std normalize here
        return rescale_image(image)


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
        FishFovDataLayer.setup(self, bottom, top)
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


class DataMapLayer(caffe.Layer):

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - data_dir: path to training images dir
        - split: train / test: data set to load image names from
        - tops: list of tops to output from {image, label}
        - seed: seed for randomization (default: None / current time)
        - samples_per_class: the number of samples to take during each iteration
        - channels_of_interest: tuple denoting the channels of interest to extract from the
          source image. (default: (1,2))
        - kernel: size of kernel to load images of
        - stats_path: path to the vsi_stats.json file that contains normalization stats
        - normalization: {static, median, mean} select normalization mode, defaults to static

        example: params = dict(data_dir="/path/to/fish_training_data", split="train",
                                tops=['image', 'label'])
        """
        params = eval(self.param_str)
        self.data_dir = path.expanduser(params['data_dir'])
        self.split = params['split']
        self.tops = params['tops']
        self.kernel = params['kernel']
        self.seed = params.get('seed', None)
        self.samples_per_class = params.get('samples_per_class', 1)
        self.channels_of_interest = params.get('channels_of_interest', (1,2))
        self.stats_path = params.get('stats_path', None)
        self.normalization = params.get('normalization', 'static').lower()
        self.profiler = Profile()

        # store top data for reshape + forward
        self.data = {}

        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.".format(len(self.tops)))
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        if self.stats_path is not None:
            self.stats_dict = json.load(open(self.stats_path))
        elif self.normalization != 'static':
            raise Exception("stats_path must be specified for normalization modes other than static")

        index_path = path.join(self.data_dir, '{}_index.json'.format(self.split))
        index = json.load(open(index_path))
        self.source_image_mask_pairs = \
            [(self.load_source_image(entry), self.load_mask(entry)) for entry in index.itervalues()]

        # load image indices for images and labels
        self.initialize_data_map()

        #if self.samples_per_class > len(self.unique_labels):
            #raise Exception("samples_per_class cannot exceed the number of instances of each class.")
        if self.samples_per_class < 1:
            raise Exception("samples_per_class cannot be less than 1.")

    def initialize_data_map(self):
        """
        The data map contains a pair of parallel arrays for each unique label in the data set.
        The image array contains kernel size views into the source image, and the label array contains
         the class label corresponding to that view in the source mask
        """
        self.data_map = {}
        images, labels = map(list, self.get_data_seqs())
        self.unique_labels = set(labels)
        self.num_labels = len(self.unique_labels)
        for target_label in set(labels): # iterate over unique labels
            # filter the image label pairs corresponding to the target label
            matching_ims, matching_labs = izip(
                *((image, label) \
                for image, label in izip(images, labels)
                if label == target_label)
            )
            # add numpy arrays containing the matching arrays and labels to the data map
            self.data_map[target_label] = {
                'image': list(matching_ims),
                'label': list(matching_labs)
            }
            # store the count since we'll be accessing that frequently
            self.data_map[target_label]['count'] = len(self.data_map[target_label]['label'])
            self.data_map[target_label]['seq_generator'] = \
                math.IntSequenceSampler(self.data_map[target_label]['count'], self.samples_per_class)

    def get_data_seqs(self):
        """
        Returns a sequence of (image_view, label) pairs that exhaustively covers the loaded image, mask pairs
        """
        image_label_pairs = chain(*imap(self.get_image_label_pair_seq, self.source_image_mask_pairs))
        image_seq, label_seq = zip(*image_label_pairs)
        return image_seq, label_seq

    def get_image_label_pair_seq(self, source_image_mask_pair):
        """
        Returns a sequence of (image_view, label) pairs for each possible view in the given
         (source_image, mask) pair
        """
        source_image, mask = source_image_mask_pair
        for i in range(0, source_image.shape[-2] - self.kernel + 1):
            for j in range(0, source_image.shape[-1] - self.kernel + 1):
                image = source_image[..., i:i+self.kernel, j:j+self.kernel]
                label = int(mask[..., i, j])
                yield image, label # add a new axis for stacking images

    def normalize_image(self, image, entry):
        """
        Normalize the given image according to the specified normalization mode
        """
        if self.normalization == 'static':
            return rescale_image(image)

        elif self.normalization == 'mean':
            norm_stats = get_normalization_stats(entry)
            offset = [c['mean'] for c in norm_stats['channel_stats']]
            return rescale_image(image, offset=offset)

        elif self.normalization == 'median':
            norm_stats = get_normalization_stats(entry)
            offset = [c['median'] for c in norm_stats['channel_stats']]
            return rescale_image(image, offset=offset)

        else:
            raise Exception('Encountered unknown normalization method: {}'.format(self.normalization))

    def get_normalization_stats(self, entry):
        """
        Get the appropriate nomalization stats dict from the master stats dict for the given entry
        """
        vsi_name = path.basename(entry['vsi_filename'])
        return self.stats_dict[vsi_name]

    def load_source_image(self, entry):
        """
        Loads the source image specified by the given entry
        """
        importer = data_io.SourceImageImporter(path.join(self.data_dir, entry['image']))
        importer.set_channels_of_interest(self.channels_of_interest)
        importer.set_transpose(2,0,1)
        image = importer.import_image().astype(numpy.float32)
        # TODO: Mean/std normalize here
        return self.normalize_image(image, entry)[numpy.newaxis, ...]

    def load_mask(self, entry):
        """
        Loads the label mask specified by the given entry
        """
        importer = data_io.SourceImageImporter(path.join(self.data_dir, entry['label']))
        mask = importer.import_image()
        mask = (mask > 0).astype(numpy.uint8)
        return mask


    #def reshape(self, bottom, top):
        #self.profiler.runcall(DataMapLayer._reshape, self, bottom, top)
        #self.profiler.print_stats()

    def reshape(self, bottom, top):
        # load data for tops and reshape tops to fit (1 is the batch dim)
        sample_inds = {
            label: self.data_map[label]['seq_generator'].get_samples()
            for label in self.unique_labels
        }

        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, sample_inds)
            top[i].reshape(*self.data[t].shape)

    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

    def backward(self, top, propagate_down, bottom):
        pass

    def load(self, top, inds):
        if top not in self.tops:
            raise Exception("Unknown output type: {}".format(top))

        data_seq = chain(
            *(subsample_list(self.data_map[label][top], inds[label])
            for label in self.unique_labels)
        )
        if top == 'image':
            return numpy.concatenate(list(data_seq))
        elif top == 'label':
            return numpy.asarray(list(data_seq))
