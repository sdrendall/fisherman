import json
import caffe
import numpy
from itertools import izip, imap, chain
from numpy import random
from os import path
from fisherman import math
from skimage import io
from cProfile import Profile

SCALE = (1/743.13, 1/595.10) # Value to scale each channel by, typically 1/std
OFFSET = (323.4328, 244.4754) # Value to offset each channel by, typically the mean

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
            [(self.load_source_image(entry), self.load_mask(entry)) for entry in index]

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
        # Load each image channel by the filename specified in index.json
        im_ch1 = io.imread(path.join(self.data_dir, 'ch1', entry['image']))
        im_ch3 = io.imread(path.join(self.data_dir, 'ch3', entry['image']))
        # Stack the image channels along the first axis results in (ch x h x w)
        image = numpy.stack((im_ch1, im_ch3))
        # Cast to float32
        image = image.astype(numpy.float32)
        # TODO: Mean/std normalize here
        return self.normalize_image(image, entry)[numpy.newaxis, ...]

    def load_mask(self, entry):
        """
        Loads the label mask specified by the given entry
        """
        mask_path = path.join(self.data_dir, 'markup', entry['label'])
        mask = io.imread(mask_path)
        mask = mask.astype(numpy.uint8)
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
