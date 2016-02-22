import numpy
import caffe
import lmdb
import json
import xml.etree.ElementTree as ET
from os import path
from matplotlib import pyplot as plt
from skimage import io
from exceptions import NameError, IndexError, TypeError
from fisherman import math
from fisherman import generators as gen
from collections import Iterable
from itertools import cycle, izip


class TrainingExample(object):
    """ A training example for cell detection.  Consists of an image, and a label """

    def __init__(self, image, label, tag=None):
        """
        @param: image: A numpy array containing the image of the training example.
        @param: label: The label associated with this image.
        """
        self.image = image
        self.label = label
        if tag == None:
            self.tag = str(label)
        else:
            self.tag = tag

    def set_tag(self, new_tag):
        self.tag = str(new_tag)

    def get_tag(self):
        return self.tag

    def get_shape(self):
        return self.image.shape

    def get_centroid(self):
        """
        Returns the coordinates of the center pixel of this image, i.e. self.image.shape/2
        """
        return tuple(numpy.floor(numpy.asarray(self.image.shape)/2))

    def to_datum(self):
        # Format the training example as a caffe datum
        #datum = caffe.proto.caffe_pb2.Datum()
        #datum.height = self.image.shape[0]
        #datum.width = self.image.shape[1]
        #datum.channels = self.image.shape[2]
        #datum.label = int(self.label)

        # Caffe expects data in the shape (Channels, Height, Width)
        #  Skimage loads the images in the form (Height, Width, Channels)
        #  The axes must be rearranged before the data is dumped to the db
        #datum.data = self.image.transpose(2,0,1).tostring()

        # The built in caffe method properly handles different data types

        return caffe.io.array_to_datum(self.image.transpose(2,0,1), label=int(self.label))


    def display(self):
        """
        Displays the training example and its label.
        """
        plt.figure()
        plt.imshow(self.image)
        plt.title('%s' % 'Positive' if (self.label == 1) else 'Negative')
        plt.show(block=False)

    def write(self, save_path):
        """
        Not sure if I need this
        """
        save_path = path.expanduser(save_path)
        io.imsave(save_path, self.image)


class TrainingSet(object):
    """ 
    An image with a corresponding set of labels.
    Individual training examples can be extracted by iterating over a training set

    :attribute image: A MxNxC image that example images are to be extracted from.
    :attribute labels: A sequence of tuples in the following format: (centroid, label)
        centroids are numpy arrays containing the row, column index of the labelled cell
        labels are either 1 or 0, corresponding to whether the cell is positive or negative
        ALL COORDINATES MUST BE FLOATING POINT VALUES
    """

    def __init__(self, image, labels, example_shape=(50,50), tag_generator=gen.counter()):
        self.set_image(image)
        self.set_labels(labels)
        self.set_example_shape(example_shape)
        self.tag_generator = tag_generator

    def __iter__(self):
        """
        Training Sets are iterable.  Iterating over a Training Set produces a sequence of
        (image, label) tuples that correspond to individual training examples
        """
        for centroid, label in self.labels:
            try:
                # The TrainingExample's tag is set after it is initialized (rather than while) so that
                #  the tag_generator will not be incremented unless the TrainingExample is successfully created
                training_image = self._extract_training_image(centroid)
                example = TrainingExample(training_image, label)
                example.set_tag(self.tag_generator.next())
                yield example
            except IndexError:
                print "Warning! Could not extract example with centroid: ", centroid, "\n"

    def _extract_training_image(self, centroid):
        """
        Extracts a training example centered at centroid from the source image
         with the dimensions specified by self.example_shape
        """
        (start_row, end_row, start_col, end_col) = self.get_example_boundries(centroid)
        if not self.centroid_in_bounds(centroid):
            print("Requested example image exceeds the boundries of the source image")
            print "Image Shape: ", self.image.shape
            print "Requested example boundries: %r:%r, %r:%r" % (start_row, end_row, start_col, end_col)
            raise IndexError

        return self.image[start_row:end_row, start_col:end_col, :]

    def centroid_in_bounds(self, centroid):
        """
        Asserts that the example boundries around the given centroid are within the source image
        """
        (start_row, end_row, start_col, end_col) = self.get_example_boundries(centroid)
        return not (start_row < 0 or start_col < 0 or end_row > self.image.shape[0] or end_col > self.image.shape[1])

    def get_example_boundries(self, centroid):
        """
        Gets the bounding box for the exmaple with the given centroid in the format:
        (start_row, end_row, start_col, end_col)
        """
        centroid = numpy.asarray(centroid, dtype=numpy.float32)
        (start_row, start_col) = centroid - numpy.floor(self.example_shape/2)
        (end_row, end_col) = centroid + numpy.ceil(self.example_shape/2)

        return (start_row, end_row, start_col, end_col)

    def set_image(self, image):
        """
        Sets the image to be used in this training set
        """
        self.image = image

    def set_labels(self, labels):
        """
        Sets the labels for this training set
        
        This deep copies the data from labels into self.labels
        Centroids are rounded then cast to float32
        """
        self.labels = [(numpy.asarray(map(round, tuple(centroid)), dtype=numpy.float32), label) 
            for centroid, label in labels]
    
    def set_tag_generator(self, generator):
        """
        Sets a new tag generator for the TrainingSet
        """
        self.tag_generator = generator

    def set_example_shape(self, shape):
        """
        Specifies a new (2d) shape for training examples.
        Note that training examples always include all channels present in the source image
        """
        assert(len(shape) == 2)
        self.example_shape = numpy.asarray(shape, dtype=numpy.float32)


class SourceImageImporter(object):
    """
    Imports source images
    This will provide some formatting later on
    """

    def __init__(self, image_path, channels_of_interest='all'):
        self.image_path = path.expanduser(image_path)
        self.set_channels_of_interest(channels_of_interest)
        self.transpose = None

    def import_image(self, normalize=False):
        """
        Loads the image from the image_path
        Only imports the desired channels from the source image

        If normalize is True, image is cast to float, and all values are normalized from 0 to 1
        """
        image = io.imread(self.image_path)

        if self.transpose is not None:
            image = image.transpose(*self.transpose)

        # TODO Temp
        if image.shape[0] == 2:
            image = image.transpose(1, 2, 0)

        if self.channels_of_interest is not 'all':
            image = image[:,:,self.channels_of_interest]

        if normalize:
            return math.normalize_by_channel(image)
        else:
            return image

    def set_transpose(self, *transpose):
        """
        Sets a transpose to be applied to the image immediately after it is loaded

        Syntax is similar to numpy.ndarray.transpose
        """
        self.transpose = transpose
        
    def set_channels_of_interest(self, channels):
        """
        Specifies the channels from the input image that should be imported.
        May be a list or tuple of integers corresponding to the index of the channels of interest, 
         or the string 'all' to specify that all channels should be loaded (the default behavior is all)
        """
        if not (type(channels) in (tuple, list) or channels is 'all'):
            message = "channels of interest may be a list or tuple of integers" \
                      " corresponding to the index of the channels," \
                      "or the string 'all' to specify that all channels should be loaded " \
                      "(the default behavior is all)"

            message += '\nGot Type: %r' % type(channels)
            raise TypeError(message)

        self.channels_of_interest = channels


class LabelImporter(object):
    """
    Imports Labels from a JSON or XML file created using CellCounter
    """

    def __init__(self, label_path, label_type=None):
        """
        Ensures that the label_path of a valid filetype
        """

        if label_type is None or label_type.lower() is not 'xml':
            filetype = label_path.split('.')[-1]
            if filetype.lower() == 'xml':
                label_type = 'xml'
            else:
                print('Label data must be of filetype JSON or XML')
                raise NameError
                
        self.label_path = path.expanduser(label_path)
        self.label_type = label_type

    def import_data(self):
        """
        Imports and formats the data from the label file.
        Returns a list of Centroid, label tuples
        """

        tree = ET.parse(self.label_path)
        root = tree.getroot()
        label_list = self._extract_label_list_from_xml_root(root)

        return label_list

    def _extract_label_list_from_xml_root(self, root):
        """
        Extracts data from the root element of the xml file generated by CellCounter.
        Data is formated as a list of (centroid, label) tuples
        """
        marker_sets = root.find('Marker_Data').findall('Marker_Type')
        # The first marker type always contains positive examples, and the second contains negative examples
        label_list = [(self._extract_centroid_from_marker_element(marker), 1) for marker in marker_sets[0].findall('Marker')]

        # Some images have no negative examples
        if len(marker_sets) > 1:
            label_list += [(self._extract_centroid_from_marker_element(marker), 0) for marker in marker_sets[1].findall('Marker')]

        return label_list
    
    def _extract_centroid_from_marker_element(self, marker):
        """
        Extracts the X and Y coordinates from the marker element, and returns a numpy array
         in [(row, column)] format, such that the array indexs the centroid of the given training example
         in the numpy array storing the source image
        """
        row = marker.find('MarkerY').text
        col = marker.find('MarkerX').text
        return numpy.asarray([float(row), float(col)])


class CaffeDataFactory(object):

    def __init__(self, reg_fcn=None):
        self._reg_fcn = None
        if reg_fcn is not None:
            self.set_regularization_function(reg_fcn)

    def set_regularization_function(self, reg_fcn):
        """
        Sets a regularization function that is applied to each training example
         before it is added to the database
        """
        assert(callable(reg_fcn))
        self._reg_fcn = reg_fcn

    def regularize_data(self, array):
        """
        Regularizes the given array using the specified regularization function if one exists
        """
        if self._reg_fcn is None:
            return array
        else:
            return self._reg_fcn(array)


class LMDatabaseFactory(CaffeDataFactory):
    """
    Used to add TrainingSets to an lmdb database that can be used as training data for a caffe net
    """
    def __init__(self, db_path, map_size, reg_fcn=None):
        CaffeDataFactory.__init__(self, reg_fcn=reg_fcn)
        self.db = lmdb.open(db_path, map_size=map_size)

    # TODO Refactor to add_example_sequence
    def add_training_set(self, training_set):
        """
        Adds each training example in the training set to the database in a single transaction
        """
        with self.db.begin(write=True) as txn:
            for example in training_set:
                txn.put(example.tag, example.to_datum().SerializeToString())

    def add_training_example(self, training_example):
        """
        Adds a single training example to the database

        reg_fcn (optional): A f(ndarray) -> ndarray that is applied to the 
         data in training_example.image before
        """
        # Write the datum to the database
        datum = training_example.to_datum
        with self.db.begin(write=True) as txn:
            txn.put(training_example.tag, datum.SerializeToString())


class H5Factory(CaffeDataFactory):
    pass


def convert_raw_datum_to_training_example(raw_datum):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    flattened_image = numpy.fromstring(datum.data, dtype=numpy.uint8)
    image = flattened_image.reshape(datum.channels, datum.height, datum.width).swapaxes(0,2).swapaxes(1,2)
    label = datum.label 

    return TrainingExample(image, label)


def get_datum_stats(db):
    """
    Returns a list of tuples containing the (mean, min, max, label) of each datum in the given database
    """
    with db.begin() as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        return [(
                data_array.mean(),
                data_array.max(),
                data_array.min(),
                datum.label
            ) for _, data_array in (
                (
                    datum.ParseFromString(raw_datum),
                    numpy.fromstring(
                        datum.data,
                        dtype=numpy.uint8
                    ) 

                ) for _, raw_datum in txn.cursor()
            )
        ]

def get_datum_data(db):
    datum = caffe.proto.caffe_pb2.Datum()
    with db.begin() as txn:
        return [data_array for _, data_array in (
            (
                 datum.ParseFromString(raw_datum),
                 numpy.fromstring(
                    datum.data,
                    dtype=numpy.uint8
                ).reshape(
                    datum.channels,
                    datum.height,
                    datum.width
                ).transpose(1, 2, 0)

            ) for _, raw_datum in txn.cursor()
        )]


def main():
    from sys import argv

    if len(argv) < 3:
        print("Usage: " + argv[0] + " source_image_path label_path")
        return
    else:
        source_image_path = argv[1]
        label_path = argv[2]

    image_importer = SourceImageImporter(source_image_path)
    image_importer.set_channels_of_interest('all')
    label_importer = LabelImporter(label_path)

    image = image_importer.import_image()
    labels = label_importer.import_data()

    test_set = TrainingSet(image, labels)

    for n, example in enumerate(test_set):
        print n, example.get_tag()
        save_path = '~/Desktop/test_example_%d_%d.tif' % (n, example.label)
        example.write(save_path)


if __name__ == "__main__":
    main()
