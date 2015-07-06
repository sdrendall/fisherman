import numpy
import caffe
import lmdb
import json
import generators as gen
import xml.etree.ElementTree as ET
from os import path
from matplotlib import pyplot as plt
from skimage import io
from exceptions import NameError, IndexError, TypeError


class TrainingExample(object):
    """ A training example for cell detection.  Consists of an image, and a label"""

    def __init__(self, image, label, tag=None):
        """
        @param: image: A numpy array containing the image of the training example.
        @param: label: The label associated with this image.
        """
        self.image = image
        self.label = label
        if tag == None:
            self.tag = str(label)

    def set_tag(self, new_tag):
        self.tag = str(new_tag)

    def get_tag(self):
        return self.tag

    def get_shape(self):
        return self.image.shape

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

    @attribute source_image: A MxNxC image that example images are to be extracted from.
    @attribute labels: A sequence of tuples in the following format: (centroid, label)
        centroids are numpy arrays containing the row, column index of the labelled cell
        labels are either 1 or 0, corresponding to whether the cell is positive or negative
        ALL COORDINATES MUST BE FLOATING POINT VALUES
    """

    def __init__(self, source_image, labels, example_shape=numpy.asarray((50.0, 50.0)), tag_generator=gen.counter()):
        self.source_image = source_image
        self.labels = labels
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
        (start_row, start_col) = centroid - numpy.floor(self.example_shape/2)
        (end_row, end_col) = centroid + numpy.ceil(self.example_shape/2)

        # Ensure that the requested example is contained in the source image
        if (start_row < 0 or start_col < 0) or (end_row > self.source_image.shape[0] or end_col > self.source_image.shape[1]):
            print("Requested example image exceeds the boundries of the source image")
            print "Image Shape: ", self.source_image.shape
            print "Requested example boundries: %r:%r, %r,%r" % (start_row, end_row, start_col, end_col)
            raise IndexError

        return self.source_image[start_row:end_row, start_col:end_col, :]

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
        self.example_shape = numpy.asarray(map(float, shape))


class SourceImageImporter(object):
    """
    Imports source images
    This will provide some formatting later on
    """

    def __init__(self, image_path):
        self.image_path = path.expanduser(image_path)
        self.channels_of_interest = 'all' # Defaults to all channels

    def import_image(self):
        """
        Loads the image from the image_path
        Only imports the desired channels from the source image
        """
        if self.channels_of_interest is 'all':
            return io.imread(self.image_path)
        else:
            return io.imread(self.image_path)[:,:,self.channels_of_interest]

    def set_channels_of_interest(self, channels):
        """
        Specifies the channels from the input image that should be imported.
        May be a list or tuple of integers corresponding to the index of the channels of interest, 
         or the string 'all' to specify that all channels should be loaded (the default behavior is all)
        """
        #if type(channels) is not list or type(channels) is not tuple or channels is not 'all':
            #message = "channels of interest may be a list or tuple of integers corresponding to the index of the channels," \
                      #"or the string 'all' to specify that all channels should be loaded (the default behavior is all)"
            #message += '\nGot Type: %r' % type(channels)
            #raise TypeError(message)

        self.channels_of_interest = channels


class LabelImporter(object):
    """
    Imports Labels from a JSON or XML file created using CellCounter
    """

    def __init__(self, label_path, label_type=None):
        """
        Ensures that the label_path of a valid filetype
        """

        if label_type is None or (label_type.lower() not in ('json', 'xml')):
            filetype = label_path.split('.')[-1]
            if filetype.lower() == 'json':
                label_type = 'json'
            elif filetype.lower() == 'xml':
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

        if (self.label_type == 'json'):
            raw_data = json.load(open(self.label_path))
            label_list = self._extract_label_list_from_json_dict(raw_data)
        else:
            tree = ET.parse(self.label_path)
            root = tree.getroot()
            label_list = self._extract_label_list_from_xml_root(root)

        return label_list

    def _extract_label_list_from_json_dict(self, raw_data):
        """
        Formats the raw json data as a list of centroid, label tuples
        """
        markers = raw_data['CellCounter_Marker_File']['Marker_Data']['Marker_Type']

        # First set is positive, second set is negative
        label_list = [(self._extract_centroid_from_json_marker(marker), 1) for marker in markers[0]['Marker']] + \
                        [(self._extract_centroid_from_json_marker(marker), 0) for marker in markers[1]['Marker']]

        return label_list

    @staticmethod
    def _extract_centroid_from_json_marker(self, marker):
        """
        Returns centroid coordinates from the marker in row column format as numpy arrays
        """
        return numpy.asarray([float(marker['MarkerY']), float(marker['MarkerX'])])

    def _extract_label_list_from_xml_root(self, root):
        """
        Extracts data from the root element of the xml file generated by CellCounter.
        Data is formated as a list of (centroid, label) tuples
        """
        marker_sets = root.find('Marker_Data').findall('Marker_Type')
        # The first marker type always contains positive examples, and the second contains negative examples
        label_list = [(self._extract_centroid_from_marker_element(marker), 1) for marker in marker_sets[0].findall('Marker')] + \
                        [(self._extract_centroid_from_marker_element(marker), 0) for marker in marker_sets[1].findall('Marker')]

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


class LMDatabaseFactory(object):
    """
    Used to add TrainingSets to an lmdb database that can be used as training data for a caffe net
    """
    def __init__(self, db_path, map_size):
        self.db = lmdb.open(db_path, map_size=map_size)

    def add_training_set(self, training_set):
        """
        Adds each training example in the training set to the database
        """
        map(self.add_training_example, training_set)

    def add_training_example(self, training_example):
        """
        Adds a training example to the database
        """
        # Format the training example as a caffe datum
        datum = caffe.proto.caffe_pb2.Datum()
        datum.height = training_example.image.shape[0]
        datum.width = training_example.image.shape[1]
        datum.channels = training_example.image.shape[2]
        # Caffe expects data in the shape (Channels, Height, Width)
        #  Skimage loads the images in the form (Height, Width, Channels)
        #  The axes must be rearranged before the data is dumped to the db
        datum.data = training_example.image.transpose(2,0,1).tostring()
        datum.label = int(training_example.label)

        # Write the datum to the database
        with self.db.begin(write=True) as txn:
            txn.put(training_example.tag, datum.SerializeToString())


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

    source_image = image_importer.import_image()
    labels = label_importer.import_data()

    test_set = TrainingSet(source_image, labels)

    for n, example in enumerate(test_set):
        print n, example.get_tag()
        save_path = '~/Desktop/test_example_%d_%d.tif' % (n, example.label)
        example.write(save_path)


if __name__ == "__main__":
    main()
