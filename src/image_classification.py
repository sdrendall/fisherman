import caffe
import numpy
import skimage
import itertools as it

class SlidingWindowClassifier(object):
    """
    Wraps a caffe net and a transformer to perform sliding window classification on large images
    """

    def __init__(self, net, transformer=None):
        self.set_net(net)
        self.set_transformer(transformer)

    def set_net(self, net):
        """
        Sets my net
        """
        self.net = net

    def set_transformer(self, transformer):
        """
        Sets my transformer.  
        :param transformer: transformer to be set
        :return: void
        The transformer should be initialized with an input channel called 'data'
         whose shape corresponds to the shape of the input images like so:
            transformer = caffe.io.Transformer({'data': my_image.shape})
            my_sliding_window_classifier.set_transformer(transformer)

        If transformer is None, creates a 'null' transformer for my net's input size
        """
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = caffe.io.Transformer(
                {'input': self.net.blobs['data'].data.shape})

    def classify(self, image):
        """
        Applies the transformer to the given image, and classifies using the supplied net
        """ 
        transformed_image = self.transformer.preprocess('data', image)
        output = self.net.forward_all(data=numpy.asarray([transformed_image]))
        return output


class ImageChunker(object):
    """
    I split large input images into chunks for gpu processing
    This can be used to avoid memory issues when processing large images on the gpu

    Images should be in the K x H x W format expected by caffe 
    """

    def __init__(self, image, chunk_size=None):
        """
        Image must be specified at initialization.
        Image should be of form: K x H x W, where K is the channel number
        By default, chunk_size = min([H, W])
        """
        self._image = image
        self._chunk_size = numpy.asarray(image.shape[1:]).min()

        if chunk_size is not None:
            self.set_chunk_size(chunk_size)

    def __iter__(self):
        """
        Iterating over an ImageChunker yields image chunks of size chunk_size

        "image chunks" are simply numpy views into the original image array
        When the edge of the image is reached, a partially redundant view is returned
         such that the 'outer' edge of the view falls along the bounding edge of the image.
         This should result in coherent chunks of memory
        """
        for row, col in self._get_chunk_range():
            yield self._get_image_chunk(row, col)

    def _get_chunk_range(self):
        """
        Returns a sequence of (row, col) pairs corresponding to the different image chunks
         of chunk_size in the image
        """
        num_chunk_rows, num_chunk_cols = self._image.shape[1:]/self._chunk_size
        return it.product(range(0, num_chunk_rows), range(0, num_chunk_cols))
        
    def _get_image_chunk(self, chunk_row, chunk_col):
        """
        Returns a view to the image corresponding to the chunk_row and chunk_col

        If the chunk would extend beyond the image, it is instead shifted inwards
         such that it extends to the boundries
        """
        # row_range looks like [start_row, end_row]
        row_range = numpy.asarray((chunk_row, chunk_row + 1)) * self._chunk_size
        col_range = numpy.asarray((chunk_col, chunk_col + 1)) * self._chunk_size

        # Ensure that the end row and column are within the boundries of the image
        # If the range exceeds the boundries, the entire range is shifted inwards such that it
        #  ends at the boundry instead
        if row_range[1] > self._image.shape[1]:
            overflow = row_range[1] - self._image.shape[1]
            row_range -= overflow

        if col_range[1] > self._image.shape[2]:
            overflow = col_range[1] - self._image.shape[2]
            col_range -= overflow

        start_row, end_row = row_range
        start_col, end_col = col_range

        return self._image[:, start_row:end_row, start_col:end_col]

    def set_chunk_size(self, chunk_size):
        """
        Sets a new chunk size
        
        The chunk_size must be smaller than the height and width of the input image
        """
        chunk_size = numpy.asarray(chunk_size)
        assert(all(self._image.shape[1:] > chunk_size))

        self._chunk_size = chunk_size

    def get_chunk_size(self):
        return self._chunk_size.copy()


class ImageChunkerWithOutput(ImageChunker):

    def __init__(self, image, chunk_size=None, window_size=None):
        """
        Image must be specified at initialization.
        Image should be of form: K x H x W, where K is the channel number
        By default, chunk_size = min([H, W])
        By default, window_size = 1
        """
        ImageChunker.__init__(self, image)
        self._window_size = numpy.asarray(1)
        self._output = None

        if chunk_size is not None:
            self.set_chunk_size(chunk_size)

        if window_size is not None:
            self.set_window_size(window_size)

    def __iter__(self):
        """
        Iterating over an ImageChunkerWithOutput yeilds (image_chunk, output_chunk) tuples where
         the output_chunk is a view into the output array corresponding to the location in the output
         array where the results of the neural network should be stored for the given image_chunk
        """
        for row, col in self._get_chunk_range():
            yield (self._get_image_chunk(row, col), self._get_output_chunk(row, col))
    
    def _get_output_chunk(self, chunk_row, chunk_col):
        """
        Returns the chunk of the output buffer at (chunk_row, chunk_col)
        """
        pass

    def set_chunk_size(self, chunk_size):
        """
        Sets a new chunk size
        
        The chunk_size must be smaller than the height and width of the input image, 
         plus window_size - 1
        """
        assert(all(self._image.shape[1:] > (chunk_size + self._window_size - 1)))
        ImageChunker.set_chunk_size(self, chunk_size)

    def set_window_size(self, window_size):
        """
        Sets a new window size

        The window size should correspond to the kernel size of the first layer 
         of the network chunks are being fed to.
        """
        window_size = numpy.asarray(window_size)
        assert(all(self._image.shape[1:] > (window_size - self._chunk_size)))

        self._window_size = window_size

    def allocate_output(self, num_classes=2):
        """
        :param num_classes: The number of classes output by the neural network.  Defaults to 2
        
        Allocates a numpy array to hold the output of the CNN resulting from the input array
        The output shape is:
            num_classes X (H - window_size + 1) X (W - window_size + 1)
            where H and W are the height and width of the input image respectively
        """
        output_height_and_width = numpy.asarray(self._image.shape[1:]) - self._window_size + 1
        self._output = numpy.zeros([num_classes] +  list(output_height_and_width))

    def get_window_size(self):
        """
        Returns a deep copy of the current value of self._window_size

        Window size should correspond to the size of the convolutional window on the first input layer
        """
        return self._window_size.copy()
