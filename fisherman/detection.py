import caffe
import numpy
import skimage
from skimage.morphology import disk
from skimage import feature, filters, measure, morphology, io
import itertools as it
from fisherman import error_handling, math
from scipy.misc import imresize
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import gaussian_laplace


class ImageChunker(object):
    """
    I split large input images into chunks for gpu processing
    This can be used to avoid memory issues when processing large images on the gpu

    Images should be in the K x H x W format expected by caffe 
    """

    def __init__(self, image, chunk_size=None, **kwargs):
        """
        Image must be specified at initialization.
        Image should be of form: K x H x W, where K is the channel number
        By default, chunk_size = min([H, W])
        """
        self._image = image
        self._image_resolution = numpy.asarray(image.shape[1:], dtype=numpy.float32)
        self._chunk_size = numpy.asarray(self._image_resolution.min(), dtype=numpy.float32)

        # The ImageChunker tracks the current chunk row and column as it iterates. 
        self.current_chunk_row = 0
        self.current_chunk_col = 0
        
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
            self.current_chunk_col = col
            self.current_chunk_row = row
            yield self._get_image_chunk(row, col)

    def _get_chunk_range(self):
        """
        Returns a sequence of (row, col) pairs corresponding to the different image chunks
         of chunk_size in the image
        """
        num_chunk_rows, num_chunk_cols = self._get_chunk_grid_shape()
        return it.product(range(0, num_chunk_rows), range(0, num_chunk_cols))    

    def _get_chunk_grid_shape(self):
        """
        Returns a tuple containing the number of rows and columns in the stored image's "chunk grid":
            (num_chunk_rows, num_chunk_cols)
        """
        return tuple(numpy.ceil(self._image_resolution/self._chunk_size))
        
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
        chunk_size = numpy.asarray(chunk_size, dtype=numpy.float32)
        if not all(self._image_resolution >= chunk_size):
            msg = "Chunk size must be less than or equal to the image height and width\n" \
                "Image Height: %r, Image Width: %r" % tuple(self._image_resolution)
            raise error_handling.ConfigurationException(msg)

        self._chunk_size = chunk_size

    def get_chunk_size(self):
        """
        Returns the chunk_size as an int
        """
        return int(self._chunk_size)

    def get_input_shape(self):
        """
        Returns the shape of an input chunk as a tuple
        """
        return self._image.shape[0], self.get_chunk_size(), self.get_chunk_size()


class ImageChunkerWithOutput(ImageChunker):

    def __init__(self, image,
                 chunk_size=None,
                 window_size=None,
                 stride=None,
                 num_classes=None,
                 **kwargs):
        """
        :param chunk_size: integer specifying the size of image chunks to iterate over
        :param window_size: integer specifying the classifier's input window size
        :param stride: integer specifying the classifier's window stride
        :param num_classes: integer specifying the number of classes output by the classifier
        """
        ImageChunker.__init__(self, image, chunk_size=chunk_size, **kwargs)
        self._window_size = None
        self._stride = None
        self._num_output_classes = None
        self._output = None
        self._output_chunk_size = None

        if window_size is not None:
            self.set_classifier_window_size(window_size)

        if stride is not None:
            self.set_classifier_window_stride(stride)

        if num_classes is not None:
            self.set_number_of_output_classes(num_classes)

    def __iter__(self):
        """
        Iterating over an ImageChunkerWithOutput yeilds [image_chunk, output_chunk] lists where
         the output_chunk is a view into the output array corresponding to the location in the output
         array where the results of the neural network should be stored for the given image_chunk

        Raises an OutputUnallocatedException if the output has not been allocated 
            i.e. self.allocate_output() has not been called
        """
        self._assert_allocated_output()
        for row, col in self._get_chunk_range():
            yield [self._get_image_chunk(row, col), self._get_output_chunk(row, col)] 

    def _get_chunk_grid_shape(self):
        """
        Returns a tuple containing the number of rows and columns in the stored image's "chunk grid":
            (num_chunk_rows, num_chunk_cols)
        """
        non_overlapping_chunk_size = self._get_non_overlapping_chunk_size()
        return tuple(numpy.ceil((self._image_resolution - self._chunk_size)/non_overlapping_chunk_size) + 1)

    def _get_non_overlapping_chunk_size(self):
        """
        Calculates the size of a the chunk of input that is directly mapped to by the output of 
        fish net
        """
        return self._chunk_size - self._window_size + 1

    def _get_image_chunk(self, chunk_row, chunk_col):
        """
        Returns a view to the image corresponding to the chunk_row and chunk_col

        If the chunk would extend beyond the image, it is instead shifted inwards
         such that it extends to the boundries
        """
        # row_range looks like [start_row, end_row]
        non_overlapping_chunk_size = self._get_non_overlapping_chunk_size()
        start_row = chunk_row * non_overlapping_chunk_size
        start_col = chunk_col * non_overlapping_chunk_size
        end_row = start_row + self._chunk_size
        end_col = start_col + self._chunk_size

        # Ensure that the end row and column are within the boundries of the image
        # If the range exceeds the boundries, the entire range is shifted inwards such that it
        #  ends at the boundry instead
        if end_row > self._image.shape[1]:
            overflow = end_row - self._image.shape[1]
            start_row -= overflow
            end_row -= overflow

        if end_col > self._image.shape[2]:
            overflow = end_col - self._image.shape[2]
            start_col -= overflow
            end_col -= overflow

        return self._image[:, start_row:end_row, start_col:end_col]

    def _get_output_chunk(self, chunk_row, chunk_col):
        """
        Returns a view to the (chunk_row, chunk_col)th chunk of the output buffer

        If the chunk would exceed the boundries of the allocated output array, the chunk 
         is shifted inwards such that it ends at the edge of the output array
        """
        # row_range looks like [start_row, end_row]
        row_range = numpy.asarray((chunk_row, chunk_row + 1)) * self._output_chunk_size
        col_range = numpy.asarray((chunk_col, chunk_col + 1)) * self._output_chunk_size
        row_range -= chunk_row
        col_range -= chunk_col
        
        output_shape = self.get_output_shape()
        if row_range[1] > output_shape[1]:
            overflow = row_range[1] - output_shape[1]
            row_range -= overflow

        if col_range[1] > output_shape[2]:
            overflow = col_range[1] - output_shape[2]
            col_range -= overflow

        start_row, end_row = row_range
        start_col, end_col = col_range

        return self._output[:, start_row:end_row, start_col:end_col]

    def allocate_output(self):
        """
        Allocates a numpy array to hold the output of the CNN resulting from the input array
        The output shape is:
            num_classes x (H - window_size + 1) x (W - window_size + 1)
            where H and W are the height and width of the input image respectively
        """
        self._assert_chunker_config()
        self._assert_window_config()

        output_resolution = numpy.ceil((self._image_resolution - self._window_size)/self._stride) + 1
        self._output_chunk_size = numpy.ceil((self._chunk_size - self._window_size)/self._stride) + 1
        self._output = numpy.zeros([self._num_output_classes] + list(output_resolution), dtype=numpy.float32)
        return self._output

    def set_classifier_window_size(self, window_size):
        """
        Sets a new window size

        The window size describes the size of the classifier window that chunks will be fed to
        Window size must be less than or equal to the chunk_size

        The specified window size is rounded and cast to a numpy float32
        """
        window_size = numpy.round(numpy.asarray(window_size, dtype=numpy.float32))
        if window_size > self._chunk_size:
            msg = "Requested window size: %r is greater than the chunk size %r" \
                % (window_size, self.get_chunk_size())
            raise error_handling.ConfigurationException(msg)

        self._window_size = window_size

    def set_classifier_window_stride(self, stride):
        """
        Specifies the stride of the classifier window

        This can be obtained by finding the product of the strides
         of each layer in the classifier network

        The specified stride is rounded and cast to a numpy float32
        """
        self._stride = numpy.round(numpy.asarray(stride, dtype=numpy.float32))

    def set_number_of_output_classes(self, num_classes):
        """
        Specifies the number of output classes for the classifier
        """
        self._num_output_classes = num_classes

    def get_classifier_window_size(self):
        """
        Returns the classifier's window size as an int
        """
        return int(self._window_size)

    def get_classifier_window_stride(self):
        """
        Returns the classifier's window stride as an int
        """
        return int(self._stride)

    def get_number_of_output_classes(self):
        """
        Returns the number of output classes as an int
        """
        return int(self._num_output_classes)

    def get_output_shape(self):
        """
        Returns the shape of the output array

        Raises OutputUnallocatedException if the output has not yet been allocated.
        """
        self._assert_allocated_output()
        return self._output.shape

    def get_output_chunk_size(self):
        """
        Returns the size of the output chunk as an int

        Output chunk size is calculated when the output is allocated.
        Raises OutputUnallocatedException if the output has not yet been allocated.
        """
        self._assert_allocated_output()
        return int(self._output_chunk_size)

    def output_is_allocated(self):
        """
        Returns true if the output has been allocated
        """
        return self._output is not None

    @error_handling.ignore_future_warnings
    def _assert_window_config(self):
        """
        Asserts that the window_size and stride have been set
        """
        if None in (self._chunk_size, self._window_size, self._stride, self._num_output_classes):
            msg = "Window size and stride must be specified before output allocation!"
            raise error_handling.ConfigurationException(msg)

    def _assert_chunker_config(self):
        """
        Asserts that the window_size, chunk_size and stride are compatible
        """
        if (self._chunk_size - self._window_size + 1) % self._stride != 0:
            msg = "Chunk size not compatible with given window size and stride!\n" \
                  "(chunk_size - window_size + 1) % stride must equal 1 to prevent output aliasing!"
            raise TypeError(msg)


    def _assert_allocated_output(self):
        """
        Asserts that the output has been allocated
        """
        if not self.output_is_allocated():
            raise error_handling.OutputUnallocatedException


class CellDetector(object):
    """
    Detects an arbitrary number of cells in an image
    """

    def __init__(self, image=None, net=None, chunker_params=None, cell_radius=15, signal_channel=0, signal_plane_only=True):
        self.image = None
        self.fish_net = None
        self.signal_channel = None
        self._signal_plane = None
        self._mean_image = None
        self._local_max_image = None
        self._fish_net_mask = None

        self.chunker_params = {
            'chunk_size': 954, 
            'window_size': 49, 
            'stride': 6, 
            'num_classes': 2
        }

        if image is not None:
            self.set_image(image)

        if net is not None:
            self.set_net(net)

        if chunker_params is not None:
            self.update_chunker_params(chunker_params)

        self.set_cell_radius(cell_radius)
        self.set_signal_channel(signal_channel)
        self.set_compute_mask_on_signal_plane_only(signal_plane_only)

    def set_image(self, image):
        """
        Sets the image to detect cells in

        Setting a new image causes cached variables to be cleared
        """
        if len(image.shape) == 2:
            image = image[..., numpy.newaxis]
        elif len(image.shape) != 3:
            raise TypeError('Image shape must be of length 2 or 3!')

        self.image = skimage.img_as_float(image).astype(numpy.float32)
        self.clear_cached_variables()
        self._set_signal_plane()

    def set_net(self, net):
        """
        Sets the Caffe convolutional neural net to use for cell detection
        """
        self.fish_net = net

    def set_cell_radius(self, radius):
        """
        Specifies the approximate cell radius (in pixels) to use when filtering images to detect cells

        Cell identification assumes that cells have similar radii, and that cells are the brightest local
         patches of signal in regions identified as cell-containing by fish-net
        """
        self._cell_radius = int(radius)

    def set_compute_mask_on_signal_plane_only(self, boolean):
        """
        Specify whether to run fish_net on the signal plane or the entire image
        """
        self._signal_plane_only = boolean

    def set_signal_channel(self, channel):
        """
        Specifies the channel in the image that contains ISH signal
        """
        self.signal_channel = channel

        if self.image is not None:
            self._set_signal_plane()

    def _set_signal_plane(self):
        """
        Updates the signal channel with the current image and signal channel
        """
        self._signal_plane = (math.median_normalize(self.image[..., self.signal_channel], dtype=numpy.float32) * 25)

    def set_mode_cpu(self):
        """
        Use the cpu for caffe computation
        """
        caffe.set_mode_cpu()
        self.mode = 'cpu'

    def set_mode_gpu(self):
        """
        Use the gpu for caffe computation
        """
        caffe.set_mode_gpu()
        self.mode = 'gpu'

    def update_chunker_params(self, params):
        """
        Updates the parameters used to create an image chunker for image processing with a dict params.

        Valid fields:
            'chunk_size' 
            'window_size' 
            'stride'
            'num_classes'
        """
        self.chunker_params.update(params)

    def detect_cells(self):
        """
        Detects cells in self.image and returns a list of Cell objects containing the detected cells
        """
        cell_mask = self.get_fish_net_mask(cleaned=True, scaled=True)
        cell_mask = self.separate_cell_mask(cell_mask)

        return self.get_cells_from_labels(cell_mask)

    def get_fish_net_mask(self, cleaned=False, scaled=True, refresh_cache=False):
        """
        Returns the mask generated by the set fish_net.  The mask is cached so the mask is only computed the first
            time this is called.

        Params:
            cleaned: fill holes and remove speckles from the mask using morpholocial operations (default: False)
            scaled: return a mask that is the same dimensions as the image it was computed for (defaut: True)
            refresh_cache: recompute the mask (default: False)
            signal_plane_only: only use the signal plane when computing the mask (default: False)
        """
        if refresh_cache or self._fish_net_mask is None:
            self._fish_net_mask = self._get_fish_net_mask()

        return_mask = self._fish_net_mask

        if cleaned:
            return_mask = self.clean_fish_net_mask(return_mask)

        if scaled:
            return_mask = self.scale_mask_to_image(return_mask)

        return return_mask

    def _get_fish_net_mask(self):
        """
        Returns the predicted cell locations 
        """
        print "Casting fish net ....."
        if self._signal_plane_only:
            chunker = ImageChunkerWithOutput(self._signal_plane[numpy.newaxis, ...], **self.chunker_params)
        else:
            chunker = ImageChunkerWithOutput(self.image.transpose(2, 0, 1), **self.chunker_params)

        self.fish_net.blobs['data'].reshape(1, *chunker.get_input_shape())

        fish_net_out = chunker.allocate_output()
        for in_chunk, out_chunk in chunker:
            out_chunk[...] = self.fish_net.forward_all(data=in_chunk[numpy.newaxis, ...])['prob'][...]

        return fish_net_out.argmax(0)

    def clear_cached_variables(self):
        """
        Clears the cached variables used by get_cells_from_labels and
         get_peak_local_max
        """
        self._mean_image = None
        self._local_max_image = None
        self._fish_net_mask = None

    def clean_fish_net_mask(self, mask):
        """
        Gussies up the given mask with some morphological operations
        """
        print "Cleaning mask ....."
        mask = binary_fill_holes(mask)
        mask = morphology.erosion(mask, disk(2))
        return morphology.dilation(mask, disk(3))

    def scale_mask_to_image(self, mask):
        """
        Rescales the mask to the size of the image
        Assumes that the mask was generated with my fish_net
        """
        print "Scaling mask ....."
        mask = imresize(mask, size=float(self.chunker_params['stride']), interp='nearest')

        borders = (numpy.asarray(self.image.shape[:2]) - numpy.asarray(mask.shape)).astype(numpy.float32)
        print "Border Size: ", tuple(borders)
        pad_before = map(int, numpy.ceil(borders/2))
        pad_after = map(int, numpy.floor(borders/2))

        return skimage.util.pad(mask, zip(pad_before, pad_after), mode='constant')

    def separate_cell_mask(self, mask):
        """
        Attempts to separate cells in a mask using watersheding of a smoothed image
        """
        print "Separating cells ....."
        gauss_laplace_image = gaussian_laplace(self._signal_plane[..., numpy.newaxis], self._cell_radius*2/3)
        # Watershed cells from the max peaks
        start_points = self.get_peak_local_max(indices=False)
        start_points = measure.label(start_points, return_num=False)

        # The background is a single region, that watersheds against the cells
        start_points[numpy.logical_not(mask)] = start_points.max() + 1 

        ws_im = morphology.watershed(gauss_laplace_image, start_points[..., numpy.newaxis])
        ws_im[ws_im == ws_im.max()] = 0 # Background should be set to zero
        return ws_im

    def get_cells_from_labels(self, label_mask):
        """
        Returns a list of Cells, each generated from a different labeled region in label mask
        """
        print "Aggregating cells ....."
        return [self.get_cell_from_props(p) for p in measure.regionprops(label_mask)]

    def get_cell_from_props(self, props):
        """
        Returns a cell objects generated from the given props
        """
        return Cell(
            image=math.crop_from_bounding_box(props.bbox, self.image),
            mask=props.image,
            centroid=props.centroid,
            bounding_box=props.bbox
        )

    def get_peak_local_max(self, indices=True, refresh_cache=False):
        """
        Returns the locations of the local maxima in an image, either as a boolean image, or as a 
         list of tuples corresponding to the locations of peaks.

        See skimage.feature.cornor_peaks
        """
        if refresh_cache or self._local_max_image is None:
            self._local_max_image = math.get_corner_peaks_by_label(
                self.get_mean_image(),
                labels=measure.label(self.get_fish_net_mask(cleaned=True, scaled=True)),
                indices=False,
                min_distance=2*self._cell_radius,
                exclude_border=False
            )

        if indices:
            return zip(*numpy.nonzero(self._local_max_image))
        else:
            return self._local_max_image

    def get_mean_image(self, refresh_cache=False):
        """
        Returns the output of the mean filtering using a structuring element of size cell radius

        Caches the result so that it is only computed once
        
        To recompute the result, specify the refresh_cache option
        The cache can be cleared by setting self._mean_image to None
        """
        if self._mean_image is None or refresh_cache:
            self._mean_image = filters.rank.mean(self._signal_plane.astype(numpy.uint8), disk(self._cell_radius))

        return self._mean_image


class ImageSlice(object):
    """
    A Slice of an image

    Contains image data, and the bounding box of the data in the parent image
    """

    def __init__(self, image, bounding_box=None):
        self.set_image(image)

        if bounding_box is None:
            bounding_box = [0, 0] + list(image.shape[:2])

        self.set_bounding_box(bounding_box)

    def display(self, block=False):
        """
        Displays the image of this cell
        """
        display(self.image)

    def set_image(self, image):
        """
        Sets the image depicting the cell
        """
        self._image = image

    def get_image(self):
        """
        Returns the image depicting this cell
        """
        return self._image

    def set_bounding_box(self, top_left):
        """
        Specifies the top, left corner of self.image in the containing image, such that 
         the elementwise addition of self._top_left and a (row, column) index in self._image
         results in the corresponding index of the original image

        Index is stored internally as a float32 and non-integer values are rounded
        """
        self._bbox = numpy.asarray(top_left, dtype=numpy.float32).round()

    def get_bounding_box(self):
        """
        Returns the top left index as a tuple
        """
        return tuple(self._bbox)

    # Properties 
    image = property(get_image, set_image)
    bbox = property(get_bounding_box, set_bounding_box)


class Cell(ImageSlice):
    """
    The image of a cell in a larger (parent) image.

    Contains the image of the cell, and the bounding box of the cell's image
     in the parent image
    """

    def __init__(self, image, mask, centroid, bounding_box=None, **kwargs):
        ImageSlice.__init__(self, image, bounding_box=bounding_box, **kwargs)
        self.set_mask(mask)
        self.set_centroid(centroid)
    
    def set_mask(self, mask):
        """
        Casts the given mask to a bool and sets it as self.mask
        """
        self._mask = mask.astype(numpy.bool)

    def get_mask(self):
        """
        Returns a boolean mask specifying the pixels in self.image that correspond to this cell
        """
        return self._mask

    def set_centroid(self, centroid):
        """
        Sets the given centroid as a numpy.float32
        """
        self._centroid = numpy.asarray(centroid, dtype=numpy.float32)

    def get_centroid(self):
        """
        Returns the coordinates to the cell's centroid within self._image
         as a tuple of floats

        This is NOT necessarily the center pixel in the image
        """
        return tuple(self._centroid)

    # Properties
    mask = property(get_mask, set_mask)
    centroid = property(get_centroid, set_centroid)


def display(image, block=False):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image)
    plt.show(block=block)
