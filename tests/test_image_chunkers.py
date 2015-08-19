from nose import *
from nose.tools import assert_equal, assert_raises, raises
from fisherman import detection, error_handling
from scipy import signal
import numpy
import caffe
from matplotlib import pyplot as plt

# Image inputs to image chunkers should be transposed such that their dimensions follow 
#  the typical caffe input shape: K x H x W, where K is the number of channels in the image,
#  H is the height of the image, and W is the width of the image
image = numpy.random.rand(2, 100, 100)

def test_image_chunker_default_init():
    chunker = detection.ImageChunker(image)

    assert chunker._image is image
    assert all(chunker._image_resolution == numpy.asarray([100, 100], dtype=numpy.float32))
    assert_equal(chunker._chunk_size, numpy.asarray(100, dtype=numpy.float32))

def test_image_chunker_parameterized_init():
    chunker = detection.ImageChunker(image, chunk_size=20)

    assert chunker._image is image
    assert all(chunker._image_resolution == numpy.asarray([100, 100], dtype=numpy.float32))
    assert_equal(chunker._chunk_size, numpy.asarray(20, dtype=numpy.float32))
    assert_equal(chunker._chunk_size.dtype, numpy.float32)

def test_image_chunker_setters():
    chunk_size = 20
    chunker = detection.ImageChunker(image)
    chunker.set_chunk_size(chunk_size)

    assert chunker._image is image
    assert all(chunker._image_resolution == numpy.asarray([100, 100], dtype=numpy.float32))
    assert_equal(chunker._chunk_size, numpy.asarray(20, dtype=numpy.float32))
    assert_equal(chunker._chunk_size.dtype, numpy.float32)

def test_image_chunker_get_grid_shape_no_remainder():
    chunk_size = 20
    grid_shape = (5,5)
    chunker = detection.ImageChunker(image, chunk_size=chunk_size)

    assert_equal(chunker._get_chunk_grid_shape(), grid_shape)

def test_image_chunker_get_grid_shape_with_remainder():
    chunk_size = 23 
    grid_shape = (5,5)
    chunker = detection.ImageChunker(image, chunk_size=chunk_size)

    assert_equal(chunker._get_chunk_grid_shape(), grid_shape)

def test_image_chunker_get_image_chunk_internal():
    chunk_size = 20
    chunker = detection.ImageChunker(image, chunk_size=chunk_size)

    # first test a chunk entirely inside the image
    start_row = 20
    start_col = 40
    chunk = image[:, start_row:start_row + chunk_size, start_col:start_col + chunk_size]

    # this corresponds to the corresponding chunk row and column
    chunk_row = 1
    chunk_col = 2
    test_chunk = chunker._get_image_chunk(chunk_row, chunk_col)

    assert chunk.shape[1:] == (chunk_size, chunk_size)
    assert test_chunk.shape[1:] == (chunk_size, chunk_size)
    assert_same_slice(chunk, test_chunk)

def test_image_chunker_get_image_chunk_boundry():
    chunk_size = 23
    chunker = detection.ImageChunker(image, chunk_size=chunk_size)

    # test a chunk at the boundry, the fifth row and column will be pushed in
    chunk_row = 4
    chunk_col = 4
    test_chunk = chunker._get_image_chunk(chunk_row, chunk_col)
    chunk = image[:, 77:, 77:]

    assert chunk.shape[1:] == (chunk_size, chunk_size)
    assert test_chunk.shape[1:] == (chunk_size, chunk_size)
    assert_same_slice(chunk, test_chunk)

def test_image_chunker_with_output_initialization():
    chunk_size = 70
    window_size = 50
    stride = 6
    num_classes = 2
    chunker = detection.ImageChunkerWithOutput(image,
                                                          chunk_size=chunk_size,
                                                          window_size=window_size,
                                                          stride=stride,
                                                          num_classes=num_classes)
    
    assert chunker.get_classifier_window_size() == window_size
    assert chunker.get_classifier_window_stride() == stride
    assert chunker.get_number_of_output_classes() == num_classes

    assert chunker._window_size.dtype == numpy.float32
    assert chunker._stride.dtype == numpy.float32

@raises(error_handling.ConfigurationException)
def test_image_chunker_wo_allocate_output_misconfig():
    # allocate_output() should raise an error if the window_size, stride, and num_classes are not set
    chunker = detection.ImageChunkerWithOutput(image)
    output = chunker.allocate_output()

def test_image_chunker_wo_allocate_output():
    chunk_size = 70
    window_size = 50
    stride = 6
    num_classes = 2
    expected_shape = (num_classes, 10, 10)
    chunker = detection.ImageChunkerWithOutput(image)
    chunker.set_chunk_size(chunk_size)
    chunker.set_classifier_window_size(window_size)
    chunker.set_classifier_window_stride(stride)
    chunker.set_number_of_output_classes(num_classes)
    output = chunker.allocate_output()

    assert output.shape == expected_shape
    assert output.dtype == numpy.float32
    
def test_image_chunker_wo_get_output_chunk_internal():
    chunk_size = 70
    window_size = 50
    stride = 6
    num_classes = 2
    chunker = detection.ImageChunkerWithOutput(image)
    chunker.set_chunk_size(chunk_size)
    chunker.set_classifier_window_size(window_size)
    chunker.set_classifier_window_stride(stride)
    chunker.set_number_of_output_classes(num_classes)

    output = chunker.allocate_output()
    expected_chunk = output[:, :5, :5]
    test_chunk = chunker._get_output_chunk(0, 0)
    assert_same_slice(expected_chunk, test_chunk)
    
def test_image_chunker_wo_get_output_chunk_boundry():
    chunk_size = 70
    window_size = 50
    stride = 6
    num_classes = 2
    chunker = detection.ImageChunkerWithOutput(image)
    chunker.set_chunk_size(chunk_size)
    chunker.set_classifier_window_size(window_size)
    chunker.set_classifier_window_stride(stride)
    chunker.set_number_of_output_classes(num_classes)

    output = chunker.allocate_output()
    expected_chunk = output[:, -5:, -5:]
    test_chunk = chunker._get_output_chunk(10, 10)
    assert_same_slice(expected_chunk, test_chunk)

def test_image_chunker_wo_unallocated_output_tests():
    chunk_size = 70
    window_size = 50
    stride = 6
    num_classes = 2
    chunker = detection.ImageChunkerWithOutput(image)
    chunker.set_chunk_size(chunk_size)
    chunker.set_classifier_window_size(window_size)
    chunker.set_classifier_window_stride(stride)
    chunker.set_number_of_output_classes(num_classes)
    
    expected_exception = error_handling.OutputUnallocatedException
    assert_raises(expected_exception, chunker.__iter__().next)
    assert_raises(expected_exception, chunker.get_output_chunk_size)
    assert_raises(expected_exception, chunker._assert_allocated_output)
    assert not chunker.output_is_allocated()

    chunker.allocate_output()
    assert chunker.output_is_allocated()

def test_image_chunker_wo_output_chunking_coverage():
    # Tests that the output chunks span the entire output array
    chunk_size = 70
    window_size = 50
    stride = 6
    num_classes = 2
    chunker = detection.ImageChunkerWithOutput(image)
    chunker.set_chunk_size(chunk_size)
    chunker.set_classifier_window_size(window_size)
    chunker.set_classifier_window_stride(stride)
    chunker.set_number_of_output_classes(num_classes)

    output = chunker.allocate_output()
    for _, output_chunk in chunker:
        output_chunk[...] = 1

    assert output.all() # Each output value should be set to one

def test_image_chunker_convolution():
    # Apply convolution to an entire target image, and to image input and output chunk pairs
    # Results should be the same
    chunk_size = 70
    window_size = 50;
    stride = 1
    num_classes = 1
    test_image = numpy.random.rand(1,100,100).astype(numpy.float32)
    chunker = detection.ImageChunkerWithOutput(test_image)
    chunker.set_chunk_size(chunk_size)
    chunker.set_classifier_window_size(window_size)
    chunker.set_classifier_window_stride(stride)
    chunker.set_number_of_output_classes(num_classes)

    mean_kernel = (numpy.ones((window_size, window_size))/window_size**2).astype(numpy.float32)
    expected_output = signal.convolve2d(test_image[0,...], mean_kernel, mode='valid')
    expected_output = expected_output[numpy.newaxis, ...]

    output = chunker.allocate_output()
    for in_chunk, out_chunk in chunker:
        out_chunk[...] = signal.convolve2d(in_chunk[0,...], mean_kernel, mode='valid')

    # Convolving chunk by chunk should produce the same results as convolving the whole image
    assert (output.shape == expected_output.shape)
    assert (output == expected_output).all()
    
def assert_same_slice(a, b):
    """
    Asserts that a and b are the same slice (i.e. they are identical slices of the same array)

    This can be fooled by repeated data in the underlying array, but with randomly generated inputs,
     sufficiently large slices should never encounter this problem
    """
    assert a.base is b.base
    assert a.shape == b.shape 
    assert (a == b).all()
