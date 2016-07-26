#! /usr/bin/env python

import caffe
import numpy
import bioformats
import javabridge
import h5py
import ast
from os import path, environ
from time import strftime
from skimage import io
from bioformats import log4j
from fisherman import data_io, math, detection
from matplotlib.pyplot import figure, show, imshow
from argparse import ArgumentParser

MODEL_PATH = path.join(environ['FISHERMAN_ROOT'], 'models/kern_149_15/fish_net_conv_deploy_weights.caffemodel')
NET_PATH = path.join(environ['FISHERMAN_ROOT'], 'caffe/kern_149_15/fish_net_conv_deploy.prototxt')

print "MODEL_PATH: {}".format(MODEL_PATH)
print "NET_PATH: {}".format(NET_PATH)

NET_PARAMS = {
    'kernel': 149,
    'stride': 15,
    'num_classes': 2
}

def configure_argument_parser():
    parser = ArgumentParser(description="Computes the output of fish-net for the given image")
    # Positional Arguments
    parser.add_argument('image_path', type=path.expanduser, help='Path to the image to process')
    # Optional Arguments
    parser.add_argument('-D','--display', action='store_true', default=False,
        help='Display the input image and output mask using matplotlib')
    parser.add_argument('-o', '--output', type=path.expanduser, default='./fish_net_mask.tif',
        help='Location to save output mask')
    parser.add_argument('-c', '--channels', type=int, nargs='+', default=None,
        help='Channels (zero based) to use as input to network. Defaults to all')
    parser.add_argument('-s', '--scale', type=float, nargs='+', default=[1.0],
        help='Scalar or array to scale the intensity values of the input image by. If multiple values are'
             'given, a different scale will be applied to each image channel.')
    parser.add_argument('-f', '--offset', type=float, nargs='+', default=[0.0],
        help='Scalar or array to offset the intensity values of the input image by. If multiple values are'
             'given, a different offset will be applied to each image channel.')
    parser.add_argument('-O', '--hdf5_output', type=path.expanduser, default=None,
        help='Path to optional hdf5 output. File will contain the raw output probabilities in addition to the '
             'class segmentation mask.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
        help='Disable output messages')
    parser.add_argument('--gpu', action='store_true', default=False,
        help='Compute network weights using a gpu. Defaults to cpu usage')
    parser.add_argument('-v', '--vsi', action='store_true', default=False,
        help='Specifies that the input image is in the vsi format')
    parser.add_argument('-p', '--chunker_params', type=ast.literal_eval, default={'chunk_size': 648, 'window_size': NET_PARAMS['kernel'], 'stride': 1},
        help='Use an image chunker to compute the input and output. Chunker params should be specified as a string'
             'Default: {"chunk_size": 300, "stride": 1, "window_size": %d, "num_classes": 2}' % NET_PARAMS['kernel'])
    parser.add_argument('-z', '--step_size', type=int, default=1,
        help='The sampling step_size to use when computing the output. ex. step_size=1 computes a fully dense
                output. step_size=2 computes an output for every other pixel')

    return parser

def extract_input_channels(source_image, args):
    if args.channels is None:
        return source_image
    else:
        return source_image[:, :, args.channels]

def transpose_input_image(input_image):
    if len(input_image.shape) == 1:
        return input_image[numpy.newaxis, ...]
    else:
        return input_image.transpose(2, 0, 1)

def rescale_image(im, args):
    """
    Rescale the image using the specified parameters
    """
    num_channels = im.shape[0]

    if len(args.offset) == 1:
        im -= args.offset[0]
    elif len(args.offset) == num_channels:
        for i in range(0, num_channels):
            im[i, ...] -= args.offset[i]
    else:
        raise Exception("Number of offset parameters must equal 1 or the number of image channels. Offset length: {}".format(len(args.offset)))

    if len(args.scale) == 1:
        im -= args.scale[0]
    elif len(args.scale) == num_channels:
        for i in range(0, num_channels):
            im[i, ...] *= args.scale[i]
    else:
        raise Exception("Number of scale parameters must equal 1 or the number of image channels. Scale length: {}".format(len(args.scale)))

    return im

def debug_out(msg, args):
    if not args.quiet:
        print "[{}]:".format(strftime('%H:%M:%S')), msg

def compute_network_outputs(input_image, net, args):
    # Allocate output
    #   An additional stride - kernel pixels are added to the border of the output to account for the redundant final pixel
    #   produced during each pass
    output_resolution = numpy.asarray(input_image.shape[-2:]) - NET_PARAMS['kernel'] + NET_PARAMS['stride']
    output = numpy.zeros((NET_PARAMS['num_classes'], output_resolution[0], output_resolution[1]))

    # Compute network outputs
    for i in range(0, NET_PARAMS['stride'], args.step_size):
        for j in range(0, NET_PARAMS['stride'], args.step_size):
            #progress = 100.0 * (i*NET_PARAMS['stride'] + j)/NET_PARAMS['stride']**2
            #debug_out("{0:.2f}% progress".format(progress), args)
            input_view = input_image[numpy.newaxis, ..., i:, j:]
            net.blobs['data'].reshape(*input_view.shape)
            output[..., i::NET_PARAMS['stride'], j::NET_PARAMS['stride']] = \
                net.forward_all(data=input_view[numpy.newaxis, ...])['prob']

    if NET_PARAMS['stride'] == 1:
        return output
    else:
        return output[..., :-NET_PARAMS['stride'] + 1, :-NET_PARAMS['stride'] + 1]


def main():
    # Configure parser
    parser = configure_argument_parser()
    args = parser.parse_args()

    # Load input image
    if args.vsi:
        source_image = data_io.load_vsi(args.image_path)
    else:
        source_image = io.imread(args.image_path)

    input_image = extract_input_channels(source_image, args).astype(numpy.float32)
    input_image = transpose_input_image(input_image)
    input_image = rescale_image(input_image, args)

    # Configure Caffe
    net = caffe.Net(NET_PATH, MODEL_PATH, caffe.TEST)

    if args.gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    chunker_params = NET_PARAMS.copy()
    chunker_params.update(args.chunker_params)
    chunker = detection.ImageChunkerWithOutput(input_image, **chunker_params)
    output = chunker.allocate_output()

    num_chunks = chunker.get_number_of_chunks()
    for i, (input_chunk, output_chunk )in enumerate(chunker):
        progress = 100.0 * i/num_chunks
        debug_out("{0:.2f}% progress".format(progress), args)
        output_chunk[...] = compute_network_outputs(input_chunk, net, args)[...]

    mask = output.argmax(0)

    debug_out("Mask shape: {}".format(mask.shape), args)
    debug_out("Output shape: {}".format(output.shape), args)
    debug_out("Max Output: {}".format(mask.max()), args)
    debug_out("Mask mean: {}".format(mask.mean()), args)

    # Save output
    io.imsave(args.output, mask.astype(numpy.uint8))

    if args.hdf5_output is not None:
        hfile = h5py.File(args.hdf5_output, 'w')
        hfile.create_dataset("output", data=output)
        hfile.create_dataset("mask", data=mask)
        hfile.close()
    
    if args.display:
        figure()
        imshow(source_image)

        figure()
        imshow(mask)

        show()


main()
