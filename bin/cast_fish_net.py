import caffe
import numpy
import bioformats
import javabridge
import h5py
from os import path
from time import strftime
from skimage import io
from bioformats import log4j
from fisherman import data_io, math, detection
from matplotlib.pyplot import figure, show, imshow
from argparse import ArgumentParser

MODEL_PATH = '/home/sam/code/fisherman/fish_net_conv_deploy_weights.caffemodel'
NET_PATH = '/home/sam/code/fisherman/caffe/fish_net/kern_149/fish_net_conv_deploy.prototxt'
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
    parser.add_argument('-s', '--scale', type=float, nargs='+', default=1.0,
        help='Scalar or array to scale the intensity values of the input image by. If multiple values are'
             'given, a different scale will be applied to each image channel.')
    parser.add_argument('-f', '--offset', type=float, nargs='+', default=0.0,
        help='Scalar or array to offset the intensity values of the input image by. If multiple values are'
             'given, a different offset will be applied to each image channel.')
    parser.add_argument('-O', '--hdf5_output', type=path.expanduser, default=None,
        help='Path to optional hdf5 output. File will contain the raw output probabilities in addition to the '
             'class segmentation mask.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
        help='Disable output messages')
    parser.add_argument('--gpu', action='store_true', default=False,
        help='Compute network weights using a gpu. Defaults to cpu usage')


    return parser

def load_vsi(vsi_path):
    """
    Load a vsi image at the given path.  The channels that are loaded, 
     and the order in which they are loaded are currently hard coded

    Note: This requires an active jvm via javabridge 
     (i.e. javabridge.start_vm(class_path=bioformats.JARS) must have been called prior to using this function)
    """
    javabridge.start_vm(class_path=bioformats.JARS)
    log4j.basic_config()
    print "Loading %s" % vsi_path

    with bioformats.ImageReader(vsi_path) as reader:
        dapi = reader.read(c=0, rescale=False).astype(numpy.uint16)
        cfos = reader.read(c=1, rescale=False).astype(numpy.uint16)

    javabridge.kill_vm()

    #return numpy.dstack((cfos, dapi)).transpose(2,0,1)
    return cfos[numpy.newaxis, ...]

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
    for i in range(0, 2):
        scale = (0.00325218, 0.00021881)
        offset = (469.376, 4183.8239)
        im[i, ...] -= offset[i]
        im[i, ...] *= scale[i]

    return im

def debug_out(msg, args):
    if not args.quiet:
        print "[{}]:".format(strftime('%H:%M:%S')), msg

def main():
    # Configure parser
    parser = configure_argument_parser()
    args = parser.parse_args()

    # Load input image
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

    # Allocate output
    #   An additional stride - kernel pixels are added to the border of the output to account for the redundant final pixel
    #   produced during each pass
    output_resolution = numpy.asarray(input_image.shape[-2:]) - NET_PARAMS['kernel'] + NET_PARAMS['stride']
    output = numpy.zeros((NET_PARAMS['num_classes'], output_resolution[0], output_resolution[1]))

    # Compute network outputs
    for i in range(0, NET_PARAMS['stride']):
        for j in range(0, NET_PARAMS['stride']):
            progress = 100.0 * (i*NET_PARAMS['stride'] + j)/NET_PARAMS['stride']**2
            debug_out("{0:.2f}% progress".format(progress), args)
            input_view = input_image[numpy.newaxis, ..., i:, j:]
            net.blobs['data'].reshape(*input_view.shape)
            output[..., i::NET_PARAMS['stride'], j::NET_PARAMS['stride'] ] = \
                net.forward_all(data=input_view[numpy.newaxis, ...])['prob']

    output = output[..., :-NET_PARAMS['stride'] + 1, :-NET_PARAMS['stride'] + 1]
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
