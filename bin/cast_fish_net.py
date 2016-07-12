import caffe
import numpy
import bioformats
import javabridge
import h5py
from os import path, environ
from skimage import io
from bioformats import log4j
from fisherman import data_io, math, detection
from matplotlib.pyplot import figure, show, imshow
from argparse import ArgumentParser

MODEL_PATH = path.join(environ['FISHERMAN_ROOT'], 'fish_net_conv_deploy_weights.caffemodel')
NET_PATH = path.join(environ['FISHERMAN_ROOT'], '/caffe/fish_net/kern_149/fish_net_conv_deploy.prototxt')

NET_PARAMS ={
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
    parser.add_argument('-o', '--output', type=path.expanduser, default='./output_mask.tif',
        help='Location to save output mask')
    parser.add_argument('-c', '--channels', type=int, nargs='+', default=None,
        help='Channels (zero based) to use as input to network. Defaults to all')
    parser.add_argument('-s', '--scale', type=float, nargs='+', default=1.0,
        help='Scalar or array to scale the intensity values of the input image by. If multiple values are'
              'given, a different scale will be applied to each image channel.')
    parser.add_argument('-f', '--offset', type=float, nargs='+', default=0.0,
        help='Scalar or array to offset the intensity values of the input image by. If multiple values are'
              'given, a different offset will be applied to each image channel.')

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

def main():
    parser = configure_argument_parser()
    args = parser.parse_args()

    source_image = io.imread(args.image_path)
    input_image = extract_input_channels(source_image, args).astype(numpy.float32)
    input_image = transpose_input_image(input_image)
    input_image = rescale_image(input_image, args)


    caffe.set_mode_cpu()
    net = caffe.Net(NET_PATH, MODEL_PATH, caffe.TEST)

    output_resolution = numpy.asarray(input_image.shape[-2:]) - NET_PARAMS['kernel'] + NET_PARAMS['stride']
    output = numpy.zeros((NET_PARAMS['num_classes'], output_resolution[0], output_resolution[1]))

    for i in range(0, NET_PARAMS['stride']):
        for j in range(0, NET_PARAMS['stride']):
            progress = 100.0 * (i*NET_PARAMS['stride'] + j)/NET_PARAMS['stride']**2
            print "Progress: {0:.2f}".format(progress)
            input_view = input_image[numpy.newaxis, ..., i:, j:]
            net.blobs['data'].reshape(*input_view.shape)
            output[..., i::NET_PARAMS['stride'], j::NET_PARAMS['stride'] ] = \
                net.forward_all(data=input_view[numpy.newaxis, ...])['prob']

    output = output[..., :-NET_PARAMS['stride'] + 1, :-NET_PARAMS['stride'] + 1]
    output_mask = output.argmax(0)

    print "Mask shape: ", output_mask.shape
    print "Output shape: ", output.shape
    print "Max Output: ", output_mask.max()
    print "Mask mean: ", output_mask.mean()

    #hfile = h5py.File('fish_filter_output.hdf5', 'w')
    #hfile.create_dataset("output", data=output)
    #hfile.close()
    
    io.imsave(args.output, output_mask.astype(numpy.uint8))

    if args.display:
        figure()
        imshow(source_image)

        figure()
        imshow(output_mask)

        show()


main()
