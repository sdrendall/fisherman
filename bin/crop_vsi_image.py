import numpy
import javabridge
import bioformats
from argparse import ArgumentParser
from bioformats import log4j
from skimage import io
from os import path


def configure_parser():
    parser = ArgumentParser(description='Crop a subimage from a vsi at the specified location')
    parser.add_argument(
        'vsi_path', 
        type=path.expanduser,
        help='Path to the vsi file to be cropped'
    )
    parser.add_argument(
        'output_path',
        type=path.expanduser,
        help='Path where the output image should be saved'
    )
    parser.add_argument(
        'top_left',
        type=int,
        nargs=2,
        help='The top left corner to be cropped from'
    )
    parser.add_argument(
        '-s', '--crop_size',
        type=int,
        help='The size of the region to be cropped, defaults to 500',
        default=500
    )

    return parser


def crop_image(image, top_left, crop_size):
    return image[top_left[0]:top_left[0]+crop_size, top_left[1]:top_left[1]+crop_size, ...]


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
        #dapi = reader.read(c=0, rescale=False).astype(numpy.uint16)
        cfos = reader.read(c=1, rescale=False).astype(numpy.uint16)

    javabridge.kill_vm()
    #return numpy.dstack((cfos, dapi))
    return cfos


def main():
    parser = configure_parser()
    args = parser.parse_args()

    image = load_vsi(args.vsi_path)
    cropped_image = crop_image(image, args.top_left, args.crop_size)

    io.imsave(args.output_path, cropped_image)


if __name__ == '__main__':
    main()
