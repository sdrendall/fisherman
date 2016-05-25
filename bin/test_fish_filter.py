import caffe
import numpy
import bioformats
import javabridge
from bioformats import log4j
from fisherman import data_io, math, detection
from matplotlib.pyplot import figure, show, imshow

def load_vsi(vsi_path):
    """
    Load a vsi image at the given path.  The channels that are loaded, 
     and the order in which they are loaded are currently hard coded

    Note: This requires an active jvm via javabridge 
     (i.e. javabridge.start_vm(class_path=bioformats.JARS) must have been called prior to using this function)
    """
    print "Loading %s" % vsi_path
    with bioformats.ImageReader(vsi_path) as reader:
        #dapi = reader.read(c=0, rescale=False).astype(numpy.uint16)
        cfos = reader.read(c=1, rescale=False).astype(numpy.uint16)

    #return numpy.dstack((cfos, dapi))
    return cfos

def main():
    im_path = '/home/sam/Desktop/Fourth-FISH-set/FC(29)/FC(29)_Slide_19/FC(29)_19_cfos_06.vsi'
    model_path = '/home/sam/code/fisherman/fish_filter_output_iter_10000.caffemodel'
    net_path = '/home/sam/code/fisherman/caffe/fish_filter/fish_filter_conv_deploy.prototxt'

    javabridge.start_vm(class_path=bioformats.JARS)
    log4j.basic_config()

    im = load_vsi(im_path)
    im = math.median_normalize(im)

    javabridge.kill_vm()


    chunker_params = {
        'chunk_size': 300,
        'stride': 1,
        'window_size': 49,
        'num_classes': 2
    }

    chunker = detection.ImageChunkerWithOutput(im[numpy.newaxis, ...], **chunker_params)

    caffe.set_mode_gpu()
    net = caffe.Net(net_path, model_path, caffe.TEST)
    net.blobs['data'].reshape(1, *chunker.get_input_shape())

    print "Imported Image."
    print "Image shape: ", im.shape
    print "Image dtype: ", im.dtype

    output = chunker.allocate_output()
    for in_chunk, out_chunk in chunker:
        out_chunk[...] = net.forward_all(data=in_chunk[numpy.newaxis, ...])['prob'][...]

    output_mask = output.argmax(0)
    print "Mask shape: ", output_mask.shape
    print "Output shape: ", output.shape
    print "Max Output: ", output_mask.max()

    figure()
    imshow(im/im.max())
    show(block=False)

    figure()
    imshow(output_mask.astype(bool))

    show()

main()
