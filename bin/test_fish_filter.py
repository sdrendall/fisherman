import caffe
import numpy
import bioformats
import javabridge
import h5py
from skimage import io
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
        dapi = reader.read(c=0, rescale=False).astype(numpy.uint16)
        cfos = reader.read(c=1, rescale=False).astype(numpy.uint16)

    return numpy.dstack((cfos, dapi)).transpose(2,0,1)

def main():
    #im_path = '/home/sam/Desktop/Fourth-FISH-set/FC(29)/FC(29)_Slide_19/FC(29)_19_cfos_06.vsi'
    model_path = '/home/sam/code/fisherman/gpu_test/fish_filter_output_iter_90000.caffemodel'
    net_path = '/home/sam/code/fisherman/caffe/fish_filter/kern_101/fish_filter_conv_deploy.prototxt'

    #javabridge.start_vm(class_path=bioformats.JARS)
    #log4j.basic_config()

    #im = load_vsi(im_path)

    #javabridge.kill_vm()

    #im = io.imread('/home/sam/code/fisherman/data/dense_labelling/Shock(23)_15_cfos_06_training.tiff').astype(numpy.float32)
    #im = im[..., 1:].transpose(2,0,1)


    importer = data_io.SourceImageImporter('/home/sam/code/fisherman/data/dense_labelling/Shock(23)_15_cfos_06_training.tiff')
    importer.set_channels_of_interest((1,2))
    importer.set_transpose(2,0,1)
    im = importer.import_image().astype(numpy.float32)

    chunker_params = {
        'chunk_size': 225,
        'stride': 1,
        'window_size': 101,
        'num_classes': 2
    }

    chunker = detection.ImageChunkerWithOutput(im, **chunker_params)

    caffe.set_mode_cpu()
    net = caffe.Net(net_path, model_path, caffe.TEST)
    net.blobs['data'].reshape(1, *chunker.get_input_shape())

    print chunker.get_input_shape()

    print "Imported Image."
    print "Image shape: ", im.shape
    print "Image dtype: ", im.dtype

    i = 0.0
    total_iters = numpy.product(chunker._get_chunk_grid_shape())
    output = chunker.allocate_output()
    for in_chunk, out_chunk in chunker:
        print "{}% Complete...".format(100*i/total_iters)
        out_chunk[...] = net.forward_all(data=in_chunk[numpy.newaxis, ...])['prob'][...]
        i += 1

    output_mask = output.argmax(0)
    print "Mask shape: ", output_mask.shape
    print "Output shape: ", output.shape
    print "Max Output: ", output_mask.max()
    print "Mask mean: ", output_mask.mean()

    hfile = h5py.File('fish_filter_output.hdf5', 'w')
    hfile.create_dataset("output", data=output)
    hfile.close()
    
    io.imsave('fish_filter_mask_out.png', output_mask.astype(numpy.uint8))

    figure()
    imshow(im[0,...]/im[0,...].max())

    figure()
    imshow(output_mask)

    show()


main()
