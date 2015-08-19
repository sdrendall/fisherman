import javabridge
import bioformats
import caffe
import numpy
from bioformats import log4j
from fisherman import detection, math
from pylab import figure, imshow, show, cm, title
from skimage import img_as_float, io
from os import path


def median_normalize(image):
    #math.map_to_axis(lambda im: im/numpy.median(im), image, 2)
    return image/numpy.median(image)


def cast_to_uint8(image):
    return (img_as_float(image) * 255).astype(numpy.uint8)


def main():
    from sys import argv
    if len(argv) < 3:
        print "Insufficient Arguments!"
        print "Proper Usage: %s [vsi_path] [model_path]" % argv[0]
        return
    
    fisherman_path = path.expanduser('~/code/fisherman')
    net_path = path.join(fisherman_path, 'caffe/fish_net_conv_deploy.prototxt')
    model_path = path.expanduser(argv[2])

    fish_net = caffe.Net(net_path, model_path, caffe.TEST)
    chunker_params = {
        'chunk_size': 954,
        'window_size': 49
    }

    # Start and configure jvm
    javabridge.start_vm(class_path=bioformats.JARS)
    log4j.basic_config()

    print "Loading Image..."
    with bioformats.ImageReader(path=path.expanduser(argv[1])) as reader:
        cfos = reader.read(c=1, rescale=False)
        cfos = (median_normalize(cfos) * 25).astype(numpy.uint8)

        #dapi = reader.read(c=0, rescale=False)
        #dapi = cast_to_uint8(math.normalize_array(dapi))

    #input_image = numpy.dstack((cfos, dapi))
    input_image = cfos

    javabridge.kill_vm()

    print "Casting Fish Net..."
    detector = detection.CellDetector(
        image=input_image, 
        net=fish_net,
        chunker_params=chunker_params,
        cell_radius=12)

    detector.set_mode_gpu()

    raw_mask = detector.get_fish_net_mask()

    print "Post processing mask..."
    clean_mask = detector.clean_fish_net_mask(raw_mask)

    raw_mask = detector.scale_mask_to_image(raw_mask)
    print "Image shape: ", input_image.shape
    print "Raw mask shape: ", raw_mask.shape
    
    figure()
    imshow(cfos, cmap=cm.Greys_r)
    imshow(raw_mask, alpha=.35, cmap=cm.Reds)
    title('Raw Mask')

    raw_mask = None
    clean_mask = detector.scale_mask_to_image(clean_mask)

    figure()
    imshow(cfos, cmap=cm.Greys_r)
    imshow(clean_mask, alpha=.35, cmap=cm.Reds)
    title('Clean Mask')

    show()

    return
    
    io.imsave('raw_mask_out.png', numpy.dstack([raw_mask.astype(numpy.uint8), input_image]))

    io.imsave('clean_mask_out.png', numpy.dstack([clean_mask.astype(numpy.uint8), input_image]))


main()
