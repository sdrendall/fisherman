import caffe
import numpy
import javabridge
import bioformats
import skimage
from bioformats import log4j
from fisherman import detection, math
from skimage import io, color
from os import path
from pylab import figure, imshow, show, cm


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

def normalize(A):
    return (A - A.min())/(A - A.min()).max()


def main():
    from sys import argv
    if len(argv) < 3:
        print "Not Enough Arguments!!"
        print "Proper Usage: %s [path_to_image] [model_path] [optional: save_path]" % argv[0]
        return
    elif len(argv) >= 4:
        save_path = path.expanduser(argv[3])
    else:
        save_path = None

    im_path = path.expanduser(argv[1])
    model_path = path.expanduser(argv[2])
    net_path = path.expanduser('~/code/fisherman/caffe/fish_net_conv_deploy.prototxt')

    fish_net = caffe.Net(net_path, model_path, caffe.TEST)

    chunker_params = {
        'chunk_size': 954,
        'stride': 6,
        'window_size': 49,
        'num_classes': 2
    }

    detector = detection.CellDetector(
        net=fish_net,
        chunker_params=chunker_params,
        signal_channel=0,
        cell_radius=12
    )

    detector.set_mode_gpu()

    javabridge.start_vm(class_path=bioformats.JARS)
    log4j.basic_config()
    
    image = load_vsi(im_path)
    #image = io.imread(im_path).transpose(1, 2, 0)
    #image = math.median_normalize(image[..., 0]) * 25
    #image = image.astype(numpy.uint8)

    javabridge.kill_vm()

    detector.set_image(image)

    mask = detector.get_fish_net_mask(cleaned=True, scaled=True)
    io.imsave('mask_out.png', mask)
    
    labels = detector.separate_cell_mask(mask)
    print 'Detected %d cells!' % len(numpy.unique(labels))

    display_image = normalize(numpy.log(numpy.finfo(numpy.float64).eps + image.astype(numpy.float64)))
    io.imsave('labels_out.png', color.label2rgb(labels[..., 0], image=display_image, bg_label=0))

    return

    mean_image = detector.get_mean_image()
    figure()
    imshow(mean_image)

    figure()
    imshow(image, cmap=cm.Greys_r)
    imshow(mask, cmap=cm.Reds, alpha=.35)

    figure()
    imshow(color.label2rgb(labels[..., 0], image=image))
    show()


    if save_path is not None:
        print "Saving image ....."
        io.imsave(save_path, color.label2rgb(cell_mask, image=image))


main()
