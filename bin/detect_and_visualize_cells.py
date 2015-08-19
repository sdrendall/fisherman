import caffe
import numpy
from fisherman import detection, math
from skimage import io, color
from os import path
from pylab import figure, imshow, show, cm


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
        'chunk_size': 354,
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
    
    image = io.imread(im_path).transpose(1, 2, 0)
    image = math.median_normalize(image[..., 0]) * 25
    image = image.astype(numpy.uint8)

    detector.set_image(image)

    mask = detector.get_fish_net_mask(cleaned=True, scaled=True)
    labels = detector.separate_cell_mask(mask)

    io.imsave('mask_out.png', mask)
    io.imsave('labels_out.png', color.label2rgb(labels[..., 0], image=image, bg_label=0))

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
