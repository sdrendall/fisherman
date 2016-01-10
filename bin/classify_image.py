import caffe
import numpy
import matplotlib.pyplot as plt
import sys
from os import path
from skimage import io

fisherman_path = '/home/sam/Dropbox/code/fisherman'
image_path = path.join(fisherman_path, 'data/training_source_images/Shock(21)_13_cfos_01_PVT-dim.tif')
net_path = path.join(fisherman_path, 'caffe/fish_net_conv_deploy.prototxt')
model_path = path.join(fisherman_path, 'models/median_normalized/fish_net_conv_deploy_weights.caffemodel')

sys.path.append(fisherman_path)
from fisherman import detection

num_output_classes = 2
chunk_size = 354
window_size = 49
stride = 6

def display_image(image, block=True):
    plt.figure()
    plt.imshow(image)
    plt.show(block=block)

def main():
    caffe.set_mode_gpu()

    print "Configuring Net"
    net = caffe.Net(net_path, model_path, caffe.TEST)

    print "Reshaping input"
    net.blobs['data'].reshape(1,1,chunk_size,chunk_size)

    print "Loading %s" % image_path
    image = caffe.io.load_image(image_path)
    cfos = image[:,:,1][numpy.newaxis, ...]

    print "Configuring Chunker"
    chunker = detection.ImageChunkerWithOutput(
        cfos,
        num_classes=num_output_classes,
        window_size=window_size,
        chunk_size=chunk_size,
        stride=stride
    )

    chunker_params = {
        'chunk_size': chunk_size,
        'window_size': window_size,
        'stride': stride,
        'num_classes': num_output_classes
    }

    cell_detector = detection.CellDetector(net=net, cell_radius=12, signal_channel=1, chunker_params=chunker_params)
    cell_detector.set_image(image)
    cells = cell_detector.detect_cells()
    detector_mask = cell_detector.get_fish_net_mask(scaled=False)
    scaled_clean_mask = cell_detector.get_fish_net_mask(cleaned=True, scaled=True)

    display_image(detector_mask, block=False)
    display_image(scaled_clean_mask, block=True)

    print "Computing by chunks"
    output = chunker.allocate_output()
    for in_chunk, out_chunk in chunker:
        net_out = net.forward_all(data=in_chunk)
        out_chunk[...] = net_out['prob'][...]

    
    display_image(output.argmax(0))
    io.imsave(path.expanduser('~/Desktop/tile_out.tif'), output.argmax(0))

    return
    
    print "Computing serially"
    caffe.set_mode_cpu()
    net.blobs['data'].reshape(1, *cfos.shape)
    full_out = net.forward_all(data=cfos)
    io.imsave(path.expanduser('~/Desktop/full_out.tif'), output.argmax(0))
    io.imsave(path.expanduser('~/Desktop/prob_pos.tif'), output[1,...])
    io.imsave(path.expanduser('~/Desktop/prob_neg.tif'), output[0,...])

main()
