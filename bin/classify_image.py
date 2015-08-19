import caffe
import numpy
import matplotlib.pyplot as plt
import sys
from os import path
from skimage import io

fisherman_path = '/home/sam/Dropbox/code/fisherman'
image_path = path.join(fisherman_path, 'data/training_source_images/Shock(21)_13_cfos_01_PVT-dim.tif')
net_path = path.join(fisherman_path, 'caffe/fish_net_conv_deploy.prototxt')
model_path = path.join(fisherman_path, 'caffe/fish_net_conv_weights_cfos_only.caffemodel')

sys.path.append(fisherman_path)
from fisherman import detection

num_output_classes = 2
chunk_size = 350
window_size = 50
stride = 6

def display_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show(block=False)

def main():
    caffe.set_mode_gpu()
    net = caffe.Net(net_path, model_path, caffe.TEST)
    net.blobs['data'].reshape(1,1,chunk_size,chunk_size)
    image = caffe.io.load_image(image_path)
    cfos = image[:,:,1][numpy.newaxis, ...]
    chunker = detection.ImageChunkerWithOutput(
        cfos,
        num_classes=num_output_classes,
        window_size=window_size,
        chunk_size=chunk_size,
        stride=stride
    )

    output = chunker.allocate_output()
    for in_chunk, out_chunk in chunker:
        net_out = net.forward_all(data=in_chunk)
        out_chunk[...] = net_out['prob'][...]

    io.imsave(path.expanduser('~/Dropbox/code/fisherman/tile_out.tif'), output.argmax(0))
    
    caffe.set_mode_cpu()
    net.blobs['data'].reshape(1, *cfos.shape)
    full_out = net.forward_all(data=cfos)
    io.imsave(path.expanduser('~/Dropbox/code/fisherman/full_out.tif'), output.argmax(0))
    io.imsave(path.expanduser('~/Dropbox/code/fisherman/prob_pos.tif'), output[1,...])
    io.imsave(path.expanduser('~/Dropbox/code/fisherman/prob_neg.tif'), output[0,...])

main()
