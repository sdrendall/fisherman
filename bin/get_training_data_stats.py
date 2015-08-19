import numpy
from fisherman import data_io, math
from os import path
from glob import glob
import xml.etree.ElementTree as ET
from skimage import io


def main():
    from sys import argv
    if len(argv) < 2:
        print "Insufficient Arguments!"
        print "Proper Usage: %s [xml_files]"
        return

    xml_file_paths = glob(path.expanduser(argv[1]))

    training_images = []
    # Add each training set to the training db
    for xml_path in xml_file_paths:
        root = ET.parse(xml_path)
        xml_dir = path.dirname(xml_path)
        image_filename = root.find('Image_Properties').find('Image_Filename').text
        image_path = path.join(xml_dir, image_filename)

        # Load Labels
        label_importer = data_io.LabelImporter(xml_path)
        labels = label_importer.import_data()

        # Load Image
        image_importer = data_io.SourceImageImporter(image_path)
        image_importer.set_channels_of_interest((1, 2))
        image = image_importer.import_image()

        # Get training images
        training_set = data_io.TrainingSet(image, labels, example_shape=(49,49))
        training_images += [example.image for example in training_set]

    
    channel_stacks = zip(*[numpy.dsplit(ex_im, ex_im.shape[2]) for ex_im in training_images])

    for i, stack in enumerate(channel_stacks):
        print "Channel %d stats:" % i
        channel_stats = [(A.mean(), A.max() - A.min()) for A in stack]
        channel_mean, channel_range = map(
            math.mean, 
            zip(*channel_stats)
        )
        print "Mean: {}".format(channel_mean)
        print "Range: {}".format(channel_range)
        print "-----------------"

main()
