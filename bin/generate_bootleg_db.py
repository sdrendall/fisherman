
__author__ = 'sdrendall'

import numpy
from fisherman import data_io, math
from fisherman import generators as gen
from os import path
from sys import argv    
from glob import glob
import xml.etree.ElementTree as ET

def main():
    if len(argv) < 3:
        print "Insufficient Arguments!"
        print argv[0] + " [xml_files] [db_path]"
        return
    
    # Create DB
    mapsize = 20*(2**32 - 1)
    db_path = path.expanduser(argv[2])
    db_factory = data_io.LMDatabaseFactory(db_path, mapsize)
    db_factory.set_regularization_function(regularization_fcn)

    tag_generator = gen.counter(start=2297)

    xml_file_paths = glob(path.expanduser(argv[1]))

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
        image = (image_importer.import_image(normalize=True) * 255).astype(numpy.uint8)
        

        # Create TrainingSet
        training_set = data_io.TrainingSet(image, labels, example_shape=(49,49))
        training_set.set_tag_generator(tag_generator)

        # Add TrainingSet to the database
        db_factory.add_training_set(training_set)

    print "%d examples added to the database" % (tag_generator.next())


def regularization_fcn(im):
    channel_params = [
        {
            'mean': 16.7253673112,
            'range': 79.3489
        },
        {
            'mean': 14.9357596068,
            'range': 103.05703091
        }
    ]

    reg_im = numpy.dstack(
        map(lambda ch: math.mean_range_regularization(ch, params['mean'], params['range']), channel)
            for params, channel in zip(channel_params, numpy.dsplit(im, im.shape[2]))
    )

    return (64*(reg_im + 64)).astype(numpy.uint8)


main()
