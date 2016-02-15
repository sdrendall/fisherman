__author__ = 'sdrendall'

import numpy
import random
import fnmatch
from fisherman import data_io, math
from fisherman import amplification as amp
from fisherman import generators as gen
from skimage import img_as_float
from os import path, walk
from sys import argv    
from glob import glob
import xml.etree.ElementTree as ET
from pylab import figure, hist, show
from functools import partial
from itertools import product


def median_normalize(image):
    #math.map_to_axis(lambda im: im/numpy.median(im), image, 2)
    return image/numpy.median(image)


def configure_training_set_amplifier():
    flip_amp = amp.TrainingExampleAmplifier(
        amplifier_fcns=[numpy.fliplr]
    )
    rot_amp = amp.TrainingExampleAmplifier(
        amplifier_fcns=[partial(numpy.rot90, k=k) for k in range(1, 4)]
    )
    return amp.AmplifierChain([flip_amp, rot_amp])


def configure_label_amplifier():
    translations = product((7, -7), (7, -7))
    translation_fcns = (partial(translate_label, trans=trans) for trans in translations)
    return amp.SequenceAmplifier(translation_fcns)
    

def translate_label(label, trans):
    lab_out = (label[0] + numpy.asarray(trans), label[1])
    return lab_out


def search_for_matching_image(start_path, image_name):
    image_name = image_name.split(' ')[-1]
    for dirpath, dirnames, filenames in walk(start_path):
        matching_names = fnmatch.filter(filenames, image_name)
        if len(matching_names) == 1:
            print dirpath
            return path.join(dirpath, matching_names[0])
        elif len(matching_names) > 1:
            print "Multiple matches found for %s!" % image_name
            return path.join(dirpath, matching_names[0])

    print "No matching name found for %s" % image_name
    return None

def name_tag_generator(basename):
    counter = gen.counter()
    for number in counter:
        yield '%s_%d' % (basename, number)

def main():
    if len(argv) < 5:
        print "Insufficient Arguments!"
        print argv[0] + " [xml_files] [db1] [db2] [db1_size/(db1_size + db2_size)]"
        return
    
    # Create DBs
    
    mapsize = 20*(2**32 - 1)
    db1_path = path.expanduser(argv[-3])
    db2_path = path.expanduser(argv[-2])
    
    split_ratio = float(argv[-1])

    db1_factory = data_io.LMDatabaseFactory(db1_path, mapsize)
    db2_factory = data_io.LMDatabaseFactory(db2_path, mapsize)

    #xml_file_paths = glob(path.expanduser(argv[1]))
    xml_file_paths = map(path.expanduser, argv[1:-3])

    # Add each training set to the training db
    for xml_path in xml_file_paths:
        print xml_path
        root = ET.parse(xml_path)
        xml_dir = path.dirname(xml_path)
        image_basename = root.find('Image_Properties').find('Image_Filename').text
        image_filename = search_for_matching_image(xml_dir, path.basename(image_basename))
        #image_path = path.join(xml_dir, image_filename)

        if image_filename is None:
            continue
        else:
            image_path = image_filename

        # Load Labels
        label_importer = data_io.LabelImporter(xml_path)
        #label_amplifier = configure_label_amplifier()
        labels = label_importer.import_data()

        # Load Image
        image_importer = data_io.SourceImageImporter(image_path)
        image_importer.set_channels_of_interest((1,))
        #image_importer.set_transpose(1, 2, 0)
        image = image_importer.import_image().astype(numpy.float32)
        #image = median_normalize(image) * 25
        #image = (img_as_float(image) * 255).astype(numpy.uint8)

        overflow = (image[..., 0] > 255).sum().astype(numpy.float32)/image[..., 0].size

        print '----------------------------------'
        print "Max: ", image[..., 0].max()
        print "Min: ", image[..., 0].min()
        print "Overflow: ", overflow
        print '----------------------------------\n'

        # Create TrainingSet
        tag_generator = name_tag_generator(path.basename(xml_path))
        training_set = data_io.TrainingSet(image.astype(numpy.uint8), labels, example_shape=(49,49))
        training_set.set_tag_generator(tag_generator)

        #amplifier = configure_training_set_amplifier()

        # Add TrainingSet to the database
        to_db1 = list()
        to_db2 = list()
        for example in training_set:
            if random.random() < split_ratio:
                to_db1.append(example)
            else:
                to_db2.append(example)

        db1_factory.add_training_set(to_db1)
        db2_factory.add_training_set(to_db2)

if __name__ == "__main__":
    main()
