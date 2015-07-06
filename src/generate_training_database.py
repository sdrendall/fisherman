__author__ = 'sdrendall'

import data_io
from os import path
from sys import argv    
import generators as gen

def main():
    if len(argv) < 3:
        print "Insufficient Arguments!"
        print argv[0] + " [prefix_file] [db_path]"
        return
    
    # Create DB
    mapsize = 20*(2**32 - 1)
    db_path = path.expanduser(argv[2])
    db_factory = data_io.LMDatabaseFactory(db_path, mapsize)

    prefix_file_path = path.expanduser(argv[1])
    prefix_file = open(prefix_file_path)
    source_dir_path = path.dirname(prefix_file_path)

    tag_generator = gen.counter()

    # Add each training set to the training db
    for prefix in prefix_file:
        prefix = path.basename(prefix.strip())
        label_path = path.join(source_dir_path, prefix + '.xml')
        image_path = path.join(source_dir_path, prefix + '.tif')

        # Load JSON
        label_importer = data_io.LabelImporter(label_path)
        labels = label_importer.import_data()

        # Load Image
        image_importer = data_io.SourceImageImporter(image_path)
        image_importer.set_channels_of_interest((1,))
        image = image_importer.import_image()

        # Create TrainingSet
        training_set = data_io.TrainingSet(image, labels)
        training_set.set_tag_generator(tag_generator)

        # Add TrainingSet to the database
        db_factory.add_training_set(training_set)


main()
