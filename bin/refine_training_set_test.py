from fisherman import data_io, ui
from skimage import io, img_as_float
from numpy import apply_along_axis
import json
from os import path

def main():
    from sys import argv
    if len(argv) < 4:
        print "Insufficient Inputs!"
        print "%s [image_path] [labels_path] [json_output_path]" % argv[0]
        return

    image_path, labels_path, output_path = argv[1:4]
    
    image = img_as_float(io.imread(path.expanduser(image_path)))
    label_importer = data_io.LabelImporter(path.expanduser(labels_path))
    labels = label_importer.import_data()

    training_set = data_io.TrainingSet(image=image, labels=labels)
    training_set.set_example_shape((100,100))
    
    ui.refine_training_set(training_set)

    save_dict = {
        'original_image_path': image_path,
        'original_labels_path': labels_path,
        'labels': training_set.labels
    }

    json.dump(save_dict, open(output_path, 'w'))

main()
