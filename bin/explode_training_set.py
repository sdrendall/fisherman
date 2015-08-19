from fisherman import data_io
from xml.etree import ElementTree as ET
from os import path

def main():
    from sys import argv
    if len(argv) < 2:
        print "Insufficient Arguments!"
        print "Proper Usage: %s [xml_path] [optional:image_path]"
        return

    xml_path = path.expanduser(argv[1])

    if len(argv) < 3:
        im_dir = path.dirname(xml_path)
        im_path = get_image_path_from_xml(xml_path)
    else:
        im_path = path.expanduser(argv[2])

    label_importer = data_io.LabelImporter(xml_path)
    labels = label_importer.import_data()

    image_importer = data_io.SourceImageImporter(im_path)
    image_importer.set_channels_of_interest((0,))
    imge_importer.set_transpose(2,0,1)
    image = image_importer.import_image(normalize=True)

    training_set = data_io.TrainingSet(image, labels, example_shape=(49,49))

    for i, example in enumerate(training_set):
        example.write('./example_%d_%d.png' % (int(example.label), i))


def get_image_path_from_xml(xml_path):
    root = ET.parse(xml_path)
    xml_dir = path.dirname(xml_path)
    image_filename = root.find('Image_Properties').find('Image_Filename').text
    return path.join(xml_dir, image_filename)

if __name__ == '__main__':
    main()
