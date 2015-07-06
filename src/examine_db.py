import caffe
import lmdb
import numpy as np
import data_io
from os import path

def convert_raw_datum_to_training_example(raw_datum):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    flattened_image = np.fromstring(datum.data, dtype=np.uint8)
    image = flattened_image.reshape(datum.channels, datum.height, datum.width).swapaxes(0,2).swapaxes(1,2)
    label = datum.label 

    return data_io.TrainingExample(image, label)


def main():
    from sys import argv
    if len(argv) < 2:
        print "Proper Usage: %s [path/to/db]" % argv[0]

    env = lmdb.open(argv[1], readonly=True)

    with env.begin() as txn:
        cursor = txn.cursor()
        labels = [convert_raw_datum_to_training_example(value).label for _, value in cursor]

    print labels
    print np.mean(np.asarray(labels).flat)

main()
