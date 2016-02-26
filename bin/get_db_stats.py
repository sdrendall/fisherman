import caffe
import lmdb
import numpy
from fisherman import data_io, math
from os import path
from itertools import islice, imap, izip


def main():
    from sys import argv
    if len(argv) < 2:
        print "Insufficient Arguments!"
        print "Proper Usage: %s [db_path]" % argv[0]
        return
        
    db_path = path.expanduser(argv[1])
    db = lmdb.open(db_path, readonly=True)

    moments = []
    positive_count = 0.0
    with db.begin() as txn:
        for _, datum_str in txn.cursor():
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(datum_str)
            example_image = caffe.io.datum_to_array(datum)
            positive_count += datum.label
            moments.append((
                example_image.astype(numpy.float64).mean(),
                example_image.astype(numpy.float64).var() + example_image.astype(numpy.float64).mean()**2
            ))

    first, second = map(numpy.asarray, izip(*moments))
    
    mean = first.mean()
    var = second.mean() - mean**2
    std = var**0.5
    scale = 1/std

    print "Mean: ", mean
    print "Var: ", var
    print "Std: ", std
    print "Scale: ", scale
    print "16 bit scale", scale/2**16
    print "8 bit scale", scale/2**8
    print "Positive Samples: ", positive_count, positive_count/len(first)
    print "Negative Samples: ", len(first) - positive_count, (len(first) - positive_count)/len(first)
    print "Total Samples: ", len(first)

main()
