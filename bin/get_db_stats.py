import caffe
import lmdb
import numpy
from fisherman import data_io, math
from os import path
from itertools import islice, imap


def main():
    from sys import argv
    if len(argv) < 2:
        print "Insufficient Arguments!"
        print "Proper Usage: %s [db_path]" % argv[0]
        return
        
    #db_path = path.expanduser(argv[1])
    #db = lmdb.open(db_path, readonly=True)
    
    #datum_data = data_io.get_datum_data(db)
    channel_stacks = zip(*[numpy.dsplit(datum_array, datum_array.shape[2]) for datum_array in datum_data])

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
