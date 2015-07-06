import lmdb
import random

def write_kv_pair_to_db(key, value, db):
    with db.begin(write=True) as txn:
        txn.put(key, value)

def main():
    from sys import argv

    if len(argv) < 5:
        print "Insufficient Argments!"
        print "Usage: %s [master_db] [db1] [db2] [size_db1/size_master]"
        return

    master_path = argv[1]
    db1_path = argv[2]
    db2_path = argv[3]
    split_ratio = float(argv[4])

    mapsize = 2**32 - 1

    master_db = lmdb.open(master_path, readonly=True)
    db1 = lmdb.open(db1_path, mapsize)
    db2 = lmdb.open(db2_path, mapsize)

    db1_count = 0
    db2_count = 0

    with master_db.begin() as txn:
        for key, value in txn.cursor():
            if random.random() < split_ratio:
                write_kv_pair_to_db(key, value, db1)
                db1_count += 1
            else:
                write_kv_pair_to_db(key, value, db2)
                db2_count += 1


    print "DB1 count: %d" % db1_count
    print "DB2 count: %d" % db2_count

main()
