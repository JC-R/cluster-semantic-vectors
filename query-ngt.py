import argparse
import datetime
from time import process_time

import ngtpy
import numpy as np

if __name__ == "__main__":

    print(datetime.datetime.now())
    t0 = process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("--query_map", type=str, required=True)
    parser.add_argument("--vector_map", type=str, required=True)
    parser.add_argument("--queries", type=str)
    args = parser.parse_args()

    index = ngtpy.Index(args.index)

    vmap = {}
    qmap = {}

    # map vector id to query id
    with open(args.vector_map, 'r') as f:
        for line in f:
            idx, d = line.strip().split(",")
            docid = d[0:-2]
            id = int(idx)
            vmap[id] = docid
            qmap[docid] = None
    print("{} vectors - ".format(len(vmap)), end="")

    # map query id to query
    with open(args.query_map, 'r') as f:
        for line in f:
            docid, url, query = line.strip().split(",")
            if docid in qmap:
                qmap[docid] = query
    # print("{} query-map".format(len(qmap)))

    print(datetime.datetime.now())

