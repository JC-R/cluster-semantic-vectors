import os
import shutil

import ngtpy
import numpy as np

import glob
from time import process_time, time
from numbers import Number
import codecs, json
import zipfile as z
import numpy as np
import sys
import argparse
import math
import datetime
import re


class autovivify_list(dict):
    '''Pickleable class to replicate the functionality of collections.defaultdict'''

    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


def build_word_vector_matrix(vector_file, re_filter, client):

    labels = []
    # input is expected to be a zip file of individual numpy vectors, one per file
    with np.load(vector_file, mmap_mode='r') as data:
        for fname, f in zip(data.files, data.zip.filelist):
            if re_filter is None or re_filter.match(fname):
                v = np.frombuffer(data.zip.read(f), dtype=np.float32).astype(float)
                client.insert(v)
                labels.append(fname)
    return labels


def find_word_clusters(labels_array, cluster_labels):
    '''Read in the labels array and clusters label and return the set of words in each cluster'''
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[str(i)].append(labels_array[c])
    return cluster_to_words


if __name__ == "__main__":

    print(datetime.datetime.now())
    t0 = process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--is_dir", action="store_true", help="Input is a dir or wildcard")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--distance", type=str, default='L2')
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dimensions", type=int, default=768)
    args = parser.parse_args()

    re_pattern = re.compile(args.filter) if args.filter else None

    if os.path.isdir(args.output):
        print("Output directory/index already exists", end="")
        if not args.overwrite:
            exit(-1)
        else:
            print("...overwriting")
            shutil.rmtree(args.output, ignore_errors=True)

    os.makedirs(args.output, exist_ok=True)
    index_name = "{}/index.anng".format(args.output)

    # create and open an index
    ngtpy.create(path=index_name, dimension=args.dimensions, distance_type=args.distance)
    index = ngtpy.Index(index_name)

    # create a map file
    fout = open("{}/labels.csv".format(args.output), 'w')

    # insert the objects
    if args.is_dir:
        label_idx = 0
        totvectors = 0
        for filepath in glob.iglob(args.input):
            labels = build_word_vector_matrix(filepath, re_pattern, index)
            totvectors += len(labels)
            print("{} - {} vectors -> {}".format(filepath, len(labels), totvectors))
            for idx, q in enumerate(labels):
                fout.write("{},{}\n".format(label_idx + idx, q))
            label_idx += len(labels)
    else:
        labels = build_word_vector_matrix(args.input, None, index)
        for idx, q in enumerate(labels):
            print("{} - {} vectors".format(args.input, len(labels)))
            fout.write("{},{}\n".format(idx, q))

    # save the index.
    index.save()
    index.close()
    print(datetime.datetime.now())