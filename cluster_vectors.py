from __future__ import division

import glob
from typing import List, Any

from sklearn.cluster import MiniBatchKMeans, KMeans
from time import process_time, time
from numbers import Number
from pandas import DataFrame
import codecs, json
from milvus import Milvus, DataType
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


def build_word_vector_matrix(vector_file, re_filter):
    embeddings = []
    labels = []

    # input is expected to be a zip file of individual numpy vectors, one per file
    with np.load(vector_file, mmap_mode='r') as data:
        for fname, f in zip(data.files, data.zip.filelist):
            if re_filter and re_filter.match(fname):
                v = np.frombuffer(data.zip.read(f), dtype=np.float32).astype(float)
                embeddings.append(v)
                labels.append(fname)
    return embeddings, labels


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
    parser.add_argument("--k", type=int)
    parser.add_argument("--reduction", type=float)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=1000)
    parser.add_argument("--method", type=str, default="kmeans")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    re_pattern = re.compile(args.filter) if args.filter else None

    if args.reduction and args.k:
        print("Only one of [k, reduction] can be used.")
        sys.exit(-1)
    if not args.k and not args.reduction:
        print("Need one of [k, reduction]")
        sys.exit(-1)
    output = args.output if args.output else args.method

    df = None
    if args.is_dir:
        for filepath in glob.iglob(args.input):
                df1, labels1 = build_word_vector_matrix(filepath, re_pattern)
                print(filepath)
                if len(df1) == 0:
                    continue
                if df is None:
                    df = np.array(df1)
                    labels_array = labels1
                else:
                    df = np.append(df, df1, axis=0)
                    labels_array.append(labels1)

    else:
        df, labels_array = build_word_vector_matrix(args.input, None)

    k = args.k if args.k else int(math.floor(args.reduction * len(df)))

    if args.method == 'kmeans':
        kmeans_model = KMeans(init='k-means++', n_clusters=k, verbose=args.verbose, n_jobs=-1, max_iter=500)

    else:
        kmeans_model = MiniBatchKMeans(init='k-means++', n_clusters=k, verbose=args.verbose,
                                       batch_size=args.batchsize)

    print("Method: {}, k: {}, {} entities".format(args.method, k, len(df)))
    kmeans_model.fit(np.array(df))

    cluster_labels = kmeans_model.labels_
    cluster_inertia = kmeans_model.inertia_
    cluster_to_words = find_word_clusters(labels_array, cluster_labels)

    with open("clusters_words.{}.json".format(output), 'w') as json_out:
        json.dump(cluster_to_words, json_out)

    print(datetime.datetime.now())
    print(process_time() - t0)

    # for c in cluster_to_words:
    #     print (cluster_to_words[c])
    #     print("clusters_words\n")
