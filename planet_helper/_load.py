import csv
import glob
import gzip
import io
import os.path as osp
import tarfile as tar

import numpy as np
from PIL import Image


def load_data():

    filedir = osp.dirname(__file__)
    im_gz_bytes = b''

    print('assembling data archives...')
    for im_gz_fn in sorted(glob.glob(osp.join(filedir, 'data/*.part'))):
        with open(im_gz_fn, 'rb') as f:
            im_gz_bytes = im_gz_bytes + f.read()

    tarfile = tar.open(mode='r:*', fileobj=io.BytesIO(im_gz_bytes))

    members = list(tarfile)

    WH = 40
    X = np.zeros((len(members), WH, WH, 4), dtype=np.uint8)

    print('loading data...')
    for ix, item in enumerate(members):
        bio = tarfile.extractfile(item)
        X[ix] = np.array(Image.open(bio).convert('RGBA'))

    print('loading labels...')
    with gzip.open(osp.join(filedir, 'data/train_v2.csv.gz'),
                   'rb') as f:  # type: ignore
        table = csv.reader(f.read().decode('utf-8').strip().split('\n'))

    label_list_raw = [x[1] for x in list(table)[1:]]
    labelset = {l for ls in label_list_raw for l in ls.split(' ')}
    labeldict = {k: v for v, k in enumerate(labelset)}

    Y = np.zeros((len(label_list_raw), len(labeldict)), dtype=np.uint8)
    for lx, raw_label in enumerate(label_list_raw):
        for label in raw_label.split(' '):
            Y[lx, labeldict[label]] = 1

    print('done')
    assert len(Y) == len(X)
    return X, Y


if __name__ == '__main__':
    load_data()
