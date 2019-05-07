import csv
import glob
import gzip
import io
import os.path as osp
import pickle
import tarfile as tar

import numpy as np
from PIL import Image


def load_train():
    """
    The training data is a split tar archive with png images.

    Don't ask...
    """

    filedir = osp.dirname(__file__)
    pickle_cache_fn = osp.expanduser("~/planet_train_cache.p")

    if osp.exists(pickle_cache_fn):
        print("found cached files, loading them.")
        with open(pickle_cache_fn, "rb") as f:
            return pickle.load(f)

    print("assembling data archives...")
    im_gz_bytes = b""
    for im_gz_fn in sorted(glob.glob(osp.join(filedir, "data/*train*.part"))):
        with open(im_gz_fn, "rb") as f:
            im_gz_bytes += f.read()

    tarfile = tar.open(mode="r:*", fileobj=io.BytesIO(im_gz_bytes))

    members = list(tarfile)

    WH = 40
    X = np.zeros((len(members), WH, WH, 4), dtype=np.uint8)

    print("loading data...")
    for ix, item in enumerate(members):
        bio = tarfile.extractfile(item)
        X[ix] = np.array(Image.open(bio).convert("RGBA"))

    print("loading labels...")
    with gzip.open(
        osp.join(filedir, "data/train_v2.csv.gz"), "rb"
    ) as f:  # type: ignore
        table = csv.reader(f.read().decode("utf-8").strip().split("\n"))

    label_list_raw = [x[1] for x in list(table)[1:]]
    unique_labels = sorted({l for ls in label_list_raw for l in ls.split(" ")})
    label_indices = {k: v for v, k in enumerate(unique_labels)}

    Y = np.zeros((len(label_list_raw), len(label_indices)), dtype=np.uint8)
    for lx, raw_label in enumerate(label_list_raw):
        for label in raw_label.split(" "):
            Y[lx, label_indices[label]] = 1

    print("done")
    assert len(Y) == len(X)

    out = X, Y, unique_labels
    with open(pickle_cache_fn, "wb") as f:
        pickle.dump(out, f)
    return out


# for compatibility
load_data = load_train


def load_test():
    """
    The test data is a much more sensible split npz array.
    """
    filedir = osp.dirname(__file__)
    npz_cache_fn = osp.expanduser("~/planet_test_cache.npz")

    if osp.isfile(npz_cache_fn):
        print("found cached file, loading it.")
        return np.load(npz_cache_fn)["X_test"]

    npz_gz_bytes = b""

    print("assembling data archives...")
    for im_gz_fn in sorted(glob.glob(osp.join(filedir, "data/*test*.part"))):
        with open(im_gz_fn, "rb") as f:
            npz_gz_bytes += f.read()

    X_test = np.load(io.BytesIO(gzip.decompress(npz_gz_bytes)))["arr_0"]
    np.savez(npz_cache_fn, X_test=X_test)

    return X_test


if __name__ == "__main__":
    load_data()
    X = load_test()
    print(X.shape)
    print(X[0])
