"""Atomic cache writes: a reader sees either no file or a complete one, even with parallel workers."""
import os
import pickle

import numpy as np


def save_npz(path, **arrays):
    tmp = "{}.{}.tmp.npz".format(path[:-4], os.getpid())
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def save_pickle(path, obj):
    tmp = "{}.{}.tmp".format(path, os.getpid())
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)
