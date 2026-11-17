"""The two graded halves as [start, end) blocks, read from the published boundaries."""
import numpy as np


def half_blocks(f):
    g = f["graded"]
    edges = np.diff(np.concatenate(([0], g.astype(np.int8), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
