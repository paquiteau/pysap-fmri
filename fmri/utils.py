"""Utils functions for fMRI data."""
import os

import numpy as np
from numpy.random import Generator, default_rng

MAX_CPU_CORE = len(os.sched_getaffinity(0))


def ssos(img, axis=0):
    """Compute the square root of sum of square."""
    return np.sqrt(np.sum(np.square(img), axis))


def fmri_ssos(img):
    """Apply the ssos on the first axis."""
    return ssos(img, axis=0)


class DimensionMismatchError(ValueError):
    """Custom Exception for Dimension mismatch."""

    pass


def validate_rng(rng=None):
    """Validate Random Number Generator."""
    if isinstance(rng, int):
        return default_rng(rng)
    elif rng is None:
        return default_rng()
    elif isinstance(rng, Generator):
        return rng
    else:
        raise ValueError("rng shoud be a numpy Generator, None or an integer seed.")
