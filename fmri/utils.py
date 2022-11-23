"""Utils functions for fMRI data."""
import numpy as np
import os


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
