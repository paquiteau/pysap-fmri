"""Utility functions for viz module."""

import numpy as np


def normalize(img):
    """Map any images between 0 and 1."""
    m = np.min(abs(img))
    M = np.max(abs(img))
    return (abs(img) - m) / (M - m)


def ssos(img, axis=0):
    """Compute the square root of sum of square."""
    return np.sqrt(np.sum(np.square(img), axis))


def fmri_ssos(img):
    """Apply the ssos on the first axis."""
    return ssos(img, axis=0)
