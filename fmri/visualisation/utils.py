import numpy as np


def normalize(img):
    """Map any images between 0 and 1."""

    m = np.min(abs(img))
    M = np.max(abs(img))
    return (abs(img) - m) / (M - m)
