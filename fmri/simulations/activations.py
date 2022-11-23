"""Add Activations."""
import numpy as np


def repeat_volume(array, n_frames):
    """Repeat array n_frames times along the last axis."""
    return np.repeat(array[..., np.newaxis], n_frames, axis=-1)
