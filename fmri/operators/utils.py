"""Utilities for fMRI Operators."""

import numpy as np


class DimensionMismatchError(ValueError):
    """Custom Exception for Dimension mismatch."""

    pass


def validate_shape(shape, array):
    """Validate shape of array."""
    if array.shape != shape:
        raise DimensionMismatchError(
            f"array should have dimension {shape}, but has {array.shape}"
        )
    return array


def validate_smaps(shape, n_coils, smaps=None):
    """Raise Value Error if smaps does not fit dimensions."""
    if n_coils == 1:
        return 1
    if smaps is None:
        return None
    return validate_shape((n_coils, *shape), smaps)


def validate_mask(shape, n_frames=None, mask=None):
    """Raise ValueError if mask does not fit dimensions."""
    if mask == 1 or mask is None:
        return 1

    if n_frames is None:
        return validate_shape(shape, mask)
    return validate_shape((*shape, n_frames), mask)
