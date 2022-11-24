"""Utilities for fMRI Operators."""

from modopt.opt.gradient import GradBasic

from fmri.utils import DimensionMismatchError


def validate_shape(shape, array):
    """Validate shape of array."""
    if array.shape != shape:
        raise DimensionMismatchError(
            f"array should have dimension {shape}, but has {array.shape}"
        )
    return array


def validate_smaps(shape, n_coils, smaps=None):
    """Raise Value Error if smaps does not fit dimensions."""
    if n_coils == 1 and not smaps:
        return 1
    if smaps is None:
        return None
    return validate_shape((n_coils, *shape), smaps)


def validate_mask(shape, n_frames=None, mask=None):
    """Raise ValueError if mask does not fit dimensions."""
    if hasattr(mask, "__len__"):
        if n_frames is not None:
            return validate_shape((*shape, n_frames), mask)
        else:
            return validate_shape(shape, mask)
    elif mask == 1 or mask is None:
        return 1
    return validate_shape((*shape, n_frames), mask)
