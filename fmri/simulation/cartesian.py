"""Cartesian data simulation."""

import numpy as np
from scipy.stats import norm

from ..operators.fourier import CartesianSpaceFourier
from .utils import validate_rng


def get_kspace_slice_loc(
    dim_size,
    center_prop,
    accel=4,
    pdf="gaussian",
    rng=None,
):
    """Get slice index at a random position.

    Parameters
    ----------
    dim_size: int
        Dimension size
    center_prop: float
        Proportion of center of kspace to continuouly sample
    accel: float
        Undersampling/Acceleration factor
    pdf: str, optional
        Probability density function for the remaining samples.
        "gaussian" (default) or "uniform".
    rng: random state

    Returns
    -------
    np.ndarray: array of size dim_size/accel.
    """
    indexes = list(range(dim_size))
    center_start = int(dim_size * (0.5 - center_prop / 2))
    center_stop = int(dim_size * (0.5 + center_prop / 2))

    center_indexes = indexes[center_start:center_stop]
    borders = np.asarray([*indexes[:center_start], *indexes[center_stop:]])

    n_samples_borders = int(dim_size / accel - len(center_indexes))

    rng = validate_rng(rng)

    if pdf == "gaussian":
        p = norm.pdf(np.linspace(norm.ppf(0.001), norm.ppf(0.999), len(borders)))
        p /= np.sum(p)
    elif pdf == "uniform":
        p = np.ones(len(borders)) / len(borders)
    else:
        raise ValueError("Unsupported value for pdf.")
    # TODO:
    # allow custom pdf as argument (vector or function.)

    sampled_in_border = list(rng.choice(borders, size=n_samples_borders, replace=False))

    return np.array(sorted(center_indexes + sampled_in_border))


def get_cartesian_mask(
    shape,
    n_frames,
    rng=None,
    constant=False,
    center_prop=0.3,
    accel=4,
    pdf="gaussian",
):
    """
    Get a cartesian mask for fMRI kspace data.

    Parameters
    ----------
    shape: tuple
        shape of fMRI volume.
    n_frames: int
        number of frames.
    rng: Generator or int or None (default)
        Random number generator or seed.
    constant: bool
        If True, the mask is constant across time.
    center_prop: float
        Proportion of center of kspace to continuouly sample
    accel: float
        Undersampling/Acceleration factor
    pdf: str, optional
        Probability density function for the remaining samples.
        "gaussian" (default) or "uniform".
    rng: random state
    Returns
    -------
    np.ndarray: random mask for an acquisition.
    """
    rng = validate_rng(rng)

    mask = np.zeros((*shape, n_frames))
    if constant:
        mask_loc = get_kspace_slice_loc(shape[-1], center_prop, accel, pdf, rng)
        mask[..., mask_loc, :] = 1
        return mask

    for i in range(n_frames):
        mask_loc = get_kspace_slice_loc(shape[-1], center_prop, accel, pdf, rng)
        mask[..., mask_loc, i] = 1
    return mask


# TODO: add mask for classical k-t methods (CAIPI, etc..)


def simulate_kspace_data(volume_sequence, n_coils, smaps=None, mask=None):
    """Get the observed kspace_data from reference."""
    fourier_op = CartesianSpaceFourier(
        volume_sequence.shape[:-1],
        mask=mask,
        n_coils=n_coils,
        n_frames=volume_sequence.shape[-1],
        smaps=smaps,
    )

    return fourier_op.op(volume_sequence)
