"""Utilities for fMRI Operators."""

import numpy as np
from modopt.opt.gradient import GradBasic


def validate_shape(shape, array):
    """Validate shape of array."""
    if array.shape != tuple(shape):
        raise ValueError(f"array should have dimension {shape}, but has {array.shape}")
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
    if hasattr(mask, "__len__"):
        if n_frames is not None:
            return validate_shape((*shape, n_frames), mask)
        else:
            return validate_shape(shape, mask)
    elif mask == 1 or mask is None:
        return 1
    return validate_shape((*shape, n_frames), mask)


def make_gradient_operator(fourier_op, obs_data):
    """Return a Gradient operator usable by Modopt."""
    if hasattr(fourier_op, "data_consistency"):
        grad_op = GradBasic(
            op=fourier_op.op,
            trans_op=fourier_op.adj_op,
            get_grad=lambda x: fourier_op.data_consistency(x, obs_data),
            input_data=obs_data,
        )
    else:
        grad_op = GradBasic(
            op=fourier_op.op,
            trans_op=fourier_op.adj_op,
            input_data=obs_data,
        )

    return grad_op


def sigma_mad(data):
    """Estimate Variance from the Median Absolute Deviation."""
    # TODO Make it faster with numba and assume the data is already sorted.
    return np.median(np.abs(data[:] - np.median(data[:]))) / 0.6745


def sure_est(data):
    """Return an estimation of the threshold computed using the SURE method.

    The computation of the estimator is based on the formulation of `cite:donoho1994`
    and the efficient implementation of [#]_

    Parameters
    ----------
    data: numpy.array
        Noisy Data with unit standard deviation.

    Returns
    -------
    float
        Value of the threshold minimizing the SURE estimator.

    References
    ----------
    .. [#] https://pyyawt.readthedocs.io/_modules/pyyawt/denoising.html#ValSUREThresh
    """
    dataf = data.flatten()
    n = dataf.size
    data_sorted = np.sort(np.abs(dataf)) ** 2
    idx = np.arange(n - 1, -1, -1)
    tmp = np.cumsum(data_sorted) + idx * data_sorted

    risk = (n - (2 * np.arange(n)) + tmp) / n
    ibest = np.argmin(risk)
    return np.sqrt(data_sorted[ibest])
