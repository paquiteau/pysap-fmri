# -*- coding: utf-8 -*-
"""Singular Value Threshold operator."""
import warnings

import numpy as np
import scipy as sp
from modopt.opt.linear import Identity
from modopt.opt.proximity import ProximityParent
from modopt.signal.noise import thresh


class SingularValueThreshold(ProximityParent):
    r"""Singular Value Threshold operator.

    This is the proximity operator solving:
    .. math:: arg min \tau||x||_* + 1/2||x||_F

    Parameters
    ----------
    threshold: float
        Threshold value
    thresh_type: str
        Must be in  {"soft", "hard", "soft-rel", "hard-rel"}
        If contains "-rel", the threshold value is considered relative to the
        maximal singular value.
    initial_rank: int
        Initial rank to use for the SVD.
    """

    def __init__(self, threshold, initial_rank=1, thresh_type="hard"):
        self._threshold = threshold
        self._rank = initial_rank
        self._threshold_type = thresh_type.split("-")[0]
        self._rel_thresh = thresh_type.endswith("-rel")
        self._incre = 5

    def op(self, data, extra_factor=1.0):
        """Perform singular values thresholding.

        Parameters
        ----------
        data: ndarray
            A MxN array.

        Returns
        -------
        data_thresholded: ndarray
            The data with thresholded singular values.
        """
        max_rank = min(data.shape) - 2
        if self._rank > max_rank:
            warnings.warn("initial rank bigger than maximal possible one, updating.")

        compute_rank = min(self._rank + 1, max_rank)
        # Singular value are  in increasing order !
        U, S, V = sp.sparse.linalg.svds(data, k=compute_rank)

        if self._rel_thresh:
            thresh_val = self._threshold * S[-1]
        else:
            thresh_val = self._threshold
        # increase the computational rank until we found a singular value small enought.
        while (
            thresh(S[0], thresh_val, self._threshold_type) > 0
            and compute_rank < max_rank
        ):
            compute_rank = min(compute_rank + self._incre, max_rank)
            U, S, V = sp.sparse.linalg.svds(data, k=compute_rank)

        S = thresh(S, thresh_val, self._threshold_type)
        self._rank = np.count_nonzero(S)
        return (U[:, -self._rank :] * S[-self._rank :]) @ V[-self._rank :, :]

    def cost(self, data):
        """Compute cost of low rank operator.

        This is the nuclear norm of data.
        """
        return np.sum(np.abs(sp.linalg.svd(data, compute_uv=False)))


class RankConstraint(ProximityParent):
    def __init__(self, rank=1):
        self._rank = rank

    def op(self, data, extra_factor=1.0):
        max_rank = min(data.shape) - 2
        compute_rank = min(self._rank, max_rank)
        U, S, V = sp.sparse.linalg.svds(data, k=compute_rank)
        return (U[:, -self._rank :] * S[-self._rank :]) @ V[-self._rank :, :]

    def cost(self, data):
        """Compute cost of low rank operator.

        This is the nuclear norm of data.
        """
        return np.sum(
            np.abs(
                sp.sparse.linalg.svds(data, return_singular_vectors=False, k=self._rank)
            )
        )


class FlattenSVT(SingularValueThreshold):
    """Same as SingularValueThreshold but flatten spatial dimension.


    Parameters
    ----------
    threshold: float
        Threshold value
    thresh_type: str
        Must be in  {"soft", "hard", "soft-rel", "hard-rel"}
        If contains "-rel", the threshold value is considered relative to the
        maximal singular value.
    initial_rank: int
        Initial rank to use for the SVD.
    roi: ndarray
        Region of interest to apply the operator on.

    See Also
    --------
    SingularValueThreshold: Singular Value Threshold operator.
    """

    def __init__(self, threshold, initial_rank, roi=None, thresh_type="hard-rel"):
        super().__init__(threshold, initial_rank, thresh_type=thresh_type)
        self.roi = roi

    def op(self, data, roi=None, extra_factor=1.0):
        """Operator function."""
        roi = roi or self.roi
        if roi is not None:
            roi_data = data[:, roi]
            roi_results = super().op(roi_data, extra_factor)
            results = np.zeros_like(data)
            results[:, roi] = roi_results
            return results

        else:
            shape = data.shape
            results = super().op(data.reshape(shape[0], -1), extra_factor)
            results = np.reshape(results, data.shape)
            return results

    def cost(self, data):
        """Compute cost."""
        return super().cost(data.reshape(data.shape[0], -1))


class FlattenRankConstraint(RankConstraint):
    def __init__(self, rank, roi=None):
        super().__init__(rank)
        self.roi = roi

    def op(self, data, roi=None, extra_factor=1.0):
        """Rank Constraint on flatten array."""
        roi = roi or self.roi
        if roi is not None:
            roi_data = data[:, roi]
            roi_results = super().op(roi_data)
            results = np.zeros_like(data)
            results[:, roi] = roi_results
            return results

        else:
            shape = data.shape
            results = super().op(data.reshape(shape[0], -1))
            results = np.reshape(results, data.shape)
            return results

    def cost(self, data):
        """Compute cost."""
        return super().cost(data.reshape(data.shape[0], -1))
