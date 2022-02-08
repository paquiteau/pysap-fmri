# -*- coding: utf-8 -*-
"""Singular Value Threshold oerator"""

import numpy as np
import scipy as sp
from modopt.opt.linear import Identity
from modopt.opt.proximity import ProximityParent, SparseThreshold


class SingularValueThreshold(ProximityParent):
    r"""Singular Value Threshold operator.

    This is the proximity operator solving:
    .. math:: arg min \tau||x||_* + 1/2||x||_F
    """

    def __init__(self, threshold, initial_rank, thresh_type="soft"):
        self.threshold = threshold
        self.rank = initial_rank
        self.threshold_op = SparseThreshold(
            linear=Identity(), weights=threshold, thresh_type=thresh_type
        )
        self._incre = 5

    def op(self, data, extra_factor=1.0):
        """Perform singular values thresholding.

        Parameters
        ----------
        data: ndarray

        Returns
        -------
        data_thresholded: ndarray
            The data with thresholded singular values.
        """
        OK = False
        s = self.rank + 1
        u, s_val, v = sp.sparse.linalg.svds(data, k=1)
        # use a relative threshold/
        thresh = s_val[0] * self.threshold
        while not OK:
            # Sigma are the singular values, sorted in increasing order.
            U, Sigma, VT = sp.sparse.linalg.svds(data, k=s)
            OK = Sigma[0] <= thresh or s == min(data.shape) - 1
            s = min(s + self._incre, min(data.shape) - 1)
        Sigma = self.threshold_op.op(Sigma, extra_factor=s_val)
        self.rank = np.count_nonzero(Sigma)
        return (U[:, -self.rank :] * Sigma[-self.rank :]) @ VT[-self.rank :, :]


class FlattenSVT(SingularValueThreshold):
    """Same as SingularValueThreshold but flatten spatial dimension."""

    def __init__(self, threshold, initial_rank, roi=None, thresh_type="soft"):
        super().__init__(threshold, initial_rank, thresh_type="soft")
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
            results = super().op(
                np.reshape(data, (shape[0], np.prod(shape[1:]))), extra_factor
            )

            return np.reshape(results, data.shape)
