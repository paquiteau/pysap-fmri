"""Singular Value Threshold operator."""

import logging
from functools import wraps
import gc
import numpy as np
import scipy as sp

from modopt.opt.proximity import ProximityParent
from modopt.signal.noise import thresh

logger = logging.getLogger("pysap-fmri")


def memory_cleanup(gpu=None):
    """Cleanup all the memory."""
    gc.collect()
    if not gpu:
        return
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()


def with_engine(fun):
    """Decorate to handle engine type."""

    @wraps(fun)
    def wrapper(self, data, **kwargs):
        if self._engine == "gpu":
            data_ = self.xp.array(data)
        else:
            data_ = data

        res_ = fun(self, data_, **kwargs)
        if self._engine == "gpu":
            res = res_.get()
        else:
            res = res_
        del res_
        del data_
        memory_cleanup(self._engine == "gpu")
        return res

    return wrapper


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

    def __init__(self, threshold, initial_rank=1, thresh_type="hard", engine="cpu"):
        from scipy.sparse.linalg import svds

        self._threshold = threshold
        self._rank = initial_rank
        self._threshold_type = thresh_type.split("-")[0]
        self._rel_thresh = thresh_type.endswith("-rel")
        self._incre = 5

        self._engine = engine
        self.xp = np

        self.svds = svds
        if self._engine == "gpu":
            try:
                import cupy as cp
                from cupyx.scipy.sparse.linalg import svds
            except ImportError:
                logger.warn("no gpu library found, uses numpy")
            else:
                self.xp = cp
                self.svds = svds

    @with_engine
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
        if self._rank is None:
            U, S, V = self.xp.linalg.svd(data, full_matrices=False)
            St = thresh(S, np.max(S) * self._threshold, self._threshold_type)
            return (U * St) @ V
        max_rank = min(data.shape) - 2
        if self._rank > max_rank:
            logger.warn("initial rank bigger than maximal possible one, updating.")

        compute_rank = min(self._rank + 1, max_rank)
        # Singular value are  in increasing order !
        U, S, V = self.svds(data, k=compute_rank)

        if self._rel_thresh:
            thresh_val = self._threshold * self.xp.max(S)
        else:
            thresh_val = self._threshold
        # increase the computational rank until we found a singular value small enought.
        while (
            thresh(self.xp.min(S), thresh_val, self._threshold_type) > 0
            and compute_rank < max_rank
        ):
            compute_rank = min(compute_rank + self._incre, max_rank)
            U, S, V = self.svds(data, k=compute_rank)
            logger.debug(f"increasing rank to {compute_rank}, thresh_val {thresh_val}")
        S = thresh(S, thresh_val, self._threshold_type)
        self._rank = self.xp.count_nonzero(S)
        logger.debug(f"new Rank: {self._rank}, max value: {self.xp.max(S)}")

        ret = (U[:, -self._rank :] * S[-self._rank :]) @ V[-self._rank :, :]
        del U, S, V
        return ret

    @with_engine
    def cost(self, data):
        """Compute cost of low rank operator.

        This is the nuclear norm of data.
        """
        cost_val = self._threshold * self.xp.sum(
            self.svds(data, k=self._rank, return_singular_vectors=False)
        )
        return cost_val


class RankConstraint(ProximityParent):
    """Singular Value Threshold operator with a fixed rank constraint."""

    def __init__(self, rank=1):
        self._rank = rank

    def op(self, data, extra_factor=1.0):
        """Perform singular values thresholding with rank constraint."""
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

    def __init__(self, threshold, initial_rank, thresh_type="hard-rel", engine="cpu"):
        super().__init__(
            threshold, initial_rank, thresh_type=thresh_type, engine=engine
        )

    def op(self, data, extra_factor=1.0):
        """Operator function."""
        shape = data.shape
        results = super().op(data.reshape(shape[0], -1), extra_factor=extra_factor)
        results = np.reshape(results, data.shape)
        return results

    def cost(self, data):
        """Compute cost."""
        return super().cost(data.reshape(data.shape[0], -1))


class FlattenRankConstraint(RankConstraint):
    """Apply rank constraint on flatten array."""

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
