"""Reconstructor for fMRI data using full reconstruction paradigm."""

import numpy as np
import tqdm
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold

from ..operators.fourier import TimeFourier
from ..operators.svt import FlattenSVT
from .base import BaseFMRIReconstructor


class LowRankPlusSparseFMRIReconstructor(BaseFMRIReconstructor):
    """Implement the reconstruction proposed in Petrov et al.

    Parameters
    ----------
    fourier_op: SpaceFourierOperator
        The fourier operator the fmri data
    lowrank_thresh: float, default=1e-3
        threshold for the singular value threshold operator.
    sparse_thres: float, default=1e-3
        threshold for the soft-thresholding operator.
    Smaps: ndarray
        Sensitivities maps for SENSE reconstruction.

    References
    ----------
    https://doi.org/10.1016/j.neuroimage.2017.06.004
    """

    def __init__(self, fourier_op, lowrank_thresh=1e-3, sparse_thresh=1e-3, roi=None):
        super().__init__(
            fourier_op=fourier_op,
            space_linear_op=Identity(),
            space_prox_op=FlattenSVT(
                threshold=lowrank_thresh,
                initial_rank=5,
                thresh_type="soft",
                roi=roi,
            ),
            time_linear_op=TimeFourier(roi=roi),
            time_prox_op=SparseThreshold(Identity(), sparse_thresh, thresh_type="soft"),
        )
        self.sparse_thres = sparse_thresh
        self.lowrank_thres = lowrank_thresh

    def reconstruct(self, kspace_data, max_iter=30, eps=1e-5):
        """Perform the reconstruction.

        Relies on an custom iterative algorithm and do not use Modopt implementation.

        Parameters
        ----------
        kspace_data: ndarray
            The fmri kspace data.
        max_iter=30:
            number of iteration
        eps:
            precision threshold

        Returns
        -------
        M: np.ndarray
            Global fMRI estimation. M = L+S
        L: np.ndarray
            Low Rank fMRI estimation
        S: np.ndarray
            Sparse fMRI estimation
        """
        M = np.zeros((len(kspace_data), *self.fourier_op.img_shape), dtype="complex128")

        L = M.copy()
        S = M.copy()
        tmp = M.copy()
        M_old = self.fourier_op.adj_op(kspace_data)

        for itr in tqdm.tqdm(range(max_iter)):
            # singular value soft thresholding
            tmp = M_old - S
            L = self.space_prox_op.op(tmp)
            # Soft thresholding in the time sparsifying domain
            S = self.time_linear_op.adj_op(
                self.time_prox_op.op(self.time_linear_op.op(tmp))
            )
            # Data consistency: substract residual
            tmp = L + S
            M = tmp - self.fourier_op.data_consistency(tmp, kspace_data)
            if np.linalg.norm(M - M_old) <= eps * np.linalg.norm(M_old):
                print(f"convergence reached at step {itr}")
                break
            M_old = M.copy()

        return M, L, S
