"""Reconstructor for fMRI data using full reconstruction paradigm."""

import numpy as np
import tqdm
from modopt.opt.algorithms import FastADMM
from modopt.opt.linear import Identity, LinearParent
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

    def __init__(self,
                 fourier_op,
                 lowrank_thresh=1e-3,
                 sparse_thresh=1e-3,
                 roi=None):
        super().__init__(
            fourier_op=fourier_op,
            space_linear_op=Identity(),
            space_regularisation=FlattenSVT(
                threshold=lowrank_thresh,
                initial_rank=5,
                thresh_type="soft",
                roi=roi,
            ),
            time_linear_op=TimeFourier(roi=roi),
            time_regularisation=SparseThreshold(
                Identity(),
                sparse_thresh,
                thresh_type="soft"),
        )
        self.sparse_thres = sparse_thresh
        self.lowrank_thres = lowrank_thresh

    def reconstruct(self, kspace_data, max_iter=30, eps=1e-5):
        """Perform the reconstruction.

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
        M = np.zeros((len(kspace_data), *self.fourier_op.img_shape),
                     dtype="complex128")

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


class ADMMReconstructor(BaseFMRIReconstructor):
    """Implement the ADMM algorithm to solve fMRI problems."""

    def __init__(self,
                 fourier_op,
                 lowrank_thresh=1e-3,
                 sparse_thresh=1e-3,
                 roi=None,
                 smaps=None):
        super().__init__(
            fourier_op=fourier_op,
            space_linear_op=Identity(),
            space_regularisation=FlattenSVT(
                threshold=lowrank_thresh,
                initial_rank=5,
                thresh_type="soft",
                roi=roi,
            ),
            time_linear_op=TimeFourier(roi=roi),
            time_regularisation=SparseThreshold(
                Identity(),
                sparse_thresh,
                thresh_type="soft"),
            Smaps=smaps,
        )

        self.fourierSpaceTime_op = LinearParent(
            op=lambda x: self.fourier_op.op(self.time_linear_op.adj_op(x)),
            adj_op=lambda x: self.time_linear_op.op(self.fourier_op.adj_op(x)),
        )

    def _optimize_x(self, init_value, obs_value, max_iter=5, **kwargs):
        """Manually perform the FISTA Algorithm on x. Memory Efficient."""
        alpha = 1.
        alpha_old = 1.
        x_old = init_value
        for i in range(max_iter):
            x = x_old - self.step_A * self.fourier_op.data_consistency(
                x_old,
                obs_value)
            x = self.space_prox_op.op(x, extra_factor=self.step_A)

            alpha = (1. + np.sqrt(1. + 4. * alpha_old ** 2)) / 2
            x = x_old + ((alpha_old - 1) / alpha) * (x - x_old)
            x_old = x.copy()
        return x

    def _optimize_z(self, init_value, obs_value, max_iter=5, **kwargs):
        """Manually perform the FISTA Algorithm on z. Memory Efficient."""
        alpha = 1.
        alpha_old = 1.
        x_old = init_value
        for i in range(max_iter):
            x = x_old - self.step_B * self.time_linear_op.adj_op(
                self.fourier_op.data_consistency(
                    self.time_linear_op.op(x_old),
                    obs_value)
            )
            x = self.space_prox_op.op(x, extra_factor=self.step_B)

            alpha = (1. + np.sqrt(1. + 4. * alpha_old ** 2)) / 2
            x = x_old + ((alpha_old - 1) / alpha) * (x - x_old)
            x_old = x.copy()
        return x

    def reconstruct(self, kspace_data, max_iter=15, max_subiter=5, **kwargs):
        """Perform the reconstruction.

        Parameters
        ----------
        kspace_data: array_like
            The kspace data of the fMRI acquisition.
        max_iter: int, optional
            maximum number of iteration
        kwargs: dict
            Extra arguments for the initialisation of ADMM

        Returns
        -------
        M: array_like
            final estimation
        L: array_like
            low rank estimation
        S: array_like
            time-frequency sparse esimation

        See Also
        --------
        modopt.opt.algorithms.FastADMM
        BaseFMRIReconstructor: parent class

        """
        x = np.zeros((len(kspace_data), *self.fourier_op.img_shape),
                     dtype="complex64")

        self.step_A = 1.0
        self.step_B = 1.0

        ADMM = FastADMM(
            x=x,
            z=self.time_linear_op.op(x),
            u=np.zeros_like(kspace_data),
            A=self.fourier_op,
            B=self.fourierSpaceTime_op,
            c=kspace_data,
            solver1=self._optimize_x,
            solver2=self._optimize_z,
            max_iter1=max_subiter,
            max_iter2=max_subiter,
            **kwargs)

        ADMM.iterate(max_iter)

        L = ADMM.x_final
        S = self.time_linear_op.adj_op(ADMM.z_final)

        M = L + S

        return M, L, S
