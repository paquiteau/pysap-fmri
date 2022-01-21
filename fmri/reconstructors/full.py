""" Reconstructor for fMRI data using full reconstruction paradigm. """

import numpy as np
import tqdm

from modopt.opt.linear import LinearParent, Identity, LinearComposition, make_adjoint
from modopt.opt.proximity import SparseThreshold
from modopt.opt.algorithms import ForwardBackward, FastADMM
from modopt.opt.gradient import GradBasic

from ..operators.fourier import TimeFourier
from ..operators.svt import SingularValueThreshold
from ..optimizers.admm import FastADMM
from .base import BaseFMRIReconstructor




class LowRankPlusSparseFMRIReconstructor(BaseFMRIReconstructor):
    """Implement the reconstruction proposed in Petrov et al.

    Parameters:
    -----------
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

    def __init__(self, fourier_op, lowrank_thresh=1e-3, sparse_thresh=1e-3, Smaps=None):
        super().__init__(
            fourier_op=fourier_op,
            space_linear_op=Identity(),
            space_regularisation=SingularValueThreshold(
                threshold=lowrank_thresh,
                initial_rank=10,
                thresh_type="soft",
            ),
            time_linear_op=TimeFourier(),
            time_regularisation=SparseThreshold(
                Identity(),
                sparse_thresh,
                thresh_type="soft"),
            Smaps=Smaps,
        )

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

        """
        M = np.zeros((len(kspace_data), *self.fourier_op.img_shape),
                     dtype="complex128")

        L = M.copy()
        S = M.copy()

        if self.smaps is None:
            M_old = self.fourier_op.adj_op(kspace_data)
        else:
            M_old = np.sum(np.conjugate(self.smaps) *
                           self.fourier_op.adj_op(kspace_data), axis=1)
        L_old = L.copy()
        S_old = S.copy()

        for _ in tqdm.tqdm(range(max_iter)):
            # singular value soft thresholding
            L = self.space_prox_op.op(
                (M_old - S_old).reshape(self.fourier_op.n_frames,
                                        np.prod(self.fourier_op.img_shape)))
            L = np.reshape(L, (self.fourier_op.n_frames,
                           *self.fourier_op.img_shape))

            # Soft thresholding in the time sparsifying domain
            S = self.time_linear_op.adj_op(
                self.time_prox_op.op(self.time_linear_op.op(M_old - L_old)))
            # Data consistency: substract residual
            if self.smaps is None:
                M = L + S - \
                    self.fourier_op.adj_op(
                        self.fourier_op.op(L + S) - kspace_data)
            else:
                M = L + S - np.sum(
                    np.conjugate(self.smaps) * self.fourier_op.adj_op(
                        self.fourier_op.op(
                            (L + S)[:, np.newaxis, ...] * self.smaps) - kspace_data),
                    axis=1)
            if np.linalg.norm(L + S - L_old - S_old) <= eps * np.linalg.norm(L_old + S_old):
                print("convergence reached")
                break
            M_old = M.copy()
            L_old = L.copy()
            S_old = S.copy()

        return M, L, S


class ADMMReconstructor(BaseFMRIReconstructor):
    """Implement the ADMM algorithm to solve fMRI problems."""

    def __init__(self, fourier_op, lowrank_thresh=1e-3, sparse_thresh=1e-3, Smaps=None):
        super().__init__(
            fourier_op=fourier_op,
            space_linear_op=Identity(),
            space_regularisation=SingularValueThreshold(
                threshold=lowrank_thresh,
                initial_rank=10,
                thresh_type="soft",
            ),
            time_linear_op=TimeFourier(),
            time_regularisation=SparseThreshold(
                Identity(),
                sparse_thresh,
                thresh_type="soft"),
            Smaps=Smaps,
        )

        self.fourierSpaceTime_op = LinearComposition(
            fourier_op,
            make_adjoint(self.time_linear_op),
        )

    def _optimize_x(self, init_value, obs_value, max_iter=5, **kwargs):
        grad = GradBasic(op=self.fourier_op.op,
                         trans_op=self.fourier_op.op,
                         obs_data=obs_value,
                         )
        opt = ForwardBackward(init_value, grad, self.space_prox_op, **kwargs)
        opt.iterate()
        opt.retrieve_outputs()
        return opt.x_final

    def _optimize_z(self, init_value, obs_value, max_iter=5, **kwargs):
        grad = GradBasic(op=self.fourierSpaceTime_op.op,
                         trans_op=self.fourierSpaceTime_op.op,
                         obs_data=obs_value,
                         )
        opt = ForwardBackward(init_value, grad, self.time_prox_op, **kwargs)
        opt.iterate()
        opt.retrieve_outputs()

        return opt.x_final

    def reconstruct(self, kspace_data, max_iter=15, max_subiter=5, **kwargs):
        """
        Perform the reconstruction.

        Parameters
        ----------
        kspace_data: array_like
            The kspace data of the fMRI acquisition.
        max_iter: int, optional
            maximum number of iteration
        kwargs: dict
            Extra arguments for the initialisation of ADMM

        See Also
        --------
        modopt.opt.algorithms.FastADMM
        BaseFMRIReconstructor: parent class

        Returns
        -------
        M: array_like
            final estimation
        L: array_like
            low rank estimation
        S: array_like
            time-frequency sparse esimation
        """
        ADMM = FastADMM(A=self.fourier_op,
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
