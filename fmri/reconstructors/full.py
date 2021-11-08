import numpy as np

import tqdm

from modopt.opt.linear import Identity
from modopt.opt.proximity import LowRankMatrix, SparseThreshold

from .base import BaseFMRIReconstructor
from ..operators.fourier import SpaceFourier, TimeFourier



class GlobalFMRIReconstructor(BaseFMRIReconstructor):
    """ Global FMRI reconstructor"""

    def initialize_opt(self, opt_kwargs, metric_kwargs):
        raise NotImplementedError



class LowRankPlusSparseFMRIReconstructor(GlobalFMRIReconstructor):
    """ This Class implement the reconstruction proposed in Petrov et al.
    https://doi.org/10.1016/j.neuroimage.2017.06.004
    """
    def __init__(self, fourier_op, lowrank_thresh=1e-3,sparse_thresh=1e-3, Smaps=None):
        super(GlobalFMRIReconstructor, self).__init__(fourier_op=fourier_op,
                                                      space_linear_op=Identity(),
                                                      space_regularisation=LowRankMatrix(lowrank_thresh,
                                                                                         thresh_type="soft"),
                                                      time_linear_op=TimeFourier(),
                                                      time_regularisation=SparseThreshold(Identity(),
                                                                                         sparse_thresh,
                                                                                         thresh_type="soft"),
                                                      Smaps=Smaps
                                                      )


    def reconstruct(self, kspace_data, max_iter=30,eps=1e-5):
        M = np.zeros((self.fourier_op.shape[0],*self.fourier_op.shape[2:]),dtype="complex128")

        L = M.copy()
        S = M.copy()

        M_old = np.sum(np.conjugate(self.smaps)*self.fourier_op.adj_op(kspace_data),axis=1)
        L_old = L.copy()
        S_old = S.copy()

        for i in tqdm.tqdm(range(max_iter)):
            # singular value soft thresholding
            L = self.space_prox_op.op((M_old-S_old).reshape(self.fourier_op.n_frames,np.prod(self.fourier_op.img_shape)))
            L = np.reshape(L,(self.fourier_op.n_frames, *self.fourier_op.img_shape))

            # Soft thresholding in the time sparsifying domain
            S = self.time_linear_op.adj_op(self.time_prox_op.op(self.time_linear_op.op(M_old-L_old)))
            # Data consistency: substract residual
            M = L + S - np.sum(np.conjugate(self.smaps) * self.fourier_op.adj_op(
                                                                self.fourier_op.op((L+S)[:,np.newaxis,...] * self.smaps) - kspace_data),
                                axis=1)
            if np.linalg.norm(L+S - L_old - S_old) <= eps * np.linalg.norm(L_old+S_old):
                print("convergence reached")
                break
            M_old = M.copy()
            L_old = L.copy()
            S_old = S.copy()

        return M, L, S
