"""Reconstructor for fMRI data using full reconstruction paradigm."""

import numpy as np
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold, ProximityCombo
from modopt.opt.algorithms import POGM
from modopt.opt.gradient import GradBasic

from ..operators.fourier import TimeFourier, SpaceFourierBase
from ..operators.svt import FlattenSVT
from ..operators.utils import make_gradient_operator
from .base import BaseFMRIReconstructor


class JointGradient(GradBasic):
    def _get_grad_method(self, input_data):
        grad = super()._get_grad_method(input_data[0] + input_data[1])
        self.grad = grad[np.newaxis, ...]  # help for the broadcast.


class LowRankPluSparseReconstructor(BaseFMRIReconstructor):
    """Low Rank + Sparse Reconstruction of fMRI data.

    Parameters
    ----------
    fourier_op: OperatorBase
        Operator for the fourier transform of each frame
    space_linear_op: OperatorBase
        Linear operator (eg Wavelet) using for the spatial regularisation
    time_linear_op: OperatorBase
        Linear operator (eg Wavelet) using for the time regularisation
    space_prox_op: OperatorBase
        Proximal Operator for the spatial regularisation
    time_prox_op: OperatorBase
        Proximal Operator for the time regularisation

    """

    def __init__(
        self,
        fourier_op: SpaceFourierBase,
        time_linear_op: TimeFourier,
        lambda_lr: float,
        lambda_sparse: float,
    ):
        self.fourier_op = fourier_op
        self.space_prox_op = FlattenSVT(lambda_lr, 5, thresh_type="soft")
        self.time_prox_op = SparseThreshold(
            Identity(), lambda_sparse, thresh_type="soft"
        )
        self.joint_prox_op = ProximityCombo(self.space_prox_op, self.time_prox_op)

    def reconstruct(self, kspace_data, max_iter=200):

        lr_s_data = np.zeros((2, len(kspace_data, *self.fourier_op.shape)))

        self.joint_grad_op = JointGradient(
            input_data=kspace_data, op=self.fourier_op, trans_op=self.fourier_op.adj_op
        )

        opt = POGM(
            u=lr_s_data.copy(),
            x=lr_s_data.copy(),
            y=lr_s_data.copy(),
            z=lr_s_data.copy(),
            grad=self.joint_grad_op,
            prox=self.joint_prox_op,
            progress=True,
        )

        return opt.x_final[0] + opt.x_final[1], opt.x_final[0], opt.x_final[1]
