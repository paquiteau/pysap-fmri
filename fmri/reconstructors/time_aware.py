"""Reconstructor for fMRI data using full reconstruction paradigm."""
from functools import partial

import numpy as np
from modopt.opt.cost import costObj
from modopt.opt.algorithms import POGM, ForwardBackward, ADMM, FastADMM
from modopt.opt.algorithms.admm import ADMMcostObj
from modopt.opt.gradient import GradBasic
from modopt.math.matrix import PowerMethod

from ..operators.fourier import TimeFourier, SpaceFourierBase
from ..operators.svt import FlattenSVT
from ..operators.utils import InTransformSparseThreshold
from .base import BaseFMRIReconstructor


class JointGradient(GradBasic):
    def __init__(self, op, trans_op, input_data, **kwargs):
        super().__init__(input_data=input_data, op=op, trans_op=trans_op, **kwargs)
        self.single_op = op
        self.single_trans_op = trans_op

        self.op = self._op_method
        self.trans_op = self._trans_op_method

    def _op_method(self, input_data):
        return self.single_op(input_data[0] + input_data[1])

    def _trans_op_method(self, input_data):
        ret = self.single_trans_op(input_data)
        # duplicate the data, copy is necessary
        # broadcast_to method return read-only array which is not compatible
        # with the restart strategy of POGM.
        return np.repeat(0.5 * ret[np.newaxis, ...], 2, axis=0)


class JointProx(ProximityParent):
    def __init__(self, operators):
        self.operators = operators
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, input_data, extra_factor=1.0):
        res = np.zeros_like(input_data)

        for i, operator in enumerate(self.operators):
            res[i] = operator.op(input_data[i], extra_factor=extra_factor)
        return res

    def _cost_method(self, *args, **kwargs):
        return np.sum(
            [
                operator.cost(input_data)
                for operator, input_data in zip(self.operators, args[0])
            ]
        )


class LowRankPlusSparseReconstructor(BaseFMRIReconstructor):
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
        time_linear_op: TimeFourier = None,
        time_prox_op: ProximityParent = None,
        space_prox_op: ProximityParent = None,
        lambda_space: float = 0.1,
        lambda_time: float = 0.1,
        cost="auto",
    ):
        self.fourier_op = fourier_op
        if space_prox_op is None:
            self.space_prox_op = FlattenSVT(lambda_space, 5, thresh_type="hard-rel")
        else:
            self.space_prox_op = space_prox_op

        if time_prox_op is None and time_linear_op is not None:
            self.time_prox_op = InTransformSparseThreshold(
                time_linear_op, lambda_time, thresh_type="soft"
            )
        elif time_prox_op is not None:
            self.time_prox_op = time_prox_op
        else:
            raise ValueError("Either time_prox_op or time_linear_op must be provided")

        self.cost = cost

    def reconstruct(
        self,
        kspace_data: np.ndarray,
        max_iter: int = 200,
        grad_step=None,
        optimizer: str = "pogm",
        verbose: bool = False,
        **kwargs,
    ):
        if optimizer in ["pogm", "fista"]:
            self.joint_prox_op = JointProx([self.space_prox_op, self.time_prox_op])
            self.joint_grad_op = JointGradient(
                input_data=kspace_data,
                op=self.fourier_op.op,
                trans_op=self.fourier_op.adj_op,
            )
            lr_s_data = np.zeros(
                (2, self.fourier_op.n_frames, *self.fourier_op.shape),
                dtype=kspace_data.dtype,
            )
            lr_s_data[0] = self.fourier_op.adj_op(kspace_data) / 2
            lr_s_data[1] = lr_s_data[0].copy()
        else:
            lr_s_data = self.fourier_op.adj_op(kspace_data) / 2
        if self.cost == "auto":
            if optimizer in ["pogm", "fista"]:
                self.cost = costObj(
                    [self.joint_grad_op, self.joint_prox_op], verbose=False
                )
            elif optimizer == "admm":
                self.cost = [self.space_prox_op.cost, self.time_prox_op.cost]

        if grad_step is None:
            pm = PowerMethod(
                self.joint_grad_op.trans_op_op,
                lr_s_data.shape,
                data_type=kspace_data.dtype,
            )
            grad_step = pm.inv_spec_rad
        if optimizer == "pogm":
            opt = POGM(
                u=lr_s_data,
                x=lr_s_data.copy(),
                y=lr_s_data.copy(),
                z=lr_s_data.copy(),
                grad=self.joint_grad_op,
                prox=self.joint_prox_op,
                cost=self.cost,
                progress=True,
                beta_param=grad_step,
                auto_iterate=False,
                verbose=False,
            )
        elif optimizer == "fista":
            opt = ForwardBackward(
                x=lr_s_data,
                grad=self.joint_grad_op,
                prox=self.joint_prox_op,
                cost=self.cost,
                auto_iterate=False,
                beta_param=grad_step,
                verbose=False,
            )

        elif "admm" in optimizer:

            def subopt(init, obs, prox):
                """Solve the low rank subproblem"""
                opt = ForwardBackward(
                    x=init,
                    grad=GradBasic(obs, self.fourier_op.op, self.fourier_op.adj_op),
                    prox=prox,
                    beta_param=grad_step,
                    cost=None,
                    auto_iterate=False,
                    verbose=False,
                )
                opt.iterate(max_iter=10)
                return opt.x_final

            opt = {"fastadmm": FastADMM, "admm": ADMM}[optimizer](
                u=lr_s_data,
                v=lr_s_data.copy(),
                mu=np.ones_like(kspace_data),
                A=self.fourier_op,
                B=self.fourier_op,
                optimizers=(
                    partial(subopt, prox=self.space_prox_op),
                    partial(subopt, prox=self.time_prox_op),
                ),
                b=kspace_data,
                auto_iterate=False,
                verbose=False,
            )

        else:
            raise ValueError(f"Optimizer '{optimizer}' not supported")

        opt.iterate(max_iter=max_iter)
        costs = opt._cost_func._cost_list

        if optimizer in ["pogm", "fista"]:
            return (
                opt.x_final[0] + opt.x_final[1],
                opt.x_final[0],
                opt.x_final[1],
                costs,
            )
        else:
            return opt.u_final + opt.v_final, opt.u_final, opt.v_final, costs
