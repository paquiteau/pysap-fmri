"""Reconstructor for fMRI data using full reconstruction paradigm."""

from functools import partial

import numpy as np
import scipy as sp
from modopt.math.matrix import PowerMethod
from modopt.opt.algorithms import ADMM, POGM, FastADMM, ForwardBackward
from modopt.opt.cost import costObj
from modopt.opt.gradient import GradBasic
from modopt.opt.linear import Identity
from modopt.opt.proximity import ProximityParent

from ..operators.fourier import SpaceFourierBase
from ..operators.proximity import InTransformSparseThreshold
from ..operators.svt import FlattenSVT
from ..operators.time_op import TimeOperator
from .base import BaseFMRIReconstructor


class JointGradient(GradBasic):
    """Vectorize the Gradient Operator."""

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
    """Vectorize proximal operator."""

    def __init__(self, operators):
        self.operators = operators
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, input_data, extra_factor=1.0):
        res = np.empty_like(input_data)

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
        time_linear_op: TimeOperator = None,
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
            self.lambda_time = lambda_time
            self.time_prox_op = InTransformSparseThreshold(
                time_linear_op, lambda_time, thresh_type="soft"
            )
        elif time_prox_op is not None:
            self.time_prox_op = time_prox_op
            if time_linear_op is None:
                self.time_linear_op = Identity()
            else:
                self.time_linear_op = time_linear_op
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
    ):
        """Reconstruct fMRI data using the choosen optimizer."""
        return getattr(self, f"_{optimizer}")(kspace_data, max_iter, grad_step)

    def _setup_fb(self, kspace_data, grad_step=None):
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

        if self.cost == "auto":
            self.cost = costObj([self.joint_grad_op, self.joint_prox_op], verbose=False)

        if grad_step is None:
            pm = PowerMethod(
                self.joint_grad_op.trans_op_op,
                lr_s_data.shape,
                data_type=kspace_data.dtype,
            )
            grad_step = pm.inv_spec_rad
        return lr_s_data, grad_step

    def _fista(self, kspace_data, max_iter, grad_step):
        lr_s_data, grad_step = self._setup_fb(kspace_data, grad_step)
        opt = ForwardBackward(
            x=lr_s_data,
            grad=self.joint_grad_op,
            prox=self.joint_prox_op,
            cost=self.cost,
            auto_iterate=False,
            beta_param=grad_step,
            verbose=False,
        )

        opt.iterate(max_iter=max_iter)
        costs = opt._cost_func._cost_list

        return (
            opt.x_final[0] + opt.x_final[1],
            opt.x_final[0],
            opt.x_final[1],
            costs,
        )

    def _pogm(self, kspace_data, max_iter, grad_step):
        lr_s_data, grad_step = self._setup_fb(kspace_data, grad_step)

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
        opt.iterate(max_iter=max_iter)
        costs = opt._cost_func._cost_list
        return (
            opt.x_final[0] + opt.x_final[1],
            opt.x_final[0],
            opt.x_final[1],
            costs,
        )

    def _fast_admm(self, kspace_data, max_iter, grad_step):
        return self._admm(kspace_data, max_iter, grad_step, fast=True)

    def _admm(self, kspace_data, max_iter, grad_step, fast=False):
        def subopt(init, obs, prox):
            """Solve the low rank subproblem."""
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

        lr_s_data = self.fourier_op.adj_op(kspace_data) / 2
        self.cost = [self.space_prox_op.cost, self.time_prox_op.cost]
        optKlass = FastADMM if fast else ADMM
        opt = optKlass(
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

        opt.iterate(max_iter=max_iter)
        costs = opt._cost_func._cost_list
        return opt.u_final + opt.v_final, opt.u_final, opt.v_final, costs

    def _otazo(self, kspace_data, max_iter, grad_step):
        from tqdm.auto import tqdm

        nt = len(kspace_data)

        M0 = self.fourier_op.adj_op(kspace_data)
        norm_M0 = np.linalg.norm(M0.reshape(nt, -1), "fro")
        M = M0.copy()
        S = np.zeros_like(M)
        L = np.zeros_like(M)
        Lprev = M.copy()
        costs = np.zeros(max_iter)
        for i in tqdm(range(max_iter)):
            L = self.space_prox_op.op(M - S, extra_factor=grad_step)
            S = self.time_prox_op.op(M - Lprev, extra_factor=grad_step)
            resk = self.fourier_op.op(L + S) - kspace_data
            M = L + S
            M -= grad_step * self.fourier_op.adj_op(resk)
            Lprev = L.copy()
            costs[i] = (
                self.space_prox_op.cost(L)
                + self.time_prox_op.cost(S)
                + np.linalg.norm(resk.reshape(nt, -1), "fro")
                ** 2  # 0.5 factor omitted as in matlab code.
            )
            if np.linalg.norm((M - M0).reshape(nt, -1), "fro") < 1e-3 * norm_M0:
                break
        costs = costs[: i + 1]

        return (L + S, L, S, costs)

    def _otazo_raw(self, kspace_data, max_iter, grad_step):
        from tqdm.auto import tqdm

        nt = len(kspace_data)

        def flatten2norm(x):
            return np.linalg.norm(x.reshape(nt, -1), "fro")

        def softthresh(x, thresh):
            return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

        M0 = self.fourier_op.adj_op(kspace_data)
        norm_M0 = np.linalg.norm(M0.reshape(nt, -1), 2)
        M = M0.copy()
        S = np.zeros_like(M)
        L = np.zeros_like(M)
        Lprev = M.copy()
        costs = np.zeros(max_iter)
        lambda_l = self.space_prox_op._threshold
        lambda_s = self.time_prox_op.weights
        print(lambda_l, lambda_s)
        for i in tqdm(range(max_iter)):
            # L = self.space_prox_op.op(M - S)
            Ut, St, Vt = sp.linalg.svd((M - S).reshape(nt, -1), full_matrices=False)
            St = softthresh(St, np.max(St) * lambda_l * grad_step)
            L = np.reshape((Ut * St) @ Vt, (nt, *self.fourier_op.shape))
            # S = self.time_prox_op.op(M - Lprev)

            Sf = sp.fft.fft(
                sp.fft.fftshift((M - Lprev).reshape(nt, -1), axes=0),
                axis=0,
                norm="ortho",
            )
            Sf = softthresh(Sf, lambda_s * grad_step)

            S = sp.fft.ifftshift(sp.fft.ifft(Sf, axis=0, norm="ortho"), axes=0).reshape(
                nt, *self.fourier_op.shape
            )

            S = self.time_prox_op._linear.adj_op(
                softthresh(
                    self.time_prox_op._linear.op(M - Lprev),
                    self.time_prox_op.weights,
                )
            )
            resk = self.fourier_op.op(L + S) - kspace_data
            M = L + S - grad_step * self.fourier_op.adj_op(resk)
            Lprev = L.copy()
            # lambda_l * np.sum(St)
            # + lambda_s
            # * np.sum(np.abs(sp.fft.fft(S.reshape(nt, -1), axis=1, norm="ortho")))
            # !! 0.5 factor was omitted as in matlab code.
            costs[i] = 0.5 * np.linalg.norm(resk.reshape(nt, -1), "fro") ** 2
            if np.linalg.norm((M - M0).reshape(nt, -1), "fro") < 1e-3 * norm_M0:
                break
        costs = costs[: i + 1]
        return (L + S, L, S, costs)
