"""Proximity Operators."""

from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
from modopt.opt.gradient import GradBasic
from modopt.opt.cost import costObj
from modopt.opt.algorithms import ForwardBackward, POGM


import numpy as np


from .utils.proxtv import tv_taut_string, vec_tv_mm, vec_gtv, jit_module


class InTransformSparseThreshold(SparseThreshold):
    """Sparse Thresholding in a transform domain."""

    def _op_method(self, input_data, extra_factor=1.0):
        return self._linear.adj_op(
            super()._op_method(self._linear.op(input_data), extra_factor=extra_factor)
        )


class ProxTV1d:
    """Proximity operator for Total Variation 1D, applied along the first axis.

    Parameters
    ----------
    method: str or callable
        Algorithm use to compute the proximity operator. Available are:
        'fista', 'POGM', 'chambolle_pock', 'condat', 'tv_mm', 'gtv_mm'.
        If callable, it should be a function that takes the data as input and
        apply a proximal operator.
    lambda_tv: float
        Regularization parameter.
    lambda_max: float, default None
        Maximum value of the regularization parameter. If None, it is computed.
    max_iter: int, default 100
        Maximum number of iterations for the algorithm.
    tolerance: float, default 1e-4
        Tolerance for the algorithm.
    **kwargs: dict
        Additional parameters for the algorithm.

    Notes
    -----
    For the 'fista' and 'pogm' methods, the algorithm solves a LASSO problem,
    where the data is centered before applying the algorithm. See [1] for more details.
    """

    def __init__(
        self, lambda_tv, lambda_max=None, method="condat", max_iter=100, **kwargs
    ):
        self.lambda_tv = lambda_tv
        self.lambda_max = lambda_max
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.dtype = None

        jit_module()

        if callable(method):
            self.method = method
        else:
            try:
                self.method = getattr(self, "_" + method)
            except AttributeError as e:
                raise ValueError(f"Unknown method: {method}") from e

        self._center_synth_mat = None

    @property
    def l_reg(self):
        """Regularization parameter."""
        if self.lambda_max is None:
            return np.asarray(self.lambda_tv, dtype=self.dtype)
        return np.asarray(self.lambda_tv * self.lambda_max, dtype=self.dtype)

    def op(self, data, extra_factor=1.0):
        """Proximity operator for Total Variation 1D.

        Parameters
        ----------
        data: np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Output data.
        """
        if np.iscomplexobj(data):
            self.dtype = data.real.dtype
        else:
            self.dtype = data.dtype
        extra_factor = np.asarray(extra_factor, dtype=self.dtype)
        return self.method(data, extra_factor)

    def _forward_backward_setup(self, data, lambda_reg):
        """Set up ModOpt Operator for a Forward-Backward algorithm.

        Parameters
        ----------
        data: np.ndarray
            Input data.

        Returns
        -------
        tuple
            Gradient operator, proximal operator, cost function.
        """
        if self._center_synth_mat is None or self._center_synth_mat.shape != (
            data.shape[0],
            np.prod(data.shape[1:]),
        ):
            mat = np.zeros((len(data) + 1, len(data)))
            mat[1:, :] = np.tri(len(data))
            mat -= np.mean(mat, axis=0)
            self._center_synth_mat = mat

        grad_op = GradBasic(
            input_data=data - np.mean(data, axis=0),
            op=self._center_synth_math.dot,
            trans_op=self._center_synth_mat.T.dot,
        )
        prox_op = SparseThreshold(
            Identity(), weights=lambda_reg, thresh_type="soft", thresh=self.l_reg
        )
        cost_op = costObj(
            [grad_op, prox_op],
            tolerance=self.kwargs.get("tolerance", 1e-5),
            verbose=False,
            cost_interval=5,
        )
        return grad_op, prox_op, cost_op

    def _pogm(self, data, extra_factor=1.0):
        grad_op, prox_op, cost_op = self._forward_backward_setup(
            data, extra_factor * self.l_reg
        )
        pogm = POGM(
            u=np.zeros_like(data),
            x=np.zeros_like(data),
            y=np.zeros_like(data),
            z=np.zeros_like(data),
            grad=grad_op,
            prox=prox_op,
            cost=cost_op,
            auto_iterate=False,
            **self.kwargs,
        )
        pogm.iterate(max_iter=self.max_iter)

        return np.reshape(self._center_synth_mat @ pogm.x_final, data.shape)

    def _fista(self, data, extra_factor=1.0):
        grad_op, prox_op, cost_op = self._forward_backward_setup(data, extra_factor)

        fista = ForwardBackward(
            x=np.zeros_like(data),
            grad=grad_op,
            prox=prox_op,
            cost=cost_op,
            auto_iterate=False,
            **self.kwargs,
        )
        fista.iterate(max_iter=self.max_iter)

        return np.reshape(self._center_synth_mat @ fista.x_final, data.shape)

    def _condat(self, data, extra_factor=1.0):
        dataflatten = data.reshape(data.shape[0], -1)
        if np.iscomplexobj(data):
            ret = np.zeros_like(dataflatten)
            ret.real = tv_taut_string(dataflatten.real, extra_factor * self.l_reg)
            ret.imag = tv_taut_string(dataflatten.imag, extra_factor * self.l_reg)
        else:
            ret = tv_taut_string(dataflatten, extra_factor * self.l_reg)

        return ret.reshape(data.shape)

    def _tv_mm(self, data, extra_factor=1.0):
        flat = data.reshape(data.shape[0], -1)
        if np.iscomplexobj(data):
            ret = np.zeros_like(flat)
            ret.real = vec_tv_mm(flat.real, extra_factor * self.l_reg, 100, 1e-3)
            ret.imag = vec_tv_mm(flat.imag, extra_factor * self.l_reg, 100, 1e-3)
        else:
            ret = vec_tv_mm(flat, extra_factor * self.l_reg)
        return ret.reshape(data.shape)

    def _gtv_mm(self, data, extra_factor=1.0):
        K = np.int16(self.kwargs.get("K", 3))
        flat = data.reshape(data.shape[0], -1)
        if np.iscomplexobj(data):
            ret = np.zeros_like(flat)
            ret.real = vec_gtv(flat.real, extra_factor * self.l_reg, K, 100, 1e-3)
            ret.imag = vec_gtv(flat.imag, extra_factor * self.l_reg, K, 100, 1e-3)
        else:
            ret = vec_gtv(data, extra_factor * self.l_reg, K, 100, 1e-3)
        return ret.reshape(data.shape)

    def cost(self, *args, **kwargs):
        """Cost function for Total Variation 1D."""
        return np.sum(np.abs(np.diff(args[0], axis=0)))

    @classmethod
    def get_lambda_max(cls, y):
        """Compute the maximum value of the regularization parameter.

        Parameters
        ----------
        y: np.ndarray
            Input data.

        Returns
        -------
        float
            Maximum value of the regularization parameter.
        """
        return np.max(np.abs(y - y.mean(axis=0)))


class MultiScaleLowRankSparse:
    """
    A double proximal operator that regularizes a series of image using spatial wavelet.

    Two priors are combined: A Low-Rank Prior on the approximation coefficients
    and a Sparse Prior on the details coefficients.

    Parameters
    ----------
    lambda_lr: float
        Regularization parameter for the low-rank prior.
    lambda_sp: float
        Regularization parameter for the sparse prior.
    linear_op: class
        Linear operator to apply to the data.
    """

    def __init__(self, lambda_lr, lambda_sp, prox_lr, prox_sp, linear_op):
        self.linear_op = linear_op
        self.lambda_lr = lambda_lr
        self.lambda_sp = lambda_sp
        self.prox_lr = prox_lr
        self.prox_sp = prox_sp

    @property
    def lambda_lr(self):
        """Low rank regularisation parameter."""
        return self._lambda_lr

    @lambda_lr.setter
    def lambda_lr(self, value):
        self._lambda_lr = value
        self.prox_lr._threshold = value

    @property
    def lambda_sp(self):
        """Time Sparsity regularisation parameter."""
        return self._lambda_sp

    @lambda_sp.setter
    def lambda_sp(self, value):
        self._lambda_sp = value
        self.prox_sp.weights = value

    def op(self, data):
        """
        Compute Forward Operator.

        Parameters
        ----------
        data: np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Regularized data.
        """
        coeffs = [] * len(data.shape[0])
        # TODO: Run in parallel
        for i in range(data.shape[0]):
            coeffs[i] = self.linear_op.op(data[i, :])

        if self.cf_shape is None:
            self.cf_shape = self.linear_op.cf_shape

        coeffs = np.array(coeffs)  # Expensive ?
        # Extract the coarse scale
        coarse_size = self.cf_shape[0]
        if self.lambda_lr:
            coarse = coeffs[:, np.prod(*coarse_size)]
            # Compute the low-rank approximation
            coarse_lr = self.prox_lr.op(coarse)

        if self.lambda_sp:
            # Compute the sparse approximation
            details = coeffs[:, np.prod(*coarse_size) :]

            details_sp = self.prox_sp.op(details)

        # Reconstruct the data
        coeffs = np.concatenate((coarse_lr, details_sp), axis=1)

        data_reg = np.zeros_like(data)
        for i in range(data.shape[0]):
            data_reg[i] = self.linear_op.adj_op(coeffs[i, :])

        return data_reg

    @classmethod
    def init_lr(cls, lambda_lr, wavelet_name="sym8", n_scale=3):
        """Initialize the prox with only the LowRank prior."""
        raise NotImplementedError

    def init_lrsp(cls, lambda_lr, lambda_sp, wavelet_name="sym8", n_scale=3):
        """Initialize the prox with a LowRank and Sparse (l1) prior."""
        raise NotImplementedError

    def cost(self, *args, **kwargs):
        """Cost function for the LowRankSparse proximal operator."""
        pass
