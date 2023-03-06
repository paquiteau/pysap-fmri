from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
from modopt.opt.gradient import GradBasic
from modopt.opt.cost import costObj
from modopt.opt.algorithms import ForwardBackward, POGM

import numpy as np

from .proxtv import prox_tv1d


class ProxTV1d:
    """Proximity operator for Total Variation 1D, applied along the first axis.

    Parameters
    ----------
    method: str or callable
        Algorithm use to compute the proximity operator.
        Available methods are: 'fista', 'POGM', 'chambolle_pock', 'condat'.
        If callable, it should be a function that takes the data as input and
        returns the proximity operator.
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
    For the 'fista' and 'pogm' methods, the algorithm solves a form of the LASSO problem, where the data is centered
    before applying the algorithm. See [1] for more details.
    method='chambolle_pock' follows the implementation of [2].
    method='condat' follows the implementation of [3].



    References
    ----------
    .. [1]

    """

    def __init__(self, lambda_tv, lambda_max=None, method="condat" max_iter=100, **kwargs):
        self.lambda_tv = lambda_tv
        self.lambda_max = lambda_max
        self.max_iter = max_iter
        self.kwargs = kwargs

        if callable(method):
            self.method = method
        else:
            try:
                self.method = getattr(self, "_" + method)
            except AttributeError:
                raise ValueError("Unknown method: {}".format(method))

        self._center_synth_mat = None

    @property
    def lambda_reg(self):
        """Regularization parameter."""
        if self.lambda_max is None:
            return self.lambda_tv
        return self.lambda_tv * self.lambda_max

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
        return self.method(data, extra_factor)

    def _forward_backward_setup(self, data, lambda_reg):
        """Setup ModOpt Operator for a Forward-Backward algorithm.
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
            Identity(), weights=lambda_reg, thresh_type="soft", thresh=self.lambda_reg
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
            data, extra_factor * self.lambda_reg
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
        return prox_tv1d(dataflatten, extra_factor * self.lambda_reg).reshape(
            data.shape
        )

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
