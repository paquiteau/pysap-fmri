""""Gradient operators for MRI reconstruction.

Adapted from pysap-mri and Modopt libraries.
"""

from functools import cached_property

import numpy as np
import cupy as cp
from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic, GradParent
from modopt.base.backend import get_backend, get_array_module


def check_lipschitz_cst(f, x_shape, x_dtype, lipschitz_cst, max_nb_of_iter=10):
    """
    Check lipschitz constant.

    This methods check that for random entrees the lipschitz constraint verify:

    ||f(x)-f(y)|| < lipschitz_cst ||x-y||

    Parameters
    ----------
    f: callable
        This lipschitzien function
    x_shape: tuple
        Input data shape
    lipschitz_cst: float
        The Lischitz constant for the function f
    max_nb_of_iter: int
        The number of time the constraint must be satisfied

    Returns
    -------
    out: bool
        If is True than the lipschitz_cst given in argument seems to be an
        upper bound of the real lipschitz constant for the function f
    """
    is_lips_cst = True
    n = 0

    while is_lips_cst and n < max_nb_of_iter:
        n += 1
        x = np.random.randn(*x_shape).astype(x_dtype)
        y = np.random.randn(*x_shape).astype(x_dtype)
        is_lips_cst = np.linalg.norm(f(x) - f(y)) <= (
            lipschitz_cst * np.linalg.norm(x - y)
        )

    return is_lips_cst


class GradBaseMRI(GradBasic):
    """
    Base Gradient class for all gradient operators.

    Implements the gradient of following function with respect to x:
    .. math:: ||M x - y|| ^ 2.

    Parameters
    ----------
    data: np.ndarray
        input data array. this is y
    operator : function
        a function that implements M
    trans_operator : function
        a function handle that implements M ^ T
    shape : tuple
        shape of observed  data y
    lipschitz_cst : int default None
        The lipschitz constant for for given operator.
        If not specified this is calculated using PowerMethod
    lips_calc_max_iter : int default 10
        Number of iterations to calculate the lipschitz constant
    num_check_lips : int default 10
        Number of iterations to check if lipschitz constant is correct
    verbose: int, default 0
        verbosity for debug prints. when 1, prints if lipschitz
        constraints are satisfied
    """

    def __init__(
        self,
        operator,
        trans_operator,
        shape,
        lips_calc_max_iter=10,
        lipschitz_cst=None,
        num_check_lips=10,
        verbose=0,
        dtype="np.float32",
        input_data_writeable=False,
        compute_backend="numpy",
    ):
        # Initialize the GradBase with dummy data
        self._cost_method = self.cost
        self._cost = self.cost
        super().__init__(
            np.array(0),
            operator,
            trans_operator,
            input_data_writeable=input_data_writeable,
        )
        self.xp, _ = get_backend(compute_backend)
        if lipschitz_cst is not None:
            self.spec_rad = lipschitz_cst
            self.inv_spec_rad = 1.0 / self.spec_rad
        else:
            calc_lips = PowerMethod(
                self.trans_op_op,
                shape,
                data_type=dtype,
                auto_run=False,
                compute_backend=compute_backend,
            )
            calc_lips.get_spec_rad(extra_factor=1.1, max_iter=lips_calc_max_iter)
            self.spec_rad = calc_lips.spec_rad
            self.inv_spec_rad = calc_lips.inv_spec_rad
        if verbose > 0:
            print("Lipschitz constant is " + str(self.spec_rad))
        if num_check_lips > 0:
            is_lips = check_lipschitz_cst(
                f=self.trans_op_op,
                x_shape=shape,
                x_dtype=dtype,
                lipschitz_cst=self.spec_rad,
                max_nb_of_iter=num_check_lips,
            )
            if not is_lips:
                raise ValueError("The lipschitz constraint is not satisfied")
            else:
                if verbose > 0:
                    print("The lipschitz constraint is satisfied")

    def _cost(self, *args, **kwargs):
        """Calculate gradient component of the cost.

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            Gradient cost component

        """
        cost_val = 0.5 * self.xp.linalg.norm(self.obs_data - self.op(args[0])) ** 2

        if "verbose" in kwargs and kwargs["verbose"]:
            print(" - DATA FIDELITY (X):", cost_val)
        if isinstance(cost_val, self.xp.ndarray):
            return cost_val.item()
        return cost_val


class GradAnalysis(GradBaseMRI):
    """Gradient Analysis class.

    This class defines the grad operators for:
    (1/2) * sum(||F x - yl||^2_2,l).

    Attributes
    ----------
    fourier_op: an object of class in mri.operators.fourier
        a Fourier operator from FFT, NonCartesianFFT or Stacked3DNFFT
        This is F in above equation.
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1.
    """

    def __init__(self, fourier_op, verbose=0, **kwargs):
        n_channels = fourier_op.n_coils if not fourier_op.uses_sense else 1
        data_shape = (n_channels, *fourier_op.shape)
        super().__init__(
            operator=fourier_op.op,
            trans_operator=fourier_op.adj_op,
            shape=data_shape,
            verbose=verbose,
            **kwargs,
        )
        self.fourier_op = fourier_op


class GradSynthesis(GradBaseMRI):
    """Gradient Synthesis class.

    This class defines the grad operators for:
    (1/2) * sum(||F Psi_t alpha - yl||^2_2,l).

    Attributes
    ----------
    fourier_op: an object of class in mri.operators.fourier
        a Fourier operator from FFT, NonCartesianFFT or Stacked3DNFFT
        This is F in above equation.
    linear_op: an object of class in mri.operators.linear
        a linear operator from WaveltN or WaveletUD2
        This is Psi in above equation.
    verbose: int, default 0
        Debug verbosity. Prints debug information during initialization if 1.
    """

    def __init__(self, linear_op, fourier_op, verbose=0, **kwargs):
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        n_channels = fourier_op.n_coils if not fourier_op.uses_sense else 1
        coef = linear_op.op(np.squeeze(np.zeros((n_channels, *fourier_op.shape))))
        self.linear_op_coeffs_shape = coef.shape
        self.shape = coef.shape
        super().__init__(
            self._op_method,
            self._trans_op_method,
            self.linear_op_coeffs_shape,
            verbose=verbose,
            **kwargs,
        )

    def _op_method(self, data):
        return self.fourier_op.op(self.linear_op.adj_op(data))

    def _trans_op_method(self, data):
        return self.linear_op.op(self.fourier_op.adj_op(data))


class CustomGradAnalysis(GradParent):
    """Custom Gradient Analysis Operator."""

    def __init__(self, fourier_op, obs_data, obs_data_gpu=None, lazy=True):
        self.fourier_op = fourier_op
        self._grad_data_type = np.complex64
        self._obs_data = obs_data
        if obs_data_gpu is None:
            self.obs_data_gpu = cp.array(obs_data)
        elif isinstance(obs_data_gpu, cp.ndarray):
            self.obs_data_gpu = obs_data_gpu
        else:
            raise ValueError("Invalid data type for obs_data_gpu")
        self.lazy = lazy
        self.shape = fourier_op.shape

    def get_grad(self, x):
        """Get the gradient value"""
        if self.lazy:
            self.obs_data_gpu.set(self.obs_data)
        self.grad = self.fourier_op.data_consistency(x, self.obs_data_gpu)
        return self.grad

    @cached_property
    def spec_rad(self):
        return self.fourier_op.get_lipschitz_cst()

    def inv_spec_rad(self):
        return 1.0 / self.spec_rad

    def cost(self, x, *args, **kwargs):
        xp = get_array_module(x)
        cost = xp.linalg.norm(self.fourier_op.op(x) - self.obs_data)
        if xp != np:
            return cost.get()
        return cost
