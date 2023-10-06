# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2018                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################

import numpy as np
from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic


def check_lipschitz_cst(f, x_shape, lipschitz_cst, max_nb_of_iter=10):
    """
    This methods check that for random entrees the lipschitz constraint are
    statisfied:

    * ||f(x)-f(y)|| < lipschitz_cst ||x-y||

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
        x = np.random.randn(*x_shape)
        y = np.random.randn(*x_shape)
        is_lips_cst = np.linalg.norm(f(x) - f(y)) <= (
            lipschitz_cst * np.linalg.norm(x - y)
        )

    return is_lips_cst


class GradBaseMRI(GradBasic):
    """Base Gradient class for all gradient operators
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
    ):
        # Initialize the GradBase with dummy data
        super().__init__(
            np.array(0),
            operator,
            trans_operator,
        )
        if lipschitz_cst is not None:
            self.spec_rad = lipschitz_cst
            self.inv_spec_rad = 1.0 / self.spec_rad
        else:
            calc_lips = PowerMethod(
                self.trans_op_op, shape, data_type=np.complex, auto_run=False
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
                lipschitz_cst=self.spec_rad,
                max_nb_of_iter=num_check_lips,
            )
            if not is_lips:
                raise ValueError("The lipschitz constraint is not satisfied")
            else:
                if verbose > 0:
                    print("The lipschitz constraint is satisfied")


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
        if fourier_op.n_coils != 1:
            data_shape = (fourier_op.n_coils, *fourier_op.shape)
        else:
            data_shape = fourier_op.shape
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
        coef = linear_op.op(
            np.squeeze(np.zeros((linear_op.n_coils, *fourier_op.shape)))
        )
        self.linear_op_coeffs_shape = coef.shape
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
