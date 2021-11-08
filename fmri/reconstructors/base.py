import warnings
import numpy as np

from modopt.opt.linear import Identity
from modopt.opt.algorithms import POGM, ForwardBackward


OPTIMIZERS = {'pogm': 'synthesis',
              'fista':  'analysis',
               None: None}


class BaseFMRIReconstructor(object):
    """ This class hold common attributes and methods for fMRI reconstruction """

    def __init__(self, fourier_op, space_linear_op, space_regularisation=None,
                 time_linear_op=None, time_regularisation=None, Smaps=None, optimizer='pogm', verbose=0,):
        self.fourier_op = fourier_op
        self.space_linear_op = space_linear_op or Identity
        self.time_linear_op = space_linear_op or Identity
        self.opt_name = optimizer
        self.grad_formulation = OPTIMIZERS[optimizer]
        self.smaps = Smaps
        self.verbose = verbose

        if space_regularisation is None:
            warnings.warn("The in space regulariser is not set. Setting to identity. "
                          "Note that optimization is just a gradient descent in space")
            self.space_prox_op = Identity()
        else:
            self.space_prox_op = space_regularisation

        if time_regularisation is None:
            warnings.warn("The in-time regularizer is not set. Setting to identity. "
                          "Note that frame will be reconstruct independently.")
            self.time_prox_op = Identity()
        else:
            self.time_prox_op = time_regularisation

    def reconstruct(self, kspace_data, *args, **kwargs):
        raise NotImplementedError

    def initialize_opt(self, opt_kwargs, metric_kwargs):
        if self.smaps is not None:
            x_init = np.zeros(self.smaps.shape[1:],dtype="complex128")
        else:
            x_init = np.zeros(self.fourier_op.shape,dtype="complex128")
        if self.grad_formulation == "synthesis":
            alpha_init = self.space_linear_op.op(x_init)
        beta = self.grad_op.inv_spec_rad
        if self.opt_name == "pogm":
            opt = POGM(
                u=alpha_init,
                x=alpha_init,
                y=alpha_init,
                z=alpha_init,
                grad=self.grad_op,
                prox=self.space_prox_op,
                linear=self.space_linear_op,
                beta_param=beta,
                sigma_bar=opt_kwargs.pop('sigma_bar',0.96),
                auto_iterate=opt_kwargs.pop("auto_iterate",False),
                **opt_kwargs,
                **metric_kwargs
            )
        elif self.opt_name == "fista":
            opt = ForwardBackward(
                x=x_init,
                grad=self.grad_op,
                prox=self.space_prox_op,
                linear=self.space_linear_op,
                beta_param=beta,
                lambda_param=opt_kwargs.pop("lambda_param",1.0),
                auto_iterate=opt_kwargs.pop("auto_iterate",False),
                **opt_kwargs,
                **metric_kwargs,
            )
        else:
            raise ValueError(f"Optimizer {self.opt_class} not implemented")
        return opt
