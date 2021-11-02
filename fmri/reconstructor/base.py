import warnings
import numpy as np
import tqdm

from modopt.opt.linear import Identity

from modopt.opt.algorithms import POGM, ForwardBackward

from mri.operators.gradient.gradient import GradAnalysis, GradSynthesis, GradSelfCalibrationAnalysis, GradSelfCalibrationSynthesis

OPTIMIZERS = {'pogm': 'analysis',
              'fista':  'analysis'}


class BaseFMRIReconstructor(object):
    """ This class hold common attributes and methods for fMRI reconstruction """

    def __init__(self, fourier_op, space_linear_op, space_regularisation,
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

        if time_regularisation is None:
            warnings.warn("The in-time regularizer is not set. Setting to identity. "
                          "Note that frame will be reconstruct independently.")
            self.time_prox_op = Identity()

    def reconstruct(self, kspace_data, *args, **kwargs):
        raise NotImplementedError

    def initialize_opt(self, opt_kwargs, metric_kwargs):
        x_init = np.zeros((self.grad_op.linear_op.n_coils,
                           *self.grad_op.fourier_op.shape),dtype="complex128")
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
                **metric_kwargs
            )
        elif self.opt_name == "fista":
            opt = ForwardBackward(
                x=x_init,
                grad=self.grad_op,
                prox=self.prox_op,
                linear=self.space_linear_op,
                beta_param=beta,
                lambda_param=opt_kwargs.pop("lambda_param",1.0),
                **metric_kwargs,
            )
        else:
            raise ValueError(f"Optimizer {self.opt_class} not implemented")
        return opt



class SequentialFMRIReconstructor(BaseFMRIReconstructor):
    """ Sequential Reconstruction of fMRI data.
    Time frame are reconstructed in a row, the previous frame estimation is used as initialization for the next one."""

    def __init__(self, fourier_op, space_linear_op, space_regularisation, optimizer="pogm"):
        super(self).__init__( fourier_op, space_linear_op, space_regularisation, optimizer=optimizer)

        if self.grad_formulation == 'analysis':
            if self.smaps is None:
                self.grad_op = GradAnalysis(self.fourier_op, self.verbose)
            else:
                self.grad_op = GradSelfCalibrationAnalysis(self.fourier_op,self.smaps,self.verbose)
        elif self.grad_formulation == 'synthesis':
            if self.smaps is None:
                self.grad_op = GradSynthesis(self.fourier_op, self.verbose)
            else:
                self.grad_op = GradSelfCalibrationSynthesis(self.fourier_op,self.space_linear_op, self.smaps, self.verbose)
        else:
            raise ValueError("Unknown Gradient formuation")


        
    def reconstruct(self, kspace_data, x_init=None):
        if self.fourier_op.n_coils != kspace_data.shape[1]:
            raise ValueError("The kspace data should have shape N_frame x N_coils x N_samples. "
                             "Also, the provided number of coils should match.")

        final_estimate = np.zeros((len(kspace_data),*self.fourier_op.shape))
        for i in tqdm.tqdm(range(len(kspace_data))):
            self.opt.grad._obs_data=kspace_data
            self.opt.iterate()
            final_estimate[i,...] = self.opt.x_final
        return final_estimate


class ParallelFMRIReconstructor(BaseFMRIReconstructor):
    """ Parallel Reconstruction of fMRI data.
    Time frame are reconstructed independently, and in parallel to speed up the reconstruction
    """
    def __init__(self, fourier_op, space_linear_op, space_regularisation, optimizer="pogm"):
        super(self).__init__(fourier_op, space_linear_op, space_regularisation, optimizer=optimizer)


    def reconstruct(self, kspace_data, x_init=None):
