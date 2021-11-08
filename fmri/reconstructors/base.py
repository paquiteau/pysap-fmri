import warnings
import numpy as np
import tqdm

from modopt.opt.linear import Identity

from modopt.opt.algorithms import POGM, ForwardBackward

from mri.operators.gradient.gradient import GradAnalysis, GradSynthesis, GradSelfCalibrationAnalysis, GradSelfCalibrationSynthesis

OPTIMIZERS = {'pogm': 'synthesis',
              'fista':  'analysis'}


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



class SequentialFMRIReconstructor(BaseFMRIReconstructor):
    """ Sequential Reconstruction of fMRI data.
    Time frame are reconstructed in a row, the previous frame estimation is used as initialization for the next one."""

    def __init__(self, fourier_op, space_linear_op, space_regularisation, optimizer="pogm",Smaps=None,verbose=0):
        super().__init__(fourier_op, space_linear_op, space_regularisation, optimizer=optimizer,Smaps=Smaps,verbose=verbose)

        if self.grad_formulation == 'analysis':
            if self.smaps is None:
                print("grad_analysis", flush=True)
                self.grad_op = GradAnalysis(self.fourier_op, verbose=self.verbose)
            else:
                print("grad_self_analysis", flush=True)
                self.grad_op = GradSelfCalibrationAnalysis(fourier_op=self.fourier_op,
                                                           Smaps=self.smaps,
                                                           verbose=self.verbose,
                                                           )

        elif self.grad_formulation == 'synthesis':
            if self.smaps is None:
                self.grad_op = GradSynthesis(self.space_linear_op, self.fourier_op, self.verbose)
            else:
                self.grad_op = GradSelfCalibrationSynthesis(self.fourier_op,self.space_linear_op, self.smaps, self.verbose)
        else:
            raise ValueError("Unknown Gradient formuation")

        self.opt = self.initialize_opt({"cost":None}, dict())

        
    def reconstruct(self, kspace_data, x_init=None, max_iter_per_frame=15):
        if self.fourier_op.n_coils != kspace_data.shape[1]:
            raise ValueError("The kspace data should have shape N_frame x N_coils x N_samples. "
                             "Also, the provided number of coils should match.")
        if self.smaps is not None:
            final_estimate = np.zeros((len(kspace_data),*self.smaps.shape[1:]),dtype="complex128")
        else:
            final_estimate = np.zeros((len(kspace_data),self.fourier_op.n_coils, *self.fourier_op.shape),dtype="complex128")
        
        for i in tqdm.tqdm(range(len(kspace_data))):
            self.opt._grad._obs_data=kspace_data[i,...]
            self.opt.iterate(max_iter=max_iter_per_frame)
            if self.grad_formulation == "synthesis":
                final_estimate[i,...] = self.space_linear_op.adj_op(self.opt.x_final)
            else:
                final_estimate[i,...] = self.opt.x_final
        return final_estimate


class ParallelFMRIReconstructor(BaseFMRIReconstructor):
    """ Parallel Reconstruction of fMRI data.
    Time frame are reconstructed independently, and in parallel to speed up the reconstruction
    """
    def __init__(self, fourier_op, space_linear_op, space_regularisation, optimizer="pogm"):
        super(self).__init__(fourier_op, space_linear_op, space_regularisation, optimizer=optimizer)


    def reconstruct(self, kspace_data, x_init=None):
        pass
