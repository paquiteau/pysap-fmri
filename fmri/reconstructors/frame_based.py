import progressbar
import numpy as np

from mri.operators.gradient.gradient import GradAnalysis, GradSynthesis, GradSelfCalibrationAnalysis, GradSelfCalibrationSynthesis
from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps
from .base import BaseFMRIReconstructor

class SequentialFMRIReconstructor(BaseFMRIReconstructor):
    """ Sequential Reconstruction of fMRI data.
    Time frame are reconstructed in a row, the previous frame estimation is used as initialization for the next one."""

    def __init__(self, fourier_op, space_linear_op, space_regularisation, optimizer="pogm",Smaps=None,verbose=0):
        super().__init__(fourier_op, space_linear_op, space_regularisation, optimizer=optimizer,Smaps=Smaps,verbose=verbose)

        if self.grad_formulation == 'analysis':
            if self.smaps is None:
                self.grad_op = GradAnalysis(self.fourier_op, verbose=self.verbose)
            else:
                self.grad_op = GradSelfCalibrationAnalysis(fourier_op=self.fourier_op,
                                                           Smaps=self.smaps,
                                                           verbose=self.verbose,
                                                           )
        elif self.grad_formulation == 'synthesis':
            if self.smaps is None:
                self.grad_op = GradSynthesis(self.space_linear_op, self.fourier_op, self.verbose)
            else:
                self.grad_op = GradSelfCalibrationSynthesis(self.fourier_op, self.space_linear_op, self.smaps, self.verbose)
        else:
            raise ValueError("Unknown Gradient formuation")



    def reconstruct(self, kspace_data, x_init=None, max_iter_per_frame=15, reset_opt=True, recompute_smaps=False, smaps_kwargs=None):

        if x_init is None:
            if self.smaps is None:
                x_init = np.zeros((self.fourier_op.n_coils, *self.fourier_op.shape),dtype="complex64")
            else:
                x_init = np.zeros(self.fourier_op.shape,dtype="complex64")
        if self.fourier_op.n_coils != kspace_data.shape[1]:
            raise ValueError("The kspace data should have shape N_frame x N_coils x N_samples. "
                             "Also, the provided number of coils should match.")
        if smaps_kwargs is None and recompute_smaps == True:
            smaps_kwargs = dict()

        if self.smaps is not None:
            final_estimate = np.zeros((len(kspace_data), *self.smaps.shape[1:]), dtype=x_init.dtype)
        else:
            final_estimate = np.zeros((len(kspace_data), self.fourier_op.n_coils, *self.fourier_op.shape),dtype=x_init.dtype)

        opt = self.initialize_opt(x_init=x_init, synthesis_init=False, opt_kwargs={"cost":None}, metric_kwargs=dict())
        next_init = x_init
        for i in progressbar.progressbar(range(len(kspace_data))):
            # at each step a new frame is loaded
            self.grad_op._obs_data=kspace_data[i,...]
            # reset Smaps and optimizer if required.
            if recompute_smaps and self.smaps is not None:
                self.grad.Smaps = get_Smaps(kspace_data[i,...],
                                  img_shape=self.fourier_op.shape,
                                  samples=self.fourier_op.samples,
                                  **smaps_kwargs)

            if reset_opt:
                opt = self.initialize_opt(x_init=next_init,synthesis_init=False,  opt_kwargs={"cost":None}, metric_kwargs=dict())
            # if no reset, the internal state is kept. (dual variable, dynamic step size)
            opt.iterate(max_iter=max_iter_per_frame)

            # Prepare for next iteration and save results
            if self.grad_formulation == "synthesis":
                next_init = self.space_linear_op.adj_op(opt.x_final)
                final_estimate[i,...] = next_init
            else:
                final_estimate[i,...] = opt.x_final
                next_init = self.opt_x_final
        return final_estimate


class ParallelFMRIReconstructor(BaseFMRIReconstructor):
    """ Parallel Reconstruction of fMRI data.
    Time frame are reconstructed independently, and in parallel to speed up the reconstruction
    """
    def __init__(self, fourier_op, space_linear_op, space_regularisation, optimizer="pogm"):
        super(self).__init__(fourier_op, space_linear_op, space_regularisation, optimizer=optimizer)


    def reconstruct(self, kspace_data, x_init=None):
        raise NotImplementedError
