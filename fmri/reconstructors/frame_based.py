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

    def get_grad_op(self, fourier_op=None):

        fourier_op = self.fourier_op if fourier_op is None else fourier_op

        if self.grad_formulation == 'analysis':
            if self.smaps is None:
                return GradAnalysis(fourier_op, verbose=self.verbose)
            else:
                return GradSelfCalibrationAnalysis(fourier_op=fourier_op,
                                                           Smaps=self.smaps,
                                                           verbose=self.verbose,
                                                           )
        elif self.grad_formulation == 'synthesis':
            if self.smaps is None:
                return GradSynthesis(self.space_linear_op, fourier_op, self.verbose)
            else:
                return GradSelfCalibrationSynthesis(fourier_op, self.space_linear_op, self.smaps, self.verbose)
        else:
            raise ValueError("Unknown Gradient formuation")

    def reconstruct(self, kspace_data, x_init=None, max_iter_per_frame=15, reset_opt=True, recompute_smaps=False, smaps_kwargs=None,warm_x=True):

        grad_op = self.get_grad_op()

        if getattr(self.fourier_op.impl,'uses_sense', False) or self.smaps is not None:
            if x_init is None:
                x_init = np.squeeze(np.zeros(self.fourier_op.shape,dtype="complex64"))
            final_estimate = np.zeros((len(kspace_data), *self.fourier_op.shape), dtype=x_init.dtype)
        else:
            if x_init is None:
                x_init = np.squeeze(np.zeros((self.fourier_op.n_coils, *self.fourier_op.shape),dtype="complex64"))
            final_estimate = np.zeros((len(kspace_data), self.fourier_op.n_coils, *self.fourier_op.shape),dtype=x_init.dtype)

        if self.fourier_op.n_coils != kspace_data.shape[1]:
            raise ValueError("The kspace data should have shape N_frame x N_coils x N_samples. "
                             "Also, the provided number of coils should match.")
        if smaps_kwargs is None and recompute_smaps == True:
            smaps_kwargs = dict()

        opt = self.initialize_opt(grad_op, x_init=x_init, synthesis_init=False, opt_kwargs={"cost":None}, metric_kwargs=dict())
        next_init = x_init
        totalPB =  progressbar.ProgressBar(max_val=len(kspace_data))
        totalPB.start()
        for i in range(len(kspace_data)):
            # at each step a new frame is loaded
            grad_op._obs_data=kspace_data[i,...]
            # reset Smaps and optimizer if required.
            if recompute_smaps and (self.smaps is not None or getattr(self.fourier_op.impl,'use_sense', False)) :
                Smaps, _ = get_Smaps(kspace_data[i,...],
                                            img_shape=self.fourier_op.shape,
                                            samples=self.fourier_op.samples,
                                            min_samples=self.fourier_op.samples.min(axis=0),
                                            max_samples=self.fourier_op.samples.max(axis=0),
                                            density_comp=self.fourier_op.density_comp,
                                            **smaps_kwargs)
                if getattr(self.fourier_op.impl,'uses_sense', False):
                    self.grad_op.fourier_op.impl.setSense(Smaps)
                else:
                    self.grad_op.Smaps = Smaps
                    
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
