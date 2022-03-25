"""
Frame based reconstructors.

this reconstructor consider the time frames (nostly) independently.

"""
import copy
from IPython.display import clear_output

from joblib import Parallel, delayed
import progressbar
import numpy as np

from mri.operators.gradient.gradient import GradAnalysis, GradSynthesis,\
    GradSelfCalibrationAnalysis, GradSelfCalibrationSynthesis
from mri.operators.fourier.non_cartesian import gpuNUFFT

from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps
from .base import BaseFMRIReconstructor
from .utils import initialize_opt, OPTIMIZERS


class SequentialFMRIReconstructor(BaseFMRIReconstructor):
    """Sequential Reconstruction of fMRI data.

    Time frame are reconstructed in a row, the previous frame estimation
    is used as initialization for the next one.

    See Also
    --------
    BaseFMRIReconstructor: parent class
    """

    def __init__(self, *args, optimizer="pogm", **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_name = optimizer
        self.grad_formulation = OPTIMIZERS[optimizer]

    def get_grad_op(self, fourier_op, **kwargs):
        """Create gradient operator specific to the problem."""
        if self.grad_formulation == 'analysis':
            return GradAnalysis(fourier_op=fourier_op,
                                verbose=self.verbose,
                                **kwargs)
        if self.grad_formulation == 'synthesis':
            return GradSynthesis(linear_op=self.space_linear_op,
                                 fourier_op=fourier_op,
                                 verbose=self.verbose,
                                 **kwargs)
        raise ValueError("Unknown Gradient formuation")

    def reconstruct(self, kspace_data, x_init=None, max_iter_per_frame=15,
                    reset_opt=True, grad_kwargs=None,
                    warm_x=True):
        """Reconstruct using sequential method."""
        grad_kwargs = {} if grad_kwargs is None else grad_kwargs
        if x_init is None:
            x_init = np.zeros(self.fourier_op.fourier_ops[0].shape,
                              dtype="complex64")
        final_estimate = np.zeros(
            (len(kspace_data), *self.fourier_op.fourier_ops[0].shape),
            dtype=x_init.dtype)

        if self.fourier_op.n_coils != kspace_data.shape[1]:
            raise ValueError("The kspace data should have shape"
                             "N_frame x N_coils x N_samples. "
                             "Also, the number of coils should match.")

        next_init = x_init
        for i in range(len(kspace_data)):
            # only recreate gradient if the trajectory change.
            grad_op = self.get_grad_op(
                    self.fourier_op.fourier_ops[i],
                    **grad_kwargs)

            # at each step a new frame is loaded
            grad_op._obs_data = kspace_data[i, ...]
            # reset Smaps and optimizer if required.
            if reset_opt:
                opt = initialize_opt(opt_name=self.opt_name,
                                     grad_op=grad_op,
                                     linear_op=self.space_linear_op,
                                     prox_op=self.space_prox_op,
                                     x_init=next_init,
                                     synthesis_init=False,
                                     opt_kwargs={"cost": None},
                                     metric_kwargs={})
            # if no reset, the internal state is kept.
            # (e.g. dual variable, dynamic step size)
            if i == 0 and warm_x:
                opt.iterate(max_iter=3 * max_iter_per_frame)
            else:
                opt.iterate(max_iter=max_iter_per_frame)

            # Prepare for next iteration and save results
            if self.grad_formulation == "synthesis":
                img = self.space_linear_op.adj_op(opt.x_final)
            else:
                img = opt.x_final
            next_init = img if warm_x else x_init
            final_estimate[i, ...] = img
            # Progressbar update
            clear_output(wait=True)
        return final_estimate
