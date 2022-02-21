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
        elif self.grad_formulation == 'synthesis':
            return GradSynthesis(linear_op=self.space_linear_op,
                                 fourier_op=fourier_op,
                                 verbose=self.verbose,
                                 **kwargs)
        else:
            raise ValueError("Unknown Gradient formuation")

    def reconstruct(self, kspace_data, x_init=None, max_iter_per_frame=15,
                    reset_opt=True, recompute_smaps=False, grad_kwargs=None,
                    smaps_kwargs=None, warm_x=True):
        """Reconstruct using sequential method."""
        grad_kwargs = dict() if grad_kwargs is None else grad_kwargs
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
        if smaps_kwargs is None and recompute_smaps:
            smaps_kwargs = dict()

        next_init = x_init

        if self.fourier_op.is_repeating:
            grad_op = self.get_grad_op(
                self.fourier_op.fourier_ops[0],
                **grad_kwargs)
        for i in range(len(kspace_data)):
            if not self.fourier_op.is_repeating:
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
                                     metric_kwargs=dict())
            # if no reset, the internal state is kept.
            # (dual variable, dynamic step size)
            if i == 0:
                opt.iterate(max_iter=3 * max_iter_per_frame)
            else:
                opt.iterate(max_iter=max_iter_per_frame)

            # Prepare for next iteration and save results
            if self.grad_formulation == "synthesis":
                img = self.space_linear_op.adj_op(opt.x_final)
                next_init = img if warm_x else x_init
                final_estimate[i, ...] = img
            else:
                final_estimate[i, ...] = opt.x_final
                next_init = opt.x_final if warm_x else x_init
            # Progressbar update
            clear_output(wait=True)
        return final_estimate


class ParallelFMRIReconstructor(SequentialFMRIReconstructor):
    """
    Parallel Reconstruction of fMRI data.

    Time frame are reconstructed independently, and in parallel to speed up the reconstruction
    """

    def reconstruct(self,
                    kspace_data,
                    x_init=None,
                    max_iter_per_frame=30,
                    n_jobs=3,
                    smaps_kwargs=None):
        """Reconstruct using Parallel method."""
        def oneframe(kspace_data, idx, x_init, fourier_kwargs, grad_kwargs,
                     smaps_kwargs, linear_op, prox_op, opt_name):
            _kspace_data = kspace_data[idx, ...]
            fourier_cls = fourier_kwargs.pop('__class__')
            fourier_kwargs_kwargs = fourier_kwargs.pop('kwargs')
            fourier_op = fourier_cls(**fourier_kwargs, **fourier_kwargs_kwargs)
            Smaps = None
            if smaps_kwargs:
                Smaps, _ = get_Smaps(_kspace_data,
                                     img_shape=fourier_op.shape,
                                     samples=fourier_op.samples,
                                     min_samples=fourier_op.samples.min(
                                         axis=0),
                                     max_samples=fourier_op.samples.max(
                                         axis=0),
                                     density_comp=fourier_op.density_comp,
                                     **smaps_kwargs)
                if getattr(fourier_op.impl, 'uses_sense', False):
                    fourier_op.impl.operator.set_smaps(Smaps)
                else:
                    grad_kwargs['Smaps'] = Smaps

            grad_cls = grad_kwargs.pop('__class__')
            grad_op = grad_cls(fourier_op=fourier_op, **grad_kwargs)
            grad_op._obs_data = _kspace_data
            if x_init is None:
                if getattr(fourier_op.impl, 'uses_sense', False) or Smaps is not None:
                    x_init = np.squeeze(
                        np.zeros(fourier_op.shape, dtype="complex64"))
                else:
                    x_init = np.squeeze(
                        np.zeros((fourier_op.n_coils, *fourier_op.shape),
                                 dtype="complex64"))
            opt = initialize_opt(opt_name, grad_op, linear_op, prox_op,
                                 x_init=x_init,
                                 synthesis_init=False,
                                 opt_kwargs={"cost": None},
                                 metric_kwargs=dict())
            opt.iterate(max_iter=max_iter_per_frame)
            if hasattr(grad_op, 'linear_op'):
                return grad_op.linear_op.adj_op(opt.x_final)
            return opt.x_final

        smaps_kwargs = smaps_kwargs or dict()
        # 2/ extract info from fourier and grad-op to properly clone them
        fourier_attrs = ['__class__', 'shape', 'samples', 'n_coils']
        if isinstance(self.fourier_op.impl, gpuNUFFT):
            fourier_attrs += ['implementation', 'density_comp', 'kwargs']
        fourier_kwargs = {attr: copy.deepcopy(
            getattr(self.fourier_op, attr)) for attr in fourier_attrs}

        grad_kwargs = {'verbose': self.verbose}
        if self.grad_formulation == 'analysis':
            if self.smaps is None:
                grad_kwargs['__class__'] = GradAnalysis
            else:
                grad_kwargs['__class__'] = GradSelfCalibrationAnalysis
                grad_kwargs['Smaps'] = self.smaps
        elif self.grad_formulation == 'synthesis':
            if self.smaps is None:
                grad_kwargs['__class__'] = GradSynthesis
                grad_kwargs['linear_op'] = self.space_linear_op
            else:
                grad_kwargs['__class__'] = GradSelfCalibrationSynthesis
                grad_kwargs['linear_op'] = self.space_linear_op
                grad_kwargs['Smaps'] = self.smaps

        # remove any reference to self.
        linear_op = self.space_linear_op
        prox_op = self.space_prox_op
        opt_name = self.opt_name

        # # 3/ Reconstruct each time frame independently
        results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(oneframe)(kspace_data, j,
                              x_init,
                              fourier_kwargs,
                              grad_kwargs, smaps_kwargs,
                              linear_op, prox_op, opt_name)
            for j in range(len(kspace_data)))
        final_estimates = np.ascontiguousarray(results)
        return final_estimates
