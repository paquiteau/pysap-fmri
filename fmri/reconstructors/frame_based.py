import sys
import copy
from IPython.display import clear_output

from joblib import Parallel, delayed
import progressbar
import numpy as np

from mri.operators.gradient.gradient import GradAnalysis, GradSynthesis,\
    GradSelfCalibrationAnalysis, GradSelfCalibrationSynthesis
from mri.operators.fourier.non_cartesian import NonCartesianFFT

from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps
from .base import BaseFMRIReconstructor
from .utils import initialize_opt


class SequentialFMRIReconstructor(BaseFMRIReconstructor):
    """ Sequential Reconstruction of fMRI data.
    Time frame are reconstructed in a row, the previous frame estimation is used as initialization for the next one."""

    def __init__(self, fourier_op, space_linear_op, space_regularisation, optimizer="pogm", Smaps=None, verbose=0):
        super().__init__(fourier_op, space_linear_op, space_regularisation,
                         optimizer=optimizer, Smaps=Smaps, verbose=verbose)

    def get_grad_op(self, **kwargs):
        if self.grad_formulation == 'analysis':
            if self.smaps is None:
                return GradAnalysis(fourier_op=self.fourier_op,
                                    verbose=self.verbose,
                                    **kwargs)
            else:
                return GradSelfCalibrationAnalysis(fourier_op=self.fourier_op,
                                                   Smaps=self.smaps,
                                                   verbose=self.verbose,
                                                   **kwargs)
        elif self.grad_formulation == 'synthesis':
            if self.smaps is None:
                return GradSynthesis(linear_op=self.space_linear_op,
                                     fourier_op=self.fourier_op,
                                     verbose=self.verbose,
                                     **kwargs)
            else:
                return GradSelfCalibrationSynthesis(fourier_op=self.fourier_op,
                                                    linear_op=self.space_linear_op,
                                                    Smaps=self.smaps,
                                                    verbose=self.verbose,
                                                    **kwargs)
        else:
            raise ValueError("Unknown Gradient formuation")

    def reconstruct(self, kspace_data, x_init=None, max_iter_per_frame=15, reset_opt=True, recompute_smaps=False, grad_kwargs=None, smaps_kwargs=None, warm_x=True):

        grad_kwargs = dict() if grad_kwargs is None else grad_kwargs
        grad_op = self.get_grad_op(**grad_kwargs)

        if getattr(self.fourier_op.impl, 'uses_sense', False) or self.smaps is not None:
            if x_init is None:
                x_init = np.squeeze(
                    np.zeros(self.fourier_op.shape, dtype="complex64"))
            final_estimate = np.zeros(
                (len(kspace_data), *self.fourier_op.shape), dtype=x_init.dtype)
        else:
            if x_init is None:
                x_init = np.squeeze(
                    np.zeros((self.fourier_op.n_coils, *self.fourier_op.shape), dtype="complex64"))
            final_estimate = np.zeros(
                (len(kspace_data), self.fourier_op.n_coils, *self.fourier_op.shape), dtype=x_init.dtype)

        if self.fourier_op.n_coils != kspace_data.shape[1]:
            raise ValueError("The kspace data should have shape N_frame x N_coils x N_samples. "
                             "Also, the provided number of coils should match.")
        if smaps_kwargs is None and recompute_smaps:
            smaps_kwargs = dict()

        opt = self.initialize_opt(grad_op, x_init=x_init, synthesis_init=False, opt_kwargs={
                                  "cost": None}, metric_kwargs=dict())
        next_init = x_init
        totalPB = progressbar.ProgressBar(max_val=len(kspace_data))
        totalPB.start()
        for i in range(len(kspace_data)):
            # at each step a new frame is loaded
            grad_op._obs_data = kspace_data[i, ...]
            # reset Smaps and optimizer if required.
            if (recompute_smaps and
                    (self.smaps is not None or getattr(self.fourier_op.impl, 'use_sense', False))):
                Smaps, _ = get_Smaps(kspace_data[i, ...],
                                     img_shape=self.fourier_op.shape,
                                     samples=self.fourier_op.samples,
                                     min_samples=self.fourier_op.samples.min(
                                         axis=0),
                                     max_samples=self.fourier_op.samples.max(
                                         axis=0),
                                     density_comp=self.fourier_op.density_comp,
                                     **smaps_kwargs)
                if getattr(self.fourier_op.impl, 'uses_sense', False):
                    grad_op.fourier_op.impl.operator.set_smaps(Smaps)
                else:
                    grad_op.Smaps = Smaps

            if reset_opt:
                opt = self.initialize_opt(grad_op,
                                          x_init=next_init,
                                          synthesis_init=False,
                                          opt_kwargs={"cost": None},
                                          metric_kwargs=dict())
            # if no reset, the internal state is kept.
            # (dual variable, dynamic step size)
            opt.iterate(max_iter=max_iter_per_frame)

            # Prepare for next iteration and save results
            if self.grad_formulation == "synthesis":
                img = self.space_linear_op.adj_op(opt.x_final)
                next_init = img if warm_x else x_init
                final_estimate[i, ...] = img
            else:
                final_estimate[i, ...] = opt.x_final
                next_init = self.opt_x_final if warm_x else x_init
            # Progressbar update
            clear_output(wait=True)
            totalPB.update(i)
        totalPB.finish()
        return final_estimate


class ParallelFMRIReconstructor(SequentialFMRIReconstructor):
    """
    Parallel Reconstruction of fMRI data.

    Time frame are reconstructed independently, and in parallel to speed up the reconstruction
    """

    def reconstruct(self, kspace_data, x_init=None, max_iter_per_frame=15, n_jobs=3, smaps_kwargs=None):

        def oneframe(kspace_data, idx, x_init, fourier_kwargs, grad_kwargs,
                     smaps_kwargs, linear_op, prox_op, opt_name):
            kspace_data = kspace_data[idx, ...]
            fourier_cls = fourier_kwargs.pop('__class__')
            fourier_kwargs_kwargs = fourier_kwargs.pop('kwargs')
            fourier_op = fourier_cls(**fourier_kwargs, **fourier_kwargs_kwargs)

            if smaps_kwargs:
                Smaps, _ = get_Smaps(kspace_data,
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
            grad_op = grad_cls(fourier_op=fourier_op, **
                               grad_kwargs, num_check_lips=20)

            opt = initialize_opt(opt_name, grad_op, linear_op, prox_op,
                                 x_init=x_init,
                                 synthesis_init=False,
                                 opt_kwargs={"cost": None},
                                 metric_kwargs=dict())
            opt.iterate(max_iter=max_iter_per_frame)
            if hasattr(grad_op, 'linear_op'):
                return grad_op.linear_op.adj_op(opt.x_final)
            else:
                return opt.x_final

        # 1/ setup initial variable and final array
        if x_init is None and  \
           (getattr(self.fourier_op.impl, 'uses_sense', False) or self.smaps is not None):
            x_init = np.squeeze(
                np.zeros(self.fourier_op.shape, dtype="complex64"))
        elif x_init is None:
            x_init = np.squeeze(
                np.zeros((self.fourier_op.n_coils, *self.fourier_op.shape), dtype="complex64"))

        if self.fourier_op.n_coils != kspace_data.shape[1]:
            raise ValueError("The kspace data should have shape N_frame x N_coils x N_samples. "
                             "Also, the provided number of coils should match.")

        if smaps_kwargs is None:
            smaps_kwargs = dict()
        # 2/ extract info from fourier and grad-op to properly clone them
        fourier_attrs = ['__class__', 'shape', 'samples', 'n_coils']
        if isinstance(self.fourier_op, NonCartesianFFT):
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
        print(x_init.shape)
        linear_op = copy.deepcopy(self.space_linear_op)
        prox_op = copy.deepcopy(self.space_prox_op)
        opt_name = copy.deepcopy(self.opt_name)

        # 3/ Reconstruct each time frame independently
        results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(oneframe)(kspace_data, j,
                              x_init,
                              fourier_kwargs,
                              grad_kwargs, smaps_kwargs,
                              linear_op, prox_op, opt_name,
                              )
            for j in range(len(kspace_data)))
        final_estimates = np.ascontiguousarray(results)
        return final_estimates
