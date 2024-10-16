"""
Frame based reconstructors.

this reconstructor consider the time frames (nostly) independently.

"""

import cupy as cp
import logging

import gc
from functools import cached_property

from modopt.base.backend import get_backend, get_array_module
import numpy as np
import copy
from tqdm.auto import tqdm, trange

from ..operators.gradient import GradAnalysis, GradSynthesis, CustomGradAnalysis
from .base import BaseFMRIReconstructor
from .utils import OPTIMIZERS, initialize_opt

from modopt.opt.algorithms import POGM
from modopt.opt.linear import Identity
from modopt.opt.gradient import GradParent
from ..optimizer import AccProxSVRG, MS2GD

logger = logging.getLogger("pysap-fmri")


class SequentialReconstructor(BaseFMRIReconstructor):
    """Sequential Reconstruction of fMRI data.

    Time frame are reconstructed in a row, the previous frame estimation
    is used as initialization for the next one.

    See Also
    --------
    BaseFMRIReconstructor: parent class
    """

    def __init__(self, *args, optimizer="pogm", progbar_disable=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_name = optimizer
        self.grad_formulation = OPTIMIZERS[optimizer]
        self.progbar_disable = progbar_disable

    def get_grad_op(self, fourier_op, dtype, **kwargs):
        """Create gradient operator specific to the problem."""
        if self.grad_formulation == "analysis":
            return GradAnalysis(
                fourier_op=fourier_op, verbose=self.verbose, dtype=dtype, **kwargs
            )
        if self.grad_formulation == "synthesis":
            return GradSynthesis(
                linear_op=self.space_linear_op,
                fourier_op=fourier_op,
                verbose=self.verbose,
                dtype=dtype,
                **kwargs,
            )
        raise ValueError("Unknown Gradient formuation")

    def reconstruct(
        self,
        kspace_data,
        x_init=None,
        max_iter_per_frame=15,
        grad_kwargs=None,
        compute_backend="numpy",
        restart_strategy="warm",
    ):
        """Reconstruct using sequential method."""
        self.compute_backend = compute_backend
        grad_kwargs = {} if grad_kwargs is None else grad_kwargs
        xp, _ = get_backend(compute_backend)
        final_estimate = np.zeros(
            (len(kspace_data), *self.fourier_op.shape),
            dtype=kspace_data.dtype,
        )
        opt_kwargs = {"cost": "auto"}
        if x_init is None:
            x_init = xp.zeros(self.fourier_op.shape, dtype="complex64")

            if restart_strategy == "warm-mean":
                x_init = xp.array(np.mean(self.fourier_op.adj_op(kspace_data), axis=0))
            if restart_strategy == "warm-pca":
                x_inits = self.fourier_op.adj_op(kspace_data)
                x_inits = x_inits.reshape(x_inits.shape[0], -1)
                from scipy.sparse.linalg import svds

                U, S, V = svds(x_inits, k=1)
                x_init = V.reshape(self.fourier_op.shape)
        next_init = x_init
        # Starting the loops
        progbar_main = trange(len(kspace_data), disable=self.progbar_disable)
        progbar = tqdm(total=max_iter_per_frame, disable=self.progbar_disable)
        for i in progbar_main:
            x_iter = self._reconstruct_frame(
                kspace_data,
                i,
                next_init,
                grad_kwargs,
                opt_kwargs,
                max_iter_per_frame,
                progbar,
            )
            # Prepare for next iteration and save results
            next_init = x_iter if restart_strategy == "warm" else x_init.copy()
            if compute_backend == "cupy":
                final_estimate[i, ...] = x_iter.get()
            else:
                final_estimate[i, ...] = x_iter
            # Progressbar update
        progbar.close()

        logger.info("final prox weight: %f ", xp.unique(self.space_prox_op.weights))
        return final_estimate

    def _reconstruct_frame(
        self, kspace_data, i, x_init, grad_kwargs, opt_kwargs, n_iter, progbar
    ):

        xp, _ = get_backend(self.compute_backend)
        # only recreate gradient if the trajectory change.
        grad_op = self.get_grad_op(
            self.fourier_op.fourier_ops[i],
            dtype=kspace_data.dtype,
            input_data_writeable=True,
            compute_backend=self.compute_backend,
            **grad_kwargs,
        )

        # at each step a new frame is loaded
        grad_op._obs_data = xp.array(kspace_data[i, ...])
        # reset Smaps and optimizer if required.
        opt = initialize_opt(
            opt_name=self.opt_name,
            grad_op=grad_op,
            linear_op=copy.deepcopy(self.space_linear_op),
            prox_op=copy.deepcopy(self.space_prox_op),
            x_init=x_init,
            synthesis_init=False,
            opt_kwargs=opt_kwargs,
            metric_kwargs={},
            compute_backend=self.compute_backend,
        )
        # if no reset, the internal state is kept.
        if progbar is not None:
            progbar.reset(total=n_iter)
        opt.iterate(max_iter=n_iter, progbar=progbar)
        if self.grad_formulation == "synthesis":
            img = self.space_linear_op.adj_op(opt.x_final)
        else:
            img = opt.x_final
        return img


class DoubleSequentialReconstructor(SequentialReconstructor):
    """
    Sequential Reconstruction of fMRI data done in two steps.

    First a classical Sequential Reconstruction using a small number of iteration
    per frame is done, to get a rough estimate of the data.
    Then a second Sequential Reconstruction is done using the first estimate as
    initialization. This second reconstruction can be understand as a fine-tuning
    of the first estimate on each frame data. This can be done in parallel.

    """

    def reconstruct(
        self,
        kspace_data,
        x_init=None,
        max_iter_per_frame=15,
        grad_kwargs=None,
        compute_backend="numpy",
        restart_strategy="warm",
    ):
        """Reconstruct using sequential method."""
        self.compute_backend = compute_backend
        grad_kwargs = {} if grad_kwargs is None else grad_kwargs
        opt_kwargs = {"cost": "auto"}
        xp, _ = get_backend(compute_backend)
        final_estimate = np.zeros(
            (len(kspace_data), *self.fourier_op.shape),
            dtype=kspace_data.dtype,
        )

        if x_init is None:
            x_init = xp.zeros(self.fourier_op.shape, dtype="complex64")
        next_init = x_init
        # Starting the loops
        progbar_main = trange(len(kspace_data), disable=self.progbar_disable)
        progbar = tqdm(total=max_iter_per_frame, disable=self.progbar_disable)
        for i in progbar_main:
            x_iter = self._reconstruct_frame(
                kspace_data,
                i,
                next_init,
                grad_kwargs,
                opt_kwargs,
                max_iter_per_frame,
                progbar,
            )
            # Prepare for next iteration and save results
            next_init = x_iter
            # Progressbar update
        progbar_main.reset(total=len(kspace_data))
        for i in progbar_main:
            x_iter = self._reconstruct_frame(
                kspace_data,
                i,
                next_init,
                grad_kwargs,
                {"cost": "auto"},
                max_iter_per_frame,
                None,
            )

            if compute_backend == "cupy":
                final_estimate[i, ...] = x_iter.get()
            else:
                final_estimate[i, ...] = x_iter

        progbar.close()
        return final_estimate


class StochasticSequentialReconstructor(BaseFMRIReconstructor):
    """Stochastic Sequential Reconstruction of fMRI data."""

    def __init__(
        self,
        fourier_op,
        space_linear_op,
        space_prox_op,
        space_prox_op_refine=None,
        progbar_disable=False,
        compute_backend="numpy",
        **kwargs,
    ):
        super().__init__(fourier_op, space_linear_op, space_prox_op, **kwargs)

        if space_prox_op_refine is None:
            self.space_prox_op_refine = space_prox_op
        else:
            self.space_prox_op_refine = space_prox_op_refine

        self.progbar_disable = progbar_disable
        self.compute_backend = compute_backend

    def reconstruct(
        self,
        kspace_data,
        x_init=None,
        max_iter_per_frame=15,
        max_iter_stochastic=20,
        grad_kwargs=None,
        algorithm="accproxsvrg",
        progbar_disable=False,
        algorithm_kwargs=None,
    ):
        """Reconstruct using sequential method."""
        self.progbar_disable = progbar_disable

        if algorithm_kwargs is None:
            algorithm_kwargs = {}

        xp, _ = get_backend(self.compute_backend)
        # Create the gradients operators
        grad_list = []
        tmp_ksp = cp.zeros_like(kspace_data[0])
        for i, fop in enumerate(self.fourier_op.fourier_ops):
            # L = fop.get_lipschitz_cst()

            # g = GradSynthesis(
            #     linear_op=self.space_linear_op,
            #     fourier_op=fop,
            #     verbose=self.verbose,
            #     dtype=kspace_data.dtype,
            #     lipschitz_cst=L,
            #     num_check_lips=0,  # trust me
            #     input_data_writeable=True,
            # )
            # g._obs_data = kspace_data[i, ...]
            g = CustomGradAnalysis(fop, kspace_data[i, ...], obs_data_gpu=tmp_ksp)
            grad_list.append(g)

        max_lip = max(g.spec_rad for g in grad_list)

        if algorithm == "accproxsvrg":

            opt = AccProxSVRG(
                x=xp.zeros(grad_list[0].shape, dtype="complex64"),
                grad_list=grad_list,
                prox=self.space_prox_op,
                step_size=1.0 / 2 * max_lip,
                auto_iterate=False,
                cost=None,
                compute_backend=self.compute_backend,
                **algorithm_kwargs,
            )

        elif algorithm == "m2sg":

            opt = MS2GD(
                x=xp.zeros(self.fourier_op.shape, dtype="complex64"),
                grad_list=grad_list,
                prox=self.space_prox_op,
                step_size=1.0 / max_lip,
                auto_iterate=False,
                cost=None,
                **algorithm_kwargs,
            )

        opt.iterate(max_iter=max_iter_stochastic)

        x_anat = opt.x_final.squeeze()

        progbar_main = trange(len(kspace_data), disable=self.progbar_disable)
        progbar = tqdm(total=max_iter_per_frame, disable=self.progbar_disable)
        final_img = np.zeros(
            (len(kspace_data), *self.fourier_op.shape),
            dtype=self.fourier_op.cpx_dtype,
        )
        del opt
        gc.collect()
        for i in progbar_main:  # Parallel

            opt = POGM(
                x_anat,
                x_anat,
                x_anat,
                x_anat,
                grad=grad_list[i],
                prox=self.space_prox_op_refine,
                linear=Identity(),
                beta=grad_list[i].inv_spec_rad,
                compute_backend=self.compute_backend,
                auto_iterate=False,
                cost=None,
            )
            opt.iterate(progbar=progbar, max_iter=max_iter_per_frame)

            progbar.reset(total=max_iter_per_frame)
            img = opt.x_final

            if self.compute_backend == "cupy":
                final_img[i] = img.get().squeeze()
            else:
                final_img[i] = img

        return final_img, x_anat
