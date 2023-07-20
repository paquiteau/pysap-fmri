"""Fourier Operator for fMRI data."""

import numpy as np
import scipy as sp
from mri.operators.fourier.cartesian import FFT
from mrinufft import get_operator

# from .fft import FFT
from .utils import validate_smaps


class SpaceFourierBase:
    """Spatial Fourier Transform on fMRI data.

    This operator applies a sequence of fourier operator to a sequence of volumic data (frames).

    Parameters
    ----------
    shape: tuple
        Dimensions of the FFT
    n_frames: int
        Number of frames for the reconstruction
    n_coils: int
        Number of coils for pMRI, default 1.
    smaps: np.ndarray
        Sensitivity Maps, shared across time.

    Attributes
    ----------
    fourier_ops: list
        List of Fourier Operator
    """

    def __init__(self, shape, n_coils=1, n_frames=1, smaps=None, fourier_ops=None):
        self.n_frames = n_frames
        self.n_coils = n_coils
        self.smaps = validate_smaps(shape, n_coils, smaps)
        self.shape = shape
        self._fourier_ops = [None] * n_frames

        if fourier_ops is not None:
            self.fourier_ops = fourier_ops

    @property
    def fourier_ops(self):
        return self._fourier_ops

    @fourier_ops.setter
    def fourier_ops(self, inputs):
        if isinstance(inputs, list):
            if len(inputs) != self.n_frames:
                raise ValueError(
                    "The number of operator provided is not consistent with the number of frames."
                )
            self._fourier_ops = inputs
        else:
            self._fourier_ops = [inputs] * self.n_frames

    def op(self, data):
        """Forward Operator method."""
        raise NotImplementedError

    def adj_op(self, adj_data):
        """Adjoint Operator method."""
        raise NotImplementedError


class CartesianSpaceFourierGlobal:
    """Cartesian Fourier Transform on fMRI data.

    Parameters
    ----------
    shape: tuple
        Dimensions of the FFT
    mask: np.ndarray
        ND array sampling mask
    n_frames: int
        Number of frames for the reconstruction
    n_coils: int
        Number of coils for pMRI, default 1.
    smaps: np.ndarray
        Sensitivity Maps, shared across time.
    """

    def __init__(self, shape, mask=None, n_coils=1, n_frames=1, smaps=None, n_jobs=-1):
        self.shape = shape
        self.n_coils = n_coils
        self.n_frames = n_frames
        self.smaps = smaps
        self.uses_sense = smaps is not None
        self.n_jobs = n_jobs
        if mask is None:
            self.mask = np.ones(n_frames)
        elif mask.shape == shape:
            # common mask for all frames.
            self.mask = [mask] * n_frames
        elif mask.shape == (n_frames, *shape):
            # custom mask for every frame.
            self.mask = mask
        else:
            raise ValueError("incompatible mask format")

    def op(self, img):
        """Forward Operator method."""
        img2 = np.zeros((self.n_frames, self.n_coils, *self.shape), dtype=img.dtype)
        for ii in range(self.n_coils):
            for i in range(self.n_frames):
                img2[i, ii, ...] = img[i] * self.smaps[ii]

        axes = tuple(np.arange(2, img2.ndim))
        multicoil_ksp = sp.fft.ifftshift(
            sp.fft.ifftn(
                sp.fft.fftshift(img2, axes=axes),
                axes=axes,
                norm="ortho",
                workers=self.n_jobs,
            ),
            axes=axes,
        )
        multicoil_ksp = multicoil_ksp * self.mask[:, None, ...]
        return multicoil_ksp

    def adj_op(self, x):
        x = x * self.mask[:, None, ...]
        axes = tuple(np.arange(2, x.ndim))
        img = sp.fft.fftshift(
            sp.fft.fftn(
                sp.fft.ifftshift(x, axes=axes),
                axes=axes,
                norm="ortho",
                workers=self.n_jobs,
            ),
            axes=axes,
        )
        return np.sum(img * self.smaps[None, ...].conj(), axis=1)


class FFT_Sense(FFT):
    def __init__(self, shape, n_coils=1, mask=None, smaps=None, n_jobs=1):
        super().__init__(shape, n_coils=n_coils, mask=mask, n_jobs=n_jobs)
        self.smaps = smaps

    @property
    def uses_sense(self):
        return self.smaps is not None

    def op(self, img):
        """Forward Operator method."""
        img2 = np.zeros((self.n_coils, *self.shape), dtype=img.dtype)
        for i in range(self.n_coils):
            img2[i, ...] = img * self.smaps[i]
        return super().op(img2)

    def adj_op(self, x):
        x = super().adj_op(x)
        return np.sum(x * self.smaps.conj(), axis=0)


class CartesianSpaceFourier(SpaceFourierBase):
    def __init__(self, shape, mask, n_coils=1, n_frames=1, smaps=None, n_jobs=1):
        self.shape = shape
        self.n_coils = n_coils
        self.n_frames = n_frames
        self.smaps = smaps
        self.uses_sense = smaps is not None
        self.n_jobs = n_jobs

        if mask is None:
            self.mask = np.ones(n_frames)
        elif mask.shape == shape:
            # common mask for all frames.
            self.mask = [mask] * n_frames
        elif mask.shape == (n_frames, *shape):
            # custom mask for every frame.
            self.mask = mask
        else:
            raise ValueError("incompatible mask format")

        self.fourier_ops = [
            FFT_Sense(
                shape, n_coils=n_coils, mask=self.mask[i], smaps=smaps, n_jobs=n_jobs
            )
            for i in range(n_frames)
        ]

    def op(self, data):
        """Forward Operator method."""
        adj_data = np.squeeze(
            np.zeros(
                (self.n_frames, self.n_coils, *self.shape),
                dtype=data.dtype,
            )
        )
        for i in range(self.n_frames):
            adj_data[i] = self.fourier_ops[i].op(data[i])
        return adj_data

    def adj_op(self, adj_data):
        """Adjoint Operator method."""
        data = np.squeeze(
            np.zeros(
                (self.n_frames, self.n_coils if self.smaps is None else 1, *self.shape),
                dtype=adj_data.dtype,
            )
        )

        for i in range(self.n_frames):
            data[i] = self.fourier_ops[i].adj_op(adj_data[i])
        return data


class NonCartesianSpaceFourier(SpaceFourierBase):
    """Spatial Fourier Transform on fMRI data.

    Parameters
    ----------
    samples: np.ndarray
        2D or 3D array of samples coordinates for non cartesian fourier
    n_frames: int
        Number of frames for the reconstruction
    n_coils: int
        Number of coils for pMRI, default 1.
    smaps: np.ndarray
        Sensitivity Maps, shared across time.
    backend:  str
        A backend library implemented by mri-nufft. ex "finufft" or "cufinufft"
    density: bool or numpy.ndarray
        If true, estimate the density.
    Attributes
    ----------
    fourier_ops: list
        List of NonCartesianFFT Operator
    """

    def __init__(
        self,
        samples,
        shape,
        n_coils=1,
        n_frames=1,
        smaps=None,
        density=True,
        backend="cufinufft",
        **kwargs,
    ):
        MRI_operator_klass = get_operator(backend)
        super().__init__(shape, n_coils, n_frames, smaps)

        if samples.ndim == 2 and n_frames == 0:
            raise ValueError(
                "2D array of samples provided, but n_frames is not specified."
            )
        if samples.ndim == 2:
            self.samples = np.repeat(samples[None, ...], n_frames, axis=0)
            self.n_samples_per_frame = samples.shape[0]
        elif samples.ndim == 3:
            self.samples = samples
            self.n_samples_per_frame = samples.shape[1]
        else:
            raise ValueError("samples array should be 2D or 3D.")

        if density is True and samples.ndim == 2:
            density = MRI_operator_klass.estimate_density(samples, self.shape)

        for i in range(n_frames):
            self.fourier_ops[i] = MRI_operator_klass(
                self.samples[i],
                shape,
                n_coils=n_coils,
                smaps=smaps,
                density=density,
                **kwargs,
            )

    def op(self, data):
        """Forward Operator method."""
        adj_data = np.squeeze(
            np.zeros(
                (self.n_frames, self.n_coils, self.n_samples_per_frame),
                dtype=data.dtype,
            )
        )
        for i in range(self.n_frames):
            adj_data[i] = self.fourier_ops[i].op(data[i])
        return adj_data

    def adj_op(self, adj_data):
        """Adjoint Operator method."""
        data = np.squeeze(
            np.zeros(
                (self.n_frames, self.n_coils if self.smaps is None else 1, *self.shape),
                dtype=adj_data.dtype,
            )
        )

        for i in range(self.n_frames):
            data[i] = self.fourier_ops[i].adj_op(adj_data[i])
        return data
