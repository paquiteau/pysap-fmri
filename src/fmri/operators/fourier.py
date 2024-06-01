"""Fourier Operator for fMRI data.

Implements basic Cartesian Operators. For Non Cartesian Operator use MRI-NUFFT
and RepeatOperator.
"""

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from mrinufft import get_operator
from modopt.base.backend import get_array_module

try:
    from mrinufft.operators.interfaces.gpunufft import make_pinned_smaps
except ImportError:
    make_pinned_smaps = None

from .utils.fft import fft, ifft

MRINUFFT_AVAILABLE = True
CUPY_AVAILABLE = True

try:
    import cupyx as cx
except ImportError:
    CUPY_AVAILABLE = False


class LazyFourierOps:
    """Lazy Fourier Operators."""

    def __init__(self, factory, params):
        self._factory = factory
        self._params = params

    def __getitem__(self, idx):
        """Get the operator at index idx."""
        return self._factory(**self._params[idx])

    def __len__(self):
        """Return the number of operators."""
        return len(self._params)


class SpaceFourierBase(ABC):
    """Space Fourier operator Base Class."""

    def __init__(self):
        self.smaps = None

    @abstractmethod
    def op(self, img):
        """Forward operator."""
        pass

    @abstractmethod
    def adj_op(self, data):
        """Adjoint operator."""
        pass


class CartesianSpaceFourier(SpaceFourierBase):
    """A Fourier Operator in space."""

    def __init__(self, shape, n_frames, mask, n_coils=1, smaps=None):
        """Initialize the Fourier operator.

        Parameters
        ----------
        shape : tuple
            Shape of the image.
        n_frames : int
            Number of frames.
        mask : array
            Mask of the image.
        n_coils : int
            Number of coils.
        """
        self.shape = shape
        self.n_frames = n_frames

        if mask is None:
            self.mask = np.ones((n_frames, *shape))
        elif mask.shape == shape:
            self.mask = np.repeat(mask[np.newaxis, ...], n_frames, axis=0)
        elif mask.shape == (n_frames, *shape):
            self.mask = mask
        else:
            raise ValueError("Mask shape not understood")
        self.n_coils = n_coils
        self.smaps = smaps

        self.fourier_ops = LazyFourierOps(
            FFT_Sense,
            [
                {
                    "shape": self.shape,
                    "mask": self.mask[i],
                    "smaps": self.smaps,
                    "n_coils": self.n_coils,
                }
                for i in range(n_frames)
            ],
        )

    def op(self, img):
        """Forward operator.

        Parameters
        ----------
        img : array
            Image in space.

        Returns
        -------
        ksp : array
            kspace data.
        """
        axes = tuple(range(-len(self.shape), 0))
        if self.n_coils > 1:
            if self.smaps is not None:
                img2 = np.repeat(img[:, np.newaxis, ...], self.n_coils, axis=1)
                img2 *= self.smaps
                ksp = fft(img2, axis=axes)
            else:
                ksp = fft(img, axis=axes)
            return ksp * self.mask[:, np.newaxis, ...]
        else:
            ksp = fft(img, axis=axes)
            return ksp * self.mask

    def adj_op(self, kspace_data):
        """Apply the adjoint operator.

        Parameters
        ----------
        kspace_data : array
            kspace data.

        Returns
        -------
        img : array
            Image in space.
        """
        axes = tuple(range(-len(self.shape), 0))
        if self.n_coils > 1:
            img = ifft(kspace_data, axis=axes)
            if self.smaps is None:
                return img
            return np.sum(img * np.conj(self.smaps), axis=1)
        else:
            return ifft(kspace_data, axis=axes)


class RepeatOperator(SpaceFourierBase):
    """Use a sequence of operator for multiple frames."""

    def __init__(self, fourier_ops):
        self.fourier_ops = list(fourier_ops)

    def op(self, images):
        """Apply the forward operator."""
        final_ksp = np.empty(
            (len(images), self.n_coils, self.n_samples), dtype=np.complex64
        )
        for i in range(len(images)):
            final_ksp[i] = self.fourier_ops[i].op(images[i])
        return final_ksp

    def adj_op(self, coeffs):
        """Apply Adjoint Operator."""
        c = 1 if self.uses_sense else self.n_coils
        xp = get_array_module(coeffs)
        final_image = xp.empty((self.n_frames, c, *self.shape), dtype=np.complex64)
        for i in range(len(coeffs)):
            final_image[i] = self.fourier_ops[i].adj_op(coeffs[i])
        return final_image.squeeze()

    def __getattr__(self, attrName):
        """Pass the attributes to the first operator."""
        return getattr(self.fourier_ops[0], attrName)

    @property
    def n_frames(self):
        """Number of frames."""
        return len(self.fourier_ops)


class LazySpaceFourier(RepeatOperator):
    """A dedicated Space Fourier operator based on cufinufft.

    Requires a workable installation of cupy and cufinufft.

    Parameters
    ----------
    samples: np.ndarray
        k-space samples location of shape (n_frames, n_samples, 3)
    shape: tuple
        image domain shape
    n_frames: int
        number of frames
    n_coils: int
        number of coils
    smaps: np.ndarray
        sensitivity maps of shape (n_coils, *shape)
    density:
        density compensation profile of shape (n_frames, n_samples)
    **kwargs:
        extra kwargs
    """

    def __init__(
        self,
        backend,
        samples,
        shape,
        n_frames,
        n_coils,
        smaps,
        density="cell_count",
        **kwargs,
    ):
        if len(samples) != n_frames:
            raise ValueError("size of samples and frames do not match")
        params = [
            dict(
                samples=samples[i],
                shape=shape,
                n_coils=n_coils,
                smaps=smaps,
                density=density,
                **kwargs,
            )
            for i in range(n_frames)
        ]
        self.uses_sense = smaps is not None
        self.shape = shape
        self.n_coils = n_coils
        self.smaps = smaps

        self.fourier_ops = LazyFourierOps(get_operator(backend), params)


class PooledgpuNUFFTSpaceFourier(SpaceFourierBase):
    """A dedicated Space Fourier operator based on gpuNUFFT.

    Requires a workable installation of Cupy and gpuNUFFT and mrinufft.

    Parameters
    ----------
    samples: np.ndarray
        k space samples location of shape (n_frames, n_samples, 3)
    shape: tuple
        image domain shape
    n_frames: int
        number of frames
    n_coils: int
        number of coils
    pool_size: int
        number of operators to pool
    smaps: np.ndarray
        sensitivity maps of shape (n_coils, *shape)
    density: bool or np.ndarray or list
        density compensation profile of shape (n_frames, n_samples)
    **kwargs:
        extra kwargs for gpunufft
    """

    def __init__(
        self,
        samples,
        shape,
        n_frames,
        n_coils,
        pool_size,
        smaps,
        density,
        **kwargs,
    ):
        if not CUPY_AVAILABLE:
            raise RuntimeError("Cupy is not available")

        self.n_samples = samples.shape[1]
        self.n_frames = n_frames
        self.n_coils = n_coils
        self.shape = shape

        if samples.shape != (n_frames, self.n_samples, len(shape)):
            raise ValueError("size of samples and frames do not match")
        self.samples = samples
        self._init_pinned_data(pool_size, smaps)
        self._init_density(density)
        self._init_operators(**kwargs)

    def _init_pinned_data(self, pool_size, smaps):
        # Prepare the smaps
        pinned_smaps = make_pinned_smaps(smaps)
        self.pinned_smaps = pinned_smaps

        self.pooled_kspace = [None] * pool_size
        self.pooled_image = [None] * pool_size
        self.pool_size = pool_size
        c = 1 if pinned_smaps is not None else self.n_coils
        for i in range(pool_size):
            self.pooled_kspace[i] = cx.zeros_pinned(
                (self.n_coils, self.n_samples), dtype=np.complex64
            )
            self.pooled_image[i] = cx.zeros_pinned(
                (np.prod(self.shape), c), dtype=np.complex64
            )

    def _init_density(self, density):
        # Format the density
        # TODO use pattern matching ?
        if isinstance(density, np.ndarray):
            if len(density) == self.n_samples:
                density = np.repeat(density[np.newaxis, ...], self.n_frames, axis=0)
            elif density.shape != (self.n_frames, self.n_samples):
                raise ValueError("Density shape not understood")
        elif isinstance(density, (bool, str, dict)) or callable(density):
            density = [density] * self.n_frames
        elif not (isinstance(density, Sequence) and len(density) == self.n_frames):
            raise ValueError("Density shape not understood")
        self.density = density

    def _init_operators(self, **kwargs):
        # initialize all the operators
        factory = get_operator("gpunufft")
        self.fourier_ops = [None] * self.n_frames
        for i, p_img, p_ksp in zip(
            range(self.n_frames),
            itertools.cycle(self.pooled_image),
            itertools.cycle(self.pooled_kspace),
        ):
            self.fourier_ops[i] = factory(
                self.samples[i],
                self.shape,
                n_coils=self.n_coils,
                smaps=None,
                density=self.density[i],
                pinned_smaps=self.pinned_smaps,
                pinned_kspace=p_ksp,
                pinned_image=p_img,
                **kwargs,
            )

    def op(self, images):
        """Apply the forward operator."""
        final_ksp = np.empty(
            (len(images), self.n_coils, self.n_samples), dtype=np.complex64
        )
        for i in range(len(images)):
            final_ksp[i] = self.fourier_ops[i].op(images[i])
        return final_ksp

    def adj_op(self, coeffs):
        """Apply Adjoint Operator."""
        c = 1 if self.pinned_smaps is not None else self.n_coils
        final_image = np.empty((self.n_frames, c, *self.shape), dtype=np.complex64)
        for i in range(len(coeffs)):
            final_image[i] = self.fourier_ops[i].adj_op(coeffs[i])
        return final_image

    def get_grad(self, data, obs_data):
        """Compute the gradient operation."""
        c = 1 if self.pinned_smaps is not None else self.n_coils
        final_image = np.empty((self.n_frames, c, *self.shape), dtype=np.complex64)

        for i in range(len(data)):
            ksp = self.fourier_ops[i].op(data[i])
            final_image[i] = self.fourier_ops[i].adj_op(ksp - obs_data[i])
        return final_image

    def __getattr__(self, attrName):
        """Pass the attributes to the first operator."""
        return getattr(self.fourier_ops[0], attrName)


class FFT_Sense(SpaceFourierBase):
    """Apply the FFT with potential Smaps support.

    Parameters
    ----------
    shape : tuple
        Shape of the image.
    n_coils : int
        Number of coils.
    mask : array
        Mask of the image.
    smaps : array
        Sensitivity maps.
    """

    def __init__(self, shape, n_coils, mask, smaps):
        self.shape = shape
        self.n_coils = n_coils
        self.mask = mask
        self.smaps = smaps

    @property
    def uses_sense(self):
        """True if operator uses smaps."""
        return self.smaps is not None

    def op(self, img):
        """Apply the forward operator."""
        axes = tuple(range(-len(self.shape), 0))
        if self.n_coils > 1:
            if self.smaps is not None:
                img2 = np.repeat(img[np.newaxis, ...], self.n_coils, axis=0)
                img2 *= self.smaps
                ksp = fft(img2, axis=axes)
            else:
                ksp = fft(img, axis=axes)
            return ksp * self.mask[np.newaxis, ...]
        else:
            return fft(img, axis=axes) * self.mask

    def adj_op(self, ksp):
        """Apply the adjoint operator."""
        axes = tuple(range(-len(self.shape), 0))
        if self.n_coils > 1:
            img = ifft(ksp, axis=axes)
            if self.smaps is None:
                return img
            return np.sum(img * np.conj(self.smaps), axis=0)
        else:
            return ifft(ksp, axis=axes)
