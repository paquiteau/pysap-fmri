"""Fourier Operator for fMRI data."""

import numpy as np

from .fft import fft, ifft


class CartesianSpaceFourier:
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
        self.uses_sense = smaps is not None

    def op(self, img):
        axes = tuple(range(-len(self.shape), 0))
        print("global", axes, img.shape)
        if self.n_coils > 1:
            if self.smaps is not None:
                img2 = np.repeat(img[:, np.newaxis, ...], self.n_coils, axis=1)
                img2 *= self.smaps
                ksp = fft(img2, axis=axes)
            else:
                ksp = fft(img, axis=axes)
            print("global", ksp.shape, self.mask.shape)
            return ksp * self.mask[:, np.newaxis, ...]
        else:
            ksp = fft(img, axis=axes)
            print("global", ksp.shape, self.mask.shape, "n_coils=1")
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


class RepeatOperator:
    def __init__(self, fourier_ops):
        self.fourier_ops = fourier_ops

    def op(self, data):
        return np.asarray(
            [
                self.fourier_ops[i].op(data[i]).copy()
                for i in range(len(self.fourier_ops))
            ]
        )

    def adj_op(self, data):
        return np.asarray(
            [
                self.fourier_ops[i].adj_op(data[i]).copy()
                for i in range(len(self.fourier_ops))
            ]
        )


class FFT_Sense:
    def __init__(self, shape, n_coils, mask, smaps):
        self.shape = shape
        self.n_coils = n_coils
        self.mask = mask
        self.smaps = smaps
        self.uses_sense = smaps is not None

    def op(self, img):
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
        axes = tuple(range(-len(self.shape), 0))
        if self.n_coils > 1:
            img = ifft(ksp, axis=axes)
            if self.smaps is None:
                return img
            return np.sum(img * np.conj(self.smaps), axis=0)
        else:
            return ifft(ksp, axis=axes)
